"""Controlled failure evidence stays labeled and cannot execute live tools."""
import copy
from datetime import datetime, timezone
from functools import wraps
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic_ai import ModelRetry
from agent_app import benchmark
from agent_app.benchmark_scenarios import controlled_tools, fixture_exercised
from agent_app.config import Settings
from agent_app.service import Session, ask
from tests.fake_model import FakeModel, SPECTROGRAM_PROMPT


class ScenarioTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cases = benchmark.selected_cases(Path(__file__).parents[1]/"benchmarks/conversation_geo_full.json")
        self.window = dict(search="BEHALL_SYS001", start_utc=datetime(2026,11,6,22,tzinfo=timezone.utc),
                           end_utc=datetime(2026,11,6,23,tzinfo=timezone.utc))

    def functions(self, case):
        def forbidden(**kwargs):
            raise AssertionError("Controlled scenario escaped to live I/O")
        functions = [wraps(fn)(lambda **kwargs: forbidden(**kwargs)) for fn in benchmark.TOOLS.values()]
        return {fn.__name__:fn for fn in controlled_tools(functions,case)}

    def test_full_matrix_counts_and_fixture_labels(self):
        self.assertEqual(len(self.cases),10)
        self.assertEqual(sum(len(c['turns']) for c in self.cases)*3*4,204)
        self.assertEqual([c['id'] for c in self.cases if c.get('controlled_fixture')],['P08','P09','P10'])
        with self.assertRaises(ValueError):
            benchmark.validate_fixture({'id':'bad','controlled_fixture':'pretend_success'})

    async def test_all_non_time_tools_are_intercepted_and_preserve_signatures(self):
        for case in self.cases[-3:]:
            functions = self.functions(case)
            for name, fn in functions.items():
                if name in {'get_current_utc_time','inspect_time_window','convert_local_time'}:
                    continue
                self.assertEqual(inspect.signature(fn),inspect.signature(benchmark.TOOLS[name]))
                kwargs = {}
                for key, param in inspect.signature(fn).parameters.items():
                    if key in self.window: kwargs[key]=self.window[key]
                    elif key == "country": kwargs[key] = "Belgium"
                    elif key == "direction": kwargs[key] = "north"
                    elif key == "action": kwargs[key] = "spectrogram"
                    elif param.default is inspect.Parameter.empty: self.fail(key)
                result = await fn(**kwargs)
                self.assertFalse(result['live_query_made'])
                self.assertFalse(result['downloaded'])
                self.assertEqual(result['evidence_mode'],'controlled_fixture')

    async def test_fixture_distinctions_pagination_validation_and_seed_integrity(self):
        none, missing, failed = [self.functions(c) for c in self.cases[-3:]]
        self.assertEqual((await none['list_receivers']())['count'],0)
        original=copy.deepcopy(self.cases[-2]['fixture_catalog_seed'])
        result=await missing['list_receivers'](detail='coordinates')
        result['receivers'][0]['coordinates']['latitude_degrees']=0
        self.assertEqual(self.cases[-2]['fixture_catalog_seed'],original)
        self.assertEqual((await missing['list_receivers'](offset=1))['receivers'],[])
        with self.assertRaises(ModelRetry): await missing['list_receivers'](limit=50)
        self.assertEqual((await missing['get_spectrogram'](**self.window))['status'],'no_spectrogram_found')
        self.assertEqual((await missing['get_wav'](**self.window))['status'],'unknown')
        self.assertEqual((await failed['get_spectrogram'](**self.window))['status'],'error')
        self.assertFalse(fixture_exercised(self.cases[-2],[]))

    async def test_real_harness_records_missing_png_without_attaching_fake_media(self):
        case=self.cases[-2]
        with patch('agent_app.brams_tools._load_spectrogram',side_effect=AssertionError('live archive called')):
            reply=await ask(SPECTROGRAM_PROMPT,Session(),Settings(),transport=FakeModel().transport(),
                            tool_functions=controlled_tools(list(benchmark.TOOLS.values()),case))
        self.assertTrue(fixture_exercised(case,reply.trace))
        self.assertEqual(reply.images,[])
        self.assertIn('no_spectrogram_found',reply.text)

    async def test_fixture_capture_is_rejected_before_any_tool_call(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'not real-tool captures'):
                await benchmark.capture(self.cases,Path(temp)/'ref.json',Path('unused.json'))
            self.assertEqual(list(Path(temp).iterdir()),[])

    def test_report_splits_live_and_controlled_denominators_and_keeps_ungraded_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            benchmark.write_json(root/'cases.json',{'cases':[
                {'id':'live','turns':[{},{}]}, {'id':'controlled','turns':[{}]}]})
            benchmark.write_json(root/'manifest.json',{'selected_models':['ornith'],'repetitions':3,
                'case_ids':['live','controlled'],'case_turn_counts':{'live':2,'controlled':1},
                'case_evidence_modes':{'live':'live','controlled':'controlled_fixture'},'mode':'live'})
            rows=[]
            for case, mode in [('live','live'),('controlled','controlled_fixture')]:
                rows.append({'model_id':'ornith','case_id':case,'repetition':1,'turn':1,'category':'test',
                    'prompt':'fixture','status':'completed','evidence_mode':mode,'elapsed_seconds':1,
                    'score':{'native_completion':True,'controlled_fixture_exercised':mode=='controlled_fixture'}})
            import json
            (root/'turns.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            benchmark.report(root)
            report=benchmark.read_json(root/'summary.json')
            groups={r['evidence_mode']:r for r in report['protocols']}
            self.assertEqual(groups['live']['planned_turns'],6)
            self.assertEqual(groups['controlled_fixture']['planned_turns'],3)
            self.assertEqual(groups['controlled_fixture']['controlled_fixture_exercised_turns'],1)
            self.assertIsNone(groups['live']['model_reviewed_pass_rate'])
            self.assertIn('trajectory_0_to_2',(root/'manual_review.csv').read_text())
