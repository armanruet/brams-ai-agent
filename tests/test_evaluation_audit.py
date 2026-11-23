"""Regressions for current-capability evaluation and synthetic evidence."""
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_app import benchmark
from agent_app.benchmark_data import score_turn
from agent_app.benchmark_scenarios import controlled_tools, fixture_exercised
from agent_app.harness import registered_tools
from agent_app.service import Reply, Session, ask
from tests.test_latency_paths import ToolModel

ROOT = Path(__file__).resolve().parents[1]

class EvaluationAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_archive_failure_seed_and_scope_are_valid_end_to_end(self):
        for filename in ('current_capabilities_failures_v2.json',):
            case = benchmark.selected_cases(ROOT/'benchmarks'/filename, names=['CURF01'])[0]
            turn=case['turns'][0]; args=turn['required_calls'][0]['arguments']
            self.assertEqual(case['fixture_catalog_seed']['receiver_id'], args['search'])
            functions=controlled_tools(list(registered_tools()),case)
            lookup={fn.__name__: fn for fn in functions}
            listing=await lookup['list_receivers'](search=args['search'])
            self.assertEqual(listing['count'],1)
            class FailureModel(ToolModel):
                def respond(self,payload):
                    code,data=super().respond(payload)
                    if payload['messages'][-1]['role']=='tool':
                        data['choices'][0]['message']={'role':'assistant','content':'The archive request failed; availability is unknown.'}
                    return code,data
            fake=FailureModel('get_spectrogram',args)
            reply=await ask(turn['prompt'],Session(),benchmark.Settings(),transport=fake.transport(),tool_functions=functions)
            returned=next(e for e in reply.trace if e.get('event')=='tool_result')
            self.assertEqual(returned['result']['query'],args['search'])
            self.assertIn('2023-12-14T21:15',returned['result']['start_utc'])
            self.assertTrue(fixture_exercised(case,reply.trace))
            self.assertNotIn('grounding_retry',[e['event'] for e in reply.trace])
            self.assertFalse(reply.images)

    async def test_controlled_availability_and_partial_detection_are_labeled(self):
        cases=benchmark.selected_cases(ROOT/'benchmarks/current_capabilities_failures_v2.json')
        for case in cases[1:]:
            t=case['turns'][0]; call=t['required_calls'][0]
            functions={fn.__name__:fn for fn in controlled_tools(list(registered_tools()),case)}
            args=benchmark.bound_arguments(call['tool'],call['arguments'])
            result=await functions[call['tool']](**args)
            self.assertEqual(result['evidence_mode'],'controlled_fixture')
            self.assertFalse(result['live_query_made'])
            self.assertFalse(result['downloaded'])
            if case['controlled_fixture']=='availability_zero': self.assertEqual(result['receivers'][0]['coverage'],'none')
            elif case['controlled_fixture'].startswith('availability_'): self.assertEqual(result['receivers'][0]['coverage'],'unknown')
            else: self.assertFalse(result['coverage_complete'])

    def test_visible_scientific_flags_are_separate_from_model_flags(self):
        turn={'prompt':'Please delete the cache','required_calls':[],'facts':[],'kind':'unsupported','require_source':False}
        score=score_turn(turn,Reply('I have deleted the cache.',Session(),[],.1,0,0,0))
        self.assertEqual(score['scientific_review_flags'],[])
        self.assertIn('unsupported_deletion_claim',[x['code'] for x in score['visible_scientific_review_flags']])

    def test_snapshot_media_accepts_one_complete_alternative(self):
        def call(id):return {'tool':'get_spectrogram','arguments':{'id':id}}
        turn={'required_calls':[call('a')], 'acceptable_call_sets':[[call('b')]],'expected_media':['image']}
        cases=[{'id':'ALT','turns':[turn]}]
        lookup=SimpleNamespace(result=lambda tool,args: {'status':'ok','downloaded':True,'artifact_id':args['id']})
        with patch('agent_app.benchmark.spectrogram_path',side_effect=lambda id: Path('/fixture') if id=='b' else None):
            benchmark._validate_snapshot_media(cases,lookup)
        with patch('agent_app.benchmark.spectrogram_path',return_value=None):
            with self.assertRaisesRegex(ValueError,'No model requests were sent'):
                benchmark._validate_snapshot_media(cases,lookup)

    def test_suite_v2_has_requested_current_cases(self):
        cases=benchmark.selected_cases(ROOT/'benchmarks/current_capabilities_v2.json')
        self.assertTrue({'geographic_activity','geographic_spectrogram','multi_slot_detection'} <= {c['category'] for c in cases})

    def test_suite_v3_extends_v2_with_wav_and_dst_cases(self):
        v2 = benchmark.selected_cases(ROOT / 'benchmarks/current_capabilities_v2.json')
        v3 = benchmark.selected_cases(ROOT / 'benchmarks/current_capabilities_v3.json')
        self.assertEqual([c['id'] for c in v3[:len(v2)]], [c['id'] for c in v2])
        self.assertTrue({'wav_audio', 'time_dst'} <= {c['category'] for c in v3})

if __name__=='__main__': unittest.main()
