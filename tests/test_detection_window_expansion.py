"""Only verified, explicit multi-slot requests expand model tool arguments."""
import unittest
from unittest.mock import patch
from pydantic_ai.agent import CallToolsNode
from pydantic_ai.messages import ModelResponse, ToolCallPart
from agent_app.config import Settings
from agent_app.service import Session, _expand_explicit_detection_window, ask
from tests.test_latency_paths import ToolModel, detection_data


def node_for(**overrides):
    args = {'search': 'Kirchberg', 'start_utc': '2026-10-22T22:00:00Z',
            'end_utc': '2026-10-22T23:00:00Z', 'max_slots': 1, 'direct_answer': False}
    args.update(overrides)
    return CallToolsNode(model_response=ModelResponse(parts=[ToolCallPart('detect_meteors', args, 'call1')]))


class DetectionWindowExpansionTests(unittest.IsolatedAsyncioTestCase):
    def test_expands_matching_explicit_hour_but_not_one_recording(self):
        prompt = 'Check Kirchberg station for meteors on 2026-10-22 from 22:00 to 23:00 UTC.'
        node = node_for()
        event = _expand_explicit_detection_window(node, prompt, [])
        self.assertEqual(event['model_max_slots'], 1)
        self.assertEqual(event['executed_max_slots'], 12)
        self.assertEqual(node.model_response.parts[0].args_as_dict()['max_slots'], 12)
        node = node_for()
        self.assertIsNone(_expand_explicit_detection_window(node, 'Check just one spectrogram for meteors at Kirchberg from 22:00 to 23:00 UTC.', []))
        self.assertEqual(node.model_response.parts[0].args_as_dict()['max_slots'], 1)

    def test_does_not_expand_mismatched_or_unverified_relative_scope(self):
        node = node_for(search='Humain')
        self.assertIsNone(_expand_explicit_detection_window(node,
            'Check Kirchberg station for meteors on 2026-10-22 from 22:00 to 23:00 UTC.', []))
        node = node_for()
        self.assertIsNone(_expand_explicit_detection_window(node,
            'Check Kirchberg station for meteors yesterday from 10-11 pm.', []))

    def test_relative_range_requires_matching_clock_and_time(self):
        clock = [{'event': 'tool_result', 'name': 'get_current_utc_time',
                  'result': {'utc_time': '2026-10-23T12:00:00Z'}}]
        prompt = 'Check Kirchberg station for meteors yesterday from 10-11 pm.'
        node = node_for()
        self.assertEqual(_expand_explicit_detection_window(node, prompt, clock)['executed_max_slots'], 12)
        node = node_for(start_utc='2026-10-22T10:00:00Z', end_utc='2026-10-22T11:00:00Z')
        self.assertIsNone(_expand_explicit_detection_window(node, prompt, clock))

    def test_explicit_limits_and_negation_never_expand(self):
        base = 'Check Kirchberg station for meteors on 2026-10-22 from 22:00 to 23:00 UTC'
        for suffix in (', no more than three recordings.', ', but do not scan the full hour.',
                       ', stop after one hit.', ', scan two recordings.', ', use a few recordings.'):
            with self.subTest(suffix=suffix):
                node = node_for()
                self.assertIsNone(_expand_explicit_detection_window(node, base + suffix, []))
                self.assertEqual(node.model_response.parts[0].args_as_dict()['max_slots'], 1)

    def test_an_explicit_artifact_or_referenced_image_is_not_expanded(self):
        node = node_for(artifact_id='a' * 32)
        prompt = 'Check Kirchberg station for meteors on 2026-10-22 from 22:00 to 23:00 UTC.'
        self.assertIsNone(_expand_explicit_detection_window(node, prompt, []))
        node = node_for()
        self.assertIsNone(_expand_explicit_detection_window(node,
            'Analyze this spectrogram for meteors at Kirchberg from 22:00 to 23:00 UTC on 2026-10-22.', []))

    def test_textual_date_must_match_before_expansion(self):
        prompt = 'Check Kirchberg station for meteors on 22 October 2026 from 22:00 to 23:00 UTC.'
        self.assertEqual(_expand_explicit_detection_window(node_for(), prompt, [])['executed_max_slots'], 12)
        wrong_day = node_for(start_utc='2026-10-23T22:00:00Z', end_utc='2026-10-23T23:00:00Z')
        self.assertIsNone(_expand_explicit_detection_window(wrong_day, prompt, []))

    def test_expands_explicit_receiver_id_when_max_slots_omitted(self):
        prompt = 'Check BEHUMA_SYS001 for meteor echoes on 2023-12-14 from 21:00 to 22:00 UTC.'
        omitted = {'search': 'BEHUMA_SYS001', 'start_utc': '2023-12-14T21:00:00Z',
                   'end_utc': '2023-12-14T22:00:00Z', 'direct_answer': True}
        node = CallToolsNode(model_response=ModelResponse(parts=[ToolCallPart('detect_meteors', omitted, 'call_id')]))
        event = _expand_explicit_detection_window(node, prompt, [])
        self.assertEqual((event['model_max_slots'], event['executed_max_slots']), (1, 12))
        self.assertEqual(node.model_response.parts[0].args_as_dict()['max_slots'], 12)
        wrong_receiver = {**omitted, 'search': 'BEGENT_SYS001'}
        wrong_node = CallToolsNode(model_response=ModelResponse(parts=[ToolCallPart('detect_meteors', wrong_receiver, 'call_bad')]))
        self.assertIsNone(_expand_explicit_detection_window(wrong_node, prompt, []))

    async def test_actual_tool_executes_expanded_count_and_records_adjustment(self):
        args = {'search': 'Kirchberg', 'start_utc': '2026-10-22T22:00:00Z',
                'end_utc': '2026-10-22T23:00:00Z', 'max_slots': 1, 'direct_answer': False}
        fake = ToolModel('detect_meteors', args)
        data = detection_data()
        with patch('agent_app.brams_tools.pybrams_setup_status', return_value={'ready': True, 'version': 'test'}), \
             patch('agent_app.brams_tools.vision_setup_status', return_value={'ready': True}), \
             patch('agent_app.brams_tools._detect_meteors', return_value=data) as detector:
            reply = await ask('Check Kirchberg station for meteors on 2026-10-22 from 22:00 to 23:00 UTC.',
                              Session(), Settings(), transport=fake.transport())
        self.assertEqual(detector.call_args.args[3], 12)
        adjustment = next(e for e in reply.trace if e.get('event') == 'orchestration_adjustment')
        self.assertEqual((adjustment['model_max_slots'], adjustment['executed_max_slots']), (1, 12))


if __name__ == '__main__': unittest.main()
