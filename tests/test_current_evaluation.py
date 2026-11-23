"""Current-capability suite and scoring remain separate from historical cases."""
from pathlib import Path
import unittest
from pydantic_ai.messages import ModelResponse, TextPart

from agent_app import benchmark
from agent_app.benchmark_data import SnapshotTools, call_matches, score_turn
from agent_app.service import Reply, Session

ROOT = Path(__file__).resolve().parents[1]


class CurrentEvaluationTests(unittest.TestCase):
    def test_new_suites_validate_and_keep_failures_separate(self):
        current = benchmark.selected_cases(ROOT / 'benchmarks/current_capabilities_v1.json')
        failures = benchmark.selected_cases(ROOT / 'benchmarks/current_capabilities_failures_v1.json')
        self.assertTrue(any(c['category'] == 'meteor_detection' for c in current))
        self.assertTrue(any(c['category'] == 'geographic_count' for c in current))
        self.assertTrue(all(not c.get('controlled_fixture') for c in current))
        self.assertEqual(failures[0]['controlled_fixture'], 'archive_error')

    def test_detector_attachment_is_scored_independently(self):
        turn = {'kind': 'tool', 'required_calls': [], 'facts': [], 'require_source': False,
                'expected_media': ['detection']}
        reply = Reply('Detected.', Session(), [], .1, 1, 10, 5)
        self.assertFalse(score_turn(turn, reply)['media_delivery_pass'])
        reply.detections = [{'path': '/tmp/example.png', 'label': 'Detection'}]
        score = score_turn(turn, reply)
        self.assertTrue(score['media_delivery_pass'])
        self.assertEqual(score['detection_attachments'], 1)

    def test_direct_detection_details_are_not_graded_as_model_output(self):
        turn = {'kind': 'recall', 'required_calls': [], 'facts': [], 'require_source': False}
        session = Session(messages=[ModelResponse(parts=[TextPart('details')],
                          metadata={'direct_detection_details': True})])
        score = score_turn(turn, Reply('details', session, [], .1, 0, 0, 0))
        self.assertTrue(score['direct_answer_completion'])
        self.assertFalse(score['native_completion'])

    def test_model_plan_does_not_get_credit_for_expanded_window(self):
        args = {'search': 'BEHUMA_SYS001', 'start_utc': '2023-12-14T21:00:00Z',
                'end_utc': '2023-12-14T22:00:00Z', 'max_slots': 12}
        turn = {'kind': 'tool', 'required_calls': [{'tool': 'detect_meteors', 'arguments': args}],
                'facts': [], 'require_source': False, 'expected_media': []}
        trace = [{'event': 'tool_call', 'name': 'detect_meteors', 'call_id': 'scan', 'arguments': args},
                 {'event': 'orchestration_adjustment', 'name': 'detect_meteors', 'call_id': 'scan',
                  'model_max_slots': 1, 'executed_max_slots': 12},
                 {'event': 'tool_result', 'name': 'detect_meteors', 'call_id': 'scan',
                  'result': {'status': 'ok'}}]
        score = score_turn(turn, Reply('Done.', Session(), trace, .1, 1, 10, 5))
        self.assertTrue(score['tool_plan_pass'])
        self.assertFalse(score['model_tool_plan_pass'])
        self.assertEqual(score['orchestration_adjustments'], 1)

    def test_grounding_flags_separate_model_and_corrected_visible_answer(self):
        turn = {'kind': 'tool', 'prompt': 'Check availability for LUKIRC_SYS001.',
                'required_calls': [], 'facts': [], 'require_source': False}
        trace = [{'event': 'tool_result', 'name': 'check_availability', 'call_id': 'one',
                  'result': {'status': 'ok', 'recording_availability_checked': True,
                             'receivers': [{'receiver_id': 'LUKIRC_SYS001',
                             'name': 'Kirchberg', 'coverage': 'none', 'availability_percent': 0}]}}]
        reply = Reply('Kirchberg archive coverage is none (0%).', Session(), trace, .1, 1, 10, 5,
                      model_text='Coverage is reported as UNKNOWN-equivalent "none" for the hour.')
        score = score_turn(turn, reply)
        self.assertIn('known_zero_reported_as_unknown',
                      [item['code'] for item in score['model_grounding_review_flags']])
        self.assertNotIn('known_zero_reported_as_unknown',
                         [item['code'] for item in score['visible_grounding_review_flags']])

    def test_expected_failure_is_not_scored_as_unexpected(self):
        args = {'search': 'BEHUMA_SYS001', 'start_utc': '2023-12-14T21:15:00Z',
                'end_utc': '2023-12-14T21:20:00Z'}
        turn = {'kind': 'tool', 'required_calls': [{'tool': 'get_spectrogram', 'arguments': args,
                'expected_statuses': ['error']}], 'facts': [], 'require_source': False,
                'expected_media': []}
        trace = [{'event': 'tool_call', 'name': 'get_spectrogram', 'call_id': 'one', 'arguments': args},
                 {'event': 'tool_result', 'name': 'get_spectrogram', 'call_id': 'one',
                  'result': {'status': 'error'}}]
        score = score_turn(turn, Reply('Unknown.', Session(), trace, .1, 1, 10, 5))
        self.assertTrue(score['strict_tool_plan_pass'])
        self.assertEqual(score['tool_errors'], 1)
        self.assertEqual(score['unexpected_tool_errors'], 0)
        self.assertIsNone(score['media_delivery_pass'])

    def test_captured_default_detection_threshold_matches_explicit_value(self):
        args = {'search': 'BEHUMA_SYS001', 'start_utc': '2023-12-14T21:15:00Z',
                'end_utc': '2023-12-14T21:20:00Z'}
        replay = SnapshotTools({'entries': [{'tool': 'detect_meteors', 'arguments': args,
                    'result': {'status': 'ok', 'confidence_threshold': .25}}]})
        explicit = {**args, 'confidence_threshold': .25}
        self.assertEqual(replay.result('detect_meteors', explicit)['status'], 'ok')
        self.assertTrue(call_matches({'tool': 'detect_meteors', 'arguments': args},
                        {'name': 'detect_meteors', 'arguments': explicit}, replay))
        self.assertEqual(replay.result('detect_meteors', {**args, 'confidence_threshold': .3})['status'], 'error')

    def test_larger_slot_limit_matches_only_with_verified_single_slot_coverage(self):
        args = {'search': 'BEHUMA_SYS001', 'start_utc': '2023-12-14T21:15:00Z',
                'end_utc': '2023-12-14T21:20:00Z'}
        result = {'status': 'ok', 'confidence_threshold': .25, 'coverage_complete': True,
                  'available_slots_in_window': 1, 'inspected_slots_count': 1,
                  'expected_slots_in_window': 1}
        replay = SnapshotTools({'entries': [{'tool': 'detect_meteors', 'arguments': args, 'result': result}]})
        more = {**args, 'max_slots': 12, 'confidence_threshold': .25}
        self.assertEqual(replay.result('detect_meteors', more)['status'], 'ok')
        self.assertTrue(call_matches({'tool': 'detect_meteors', 'arguments': args},
                        {'name': 'detect_meteors', 'arguments': more}, replay))
        result['coverage_complete'] = False
        replay = SnapshotTools({'entries': [{'tool': 'detect_meteors', 'arguments': args, 'result': result}]})
        self.assertEqual(replay.result('detect_meteors', more)['status'], 'error')

    def test_detection_defaults_match_across_captured_receiver_aliases(self):
        args = {'search': 'Kirchberg', 'start_utc': '2023-12-14T21:15:00Z',
                'end_utc': '2023-12-14T21:20:00Z'}
        receiver = {'receiver_id': 'LUKIRC_SYS001', 'name': 'Kirchberg', 'location_code': 'LUKIRC',
                    'antenna': 1, 'catalog_start': '2020', 'catalog_end': '2026'}
        result = {'status': 'ok', 'confidence_threshold': .25, 'coverage_complete': True,
                  'available_slots_in_window': 1, 'inspected_slots_count': 1, 'expected_slots_in_window': 1}
        replay = SnapshotTools({'entries': [
            {'tool': 'list_receivers', 'arguments': {}, 'result': {'status': 'ok', 'count': 1,
             'receivers': [receiver], 'source_url': 'https://example.invalid/catalog'}},
            {'tool': 'detect_meteors', 'arguments': args, 'result': result}]})
        explicit = {**args, 'search': 'LUKIRC_SYS001', 'confidence_threshold': .25, 'max_slots': 12}
        self.assertEqual(replay.result('detect_meteors', explicit)['status'], 'ok')
        self.assertTrue(call_matches({'tool': 'detect_meteors', 'arguments': args},
                        {'name': 'detect_meteors', 'arguments': explicit}, replay))
        self.assertEqual(replay.result('detect_meteors', {**explicit, 'confidence_threshold': .3})['status'], 'error')


if __name__ == '__main__':
    unittest.main()
