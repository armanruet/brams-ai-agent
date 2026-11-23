"""Regression tests for archive gaps and per-recording detection failures."""
from datetime import timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent_app.brams_sdk import _detect_meteors, _load_availability
from agent_app.recovery import evidence_summary, meteor_summary
from tests.test_brams_sdk import START, END, CODE, catalog
from tests import test_meteor_detection as detection_fixtures

SCAN_START = detection_fixtures.START


class AvailabilityPresenceTests(unittest.TestCase):
    def run_query(self, entries, *, end=END):
        from pybrams import Receiver
        with patch.object(Receiver, 'get_all', return_value=catalog()), \
             patch('pybrams.utils.http.post', return_value=SimpleNamespace(json=lambda: entries)):
            return _load_availability('Kirchberg', START, end)

    def test_omitted_receiver_is_unknown_not_zero(self):
        result = self.run_query([{'receiver_code': 'OTHER_SYS001', 'datetime': '2026-10-22 00:00:00', 'availability': 1}])
        self.assertEqual(result['receivers'][0]['coverage'], 'unknown')
        self.assertIsNone(result['receivers'][0]['availability_percent'])
        self.assertEqual(result['coverage_counts']['unknown'], 1)

    def test_missing_hour_preserves_positive_evidence_without_average(self):
        result = self.run_query([{'receiver_code': CODE, 'datetime': '2026-10-22 00:00:00', 'availability': 1}], end=END + timedelta(hours=1))
        row = result['receivers'][0]
        self.assertEqual(row['coverage'], 'unknown')
        self.assertIsNone(row['availability_percent'])
        self.assertEqual([hour['availability_fraction'] for hour in row['hours']], [1, None])
        self.assertTrue(row['known_recordings_in_queried_hours'])
        self.assertEqual(result['receivers_with_recordings'], 1)
        self.assertTrue(result['receivers_with_recordings_is_lower_bound'])

    def test_explicit_zero_remains_zero(self):
        result = self.run_query([{'receiver_code': CODE, 'datetime': '2026-10-22 00:00:00', 'availability': 0}])
        self.assertEqual(result['receivers'][0]['coverage'], 'none')
        self.assertEqual(result['receivers'][0]['availability_percent'], 0)

    def test_empty_and_http_failure_never_claim_zero(self):
        self.assertEqual(self.run_query([])['status'], 'no_availability_information')
        from pybrams import Receiver
        with patch.object(Receiver, 'get_all', return_value=catalog()), \
             patch('pybrams.utils.http.post', side_effect=RuntimeError('HTTP failure')):
            with self.assertRaises(RuntimeError):
                _load_availability('Kirchberg', START, END)

    def test_conflicting_duplicates_are_unknown(self):
        rows = [{'receiver_code': CODE, 'datetime': '2026-10-22 00:00:00', 'availability': val} for val in (1, 0)]
        self.assertEqual(self.run_query(rows)['receivers'][0]['coverage'], 'unknown')


class SlotIsolationTests(unittest.TestCase):
    def analysis(self):
        return {'meteor_detected': True, 'meteor_count': 1, 'high_confidence_count': 1,
                'max_confidence': 0.8, 'detections': [{'echo_index': 1, 'confidence': 0.8}],
                'inference_ms': 2, 'model_name': 'fixture', 'model_architecture': 'fixture',
                'backend': 'fixture'}

    def run_scan(self, side_effect, *, overlay_side_effect=None):
        helpers = detection_fixtures.MeteorDetectionSdkAndToolTests()
        stream = helpers._make_fake_stream(b'png', 3)
        with helpers._patch_fresh_sdk(), \
             patch('agent_app.brams_sdk._archive_files', return_value=stream), \
             patch('agent_app.vision.analyze_spectrogram_bytes', side_effect=side_effect) as infer, \
             patch('agent_app.vision.create_annotated_overlay', side_effect=overlay_side_effect or (lambda *a, **k: 'a' * 32)):
            result = _detect_meteors('Kirchberg', SCAN_START, SCAN_START + timedelta(minutes=15), max_slots=3)
        return result, infer.call_count

    def test_failed_middle_inference_retains_other_slots(self):
        result, calls = self.run_scan([self.analysis(), RuntimeError('failed'), self.analysis()])
        self.assertEqual(calls, 3)
        self.assertEqual(result['inspected_slots_count'], 2)
        self.assertEqual(result['total_meteor_count'], 2)
        self.assertFalse(result['coverage_complete'])
        self.assertEqual(result['slot_errors'][0]['stage'], 'inference')
        self.assertEqual(len(result['uncovered_slot_starts_utc']), 1)
        self.assertIn('could not be analyzed', evidence_summary([{'event':'tool_result','name':'detect_meteors','result':result}]))

    def test_all_failed_inference_is_unknown(self):
        result, calls = self.run_scan([RuntimeError('failed') for _ in range(3)])
        self.assertEqual(calls, 3)
        self.assertEqual(result['status'], 'error')
        self.assertFalse(result['analysis_performed'])
        self.assertNotIn('total_meteor_count', result)
        self.assertEqual([e['stage'] for e in result['slot_errors']], ['inference'] * 3)

    def test_overlay_failure_preserves_analysis_and_coverage(self):
        result, calls = self.run_scan([self.analysis() for _ in range(3)], overlay_side_effect=[RuntimeError('render failed'), 'a'*32, 'b'*32])
        self.assertEqual(calls, 3)
        self.assertTrue(result['coverage_complete'])
        self.assertEqual(result['total_meteor_count'], 3)
        self.assertEqual(len(result['annotated_artifacts']), 2)
        self.assertEqual(result['overlay_errors'][0]['stage'], 'overlay')
        self.assertIn('annotated image could not be created', meteor_summary(result))


if __name__ == '__main__':
    unittest.main()
