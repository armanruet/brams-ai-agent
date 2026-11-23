from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from pydantic_ai import ModelRetry

from agent_app.benchmark_data import canonical, score_turn
from agent_app.brams_sdk import _archive_files, _select_recording, _select_recordings
from agent_app.grounding import check_unbacked_claims, requires_tool_or_history
from agent_app.recovery import meteor_summary
from agent_app.service import Reply, Session, _shortcut_time_verified
from agent_app.tools import convert_local_time, get_current_utc_time
from agent_app.vision import nms_boxes


class IntegrityRegressionTests(unittest.TestCase):
    def test_grounding_needs_the_requested_domain(self):
        clock = [{"event": "tool_result", "name": "get_current_utc_time", "result": {"utc_time": "2026-11-11T00:00:00Z"}}]
        self.assertTrue(requires_tool_or_history("How many stations are in the southern part of Belgium?", clock))
        self.assertTrue(requires_tool_or_history("Help me check Kirchberg availability", clock))
        wrong = clock + [{"event": "tool_result", "name": "check_availability", "result": {"status": "ok"}}]
        self.assertTrue(requires_tool_or_history("Show a spectrogram and how many stations are there?", wrong))
        incorrect_time = [{"event": "tool_result", "name": "get_spectrogram", "result": {
            "status": "ok", "query": "Kirchberg", "start_utc": "2026-11-10T10:00:00Z",
            "end_utc": "2026-11-10T10:05:00Z"}}]
        self.assertTrue(requires_tool_or_history("Show the Kirchberg station spectrogram on 2026-11-10 from 09:00 to 09:05 UTC", incorrect_time))
        incorrect_region = [{"event": "tool_result", "name": "query_geographic_receivers", "result": {
            "status": "ok", "direction": "north", "action": "count", "site_count": 10}}]
        self.assertTrue(requires_tool_or_history("How many stations are in southern Belgium?", incorrect_region))
        region_image = [{"event": "tool_result", "name": "query_geographic_receivers", "result": {
            "status": "ok", "direction": "north", "action": "spectrogram"}}]
        self.assertTrue(requires_tool_or_history("Show a spectrogram in northern Belgium and how many stations are there?", region_image))

    def test_url_must_come_from_a_tool_result(self):
        trace = [{"event": "tool_result", "name": "list_receivers", "result": {"source_url": "https://trusted.example"}}]
        issues = check_unbacked_claims("stations", "https://trusted.example.evil.invalid/path", trace)
        self.assertIn("invented_url", [issue["code"] for issue in issues])

    def test_shortcut_requires_matching_clock_and_conversions(self):
        result = {"start_utc": "2026-11-10T09:00:00Z", "end_utc": "2026-11-10T09:05:00Z"}
        self.assertFalse(_shortcut_time_verified("Show yesterday's spectrogram", [], result))
        clock = [{"event": "tool_result", "name": "get_current_utc_time", "result": {"utc_time": "2026-11-11T12:00:00Z"}}]
        self.assertTrue(_shortcut_time_verified("Show yesterday's spectrogram", clock, result))
        self.assertFalse(_shortcut_time_verified("Show today's spectrogram", clock, result))
        self.assertFalse(_shortcut_time_verified("Show the spectrogram at Brussels time", clock, result))

    def test_daylight_saving_elapsed_duration_and_fixed_abbreviations(self):
        result = convert_local_time("2026-10-25T01:30:00", "Europe/Brussels", end_local_datetime="2026-10-25T03:30:00")
        self.assertEqual(result["duration_minutes"], 180)
        self.assertEqual(convert_local_time("2026-07-01T12:00:00", "CET")["utc_time"], "2026-07-01T11:00:00Z")
        self.assertEqual(convert_local_time("2026-01-01T12:00:00", "CEST")["utc_time"], "2026-01-01T10:00:00Z")
        with self.assertRaises(ModelRetry):
            convert_local_time("2026-10-25T01:30:00", "Europe/Brussels", end_local_datetime="2026-10-25T02:30:00")
        result = convert_local_time("2026-10-25T01:30:00", "Europe/Brussels", end_local_datetime="2026-10-25T02:30:00", end_fold=1)
        self.assertEqual(result["duration_minutes"], 120)
        self.assertIn("local_date", get_current_utc_time("Europe/Brussels"))

    def test_boundary_spillover_is_not_a_recording_for_the_next_slot(self):
        start = datetime(2026, 11, 10, 6, tzinfo=timezone.utc)
        prev = SimpleNamespace(receiver_code="X", start=start-timedelta(minutes=5),
                               end=start+timedelta(milliseconds=300), png_url="x")
        self.assertIsNone(_select_recording(iter([prev]), start, start+timedelta(minutes=5), "X", "png_url"))
        selected, count = _select_recordings(iter([prev]), start, start+timedelta(minutes=5), "X")
        self.assertEqual((selected, count), ([], 0))
        self.assertIs(_select_recording(iter([prev]), start-timedelta(minutes=3), start, "X", "png_url"), prev)

    def test_empty_archive_listing_differs_from_http_failure(self):
        from pybrams.utils.http import HttpRequestError
        api = SimpleNamespace(base_url="https://example.invalid/")
        config = SimpleNamespace(get=lambda *args: "file.php")
        start = datetime(2026, 11, 10, tzinfo=timezone.utc)
        with patch("pybrams.utils.http.post", return_value=SimpleNamespace(json=lambda: [])):
            self.assertEqual(list(_archive_files(api, config, start, start+timedelta(minutes=5), "X")), [])
        with patch("pybrams.utils.http.post", side_effect=HttpRequestError("404", status_code=404)):
            with self.assertRaises(HttpRequestError):
                list(_archive_files(api, config, start, start+timedelta(minutes=5), "X"))

    def test_detector_cap_and_summary_count(self):
        boxes = np.array([[i*3, 0, i*3+1, 1] for i in range(201)], dtype=float)
        kept, truncated = nms_boxes(boxes, np.arange(201), max_det=200, return_truncation=True)
        self.assertEqual(len(kept), 200)
        self.assertTrue(truncated)
        data = {"meteor_detected": True, "total_meteor_count": 30, "detection_limit_reached": True,
                "slots": [{"meteor_count": 30, "detections": [{"echo_index": i, "confidence": .5} for i in range(1, 26)]}]}
        answer = meteor_summary(data)
        self.assertIn("plus 25 more", answer)
        self.assertIn("At least 30", answer)
        self.assertIn("output limit", answer)

    def test_benchmark_direct_answer_key_and_visible_score(self):
        args = {"country": "Belgium", "direction": "south", "action": "count"}
        self.assertEqual(canonical("query_geographic_receivers", args),
                         canonical("query_geographic_receivers", {**args, "direct_answer": False}))
        from pydantic_ai.messages import ModelResponse, TextPart
        session = Session([ModelResponse(parts=[TextPart("There are 12 sites.")], metadata={"direct_query_geographic_receivers_summary": True})])
        trace = [{"event": "tool_result", "name": "query_geographic_receivers", "result": {"site_count": 12}}]
        reply = Reply("There are 12 sites.", session, trace, 0, 1, 0, 0)
        turn = {"required_calls": [], "facts": [{"tool": "query_geographic_receivers", "path": "site_count"}],
                "require_source": False, "kind": "normal"}
        score = score_turn(turn, reply)
        self.assertFalse(score["fact_presence_heuristic"][0]["mentioned"])
        self.assertTrue(score["visible_fact_presence_heuristic"][0]["mentioned"])
        self.assertTrue(score["direct_answer_completion"])


if __name__ == "__main__":
    unittest.main()
