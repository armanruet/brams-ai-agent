"""Synthetic checks for v2 evaluation mechanics, not new model performance results."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from pydantic_ai import ModelRetry

from agent_app import benchmark, brams_tools
from agent_app.benchmark_audit import scientific_flags
from agent_app.benchmark_data import SnapshotTools, bound_arguments, call_matches, score_turn
from agent_app.benchmark_reporting import write_report
from agent_app.benchmark_resources import ResourceSampler
from agent_app.benchmark_values import fact_value, value_mentioned
from agent_app.service import Reply, Session


def fixture_reply(trace=(), text="fixture answer"):
    return Reply(text, Session(), list(trace), .1, 2, 12, 8, model_text=text)


class ValueTests(unittest.TestCase):
    def test_scientific_notation_and_grouped_values(self):
        for text in ("2.368 × 10⁻⁶", "2.368e-6", "2.368 x10⁻⁶"):
            self.assertTrue(value_mentioned(2.367889200171257e-6, text), text)
        self.assertFalse(value_mentioned(2.367889200171257e-6, "2.368"))
        for text in ("49 969 000 Hz", "49 969 000 Hz", "49,969,000 Hz", "49.969 MHz"):
            self.assertTrue(value_mentioned(49969000, text, unit="Hz"), text)
        self.assertTrue(value_mentioned(1000, "1 kHz", unit="Hz"))
        self.assertFalse(value_mentioned(1000, "1 MHz", unit="Hz"))
        self.assertFalse(value_mentioned(5512.5, "5512 Hz", unit="Hz", rel_tol=0, abs_tol=1e-6))

    def test_timezone_equivalence_without_accepting_wrong_date_or_naive_time(self):
        expected = "2023-12-15T22:30:00Z"
        for text in ("2023-12-15 22:30 UTC", "2023‑12‑15 22:30:00 Z", "2023-12-15T23:30:00+01:00"):
            self.assertTrue(value_mentioned(expected, text), text)
        for text in ("2023-12-14 22:30 UTC", "2023-12-15 23:30 UTC", "2023-12-15 22:30"):
            self.assertFalse(value_mentioned(expected, text), text)
        self.assertFalse(value_mentioned("2023-12-14T21:15:00.809161Z", "2023-12-14T21:15:00.809Z"))

    def test_select_requested_slot_not_first_boundary_row(self):
        result = {"recordings": [{"slot_start_utc": "21:10", "duration_seconds": 300.135873},
                                  {"slot_start_utc": "21:15", "duration_seconds": 300.226576}]}
        fact = {"path": "duration_seconds", "select": {"collection": "recordings", "field": "slot_start_utc", "equals": "21:15"}}
        self.assertEqual(fact_value(result, fact), 300.226576)
        with self.assertRaises(KeyError):
            fact_value(result, {**fact, "select": {**fact["select"], "equals": "missing"}})

    def test_all_bank_ids_and_conversations_exist(self):
        cases = benchmark.selected_cases(benchmark.DEFAULT_CASES)
        self.assertEqual(len(cases), 54)
        self.assertEqual(sum(len(case["turns"]) for case in cases), 66)
        document = (benchmark.PROJECT_ROOT / "QUESTIONS.md").read_text()
        self.assertEqual(set(re.findall(r"^\| ([A-F]\d{2}) \|", document, re.M)), {case["id"] for case in cases})
        self.assertEqual(len(benchmark.selected_cases(benchmark.PROJECT_ROOT / "benchmarks/holdout.json")), 8)
        self.assertTrue(all(len(case["turns"]) == 2 for case in cases if case["tier"] in {"C", "F"}))
        self.assertEqual(len(benchmark.selected_cases(benchmark.DEFAULT_CASES, tiers=["E"])), 4)
        d07 = next(case for case in cases if case["id"] == "D07")
        recording = next(call for call in d07["turns"][0]["required_calls"] if call["tool"] == "get_recordings")
        self.assertEqual(recording["arguments"]["start_utc"], "2023-12-15T22:30:00Z")

    def test_valid_alternative_and_auxiliary_miss_are_separate_from_target(self):
        wanted = {"tool": "list_receivers", "arguments": {}}
        trace = [{"event": "tool_call", "name": "get_beacon", "arguments": {}, "call_id": "aux"},
                 {"event": "tool_result", "name": "get_beacon", "call_id": "aux", "result": {"status": "error", "error_code": "benchmark_unmatched_call"}},
                 {"event": "tool_call", "name": "list_receivers", "arguments": {}, "call_id": "gold"},
                 {"event": "tool_result", "name": "list_receivers", "call_id": "gold", "result": {"status": "ok", "count": 3}}]
        turn = {"required_calls": [wanted], "facts": [], "kind": "tool", "require_source": False}
        score = score_turn(turn, fixture_reply(trace))
        self.assertTrue(score["tool_plan_pass"])
        self.assertFalse(score["strict_tool_plan_pass"])
        self.assertEqual(score["snapshot_misses"], 1)
        alternative = {**turn, "required_calls": [{"tool": "get_current_utc_time", "arguments": {}}], "acceptable_call_sets": [[wanted]]}
        self.assertTrue(score_turn(alternative, fixture_reply(trace))["tool_plan_pass"])
        self.assertFalse(score_turn(turn, fixture_reply(trace[:-1]))["tool_plan_pass"])

    def test_source_label_is_not_a_citation_and_reference_facts_are_independent(self):
        turn = {"required_calls": [], "facts": [{"tool": "list_receivers", "arguments": {}, "path": "count", "label": "count"}], "kind": "recall", "require_source": True}
        reference = SnapshotTools({"entries": [{"tool": "list_receivers", "arguments": {}, "result": {"status": "ok", "count": 3}}]})
        reply = fixture_reply([{"event": "tool_result", "name": "list_receivers", "result": {"status": "ok", "count": 99, "source": "PyBRAMS Receiver.get_all"}}], "99 according to PyBRAMS Receiver.get_all")
        score = score_turn(turn, reply, reference=reference)
        self.assertEqual(score["fact_presence_heuristic"][0]["expected"], 3)
        self.assertFalse(score["fact_presence_heuristic"][0]["mentioned"])
        self.assertFalse(score["source_mentioned"])

    def test_claim_checks_flag_misclassification_and_wrong_samples(self):
        result = {"status": "ok", "scope": "hourly_recording_availability", "receivers": [{"receiver_id": "BEHUMA_SYS001", "coverage": "unknown"}],
                  "sample_count": 1654999, "header_summary": {"nominal_beacon_minus_lo_hz": 1000}}
        trace = [{"result": result}]
        text = "Receivers with recordings (full):\n| BEHUMA_SYS001 | Humain |\n16,549,999 samples; nominal −1 kHz"
        flags = scientific_flags(text, trace, {})
        self.assertEqual({flag["code"] for flag in flags}, {"receiver_coverage_mismatch", "sample_count_mismatch", "nominal_offset_mismatch"})
        self.assertEqual(scientific_flags("1,654,999 samples; nominal 1 kHz", trace, {}), [])

    def test_catalog_aliases_use_complete_captured_rows_without_inventing_metadata(self):
        receivers = [{"receiver_id": f"BEHUMA_SYS00{i}", "name": f"Humain {i}", "location_code": "BEHUMA",
                      "antenna": i, "catalog_start": "2015-04-23", "catalog_end": "2026-11-01"} for i in (1, 2)]
        replay = SnapshotTools({"captured_at_utc": "2026-11-01T00:00:00Z", "entries": [{
            "tool": "list_receivers", "arguments": {}, "result": {"status": "ok", "count": 2, "receivers": receivers,
                                                                       "source_url": "https://example.invalid/receiver.php"}}]})
        result = replay.result("list_receivers", {"search": "Humain"})
        self.assertEqual(result["count"], 2)
        self.assertTrue(result["derived_from_captured_catalog"])
        expected = {"tool": "list_receivers", "arguments": {"search": "BEHUMA"}}
        actual = {"name": "list_receivers", "arguments": {"search": "Humain"}}
        self.assertTrue(call_matches(expected, actual, replay))
        # Missing coordinates are never inferred from a different station.
        self.assertEqual(replay.result("list_receivers", {"search": "Humain", "detail": "coordinates"})["error_code"], "benchmark_unmatched_call")
        self.assertEqual(replay.result("list_receivers", {"search": "BEFAKE"})["count"], 0)


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_and_replay_reject_same_semantic_invalid_inputs(self):
        invalid = [("get_recordings", {"search": "BEHUMA", "start_utc": "2023-12-14T00:00:00Z", "end_utc": "2023-12-16T00:00:00Z"}),
                   ("list_receivers", {"limit": 30}),
                   ("get_psd", {"search": "BEHUMA", "start_utc": "2023-12-14T21:15:00Z", "end_utc": "2023-12-14T21:20:00Z", "noise_low_hz": 900, "noise_high_hz": 800})]
        for name, arguments in invalid:
            with self.assertRaises(ModelRetry) as live:
                await getattr(brams_tools, name)(**bound_arguments(name, arguments))
            with self.assertRaises(ModelRetry) as replay:
                SnapshotTools({"entries": []}).result(name, arguments)
            self.assertEqual(str(live.exception), str(replay.exception))

    async def test_replay_marks_capture_provenance_without_mutating_reference(self):
        args = {"search": "BEHUMA", "start_utc": "2023-12-14T21:15:00Z", "end_utc": "2023-12-14T21:20:00Z"}
        result = {"status": "ok", "download_origin": "fresh archive download", "retrieved_at_utc": "2026-11-01T13:16:08Z"}
        replay = SnapshotTools({"captured_at_utc": "2026-11-02T00:00:00Z", "entries": [{"tool": "get_wav", "arguments": args, "result": result}]})
        returned = replay.result("get_wav", args)
        self.assertFalse(returned["benchmark_provenance"]["live_query_made"])
        self.assertEqual(returned["benchmark_provenance"]["captured_at_utc"], result["retrieved_at_utc"])
        self.assertIn("no fresh download", returned["download_origin"])
        self.assertEqual(result["download_origin"], "fresh archive download")

    async def test_memory_sampling_is_optional_and_never_reports_missing_as_zero(self):
        with patch("agent_app.benchmark_resources.process_rss", return_value={}):
            async with ResourceSampler(123) as sampler:
                pass
        self.assertIsNone(sampler.result()["server"]["peak_sampled_mib"])
        self.assertEqual(sampler.result()["server"]["status"], "unavailable")
        import os
        with patch("agent_app.benchmark_resources.process_rss", side_effect=[{os.getpid(): 100, 123: 250}, {os.getpid(): 110, 123: 260}]):
            async with ResourceSampler(123) as sampler:
                pass
        self.assertEqual(sampler.result()["server"]["peak_sampled_mib"], 260)
        self.assertEqual(sampler.result()["app"]["samples"], 2)

    async def test_capture_can_reuse_real_reference_rows_without_archive_requests(self):
        async def catalog(search: str = "", detail: str = "summary", limit: int = 20, offset: int = 0) -> dict:
            raise AssertionError("Reused catalog must not query the archive.")
        case = {"id": "test", "turns": [{"required_calls": [{"tool": "list_receivers", "arguments": {}}],
                                         "facts": [{"tool": "list_receivers", "path": "count", "label": "count"}]}]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            benchmark.write_json(root / "cases.json", {"schema_version": 1, "cases": [case]})
            benchmark.write_json(root / "seed.json", {"schema_version": 1, "kind": "real_tool_capture", "captured_at_utc": "2026-11-01T00:00:00Z",
                                "entries": [{"tool": "list_receivers", "arguments": {}, "result": {"status": "ok", "count": 3}}]})
            with patch.dict(benchmark.TOOLS, {"list_receivers": catalog}):
                await benchmark.capture([case], root / "reference.json", root / "cases.json", root / "seed.json")
            data = benchmark.read_json(root / "reference.json")
            self.assertEqual(data["entries"][0]["capture_source"], "reused_real_snapshot")


class ReportTests(unittest.TestCase):
    def row(self, repetition, turn, **extra):
        return dict(model_id="fixture", case_id="F01", category="conversation", repetition=repetition, turn=turn,
                    status="completed", elapsed_seconds=2, prompt="synthetic", model_answer="fixture", displayed_answer="fixture",
                    score={"native_completion": True, "tool_plan_pass": True}, **extra)

    def test_human_quality_stays_ungraded_and_consistency_requires_complete_matrix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            benchmark.write_json(root / "manifest.json", {"repetitions": 2})
            benchmark.write_json(root / "cases.json", {"cases": [{"id": "F01", "tier": "F", "turns": [{}, {}]}]})
            rows = [self.row(1, 1), self.row(1, 2), self.row(2, 1)]
            path = root / "turns.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            summary = write_report(root)[0]
            self.assertIsNone(summary["model_reviewed_pass_rate"])
            self.assertIsNone(benchmark.read_json(root / "consistency.json")[0]["native_workflow_all_trials"])
            with path.open("a") as stream:
                stream.write(json.dumps(self.row(2, 2)) + "\n")
            write_report(root)
            self.assertTrue(benchmark.read_json(root / "consistency.json")[0]["native_workflow_all_trials"])
            with (root / "manual_review.csv").open(newline="") as stream:
                reader = csv.DictReader(stream); fields, reviews = reader.fieldnames, list(reader)
            for review in reviews:
                review.update(reviewer="Synthetic reviewer", model_verdict="PASS", assistant_verdict="PASS", notes="preserve")
            with (root / "manual_review.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(reviews)
            self.assertEqual(write_report(root)[0]["model_reviewed_pass_rate"], 1)
            self.assertTrue(benchmark.read_json(root / "consistency.json")[0]["model_reviewed_pass_all_trials"])

    def test_failed_preflight_is_visible_even_without_any_case_turns(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            benchmark.write_json(root / "manifest.json", {"selected_models": ["failed_model"], "model_status": {"failed_model": {"status": "preflight_failed"}}})
            (root / "turns.jsonl").touch()
            result = write_report(root)[0]
            self.assertEqual(result["preflight_status"], "preflight_failed")
            self.assertEqual(result["attempted_turns"], 0)
            self.assertIsNone(result["native_completion_rate"])

    def test_timeout_fallback_marks_usage_incomplete_and_panel_is_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = self.row(1, 1, timings=[{"kind": "model", "status": "cancelled"}])
            row.update(elapsed_seconds=180, model_answer='</script><img src=x onerror="alert(1)">')
            row["score"].update(native_completion=False, fallback=True)
            (root / "turns.jsonl").write_text(json.dumps(row) + "\n")
            summary = write_report(root)[0]
            self.assertTrue(summary["usage_may_be_incomplete"])
            self.assertEqual(summary["timeout_turns"], 1)
            page = (root / "report.html").read_text()
            self.assertNotIn('<img src=x', page)
            self.assertNotIn('.innerHTML', page)
            self.assertIn('\\u003c/script>', page)
            scripts = re.findall(r'<script(?: [^>]*)?>(.*?)</script>', page, re.S)
            self.assertEqual(json.loads(scripts[0])["rows"][0]["model_answer"], row["model_answer"])
            if shutil.which("node"):
                script = root / "panel.js"; script.write_text(scripts[-1])
                result = subprocess.run(["node", "--check", str(script)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_offline_rescore_preserves_originals_and_makes_no_requests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); original = root / "original"; original.mkdir()
            benchmark.write_json(original / "manifest.json", {"config": {"models": []}, "repetitions": 1})
            benchmark.write_json(original / "cases.json", {"cases": [{"id": "F01", "turns": [{"required_calls": [], "facts": [], "require_source": False, "kind": "recall"}]}]})
            path = original / "turns.jsonl"; path.write_text(json.dumps(self.row(1, 1)) + "\n")
            before = path.read_bytes()
            with patch("agent_app.benchmark.ask", new=AsyncMock()) as model:
                benchmark.rescore(original, root / "rescored")
                model.assert_not_called()
            self.assertEqual(path.read_bytes(), before)
            self.assertTrue((root / "rescored/report.html").exists())
            self.assertEqual(benchmark.read_json(root / "rescored/manifest.json")["rescoring"]["new_api_calls"], 0)
