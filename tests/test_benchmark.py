"""Synthetic protocol/unit tests only; these are not measured model benchmarks."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from agent_app import benchmark
from agent_app.benchmark_data import SnapshotTools, canonical, score_turn, value_mentioned
from agent_app.config import Settings, load_settings
from agent_app.service import Reply, Session, ask
from tests.fake_model import FakeModel


def fixture_snapshot():
    return {"entries": [{"tool": "get_current_utc_time", "arguments": {},
                         "result": {"utc_time": "2026-11-01T12:00:00Z", "source": "synthetic_clock_fixture"}},
                        {"tool": "list_receivers", "arguments": {},
                         "result": {"status": "ok", "count": 3, "location_count": 2,
                                    "receivers": [{"receiver_id": str(index)} for index in range(3)],
                                    "offset": 0, "limit": 20, "has_more": False,
                                    "source_url": "https://example.invalid/receivers"}}]}


class BenchmarkDataTests(unittest.TestCase):
    def test_local_time_iso_format_does_not_change_canonical_call(self):
        short = {"local_datetime": "2026-10-25T01:30", "timezone_name": "Europe/Brussels",
                 "end_local_datetime": "2026-10-25T03:30"}
        seconds = {**short, "local_datetime": "2026-10-25T01:30:00",
                   "end_local_datetime": "2026-10-25T03:30:00"}
        self.assertEqual(canonical("convert_local_time", short), canonical("convert_local_time", seconds))
        self.assertNotEqual(canonical("convert_local_time", short), canonical("convert_local_time", {
            **seconds, "end_local_datetime": "2026-10-25T02:30:00"}))

    def test_aware_times_and_search_are_canonical(self):
        first = canonical("check_availability", {"search": " Kirchberg ", "start_utc": "2026-10-22T02:00:00+02:00", "end_utc": "2026-10-22T03:00:00+02:00"})
        second = canonical("check_availability", {"search": "kirchberg", "start_utc": "2026-10-22T00:00:00Z", "end_utc": "2026-10-22T01:00:00Z"})
        self.assertEqual(first, second)

    def test_snapshot_copies_evidence_and_never_queries_archive(self):
        replay = SnapshotTools(fixture_snapshot())
        result = replay.result("list_receivers", {"limit": 1})
        self.assertEqual(len(result["receivers"]), 1)
        self.assertEqual(result["count"], 3)
        result["receivers"].clear()
        self.assertEqual(len(replay.result("list_receivers", {})["receivers"]), 3)
        missing = replay.result("get_beacon", {"emitter_code": "UNKNOWN"})
        self.assertEqual(missing["error_code"], "benchmark_unmatched_call")
        self.assertEqual(len(replay.misses), 1)

    def test_scoring_does_not_credit_python_verified_evidence(self):
        self.assertFalse(value_mentioned(1e-14, "Noise PSD is 0"))
        self.assertTrue(value_mentioned(1.4678e-10, "Noise PSD is 1.47e-10"))
        result = fixture_snapshot()["entries"][1]["result"]
        turn = {"kind": "tool", "required_calls": [{"tool": "list_receivers", "arguments": {}}],
                "facts": [{"tool": "list_receivers", "path": "count", "label": "count"}], "require_source": True}
        reply = Reply("3 https://example.invalid/receivers", Session(), [
            {"event": "tool_call", "name": "list_receivers", "arguments": {"limit": 1}},
            {"event": "tool_result", "name": "list_receivers", "result": result},
        ], .1, 2, 12, 8, model_text="I checked the catalog.")
        score = score_turn(turn, reply)
        self.assertTrue(score["tool_plan_pass"])
        self.assertFalse(score["source_mentioned"])
        self.assertFalse(score["fact_presence_heuristic"][0]["mentioned"])
        reply.model_text = "3 receivers. https://example.invalid/receivers"
        self.assertTrue(score_turn(turn, reply)["fact_presence_heuristic"][0]["mentioned"])

    def test_config_uses_environment_and_rejects_inline_keys(self):
        config = benchmark.configuration(benchmark.DEFAULT_CONFIG)
        model = {"id": "public", "model": "example-model", "base_url": "https://example.invalid/v1",
                 "provider": "compatible", "api_key_env": "BENCHMARK_TEST_KEY"}
        with patch.dict("os.environ", {"BENCHMARK_TEST_KEY": "test-secret"}):
            settings = benchmark.settings_for(config, model)
            self.assertEqual(settings.api_key, "test-secret")
            self.assertNotIn("test-secret", repr(settings))
            self.assertEqual(benchmark.sanitize({"error": "key test-secret"}, [settings.api_key]), {"error": "key [redacted]"})
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            benchmark.settings_for(config, {**model, "base_url": "http://example.invalid/v1"}, require_key=False)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            config["models"][0]["api_key"] = "do-not-save"
            benchmark.write_json(path, config)
            with self.assertRaisesRegex(ValueError, "environment"):
                benchmark.configuration(path)

    def test_dataset_is_valid_and_includes_conversation_cases(self):
        cases = benchmark.selected_cases(benchmark.DEFAULT_CASES)
        self.assertEqual(len(cases), 54)
        self.assertEqual(sum(len(case["turns"]) for case in cases), 66)
        self.assertEqual(len(benchmark.selected_cases(benchmark.LEGACY_CASES)), 20)
        self.assertEqual(len(benchmark.selected_models(benchmark.configuration(benchmark.DEFAULT_CONFIG))), 3)

    def test_invalid_key_reference_explains_env_setup_without_echoing_value(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            for invalid in ("literal-api-key-should-stay-private", None, 17, ["bad"], ""):
                config = benchmark.configuration(benchmark.DEFAULT_CONFIG)
                config["models"][-1]["api_key_env"] = invalid
                benchmark.write_json(path, config)
                with self.assertRaises(ValueError) as raised:
                    benchmark.configuration(path)
                message = str(raised.exception)
                self.assertIn("environment-variable NAME", message)
                self.assertIn("BENCHMARK_PUBLIC_API_KEY", message)
                self.assertIn(".env", message)
                self.assertNotIn("literal-api-key-should-stay-private", message)

    def test_reports_include_failures_and_preserve_review_annotations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = {"model_id": "fixture", "case_id": "one", "repetition": 1, "turn": 1,
                   "status": "completed", "elapsed_seconds": 2, "score": {"native_completion": True, "tool_plan_pass": True}}
            path = root / "turns.jsonl"
            path.write_text(json.dumps(row) + "\n")
            benchmark.report(root)
            review = root / "manual_review.csv"
            with review.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields, reviews = reader.fieldnames, list(reader)
            reviews[0]["notes"] = "Keep my annotation"
            with review.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(reviews)
            second = {**row, "case_id": "two", "status": "error", "score": {"native_completion": False, "tool_plan_pass": False}}
            with path.open("a") as stream:
                stream.write(json.dumps(second) + "\n")
            summary = benchmark.report(root)[0]
            self.assertEqual(summary["native_completion_rate"], .5)
            self.assertEqual(summary["completed_turn_p95_seconds"], 2)
            with review.open(newline="") as stream:
                reviews = list(csv.DictReader(stream))
            self.assertEqual(len(reviews), 2)
            self.assertEqual(reviews[0]["notes"], "Keep my annotation")


class BenchmarkProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_snapshot_explains_capture_step_without_model_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = root / "custom reference.json"
            args = benchmark.parser().parse_args([
                "run", "--snapshot", str(reference), "--output", str(root / "run"),
                "--models", "ornith",
            ])
            config = benchmark.configuration(benchmark.DEFAULT_CONFIG)
            with patch("agent_app.benchmark.ask", new=AsyncMock()) as request:
                for incomplete in (False, True):
                    if incomplete:
                        Path(str(reference) + ".partial").write_text("{}")
                    with self.assertRaises(ValueError) as raised:
                        await benchmark.run(args, config, [config["models"][0]], [])
                    message = str(raised.exception)
                    self.assertIn("reference snapshot not found", message)
                    self.assertIn("agent_app.benchmark capture", message)
                    self.assertIn(str(reference), message)
                    self.assertIn("No model requests were sent", message)
                    if incomplete:
                        self.assertIn("incomplete capture", message)
                request.assert_not_called()
            self.assertFalse((root / "run").exists())

    async def test_failed_reference_is_not_published_as_gold_data(self):
        async def failing_catalog(search: str = "", detail: str = "summary", limit: int = 20, offset: int = 0) -> dict:
            return {"status": "error", "count": None}
        case = {"id": "capture_error", "turns": [{"required_calls": [{"tool": "list_receivers", "arguments": {}}], "facts": []}]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = root / "cases.json"
            cases.write_text("{}")
            with patch.dict(benchmark.TOOLS, {"list_receivers": failing_catalog}):
                with self.assertRaisesRegex(ValueError, "Reference capture failed"):
                    await benchmark.capture([case], root / "reference.json", cases)
            self.assertFalse((root / "reference.json").exists())
            self.assertTrue((root / "reference.json.partial").exists())

    async def test_public_batch_requires_explicit_inclusion_before_any_calls(self):
        args = benchmark.parser().parse_args(["run", "--mode", "live"])
        config = benchmark.configuration(benchmark.DEFAULT_CONFIG)
        public = {"id": "public", "provider": "compatible"}
        with patch("agent_app.benchmark.ask", new=AsyncMock()) as request:
            with self.assertRaisesRegex(ValueError, "include-public"):
                await benchmark.run(args, config, [public], [])
            request.assert_not_called()

    async def test_replay_keeps_actual_tool_schemas_and_validation(self):
        fake = FakeModel()
        replay = SnapshotTools(fixture_snapshot())
        reply = await ask("CLOCK", Session(), Settings(), transport=fake.transport(), tool_functions=replay.functions())
        self.assertIn("synthetic_clock_fixture", reply.model_text)
        self.assertEqual(reply.trace[-1]["result"]["utc_time"], "2026-11-01T12:00:00Z")
        schemas = {row["function"]["name"]: row["function"]["parameters"] for row in fake.requests[0]["tools"]}
        self.assertIn("start_utc", schemas["get_psd"]["properties"])
        bad = await ask("BAD_ARGS", Session(), Settings(), transport=FakeModel().transport(), tool_functions=replay.functions())
        self.assertTrue(any(row["event"] == "validation_retry" for row in bad.trace))

    async def test_limits_are_configurable_without_changing_defaults(self):
        self.assertEqual((Settings().request_limit, Settings().tool_calls_limit), (6, 6))
        with patch.dict("os.environ", {"LLAMA_REQUEST_LIMIT": "12", "LLAMA_TOOL_CALL_LIMIT": "12"}):
            self.assertEqual((load_settings().request_limit, load_settings().tool_calls_limit), (12, 12))
        reply = await ask("DISTINCT_LOOP", Session(), Settings(request_limit=2, tool_calls_limit=2), transport=FakeModel().transport())
        self.assertEqual(reply.requests, 2)
        self.assertIsNone(reply.model_text)
        self.assertTrue(any(row["event"] == "recovery" for row in reply.trace))

    async def test_full_runner_checkpoint_and_report_with_synthetic_model(self):
        class ClockFixture(FakeModel):
            def respond(self, payload):
                payload = deepcopy(payload)
                for message in payload["messages"]:
                    if message["role"] == "user":
                        message["content"] = "CLOCK"
                return super().respond(payload)
        fake = ClockFixture()
        real_ask = ask
        async def request(*args, **kwargs):
            kwargs["transport"] = fake.transport()
            return await real_ask(*args, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = {"id": "synthetic_clock", "category": "test", "turns": [{
                "prompt": "CLOCK", "kind": "tool", "required_calls": [{"tool": "get_current_utc_time", "arguments": {}}],
                "facts": [], "require_source": True}]}
            cases_path = root / "cases.json"
            benchmark.write_json(cases_path, {"schema_version": 1, "cases": [case]})
            snapshot_path = root / "reference.json"
            # Deliberately confined to tests: synthetic references are never shipped as real data.
            benchmark.write_json(snapshot_path, {**fixture_snapshot(), "schema_version": 1, "kind": "real_tool_capture",
                                                "cases_sha256": benchmark.digest(cases_path)})
            args = benchmark.parser().parse_args(["run", "--cases", str(cases_path), "--snapshot", str(snapshot_path),
                                                   "--output", str(root / "run"), "--models", "ornith", "--repetitions", "1"])
            config = benchmark.configuration(benchmark.DEFAULT_CONFIG)
            with patch("agent_app.benchmark.ask", side_effect=request), patch("agent_app.benchmark.check_server", new=AsyncMock(return_value={"selected_model_found": True})):
                code = await benchmark.run(args, config, [config["models"][0]], [case])
            self.assertEqual(code, 0)
            manifest = benchmark.read_json(root / "run/manifest.json")
            self.assertEqual(manifest["status"], "finished")
            rows = (root / "run/turns.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 1)
            self.assertTrue(json.loads(rows[0])["score"]["tool_plan_pass"])
            self.assertTrue((root / "run/manual_review.csv").exists())

    async def test_preflight_rejects_text_only_model(self):
        with self.assertRaisesRegex(ValueError, "preflight failed"):
            await benchmark.warmup(Settings(), transport=FakeModel().transport())
