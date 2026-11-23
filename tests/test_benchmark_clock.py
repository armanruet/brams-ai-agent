"""Clock-fixture isolation and evidence labeling; no measured model calls."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch
from agent_app.benchmark import TOOLS, fixed_clock, tools_with_clock
from agent_app.service import Session, ask
from agent_app.config import Settings
from tests.fake_model import FakeModel


class BenchmarkClockTests(unittest.IsolatedAsyncioTestCase):
    def test_clock_requires_offset_and_normalizes_utc(self):
        self.assertEqual(fixed_clock("2026-11-07T13:00:00+01:00"), "2026-11-07T12:00:00Z")
        with self.assertRaises(ValueError):
            fixed_clock("2026-11-07T12:00:00")
        self.assertIsNone(tools_with_clock(None, None))

    async def test_real_harness_uses_labeled_fixture_without_mutating_production(self):
        original = TOOLS["get_current_utc_time"]
        clock = "2026-11-07T12:00:00Z"
        reply = await ask("CLOCK", Session(), Settings(), transport=FakeModel().transport(),
                          tool_functions=tools_with_clock(None, clock))
        result = next(row["result"] for row in reply.trace if row["event"] == "tool_result"
                      and row["name"] == "get_current_utc_time")
        self.assertEqual(result["utc_time"], clock)
        self.assertEqual(result["source"], "benchmark_fixed_clock")
        self.assertFalse(result["live_clock_read"])
        self.assertIs(TOOLS["get_current_utc_time"], original)
        self.assertEqual(original()["source"], "local_system_clock")

    def test_override_preserves_non_clock_functions_and_pilot_counts(self):
        functions = list(TOOLS.values())
        wrapped = tools_with_clock(functions, "2026-11-07T12:00:00Z")
        for old, new in zip(functions, wrapped):
            if old.__name__ != "get_current_utc_time":
                self.assertIs(old, new)
        from pathlib import Path
        from agent_app.benchmark import selected_cases
        cases = selected_cases(Path(__file__).parents[1]/"benchmarks/conversation_geo_pilot.json")
        self.assertEqual(len(cases), 3)
        self.assertEqual(sum(len(case["turns"]) for case in cases), 6)
        self.assertEqual(cases[2]["turns"][0]["prompt"], cases[2]["turns"][1]["prompt"])

    async def test_failed_turn_keeps_observed_trace_and_fixture_manifest(self):
        from agent_app import benchmark
        error = RuntimeError("fixture failure after a tool attempt")
        error.tool_trace = [{"event": "tool_call", "name": "get_current_utc_time", "arguments": {}, "call_id": "fixture"}]
        error.timings = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case = {"id": "fixture", "category": "test", "turns": [{"prompt": "CLOCK",
                    "kind": "tool", "required_calls": [], "facts": [], "require_source": False}]}
            benchmark.write_json(root/"cases.json", {"schema_version": 1, "cases": [case]})
            args = benchmark.parser().parse_args(["run", "--mode", "live", "--cases", str(root/"cases.json"),
                    "--output", str(root/"run"), "--repetitions", "1", "--clock", "2026-11-07T12:00:00Z"])
            config = benchmark.configuration(benchmark.DEFAULT_CONFIG)
            with patch.object(benchmark, "check_server", new=AsyncMock(return_value={})), \
                 patch.object(benchmark, "warmup", new=AsyncMock(return_value={})), \
                 patch.object(benchmark, "ask", new=AsyncMock(side_effect=error)):
                code = await benchmark.run(args, config, [config["models"][0]], [case])
            self.assertEqual(code, 1)
            row = json.loads((root/"run/turns.jsonl").read_text())
            self.assertEqual(row["trace"], error.tool_trace)
            self.assertEqual(row["status"], "error")
            manifest = benchmark.read_json(root/"run/manifest.json")
            self.assertEqual(manifest["clock_override_utc"], "2026-11-07T12:00:00Z")
