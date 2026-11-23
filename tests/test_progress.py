"""Live observations, real timing boundaries and responsive monitor callbacks."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from pydantic_ai.exceptions import ModelHTTPError

from agent_app.config import Settings
from agent_app.presentation import activity_view, history_view
from agent_app.progress import RunProgress
from agent_app.service import Session, Reply, ask
from tests.fake_model import FakeModel


class ProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_tool_start_is_published_before_it_finishes(self):
        started = asyncio.Event()
        release = asyncio.Event()
        events = []
        async def get_current_utc_time() -> dict:
            """Read the clock after a controlled synthetic delay."""
            started.set()
            await release.wait()
            from agent_app.tools import get_current_utc_time as clock
            return clock()
        fake = FakeModel()
        with patch("agent_app.harness.get_current_utc_time", new=get_current_utc_time):
            task = asyncio.create_task(ask("CLOCK", Session(), Settings(), transport=fake.transport(), on_event=events.append))
            try:
                await asyncio.wait_for(started.wait(), 2)
                self.assertFalse(task.done())
                self.assertEqual(events[-1]["kind"], "tool")
                self.assertEqual(events[-1]["status"], "running")
                await asyncio.sleep(.02)
            finally:
                release.set()
            reply = await task
        self.assertEqual([step["kind"] for step in reply.timings], ["model", "tool", "model"])
        self.assertGreaterEqual(reply.timings[1]["duration_seconds"], .02)
        self.assertTrue(all(step["duration_seconds"] >= 0 for step in reply.timings))
        self.assertLessEqual(sum(step["duration_seconds"] for step in reply.timings), reply.elapsed_seconds)
        self.assertEqual([step["status"] for step in events], ["running", "completed"]*3)
        self.assertEqual(len(fake.requests), 2)
        self.assertEqual(len(reply.trace), 2)  # Timing never changes the canonical tool trace.

    async def test_model_failure_keeps_duration_and_failed_request(self):
        fake = FakeModel()
        with self.assertRaises(ModelHTTPError) as caught:
            await ask("HTTP_ERROR", Session(), Settings(), transport=fake.transport())
        timings = caught.exception.timings
        self.assertEqual(len(timings), 1)
        self.assertEqual(timings[0]["status"], "error")
        self.assertGreaterEqual(caught.exception.elapsed_seconds, timings[0]["duration_seconds"])

    async def test_recovery_records_blocked_repeat_without_external_work(self):
        reply = await ask("LOOP", Session(), Settings(), transport=FakeModel().transport())
        tools = [step for step in reply.timings if step["kind"] == "tool"]
        self.assertEqual([step["status"] for step in tools], ["completed", "blocked"])
        self.assertEqual(sum(event["event"] == "tool_result" for event in reply.trace), 1)

    async def test_invalid_arguments_never_report_a_tool_execution(self):
        reply = await ask("BAD_ARGS", Session(), Settings(), transport=FakeModel().transport())
        tools = [step["name"] for step in reply.timings if step["kind"] == "tool"]
        self.assertEqual(tools, ["get_current_utc_time"])
        self.assertIn("validation_retry", [event["event"] for event in reply.trace])

    async def test_concurrent_runs_have_independent_observers(self):
        events_a, events_b = [], []
        await asyncio.gather(
            ask("CLOCK", Session(), Settings(), transport=FakeModel().transport(), on_event=events_a.append),
            ask("DURATION", Session(), Settings(), transport=FakeModel().transport(), on_event=events_b.append))
        self.assertEqual({step["name"] for step in events_a if step["kind"] == "tool"}, {"get_current_utc_time"})
        self.assertEqual({step["name"] for step in events_b if step["kind"] == "tool"}, {"inspect_time_window"})

    async def test_credentials_redacted_html_escaped_and_observers_cannot_break_work(self):
        secret = "fixture-private-key"
        events = []
        progress = RunProgress(events.append, secret)
        step = await progress.begin("tool", "<script>bad</script>", {"query": secret+"<img src=x onerror=bad>"})
        await progress.finish(step)
        self.assertNotIn(secret, json.dumps(events))
        self.assertNotIn(secret, json.dumps(progress.snapshot()))
        rendered = activity_view(progress.snapshot(), 1)
        self.assertNotIn("<script>", rendered)
        self.assertNotIn("<img src=x", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        def fail_observer(event):
            raise RuntimeError("fixture observer failure")
        reply = await ask("CLOCK", Session(), Settings(), transport=FakeModel().transport(), on_event=fail_observer)
        self.assertEqual(reply.requests, 2)


class MonitorUITests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        from app import build_ui
        cls.ui = build_ui(Settings())
        cls.callbacks = {fn.fn.__name__: fn.fn for fn in cls.ui.fns.values()}

    async def test_live_timer_and_activity_are_yielded_before_answer(self):
        finished = asyncio.Event()
        async def delayed_ask(*args, on_event):
            on_event({"id": 1, "kind": "model", "name": "Model request 1", "status": "running",
                      "started_at_seconds": 0, "duration_seconds": None})
            await asyncio.sleep(.6)
            finished.set()
            return Reply("Answer", Session(), [], .6, 1, 20, 5,
                         timings=[{"id": 1, "kind": "model", "name": "Model request 1", "status": "completed",
                                   "started_at_seconds": 0, "duration_seconds": .6}])
        with patch("app.ask", new=delayed_ask):
            generator = self.callbacks["respond"]("Question", [], Session(), True)
            await anext(generator)
            pending = await anext(generator)
            self.assertFalse(finished.is_set())
            self.assertIn("Running", pending[8])
            self.assertIn("Model request 1", pending[9])
            rest = [item async for item in generator]
        final = rest[-1]
        self.assertEqual(final[0][-1]["content"], "Answer")
        self.assertEqual(final[11][-1]["elapsed"], .6)
        self.assertEqual(final[11][-1]["model"], .6)

    async def test_disconnect_cancels_background_model_task(self):
        cancelled = asyncio.Event()
        async def waiting(*args, on_event):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        with patch("app.ask", new=waiting):
            generator = self.callbacks["respond"]("Question", [], Session(), True)
            await anext(generator)
            await anext(generator)
            await generator.aclose()
        self.assertTrue(cancelled.is_set())

    async def test_failed_turn_is_timed_without_changing_prior_conversation(self):
        error = RuntimeError("fixture failure")
        error.elapsed_seconds = 2.75
        session = Session(successful_turns=2)
        with patch("app.ask", new=AsyncMock(side_effect=error)):
            outputs = [item async for item in self.callbacks["respond"]("Question", [], session, True)]
        self.assertIs(outputs[-1][1].messages, session.messages)
        self.assertEqual(outputs[-1][1].successful_turns, session.successful_turns)
        self.assertEqual(outputs[-1][11][-1]["status"], "Failed")
        self.assertEqual(outputs[-1][11][-1]["elapsed"], 2.75)

    async def test_history_bounded_and_reset_clears_all_monitor_outputs(self):
        history = [{"id": i, "question": "Earlier", "elapsed": 1, "model": 1, "tool": 0, "status": "Completed"}
                   for i in range(1, 21)]
        with patch("app.ask", new=AsyncMock(return_value=Reply("Next", Session(), [], 1, 1, 2, 3))):
            outputs = [item async for item in self.callbacks["respond"]("Next", [], Session(), True, None, history)]
        updated = outputs[-1][11]
        self.assertEqual(len(updated), 20)
        self.assertEqual((updated[0]["id"], updated[-1]["id"]), (2, 21))
        self.assertEqual(history[-1]["id"], 20)
        reset = self.callbacks["reset_chat"]()
        self.assertEqual(reset[-1], [])
        self.assertIn("No requests yet", reset[-2])

    def test_history_html_escapes_questions(self):
        html = history_view([{"id": 1, "question": "<script>bad</script>", "elapsed": 1, "model": .9, "tool": .1, "status": "Completed"}])
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
