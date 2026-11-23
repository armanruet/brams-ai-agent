import unittest

import httpx
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded

from agent_app.config import DEFAULT_MODEL, Settings
from agent_app.diagnostics import check_server
from agent_app.service import Session, ask, friendly_error
from tests.fake_model import FakeModel


class PrototypeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = Settings(timeout_seconds=10)
        self.fake = FakeModel()

    async def request(self, prompt, session=None, enable_tools=True):
        return await ask(
            prompt, session or Session(), self.settings, enable_tools,
            transport=self.fake.transport(),
        )

    async def test_text_turn_uses_local_chat_endpoint_and_exact_model_id(self):
        reply = await self.request("Hello", enable_tools=False)
        payload = self.fake.requests[0]
        self.assertEqual(payload["model"], DEFAULT_MODEL)
        self.assertEqual(reply.text, "Hello from the protocol test fixture.")
        self.assertEqual(reply.session.successful_turns, 1)
        self.assertEqual(reply.trace, [])
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertNotIn("max_completion_tokens", payload)
        self.assertNotIn("temperature", payload)
        self.assertFalse(payload.get("tools"))

    async def test_clock_call_is_executed_and_result_returned_to_model(self):
        reply = await self.request("CLOCK")
        self.assertEqual(reply.requests, 2)
        self.assertEqual([event["event"] for event in reply.trace], ["tool_call", "tool_result"])
        self.assertEqual(reply.trace[-1]["result"]["source"], "local_system_clock")
        self.assertIn("utc_time", reply.text)
        self.assertEqual(self.fake.requests[1]["messages"][-1]["role"], "tool")
        self.assertFalse(self.fake.requests[0]["parallel_tool_calls"])
        for tool in self.fake.requests[0]["tools"]:
            self.assertFalse(tool["function"].get("strict", False))

    async def test_offset_interval_is_normalized_to_utc(self):
        reply = await self.request("DURATION")
        result = reply.trace[-1]["result"]
        self.assertEqual(result["start_utc"], "2026-10-28T10:00:00Z")
        self.assertEqual(result["duration_minutes"], 90)

    async def test_naive_timestamp_yields_validation_feedback_then_retry(self):
        reply = await self.request("BAD_ARGS")
        events = [event["event"] for event in reply.trace]
        self.assertIn("validation_retry", events)
        self.assertEqual(sum(event == "tool_result" for event in events), 1)
        self.assertEqual(reply.trace[-1]["name"], "get_current_utc_time")

    async def test_unknown_tool_does_not_execute_and_model_gets_retry(self):
        reply = await self.request("UNKNOWN_TOOL")
        self.assertIn("validation_retry", [e["event"] for e in reply.trace])
        returns = [e for e in reply.trace if e["event"] == "tool_result"]
        self.assertEqual([e["name"] for e in returns], ["get_current_utc_time"])

    async def test_history_preserves_tool_call_and_result_on_follow_up(self):
        first = await self.request("CLOCK")
        second = await self.request("HISTORY", first.session)
        messages = self.fake.requests[-1]["messages"]
        self.assertIn("CLOCK", second.text)
        self.assertTrue(any(message["role"] == "tool" for message in messages))
        self.assertTrue(any(message.get("tool_calls") for message in messages))
        self.assertEqual(second.session.successful_turns, 2)
        self.assertEqual(first.session.successful_turns, 1)
        self.assertEqual(second.trace, [])

    async def test_sessions_are_isolated(self):
        await self.request("My name is Alice.")
        reply = await self.request("HISTORY", Session())
        self.assertNotIn("Alice", reply.text)

    async def test_loop_is_bounded_and_original_session_is_unchanged(self):
        session = Session()
        reply = await self.request("LOOP", session)
        self.assertIn("repeated tool call", reply.warning)
        self.assertIn("UTC time:", reply.text)
        self.assertEqual(len(self.fake.requests), 2)
        self.assertEqual(sum(e["event"] == "tool_result" for e in reply.trace), 1)
        self.assertEqual(session.messages, [])
        follow_up = await self.request("HISTORY", reply.session)
        self.assertIn("LOOP", follow_up.text)

    async def test_server_error_does_not_commit_history_or_retry_sdk(self):
        session = Session()
        with self.assertRaises(ModelHTTPError):
            await self.request("HTTP_ERROR", session)
        self.assertEqual(len(self.fake.requests), 1)
        self.assertEqual(session.messages, [])

    async def test_truncated_answer_is_not_committed(self):
        with self.assertRaisesRegex(RuntimeError, "did not finish"):
            await self.request("LENGTH")
        self.assertEqual([request["max_tokens"] for request in self.fake.requests], [4096, 6144])

    async def test_model_catalog_check(self):
        result = await check_server(self.settings, transport=self.fake.transport())
        self.assertTrue(result["selected_model_found"])
        other = Settings(model="not-in-fixture")
        result = await check_server(other, transport=self.fake.transport())
        self.assertFalse(result["selected_model_found"])

    async def test_empty_message_does_not_call_model(self):
        with self.assertRaises(ValueError):
            await self.request("  ")
        self.assertEqual(self.fake.requests, [])

    def test_configuration_rejects_wrong_endpoint(self):
        with self.assertRaises(ValueError):
            Settings(base_url="http://127.0.0.1:8080/v1/chat/completions")

    def test_error_output_redacts_key(self):
        settings = Settings(api_key="secret-fixture-key")
        output = friendly_error(RuntimeError("bad secret-fixture-key"), settings)
        self.assertNotIn("secret-fixture-key", output)


if __name__ == "__main__":
    unittest.main()
