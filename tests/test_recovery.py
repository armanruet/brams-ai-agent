from unittest.mock import AsyncMock, patch
import unittest

from pydantic_ai.exceptions import UsageLimitExceeded

from agent_app.config import Settings
from agent_app.service import Session, ask
from tests.fake_model import FakeModel


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_distinct_call_loop_stays_bounded_and_returns_evidence(self):
        fake = FakeModel()
        reply = await ask("DISTINCT_LOOP", Session(), Settings(), transport=fake.transport())
        self.assertEqual(len(fake.requests), 6)
        self.assertIn("request limit", reply.warning)
        self.assertIn("minutes", reply.text)
        self.assertEqual(reply.trace[-1]["event"], "recovery")
        await ask("HISTORY", reply.session, Settings(), transport=fake.transport())
        messages = fake.requests[-1]["messages"]
        calls = {call["id"] for message in messages for call in message.get("tool_calls", [])}
        returns = {message["tool_call_id"] for message in messages if message["role"] == "tool"}
        self.assertEqual(calls, returns)

    async def test_single_pretool_token_retry_can_succeed(self):
        class TruncateOnce(FakeModel):
            def respond(self, payload):
                status, body = super().respond(payload)
                if len(self.requests) == 1:
                    body["choices"][0]["finish_reason"] = "length"
                return status, body
        fake = TruncateOnce()
        reply = await ask("Hello", Session(), Settings(), transport=fake.transport())
        self.assertEqual(reply.requests, 2)
        self.assertEqual([payload["max_tokens"] for payload in fake.requests], [4096, 6144])
        self.assertIn("recovery succeeded", reply.warning)
        self.assertEqual(reply.trace[0]["event"], "generation_retry")
        self.assertEqual(reply.session.successful_turns, 1)

    async def test_failed_run_trace_is_shown_in_gradio_without_clearing_history(self):
        from app import build_ui
        ui = build_ui(Settings())
        callback = next(fn.fn for fn in ui.fns.values() if fn.fn.__name__ == "respond")
        error = UsageLimitExceeded("fixture budget")
        error.tool_trace = [{"event": "validation_retry", "name": "get_wav", "feedback": "missing receiver"}]
        session = Session(successful_turns=2)
        with patch("app.ask", new=AsyncMock(side_effect=error)):
            outputs = [item async for item in callback("Play one", [], session, True)]
        self.assertEqual(outputs[-1][4], error.tool_trace)
        self.assertIs(outputs[-1][1].messages, session.messages)
        self.assertEqual(outputs[-1][1].successful_turns, session.successful_turns)
        self.assertEqual(len(outputs[-1][1].ui_messages), len(outputs[-1][0]))
        self.assertNotIn("Clear the chat", outputs[-1][0][-1]["content"])

    async def test_repeated_receiver_alias_does_not_redownload(self):
        from agent_app.tool_guard import guard_tool, RepeatedToolCall
        calls = []
        async def get_wav(search: str, start_utc: str):
            calls.append(search)
            return {"receiver_id": "TESTKIR_SYS001"}
        guarded = guard_tool(get_wav, set())
        await guarded("Kirchberg", "2026-10-22T00:00:00Z")
        with self.assertRaises(RepeatedToolCall):
            await guarded("TESTKIR_SYS001", "2026-10-22T00:00:00Z")
        self.assertEqual(calls, ["Kirchberg"])

    async def test_zero_tool_factual_query_retries_and_executes_tool(self):
        class LazyThenToolModel(FakeModel):
            def respond(self, payload):
                self.requests.append(payload)
                if len(self.requests) == 1:
                    # First attempt skips calling any tool and hallucinates a plain-text reply.
                    return 200, {
                        "id": "fixture_lazy",
                        "object": "chat.completion",
                        "created": 1793181600,
                        "model": "fixture",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "Ecotron has 1 receiver available."},
                            "finish_reason": "stop",
                        }],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
                    }
                # Second attempt (after grounding_retry) calls list_receivers properly.
                self.requests.pop()
                copied = dict(payload)
                copied["messages"] = [dict(m) for m in payload["messages"]]
                for m in copied["messages"]:
                    if m["role"] == "user":
                        m["content"] = "Which receivers are available at Ecotron?"
                return super().respond(copied)

        fake = LazyThenToolModel()
        with patch("agent_app.brams_tools.pybrams_setup_status", return_value={"ready": True, "version": "0.1.7"}), \
             patch("agent_app.brams_tools._load_catalog", return_value=({}, "https://example.invalid/receiver.php")):
            reply = await ask("Which receivers are available at Ecotron?", Session(), Settings(), transport=fake.transport())
        self.assertEqual(reply.trace[0]["event"], "grounding_retry")
        self.assertTrue(any(e.get("name") == "list_receivers" for e in reply.trace))
        self.assertIn("Tool execution was triggered", reply.warning)
        self.assertNotIn("Ecotron has 1 receiver available.", str(reply.session.messages))

    async def test_grounding_guard_corrects_error_reported_as_zero_and_strips_invented_urls(self):
        class HallucinatingModel(FakeModel):
            def respond(self, payload):
                self.requests.append(payload)
                last = payload["messages"][-1]
                if last["role"] != "tool":
                    return 200, {
                        "id": "call_1",
                        "object": "chat.completion",
                        "created": 1793181600,
                        "model": "fixture",
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "check_availability",
                                        "arguments": '{"search":"Kirchberg","start_utc":"2026-10-22T00:00:00Z","end_utc":"2026-10-22T01:00:00Z"}',
                                    },
                                }],
                            },
                            "finish_reason": "tool_calls",
                        }],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
                    }
                return 200, {
                    "id": "call_2",
                    "object": "chat.completion",
                    "created": 1793181600,
                    "model": "fixture",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                "There were 0 recordings found for Kirchberg (FAKEID_SYS999). "
                                "See https://hallucinated.example.org/fake.wav for details."
                            ),
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 12, "total_tokens": 24},
                }

        fake = HallucinatingModel()
        with patch("agent_app.brams_tools.pybrams_setup_status", return_value={"ready": True, "version": "0.1.7"}), \
             patch("agent_app.brams_tools._load_availability", side_effect=RuntimeError("archive timeout")):
            reply = await ask(
                "Can you check if Kirchberg receiver is available from 22/10/2026 00:00:00 to 22/10/2026 01:00:00 or not?",
                Session(),
                Settings(),
                transport=fake.transport(),
            )
        # Raw model text is preserved untouched for benchmarking, while visible reply is corrected.
        self.assertIn("https://hallucinated.example.org/fake.wav", reply.model_text)
        self.assertNotIn("https://hallucinated.example.org/fake.wav", reply.text)
        self.assertNotIn("0 recordings", reply.text)
        self.assertNotIn("0 recordings", str(reply.session.messages))
        self.assertIn("unknown", reply.text.lower())
        self.assertTrue(any(e.get("event") == "grounding_guard" for e in reply.trace))
        self.assertIsNotNone(reply.warning)
