"""Phase A regression tests: context-size budgeting, retry scoping, compaction, and overflow UX."""

import json
import os
import unittest
from unittest.mock import patch

import httpx
from pydantic_ai.agent import ModelRequestNode
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolReturnPart, UserPromptPart

from agent_app.config import Settings, load_settings
from agent_app.diagnostics import probe_local_context_size
from agent_app.service import (
    ContextLengthExceededError,
    Session,
    _limit_current_tool_results,
    _restore_tool_results,
    ask,
    friendly_error,
)
from tests.fake_model import FakeModel


class ContextBudgetTests(unittest.IsolatedAsyncioTestCase):
    def test_default_recovery_budget_and_env_context_override(self):
        self.assertEqual(Settings().recovery_max_tokens, 6144)
        with patch.dict(os.environ, {"LLAMA_MAX_TOKENS": "4096", "LLAMA_CTX_SIZE": "16384"}, clear=False):
            os.environ.pop("LLAMA_RECOVERY_MAX_TOKENS", None)
            loaded = load_settings()
        self.assertEqual(loaded.recovery_max_tokens, 6144)
        self.assertEqual(loaded.context_size, 16384)
        with self.assertRaises(ValueError):
            Settings(context_size=512)

    async def test_probe_local_context_reads_props_not_v1_models(self):
        seen_paths = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_paths.append(request.url.path)
            if request.url.path == "/props":
                return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 16384}})
            return httpx.Response(200, json={"data": [{"id": "m", "meta": {"n_ctx_train": 262144}}]})

        n_ctx = await probe_local_context_size(Settings(), transport=httpx.MockTransport(handler))
        self.assertEqual(n_ctx, 16384)
        self.assertEqual(seen_paths, ["/props"])

    async def test_preflight_rejects_oversized_prompt_without_calling_model(self):
        fake = FakeModel()
        huge = "A " * 25000
        with self.assertRaises(ContextLengthExceededError) as ctx:
            await ask(huge, Session(), Settings(context_size=16384), transport=fake.transport())
        self.assertEqual(fake.requests, [])
        self.assertEqual(ctx.exception.context_size, 16384)
        msg = friendly_error(ctx.exception, Settings())
        self.assertIn("context window", msg)
        self.assertIn("estimated", msg)
        self.assertIn("16,384", msg)
        self.assertIn("earlier conversation is preserved", msg)

    async def test_preflight_clamps_recovery_max_tokens_when_headroom_is_tight(self):
        class TruncateOnce(FakeModel):
            def respond(self, payload):
                status, body = super().respond(payload)
                if len(self.requests) == 1:
                    body["choices"][0]["finish_reason"] = "length"
                return status, body

        fake = TruncateOnce()
        # With tools (~3800 est), safety (512), prompt (~2000), ctx=11000 leaves ~4688 < 6144.
        prompt = "hello " * 1100
        reply = await ask(prompt, Session(), Settings(context_size=11000, max_tokens=2048, recovery_max_tokens=6144),
                          transport=fake.transport())
        self.assertEqual(reply.requests, 2)
        self.assertEqual(fake.requests[0]["max_tokens"], 2048)
        self.assertGreater(fake.requests[1]["max_tokens"], 2048)
        self.assertLess(fake.requests[1]["max_tokens"], 6144)

    async def test_grounding_retry_keeps_max_tokens_instead_of_recovery_budget(self):
        class UngroundedThenTool(FakeModel):
            def respond(self, payload):
                self.requests.append(payload)
                if len(self.requests) == 1:
                    msg = {"role": "assistant", "content": "Kirchberg is available."}
                    return 200, {"id": "c1", "object": "chat.completion", "created": 0, "model": payload["model"],
                                 "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]}
                if len(self.requests) == 2:
                    msg = {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "call-clock", "type": "function",
                         "function": {"name": "get_current_utc_time", "arguments": "{}"}}
                    ]}
                    return 200, {"id": "c2", "object": "chat.completion", "created": 0, "model": payload["model"],
                                 "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}]}
                msg = {"role": "assistant", "content": "Checked."}
                return 200, {"id": "c3", "object": "chat.completion", "created": 0, "model": payload["model"],
                             "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]}

        fake = UngroundedThenTool()
        # Trigger grounding retry on a BRAMS prompt; attempt 1 must NOT jump to 6144.
        with patch("agent_app.service.requires_tool_or_history", side_effect=[True, False]):
            reply = await ask("Check BEHUMA_SYS001 availability on 2023-12-14 21:00 to 22:00 UTC.",
                              Session(), Settings(), transport=fake.transport())
        self.assertEqual(fake.requests[0]["max_tokens"], 4096)
        self.assertEqual(fake.requests[1]["max_tokens"], 4096)
        self.assertTrue(any(e.get("event") == "grounding_retry" for e in reply.trace))

    def test_intra_turn_tool_compaction_covers_non_detection_tools_and_restores(self):
        huge_rows = [{"receiver_id": f"BE{i:04d}_SYS001", "coverage": "full", "extra": "x" * 200} for i in range(60)]
        payload = {"status": "ok", "receivers": huge_rows, "count": len(huge_rows), "source_url": "https://brams.aeronomie.be/v4/file.php"}
        part = ToolReturnPart(tool_name="check_availability", content=payload, tool_call_id="c1")
        request = ModelRequest(parts=[part])
        node = ModelRequestNode(request=request)
        saved = {}
        _limit_current_tool_results(node, saved, max_chars=2000)
        self.assertIn("c1", saved)
        self.assertLessEqual(len(json.dumps(part.content, default=str)), 2000)
        self.assertTrue(part.content.get("history_truncated"))
        _restore_tool_results([request], saved)
        self.assertEqual(len(part.content["receivers"]), 60)

    async def test_http_400_context_overflow_retries_once_without_history(self):
        class OverflowWithHistory(FakeModel):
            def respond(self, payload):
                self.requests.append(payload)
                if len(self.requests) == 1:
                    return 400, {"error": {"type": "exceed_context_size_error", "message": "request exceeds the available context size"}}
                msg = {"role": "assistant", "content": "Recovered without old turns."}
                return 200, {"id": "c2", "object": "chat.completion", "created": 0, "model": payload["model"],
                             "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]}

        fake = OverflowWithHistory()
        old = [
            ModelRequest(parts=[UserPromptPart("older question")]),
            ModelResponse(parts=[TextPart("older answer")]),
        ]
        reply = await ask("Follow-up question", Session(messages=old), Settings(), transport=fake.transport())
        self.assertEqual(len(fake.requests), 2)
        self.assertEqual([m for m in fake.requests[0]["messages"] if m["role"] == "system"],
                         [m for m in fake.requests[1]["messages"] if m["role"] == "system"])
        self.assertTrue(any(e.get("event") == "context_retry" for e in reply.trace))
        self.assertIn("context window", reply.warning)
        self.assertEqual(reply.text, "Recovered without old turns.")

    async def test_http_400_context_overflow_with_empty_history_raises_friendly_estimate(self):
        class AlwaysOverflow(FakeModel):
            def respond(self, payload):
                return 400, {"error": {"type": "exceed_context_size_error", "message": "exceeds the available context size"}}

        fake = AlwaysOverflow()
        with self.assertRaises(ContextLengthExceededError) as ctx:
            await ask("Hello", Session(), Settings(context_size=16384), transport=fake.transport())
        msg = friendly_error(ctx.exception, Settings())
        self.assertIn("estimated", msg)
        self.assertIn("16,384", msg)
        self.assertIn("New chat", msg)

    def test_friendly_error_distinguishes_context_400_from_generic_400(self):
        settings = Settings(provider="compatible", base_url="https://fixture.invalid/v1", model="m", api_key="secret-xyz")
        ctx_err = ModelHTTPError(status_code=400, model_name="m", body="maximum context length exceeded secret-xyz")
        msg = friendly_error(ctx_err, settings)
        self.assertIn("context window", msg)
        self.assertNotIn("secret-xyz", msg)
        other_err = ModelHTTPError(status_code=400, model_name="m", body="bad param secret-xyz")
        other_msg = friendly_error(other_err, settings)
        self.assertIn("token-limit", other_msg)
        self.assertNotIn("secret-xyz", other_msg)


if __name__ == "__main__":
    unittest.main()
