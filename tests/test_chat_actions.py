"""Message deletion against actual typed tool history; all model HTTP is mocked."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from agent_app.chat_history import action_manifest, message_ref, remove_message
from agent_app.config import Settings
from agent_app.service import Session, ask
from tests.fake_model import FakeModel


class ChatActionTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        from app import build_ui
        cls.ui = build_ui(Settings())
        cls.callbacks = {fn.fn.__name__: fn.fn for fn in cls.ui.fns.values()}

    async def request(self, prompt, transcript, session, fake):
        async def fixture(*args, **kwargs):
            return await ask(*args, **kwargs, transport=fake.transport())
        with patch("app.ask", new=fixture):
            results = [value async for value in self.callbacks["respond"](prompt, transcript, session, True)]
        return results[-1][0], results[-1][1]

    async def test_delete_older_answer_preserves_later_messages_and_tool_pairs(self):
        fake = FakeModel()
        transcript, session = await self.request("CLOCK", [], Session(), fake)
        transcript, session = await self.request("HISTORY", transcript, session, fake)
        self.assertEqual(len(session.ui_messages), 4)
        target = session.ui_messages[1]
        original_text = next(part.content for part in session.messages[target["message_index"]].parts if isinstance(part, TextPart))
        deleted = self.callbacks["delete_message"](json.dumps({"message_id": target["id"]}), transcript, session)
        remaining, updated = deleted[:2]
        self.assertEqual(len(remaining), 3)
        self.assertEqual(remaining[-2:], transcript[-2:])
        selected = updated.messages[target["message_index"]]
        self.assertNotIn(original_text, str(selected.parts))
        self.assertIn(original_text, str(session.messages[target["message_index"]].parts))
        calls = {part.tool_call_id for message in updated.messages for part in message.parts if isinstance(part, ToolCallPart)}
        returns = {part.tool_call_id for message in updated.messages for part in message.parts if isinstance(part, ToolReturnPart)}
        self.assertEqual(calls, returns)
        self.assertTrue(calls)
        # Same deleted ID cannot now remove the next row that occupies that index.
        stale = self.callbacks["delete_message"](json.dumps({"message_id": target["id"]}), remaining, updated)
        self.assertEqual(stale[0], gr.skip())
        # A subsequent real harness call still serializes the remaining tool history.
        remaining, following = await self.request("HISTORY", remaining, updated, fake)
        self.assertEqual(len(following.ui_messages), 5)
        self.assertTrue(any(row["role"] == "tool" for row in fake.requests[-1]["messages"]))

    async def test_delete_user_text_removes_it_from_model_prompt_but_keeps_protocol(self):
        fake = FakeModel()
        transcript, session = await self.request("CLOCK", [], Session(), fake)
        target = session.ui_messages[0]
        remaining, updated = remove_message(json.dumps({"message_id": target["id"]}), transcript, session)
        user_parts = [part.content for message in updated.messages for part in message.parts if isinstance(part, UserPromptPart)]
        self.assertNotIn("CLOCK", user_parts)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0], transcript[1])
        await self.request("HISTORY", remaining, updated, fake)
        users = [message["content"] for message in fake.requests[-1]["messages"] if message["role"] == "user"]
        self.assertNotIn("CLOCK", users)

    async def test_delete_user_after_grounding_retry_removes_all_prompt_copies(self):
        class RetryModel(FakeModel):
            def respond(self, payload):
                self.requests.append(payload)
                if len(self.requests) == 1:
                    message = {"role": "assistant", "content": "A receiver is available."}
                    finish = "stop"
                elif len(self.requests) == 2:
                    message = {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "retry-call", "type": "function", "function": {
                            "name": "list_receivers", "arguments": json.dumps({"search": "Ecotron"})}}]}
                    finish = "tool_calls"
                else:
                    message = {"role": "assistant", "content": "The lookup is complete."}
                    finish = "stop"
                return 200, {"id": "retry-fixture", "object": "chat.completion", "created": 0,
                             "model": "fixture", "choices": [{"index": 0, "message": message,
                             "finish_reason": finish}], "usage": {"prompt_tokens": 10,
                             "completion_tokens": 5, "total_tokens": 15}}

        fake = RetryModel()
        prompt = "Which receivers are available at Ecotron? Private phrase zebra-harbor-728."
        with patch("agent_app.brams_tools.pybrams_setup_status", return_value={"ready": True, "version": "0.1.7"}), \
             patch("agent_app.brams_tools._load_catalog", return_value=({}, "https://example.invalid/receiver.php")):
            transcript, session = await self.request(prompt, [], Session(), fake)
        self.assertEqual(len(fake.requests), 3)
        retry_payload = json.dumps(fake.requests[1])
        self.assertIn("Call the appropriate BRAMS or time tool first", retry_payload)
        self.assertEqual(sum(prompt in json.dumps(message) for message in fake.requests[1]["messages"]), 1)
        target = session.ui_messages[0]
        remaining, updated = remove_message(json.dumps({"message_id": target["id"]}), transcript, session)
        await self.request("Hello again", remaining, updated, fake)
        payload = fake.requests[-1]
        self.assertNotIn(prompt, json.dumps(payload))
        self.assertNotIn("zebra-harbor-728", json.dumps(payload))
        calls = {call["id"] for message in payload["messages"] for call in message.get("tool_calls", [])}
        returns = {message["tool_call_id"] for message in payload["messages"] if message["role"] == "tool"}
        self.assertEqual(calls, returns)
        self.assertTrue(calls)

    async def test_delete_media_only_hides_attachment_and_never_deletes_original_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "original.wav"
            path.write_bytes(b"original-fixture-bytes")
            ref = message_ref("media")
            session = Session(ui_messages=[ref])
            transcript = [{"role": "assistant", "content": {"file": {"path": str(path)}}}]
            remaining, updated = remove_message(json.dumps({"message_id": ref["id"]}), transcript, session)
            self.assertEqual(remaining, [])
            self.assertEqual(updated.ui_messages, [])
            self.assertEqual(path.read_bytes(), b"original-fixture-bytes")

    async def test_action_manifest_contains_only_ids_and_rejects_misaligned_state(self):
        ref = message_ref("user")
        session = Session(ui_messages=[ref])
        manifest = action_manifest([{"role": "user", "content": "private message"}], session)
        self.assertIn(ref["id"], manifest)
        self.assertNotIn("private message", manifest)
        self.assertNotIn("message_index", manifest)
        self.assertIsNone(remove_message(json.dumps({"message_id": ref["id"]}), [], session))
        for payload in ("bad-json", "null", "[]", '{"message_id":1}'):
            self.assertIsNone(remove_message(payload, [], session))
