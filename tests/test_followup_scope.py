"""Regression coverage for implicit media follow-ups and recovery."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart, ToolReturnPart

from agent_app.artifacts import save_spectrogram
from agent_app.config import Settings
from agent_app.service import Session, ask, _followup_scope_prompt
from tests.fake_model import FakeModel, SPECTROGRAM_PROMPT
from tests.test_spectrogram import READY, png_bytes


class FollowupScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_wrong_hour_is_rejected_and_not_reused(self):
        prior = "Please check if a spectrogram is available in Kirchberg yesterday 10:00 to 10:05 am or not"
        session = Session([ModelRequest(parts=[UserPromptPart(prior)]),
                           ModelResponse(parts=[TextPart("The requested availability is unknown.")])])

        class WrongTimeModel(FakeModel):
            def respond(self, payload):
                payload = dict(payload)
                payload["messages"] = [dict(message) for message in payload["messages"]]
                for message in payload["messages"]:
                    if message["role"] == "user":
                        message["content"] = SPECTROGRAM_PROMPT
                status, body = super().respond(payload)
                for call in body["choices"][0]["message"].get("tool_calls", []):
                    args = json.loads(call["function"]["arguments"])
                    args.update(start_utc="2026-11-13T09:00:00Z", end_utc="2026-11-13T09:05:00Z", direct_answer=True)
                    call["function"]["arguments"] = json.dumps(args)
                return status, body

        with tempfile.TemporaryDirectory() as directory, \
             patch("agent_app.artifacts.SPECTROGRAM_DIR", Path(directory)), \
             patch("agent_app.brams_tools.pybrams_setup_status", return_value=READY):
            artifact = save_spectrogram(png_bytes())
            result = {"status": "ok", "downloaded": True, "artifact_id": artifact,
                      "receiver_id": "LUKIRC_SYS001", "receiver_name": "Kirchberg"}
            with patch("agent_app.brams_tools._load_spectrogram", return_value=result):
                reply = await ask("show me the spectrogram", session, Settings(), transport=WrongTimeModel().transport())
        self.assertEqual(reply.images, [])
        self.assertIn("does not match", reply.text)
        returns = [p for m in reply.session.messages for p in m.parts if isinstance(p, ToolReturnPart)]
        self.assertEqual(returns[-1].content["status"], "error")

    def test_explicit_new_interval_is_not_overridden(self):
        old = ModelRequest(parts=[UserPromptPart("Show Kirchberg from 10:00 to 10:05 yesterday")])
        prompt = "Show the spectrogram from 11:00 to 11:05 today"
        self.assertEqual(_followup_scope_prompt(prompt, [old]), prompt)
