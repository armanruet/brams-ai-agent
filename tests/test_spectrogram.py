from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from pydantic_ai import ModelRetry

from agent_app.artifacts import save_spectrogram, spectrogram_path
from agent_app.brams_tools import get_spectrogram
from agent_app.config import Settings
from agent_app.service import Session, ask, spectrogram_images
from tests.fake_model import FakeModel, KIRCHBERG_PROMPT, SPECTROGRAM_PROMPT


START = datetime(2026, 10, 22, tzinfo=timezone.utc)
END = START + timedelta(hours=1)
READY = {"ready": True, "version": "0.1.7"}


def png_bytes():
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "black").save(buffer, format="PNG")
    return buffer.getvalue()


class SpectrogramTests(unittest.IsolatedAsyncioTestCase):
    async def test_follow_up_executes_tool_and_attaches_only_actual_saved_png(self):
        fake = FakeModel()
        with tempfile.TemporaryDirectory() as directory, \
             patch("agent_app.artifacts.SPECTROGRAM_DIR", Path(directory)), \
             patch("agent_app.brams_tools.pybrams_setup_status", return_value=READY):
            with patch("agent_app.brams_tools._load_availability", return_value={"status": "ok"}):
                first = await ask(KIRCHBERG_PROMPT, Session(), Settings(), transport=fake.transport())
            artifact_id = save_spectrogram(png_bytes())
            result = {"status": "ok", "downloaded": True, "artifact_id": artifact_id,
                      "receiver_id": "TESTKIR_SYS001", "image_source_url": "https://example.invalid/test.png"}
            with patch("agent_app.brams_tools._load_spectrogram", return_value=result) as load:
                reply = await ask(SPECTROGRAM_PROMPT, first.session, Settings(), transport=fake.transport())
            load.assert_called_once_with("Kirchberg", START, END)
            self.assertEqual(reply.images, [str(Path(directory) / f"{artifact_id}.png")])
            self.assertEqual(reply.trace[0]["name"], "get_spectrogram")
            self.assertTrue(any(m["role"] == "tool" for m in fake.requests[-2]["messages"]))
            self.assertNotIn(directory, str(fake.requests))
            self.assertIn("https://example.invalid/test.png", reply.text)

    async def test_simple_spectrogram_reply_is_concise_and_preserves_follow_up(self):
        class DirectSpectrogramModel(FakeModel):
            def respond(self, payload):
                fixture_payload = dict(payload)
                fixture_payload["messages"] = [dict(message) for message in payload["messages"]]
                for message in fixture_payload["messages"]:
                    if message["role"] == "user":
                        message["content"] = SPECTROGRAM_PROMPT
                status, body = super().respond(fixture_payload)
                message = body["choices"][0]["message"]
                for call in message.get("tool_calls", []):
                    if call["function"]["name"] == "get_spectrogram":
                        import json
                        args = json.loads(call["function"]["arguments"])
                        args["direct_answer"] = True
                        call["function"]["arguments"] = json.dumps(args)
                return status, body

        fake = DirectSpectrogramModel()
        with tempfile.TemporaryDirectory() as directory, \
             patch("agent_app.artifacts.SPECTROGRAM_DIR", Path(directory)), \
             patch("agent_app.brams_tools.pybrams_setup_status", return_value=READY):
            artifact_id = save_spectrogram(png_bytes())
            result = {"status": "ok", "downloaded": True, "artifact_id": artifact_id,
                      "receiver_id": "LUKIRC_SYS001", "receiver_name": "Kirchberg",
                      "recording_start_utc": "2026-10-22T00:00:00.319662Z",
                      "recording_end_utc": "2026-10-22T00:05:00.319662Z",
                      "image_source_url": "https://example.invalid/spectrogram.png"}
            with patch("agent_app.brams_tools._load_spectrogram", return_value=result):
                reply = await ask("Show one Kirchberg spectrogram on 2026-10-22 from 00:00 to 01:00 UTC", Session(), Settings(), transport=fake.transport())
            self.assertEqual(reply.requests, 1)
            self.assertEqual(reply.images, [str(Path(directory) / f"{artifact_id}.png")])
            self.assertIn("Kirchberg", reply.text)
            self.assertIn("22 October 2026, 00:00–00:05 UTC", reply.text)
            self.assertIn("one recording from the requested interval", reply.text)
            self.assertIn("doesn’t include meteor echo detection", reply.text)
            self.assertIn("https://example.invalid/spectrogram.png", reply.text)
            self.assertNotIn("Verified tool results", reply.text)
            self.assertNotIn(".319662", reply.text)
            followup_fake = FakeModel()
            await ask("HISTORY", reply.session, Settings(), transport=followup_fake.transport())
            messages = followup_fake.requests[-1]["messages"]
            calls = {call["id"] for message in messages for call in message.get("tool_calls", [])}
            returns = {message["tool_call_id"] for message in messages if message["role"] == "tool"}
            self.assertEqual(calls, returns)

    def test_summaries_cover_zero_partial_and_cross_midnight(self):
        from agent_app.recovery import meteor_summary, spectrogram_summary
        data = {"receiver_name": "Kirchberg", "receiver_id": "LUKIRC_SYS001",
                "recording_start_utc": "2026-10-22T23:55:00Z",
                "recording_end_utc": "2026-10-23T00:00:00Z", "meteor_detected": False,
                "total_meteor_count": 0, "confidence_threshold": 0.25,
                "inspected_slots_count": 1, "available_slots_in_window": 2,
                "coverage_complete": False}
        text = meteor_summary(data)
        self.assertIn("No meteor echoes detected", text)
        self.assertIn("22 October 2026, 23:55–23 October 2026, 00:00 UTC", text)
        self.assertIn("1 of 2 available recordings", text)
        self.assertIn("Coverage is incomplete", text)
        self.assertNotIn("https://", spectrogram_summary(data))

        multi = {
            "receiver_name": "Kirchberg",
            "receiver_id": "LUKIRC_SYS001",
            "recording_start_utc": "2026-10-22T22:00:00Z",
            "recording_end_utc": "2026-10-22T23:00:00Z",
            "meteor_detected": True,
            "total_meteor_count": 3,
            "confidence_threshold": 0.25,
            "inspected_slots_count": 12,
            "available_slots_in_window": 12,
            "coverage_complete": True,
            "slots": [
                {
                    "recording_start_utc": "2026-10-22T22:00:00Z",
                    "recording_end_utc": "2026-10-22T22:05:00Z",
                    "meteor_count": 0,
                    "high_confidence_count": 0,
                    "max_confidence": None,
                },
                {
                    "recording_start_utc": "2026-10-22T22:05:00Z",
                    "recording_end_utc": "2026-10-22T22:10:00Z",
                    "meteor_count": 2,
                    "high_confidence_count": 1,
                    "max_confidence": 0.88,
                },
                {
                    "recording_start_utc": "2026-10-22T22:20:00Z",
                    "recording_end_utc": "2026-10-22T22:25:00Z",
                    "meteor_count": 1,
                    "high_confidence_count": 0,
                    "max_confidence": 0.41,
                },
            ],
        }
        multi_text = meteor_summary(multi)
        self.assertIn("Activity by 5-minute recording:", multi_text)
        self.assertIn("22:05–22:10 UTC", multi_text)
        self.assertIn("2 echoes (1 high-confidence), max conf `0.88`", multi_text)
        self.assertIn("22:20–22:25 UTC", multi_text)
        self.assertNotIn("22:00–22:05 UTC", multi_text)

    async def test_invalid_interval_does_not_download(self):
        with patch("agent_app.brams_tools._load_spectrogram") as load:
            for search, start, end in [("", START, END), ("Kirchberg", END, START),
                                       ("Kirchberg", START, START + timedelta(days=2))]:
                with self.assertRaises(ModelRetry):
                    await get_spectrogram(search, start, end)
        load.assert_not_called()

    async def test_failure_does_not_report_download_or_attach_image(self):
        with patch("agent_app.brams_tools.pybrams_setup_status", return_value=READY), \
             patch("agent_app.brams_tools._load_spectrogram", side_effect=RuntimeError):
            result = await get_spectrogram("Kirchberg", START, END)
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["downloaded"])
        self.assertFalse(result["image_analysis_performed"])
        self.assertEqual(spectrogram_images([{"event": "tool_result", "name": "get_spectrogram", "result": result}]), [])

    def test_only_valid_png_is_stored_and_ids_cannot_select_arbitrary_files(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("agent_app.artifacts.SPECTROGRAM_DIR", Path(directory)):
            with self.assertRaises(Exception):
                save_spectrogram(b"<html>upstream error</html>")
            self.assertEqual(list(Path(directory).iterdir()), [])
            artifact_id = save_spectrogram(png_bytes())
            self.assertTrue(spectrogram_path(artifact_id).is_file())
            for value in ["../config.py", "/etc/passwd", None, "a" * 32]:
                self.assertIsNone(spectrogram_path(value))

    async def test_gradio_callback_serializes_image_and_retains_it_on_follow_up(self):
        import gradio as gr
        from app import build_ui
        from agent_app.service import Reply
        from unittest.mock import AsyncMock

        with tempfile.TemporaryDirectory() as directory, \
             patch("agent_app.artifacts.SPECTROGRAM_DIR", Path(directory)):
            path = spectrogram_path(save_spectrogram(png_bytes()))
            ui = build_ui(Settings())
            callback = next(fn.fn for fn in ui.fns.values() if fn.fn.__name__ == "respond")
            reply = Reply("Archive image", Session(), [], 1, 2, 10, 10, images=[str(path)])
            with patch("app.ask", new=AsyncMock(return_value=reply)):
                outputs = [item async for item in callback("Show one", [], Session(), True)]
            transcript = outputs[-1][0]
            self.assertIsInstance(transcript[-1]["content"], gr.Image)
            chatbot = next(component for component in ui.blocks.values() if isinstance(component, gr.Chatbot))
            serialized = chatbot.postprocess(transcript)
            self.assertEqual(serialized.root[-1].content[0].component, "image")
            # Browser-returned media messages also survive a subsequent turn.
            browser_transcript = chatbot.preprocess(serialized)
            with patch("app.ask", new=AsyncMock(return_value=Reply("Next", Session(), [], 1, 1, 1, 1))):
                outputs = [item async for item in callback("Next", browser_transcript, Session(), True)]
            self.assertEqual(len(outputs[-1][0]), 5)
