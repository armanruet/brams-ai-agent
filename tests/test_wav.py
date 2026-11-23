from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
import wave

from pydantic_ai import ModelRetry

from agent_app.artifacts import save_wav_file, wav_path
from agent_app.brams_tools import get_wav
from agent_app.config import Settings
from agent_app.recovery import wav_summary, spectrogram_summary
from agent_app.service import Session, ask, Reply, wav_attachments
from tests.fake_model import FakeModel, WAV_PROMPT, KIRCHBERG_PROMPT


START = datetime(2026, 10, 22, tzinfo=timezone.utc)
END = START + timedelta(hours=1)
READY = {"ready": True, "version": "0.1.7"}


def wav_bytes():
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00\x10\x00" * 400)
    return buffer.getvalue()


class WavTests(unittest.IsolatedAsyncioTestCase):
    def save(self, directory):
        path = Path(directory) / "input.wav"
        path.write_bytes(wav_bytes())
        return save_wav_file(path)

    def test_original_bytes_preserved_and_separate_preview_created(self):
        with tempfile.TemporaryDirectory() as directory, patch("agent_app.artifacts.WAV_DIR", Path(directory) / "out"):
            data = self.save(directory)
            self.assertEqual(wav_path(data["artifact_id"]).read_bytes(), wav_bytes())
            with wave.open(str(wav_path(data["artifact_id"], preview=True))) as preview:
                self.assertEqual(preview.getsampwidth(), 2)
                self.assertEqual(preview.getframerate(), 8000)
            self.assertEqual(data["sample_rate_hz"], 8000)
            self.assertEqual(data["duration_seconds"], 0.1)
            self.assertIsNone(wav_path("../../input.wav"))

    async def test_follow_up_attaches_audio_and_does_not_send_paths_to_model(self):
        fake = FakeModel()
        with tempfile.TemporaryDirectory() as directory, \
             patch("agent_app.artifacts.WAV_DIR", Path(directory) / "out"), \
             patch("agent_app.brams_tools.pybrams_setup_status", return_value=READY):
            with patch("agent_app.brams_tools._load_availability", return_value={"status": "ok"}):
                first = await ask(KIRCHBERG_PROMPT, Session(), Settings(), transport=fake.transport())
            data = {**self.save(directory), "status": "ok", "downloaded": True,
                    "filename": "test.wav", "receiver_id": "TESTKIR_SYS001", "receiver_name": "Kirchberg",
                    "wav_source_url": "https://example.invalid/test.wav",
                    "recording_start_utc": "2026-10-22T00:00:00Z", "recording_end_utc": "2026-10-22T00:05:00Z"}
            with patch("agent_app.brams_tools._load_wav", return_value=data) as load:
                reply = await ask(WAV_PROMPT, first.session, Settings(), transport=fake.transport())
            load.assert_called_once_with("Kirchberg", START, END)
            self.assertEqual(len(reply.audio), 1)
            self.assertNotIn(directory, str(fake.requests))
            self.assertEqual(reply.trace[0]["name"], "get_wav")

    async def test_download_survives_model_loop_or_truncated_final(self):
        for prompt in ["WAV_LOOP", "WAV_LENGTH"]:
            with self.subTest(prompt=prompt), tempfile.TemporaryDirectory() as directory, \
                 patch("agent_app.artifacts.WAV_DIR", Path(directory) / "out"), \
                 patch("agent_app.brams_tools.pybrams_setup_status", return_value=READY):
                data = {**self.save(directory), "status": "ok", "downloaded": True, "filename": "test.wav",
                        "receiver_name": "Kirchberg", "wav_source_url": "https://example.invalid/test.wav",
                        "recording_start_utc": "2026-10-22T00:00:00Z", "recording_end_utc": "2026-10-22T00:05:00Z"}
                fake = FakeModel()
                with patch("agent_app.brams_tools._load_wav", return_value=data) as download:
                    original_session = Session()
                    reply = await ask(prompt, original_session, Settings(), transport=fake.transport())
                download.assert_called_once()
                self.assertEqual(len(reply.audio), 1)
                self.assertEqual(len(fake.requests), 2)
                self.assertIn("Kirchberg", reply.text)
                self.assertIn("https://example.invalid/test.wav", reply.text)
                self.assertTrue(reply.warning)
                self.assertEqual(original_session.messages, [])
                # No incomplete tool calls may leak into subsequent requests.
                await ask("HISTORY", reply.session, Settings(), transport=fake.transport())
                messages = fake.requests[-1]["messages"]
                calls = {call["id"] for message in messages for call in message.get("tool_calls", [])}
                returns = {message["tool_call_id"] for message in messages if message["role"] == "tool"}
                self.assertEqual(calls, returns)

    async def test_invalid_range_or_receiver_does_not_download(self):
        with patch("agent_app.brams_tools._load_wav") as download:
            for search, start, end in [("", START, END), ("Kirchberg", END, START),
                                       ("Kirchberg", START, START + timedelta(days=2))]:
                with self.assertRaises(ModelRetry):
                    await get_wav(search, start, end)
        download.assert_not_called()

    async def test_gradio_audio_player_and_original_download_serialize(self):
        import gradio as gr
        from app import build_ui
        with tempfile.TemporaryDirectory() as directory, patch("agent_app.artifacts.WAV_DIR", Path(directory) / "out"):
            data = {**self.save(directory), "status": "ok", "downloaded": True, "filename": "test.wav"}
            audio = wav_attachments([{"event": "tool_result", "name": "get_wav", "result": data}])
            ui = build_ui(Settings())
            callback = next(fn.fn for fn in ui.fns.values() if fn.fn.__name__ == "respond")
            reply = Reply("Original and preview", Session(), [], 1, 2, 1, 1, audio=audio)
            with patch("app.ask", new=AsyncMock(return_value=reply)):
                outputs = [item async for item in callback("Play one", [], Session(), True)]
            chatbot = next(component for component in ui.blocks.values() if isinstance(component, gr.Chatbot))
            serialized = chatbot.postprocess(outputs[-1][0])
            self.assertEqual(serialized.root[-2].content[0].component, "audio")
            self.assertEqual(serialized.root[-1].content[0].file.mime_type, "application/octet-stream")
            self.assertEqual(serialized.root[-1].content[0].file.orig_name, "test.wav")
            browser_transcript = chatbot.preprocess(serialized)
            with patch("app.ask", new=AsyncMock(return_value=Reply("Next", Session(), [], 1, 1, 1, 1))):
                outputs = [item async for item in callback("Next", browser_transcript, Session(), True)]
            self.assertEqual(len(outputs[-1][0]), 6)

    def test_wav_summary_conversational_and_time_context_and_player_label(self):
        with tempfile.TemporaryDirectory() as directory, patch("agent_app.artifacts.WAV_DIR", Path(directory) / "out"):
            data = {
                **self.save(directory),
                "status": "ok",
                "downloaded": True,
                "filename": "RAD_BEDOUR_20261022_0000_LUKIRC_SYS001.wav",
                "receiver_id": "LUKIRC_SYS001",
                "receiver_name": "Kirchberg",
                "wav_source_url": "https://example.invalid/test.wav",
                "recording_start_utc": "2026-10-22T00:00:00Z",
                "recording_end_utc": "2026-10-22T00:05:00Z",
                "header_summary": {
                    "samplerate_hz": 5512.5,
                    "lo_frequency_hz": 49969000,
                    "beacon_frequency_hz": 49970000,
                    "nominal_beacon_minus_lo_hz": 1000,
                    "pps_count": 300,
                    "state_raw": 0,
                    "gps_state_note": "Raw state 0 is not proof of a healthy GPS fix.",
                },
            }
            audio = wav_attachments([{"event": "tool_result", "name": "get_wav", "result": data}])
            self.assertEqual(len(audio), 1)
            self.assertIn("Kirchberg (LUKIRC_SYS001)", audio[0]["label"])
            self.assertIn("22 October 2026, 00:00–00:05 UTC", audio[0]["label"])

            time_event = [{
                "event": "tool_result",
                "name": "convert_local_time",
                "result": {
                    "timezone": "Europe/Brussels",
                    "local_datetime": "2026-10-22T02:00:00+02:00",
                    "end_local_datetime": "2026-10-22T03:00:00+02:00",
                    "start_utc": "2026-10-22T00:00:00Z",
                    "end_utc": "2026-10-22T01:00:00Z",
                    "utc_offset": "+02:00",
                },
            }]
            simple = wav_summary(data, include_header=False, time_events=time_event)
            self.assertIn("Here’s the archived WAV recording for **Kirchberg** (`LUKIRC_SYS001`)", simple)
            self.assertIn("Europe/Brussels", simple)
            self.assertIn("normalized PCM16 preview player", simple)
            self.assertNotIn("Decoded `BRA1` WAV Header", simple)

            detailed = wav_summary(data, include_header=True)
            self.assertIn("Decoded `BRA1` WAV Header & Provenance:", detailed)
            self.assertIn("49969000 Hz", detailed)

            spec_with_time = spectrogram_summary({
                "status": "ok",
                "downloaded": True,
                "receiver_name": "Kirchberg",
                "receiver_id": "LUKIRC_SYS001",
                "recording_start_utc": "2026-10-22T00:00:00Z",
                "recording_end_utc": "2026-10-22T00:05:00Z",
                "start_utc": "2026-10-22T00:00:00Z",
                "end_utc": "2026-10-22T01:00:00Z",
                "image_source_url": "https://example.invalid/test.png",
            }, time_events=time_event)
            self.assertIn("Europe/Brussels", spec_with_time)
            self.assertIn("2026-10-22T00:00:00Z", spec_with_time)
