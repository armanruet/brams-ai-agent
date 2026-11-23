"""Phase C regression tests: real task cancellation across downloads, SDK_LOCK, multi-slot scans, and UI."""

import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr
import httpx

from agent_app.brams_sdk import SDK_LOCK, _detect_meteors
from agent_app.cancellation import (
    OperationCancelled,
    cancellation_scope,
    stream_download_bytes,
    stream_download_to_file,
)
from agent_app.config import Settings
from agent_app.service import Session, ask, friendly_error
from tests.fake_model import FakeModel


class _ChunkedBody(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes], on_chunk=None):
        self._chunks = chunks
        self._on_chunk = on_chunk

    def __iter__(self):
        for index, chunk in enumerate(self._chunks):
            if self._on_chunk:
                self._on_chunk(index)
            yield chunk


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    def test_stream_download_aborts_mid_chunk_and_deletes_partial_file(self):
        cancel = threading.Event()
        chunks_yielded = []

        def on_chunk(idx):
            chunks_yielded.append(idx)
            if idx == 1:
                cancel.set()

        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, stream=_ChunkedBody([b"aaaa", b"bbbb", b"cccc", b"dddd"], on_chunk))
        )
        with self.assertRaises(OperationCancelled):
            stream_download_bytes("https://brams.invalid/test.png", cancel_event=cancel, chunk_size=4, transport=transport)
        self.assertEqual(chunks_yielded, [0, 1])

        cancel2 = threading.Event()
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "partial.wav"
            transport2 = httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    stream=_ChunkedBody([b"1111", b"2222", b"3333"], lambda idx: cancel2.set() if idx == 1 else None),
                )
            )
            with self.assertRaises(OperationCancelled):
                stream_download_to_file("https://brams.invalid/test.wav", dest, cancel_event=cancel2, chunk_size=4, transport=transport2)
            self.assertFalse(dest.exists())

    def test_detect_meteors_stops_between_slots_and_runs_downloads_outside_sdk_lock(self):
        start = datetime(2026, 10, 22, 22, 0, tzinfo=timezone.utc)
        end = start + timedelta(minutes=15)
        cancel = threading.Event()
        fetched = []
        lock_states = []

        class _FakeRecording:
            def __init__(self, slot_idx: int):
                self.slot_idx = slot_idx
                self.receiver_code = "BEHUMA_SYS001"
                self.start = start + timedelta(minutes=5 * slot_idx)
                self.end = self.start + timedelta(minutes=5)
                self.png_name = f"slot_{slot_idx}.png"
                self.png_url = f"https://brams.invalid/slot_{slot_idx}.png"
                self.wav_url = None
                self.wav_name = None

            @property
            def spectrogram(self):
                lock_states.append(SDK_LOCK.locked())
                fetched.append(self.slot_idx)
                if self.slot_idx == 0:
                    cancel.set()
                return b"\x89PNG\r\n\x1a\nfake"

        class _FakeFiles:
            def __iter__(self):
                return iter([_FakeRecording(0), _FakeRecording(1), _FakeRecording(2)])

            def close(self):
                return None

        @contextmanager
        def fake_fresh_sdk():
            with SDK_LOCK:
                receiver = SimpleNamespace()
                api = SimpleNamespace(base_url="https://brams.invalid")
                cfg = SimpleNamespace(get=lambda *a: "/v4/file.php")
                yield receiver, api, cfg

        fake_catalog = {
            "BEHUMA_SYS001": SimpleNamespace(
                receiver_code="BEHUMA_SYS001", name="Humain 1", location_code="BEHUMA",
                antenna=1, start="2015-01-01", end="2026-12-31",
            )
        }
        with (
            patch("agent_app.brams_sdk._fresh_sdk", fake_fresh_sdk),
            patch("agent_app.brams_sdk._catalog_from_sdk_with_meta", return_value=(fake_catalog, "", {})),
            patch("agent_app.brams_sdk._archive_files", return_value=_FakeFiles()),
            patch("agent_app.vision.analyze_spectrogram_bytes") as mock_infer,
        ):
            with cancellation_scope(cancel):
                with self.assertRaises(OperationCancelled):
                    _detect_meteors("BEHUMA_SYS001", start, end, max_slots=3)

        self.assertEqual(fetched, [0])
        self.assertEqual(lock_states, [False])
        mock_infer.assert_not_called()
        self.assertFalse(SDK_LOCK.locked())

    async def test_ask_honors_cancel_event_before_model_call(self):
        fake = FakeModel()
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(OperationCancelled) as ctx:
            await ask("Hello", Session(), Settings(), transport=fake.transport(), cancel_event=cancel)
        self.assertEqual(fake.requests, [])
        self.assertIn("cancelled", friendly_error(ctx.exception, Settings()).lower())

    async def test_ui_stop_button_cancels_event_and_rolls_back_pending_turn(self):
        from app import build_ui

        ui = build_ui(Settings())
        stop_btn = next(
            block for block in ui.blocks.values()
            if isinstance(block, gr.Button) and getattr(block, "elem_id", None) == "stop-message"
        )
        self.assertFalse(stop_btn.visible)

        callbacks = {fn.fn.__name__: fn.fn for fn in ui.fns.values()}
        self.assertIn("cancel_chat", callbacks)
        cancel_fn = next(fn for fn in ui.fns.values() if fn.fn.__name__ == "cancel_chat")
        self.assertFalse(cancel_fn.queue)

        holder = {"event": None, "observed": {}, "started": None, "pending_len": None, "question": ""}
        prior_transcript = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
        prior_session = Session(ui_messages=[{"id": "u1", "kind": "user"}, {"id": "a1", "kind": "assistant"}])
        started_ask = asyncio.Event()

        async def slow_ask(*args, **kwargs):
            started_ask.set()
            cancel_event = kwargs["cancel_event"]
            while not cancel_event.is_set():
                await asyncio.sleep(0.02)
            raise OperationCancelled("stopped")

        frames = []
        with patch("app.ask", side_effect=slow_ask):
            gen = callbacks["respond"]("Long scan", prior_transcript, prior_session, True, None, [], holder)
            first = await anext(gen)
            frames.append(first)
            self.assertTrue(first[-1]["visible"])  # Stop button shown while running
            self.assertEqual(len(first[0]), 3)  # Pending user bubble appended

            # Simulate user clicking Stop button immediately (unqueued callback)
            cancel_out = callbacks["cancel_chat"](holder, first[0], first[1], [])
            self.assertEqual(len(cancel_out[0]), 2)  # Uncommitted bubble rolled back
            self.assertEqual(len(cancel_out[1].ui_messages), 2)
            self.assertIn("cancelled", cancel_out[3].lower())
            self.assertTrue(cancel_out[4]["interactive"])  # Send re-enabled
            self.assertFalse(cancel_out[7]["visible"])  # Stop hidden again

            async for frame in gen:
                frames.append(frame)

        final = frames[-1]
        self.assertEqual(len(final[0]), 2)
        self.assertIn("cancelled", final[3].lower())
        self.assertFalse(final[-1]["visible"])


if __name__ == "__main__":
    unittest.main()
