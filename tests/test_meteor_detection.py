"""Offline unit and protocol tests for local YOLOv11-n BRAMS meteor echo detection."""

from datetime import datetime, timedelta, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
import wave

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from pydantic_ai import ModelRetry

from agent_app.artifacts import detection_path, save_detection_image, save_spectrogram, spectrogram_path
from agent_app.brams_sdk import _detect_meteors
from agent_app.brams_tools import detect_meteors
from agent_app.config import Settings
from agent_app.recovery import evidence_summary
from agent_app.service import Reply, Session, ask
from agent_app.vision import (
    analyze_spectrogram_bytes,
    decode_yolo_output,
    extract_spectrogram_canvas,
    letterbox_rgb,
    vision_setup_status,
)
from tests.fake_model import FakeModel


START = datetime(2026, 10, 22, 22, 0, tzinfo=timezone.utc)
END = START + timedelta(hours=1)
READY_PYBRAMS = {"ready": True, "version": "0.1.7", "code": "ready", "message": "ready"}
READY_VISION = {
    "ready": True,
    "code": "ready",
    "backend": "onnxruntime",
    "model_filename": "yolo11n_brams.onnx",
    "model_name": "yolo11n_brams",
    "architecture": "YOLOv11-n",
    "input_mode": "archive_png",
    "imgsz": 640,
    "message": "ready",
}


def make_framed_brams_png(canvas_w: int = 864, canvas_h: int = 595) -> bytes:
    """Create a synthetic BIRA-IASB / PyBRAMS framed PNG (64/64/96/96 + [128,128,128] border)."""
    full_w = canvas_w + 64 + 96
    full_h = canvas_h + 64 + 96
    arr = np.zeros((full_h, full_w, 3), dtype=np.uint8)
    # 1-px gray border at x=63 and y=63
    arr[63, 64 : 64 + canvas_w] = [128, 128, 128]
    arr[64 + canvas_h, 64 : 64 + canvas_w] = [128, 128, 128]
    arr[64 : 64 + canvas_h, 63] = [128, 128, 128]
    arr[64 : 64 + canvas_h, 64 + canvas_w] = [128, 128, 128]
    # Jet-like blue interior + bright vertical meteor echo
    arr[64 : 64 + canvas_h, 64 : 64 + canvas_w] = [0, 0, 180]
    arr[250:420, 64 + 216 : 64 + 222] = [120, 255, 120]
    buf = BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def make_borderless_rmz_png(w: int = 864, h: int = 595) -> bytes:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = [0, 20, 170]
    arr[h // 2 - 2 : h // 2 + 2, :] = [255, 40, 0]
    buf = BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def make_wav_bytes(duration_s: float = 1.0, rate: int = 5512) -> bytes:
    t = np.linspace(0.0, duration_s, int(rate * duration_s), endpoint=False)
    samples = (0.3 * np.sin(2.0 * np.pi * 1000.0 * t) * 32767).astype(np.int16)
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


class _FakeDetector:
    backend = "onnxruntime"
    manifest = {"model_name": "yolo11n_brams", "architecture": "YOLOv11-n"}

    def __init__(self, boxes):
        self._boxes = boxes

    def predict(self, canvas_image, conf_threshold: float, iou_threshold: float):
        return [dict(item) for item in self._boxes if item["confidence"] >= conf_threshold]


class VisionPreprocessingAndDecodeTests(unittest.TestCase):
    def test_extract_canvas_crops_bira_frame_and_preserves_borderless_rmz(self):
        with Image.open(BytesIO(make_framed_brams_png(864, 595))) as framed:
            canvas, box = extract_spectrogram_canvas(framed)
            self.assertEqual(canvas.size, (864, 595))
            self.assertEqual(box, (64, 64, 928, 659))

        with Image.open(BytesIO(make_borderless_rmz_png(864, 595))) as raw:
            canvas2, box2 = extract_spectrogram_canvas(raw)
            self.assertEqual(canvas2.size, (864, 595))
            self.assertEqual(box2, (0, 0, 864, 595))

    def test_decode_yolo_output_handles_anchor_free_and_end_to_end_nms(self):
        img = Image.new("RGB", (864, 595), (0, 0, 160))
        tensor, transform = letterbox_rgb(img, imgsz=640, resize_mode="letterbox")
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertTrue(0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0)

        scale = transform["scale_x"]
        pad_x, pad_y = transform["pad_x"], transform["pad_y"]
        # Target box on canvas: [200, 150, 220, 350]
        cx = 210.0 * scale + pad_x
        cy = 250.0 * scale + pad_y
        bw = 20.0 * scale
        bh = 200.0 * scale

        raw_head = np.zeros((1, 5, 8400), dtype=np.float32)
        raw_head[0, :, 10] = [cx, cy, bw, bh, 0.88]
        # Duplicate overlapping candidate that NMS should suppress
        raw_head[0, :, 11] = [cx + 1.0, cy + 1.0, bw, bh, 0.61]
        # Below-threshold candidate
        raw_head[0, :, 12] = [100.0, 200.0, 15.0, 80.0, 0.12]

        decoded = decode_yolo_output(raw_head, transform, conf_threshold=0.25, iou_threshold=0.45)
        self.assertEqual(len(decoded), 1)
        self.assertAlmostEqual(decoded[0]["confidence"], 0.88, places=4)
        x1, y1, x2, y2 = decoded[0]["canvas_xyxy"]
        self.assertAlmostEqual(x1, 200.0, delta=1.5)
        self.assertAlmostEqual(y1, 150.0, delta=1.5)
        self.assertAlmostEqual(x2, 220.0, delta=1.5)
        self.assertAlmostEqual(y2, 350.0, delta=1.5)

        e2e = np.zeros((1, 10, 6), dtype=np.float32)
        e2e[0, 0] = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, 0.77, 0.0]
        decoded_e2e = decode_yolo_output(e2e, transform, conf_threshold=0.25)
        self.assertEqual(len(decoded_e2e), 1)
        self.assertAlmostEqual(decoded_e2e[0]["confidence"], 0.77, places=4)

    def test_analyze_spectrogram_maps_utc_timestamps_and_strips_pixel_boxes(self):
        fake_det = _FakeDetector([
            {"class_id": 0, "confidence": 0.42, "canvas_xyxy": [640.0, 200.0, 656.0, 360.0]},
            {"class_id": 0, "confidence": 0.91, "canvas_xyxy": [208.0, 180.0, 224.0, 390.0]},
        ])
        rec_start = datetime(2026, 10, 22, 22, 0, 0, tzinfo=timezone.utc)
        rec_end = datetime(2026, 10, 22, 22, 5, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp, \
             patch("agent_app.artifacts.DETECTION_DIR", Path(tmp)), \
             patch("agent_app.vision._get_detector", return_value=fake_det):
            res = analyze_spectrogram_bytes(
                make_framed_brams_png(864, 595),
                recording_start=rec_start,
                recording_end=rec_end,
                receiver_label="Kirchberg (LUKIRC_SYS001)",
                save_overlay=True,
            )
            self.assertTrue(res["meteor_detected"])
            self.assertEqual(res["meteor_count"], 2)
            self.assertEqual(res["high_confidence_count"], 1)
            # Chronologically ordered: center=216/864=0.25 -> +75s; center=648/864=0.75 -> +225s
            self.assertEqual(res["detections"][0]["estimated_echo_utc"], "2026-10-22T22:01:15Z")
            self.assertEqual(res["detections"][0]["confidence_tier"], "high")
            self.assertEqual(res["detections"][1]["estimated_echo_utc"], "2026-10-22T22:03:45Z")
            self.assertEqual(res["detections"][1]["confidence_tier"], "moderate")
            self.assertTrue(all("image_xyxy" not in d for d in res["detections"]))
            saved_overlay = detection_path(res["annotated_artifact_id"])
            self.assertIsNotNone(saved_overlay)
            self.assertTrue(saved_overlay.is_file())


class MeteorDetectionSdkAndToolTests(unittest.IsolatedAsyncioTestCase):
    def _make_fake_stream(self, png_data: bytes, n_slots: int = 12, fail_png: bool = False):
        rows = []
        for idx in range(n_slots):
            slot_start = START + timedelta(minutes=5 * idx)
            slot_end = slot_start + timedelta(minutes=5)
            stamp = slot_start.strftime("%Y%m%d_%H%M")

            class _Rec:
                def __init__(self, s, e, st):
                    self.receiver_code = "LUKIRC_SYS001"
                    self.start = s
                    self.end = e
                    self.year = s.year
                    self.month = s.month
                    self.day = s.day
                    self.hours = s.hour
                    self.minutes = s.minute
                    self.png_name = f"RAD_BEDOUR_{st}_LUKIRC_SYS001.png"
                    self.wav_name = f"RAD_BEDOUR_{st}_LUKIRC_SYS001.wav"
                    self.png_url = f"https://example.invalid/{self.png_name}"
                    self.wav_url = f"https://example.invalid/{self.wav_name}"

                @property
                def spectrogram(self):
                    if fail_png:
                        raise RuntimeError("PNG unavailable")
                    return png_data

                def save_raw(self, directory: Path):
                    (Path(directory) / self.wav_name).write_bytes(make_wav_bytes())

            rows.append(_Rec(slot_start, slot_end, stamp))

        def _gen():
            for item in rows:
                yield item

        return _gen()

    def _patch_fresh_sdk(self):
        from contextlib import contextmanager

        catalog = {
            "LUKIRC_SYS001": SimpleNamespace(
                receiver_code="LUKIRC_SYS001",
                name="Kirchberg",
                location_code="LUKIRC",
                antenna=1,
                start="2021-01-01",
                end="",
            )
        }
        api_stub = SimpleNamespace(base_url="https://brams.aeronomie.be/v4/")
        cfg_stub = SimpleNamespace(get=lambda section, key: "file.php")

        @contextmanager
        def fake_ctx():
            yield SimpleNamespace(get_all=lambda: catalog), api_stub, cfg_stub

        return patch("agent_app.brams_sdk._fresh_sdk", fake_ctx)

    def test_single_slot_vs_full_hour_scan_and_wav_fallback(self):
        png = make_framed_brams_png()
        fake_det = _FakeDetector([
            {"class_id": 0, "confidence": 0.84, "canvas_xyxy": [300.0, 200.0, 314.0, 380.0]}
        ])
        with tempfile.TemporaryDirectory() as tmp, \
             patch("agent_app.artifacts.DETECTION_DIR", Path(tmp)), \
             patch("agent_app.vision._get_detector", return_value=fake_det), \
             self._patch_fresh_sdk():
            # 1. Default max_slots=1 across a 12-slot hour
            with patch("agent_app.brams_sdk._archive_files", return_value=self._make_fake_stream(png, 12)):
                one = _detect_meteors("Kirchberg", START, END, max_slots=1)
            self.assertEqual(one["status"], "ok")
            self.assertEqual(one["inspected_slots_count"], 1)
            self.assertEqual(one["available_slots_in_window"], 12)
            self.assertEqual(one["uninspected_slots_count"], 11)
            self.assertFalse(one["coverage_complete"])
            self.assertEqual(len(one["uncovered_slot_starts_utc"]), 11)
            summary = evidence_summary([{"event": "tool_result", "name": "detect_meteors", "result": one}])
            self.assertIn("Coverage is incomplete", summary)
            self.assertEqual(one["total_meteor_count"], 1)
            self.assertEqual(len(one["annotated_artifacts"]), 1)
            self.assertNotIn(tmp, json.dumps(one))

            # 2. Full-hour max_slots=12 caps visual overlays at 3 while scanning all 12 slots
            with patch("agent_app.brams_sdk._archive_files", return_value=self._make_fake_stream(png, 12)):
                twelve = _detect_meteors("Kirchberg", START, END, max_slots=12)
            self.assertEqual(twelve["inspected_slots_count"], 12)
            self.assertTrue(twelve["coverage_complete"])
            self.assertEqual(twelve["uncovered_slot_starts_utc"], [])
            self.assertEqual(twelve["total_meteor_count"], 12)
            self.assertEqual(len(twelve["annotated_artifacts"]), 3)

            # 3. Automatic WAV fallback when png_url fails
            with patch("agent_app.brams_sdk._archive_files", return_value=self._make_fake_stream(png, 2, fail_png=True)):
                fallback = _detect_meteors("Kirchberg", START, END, max_slots=1)
            self.assertEqual(fallback["status"], "ok")
            self.assertEqual(fallback["slots"][0]["input_kind"], "rmz_from_wav")

    def test_artifact_id_reuse_and_provenance_validation(self):
        png = make_framed_brams_png()
        fake_det = _FakeDetector([
            {"class_id": 0, "confidence": 0.75, "canvas_xyxy": [432.0, 210.0, 445.0, 360.0]}
        ])
        with tempfile.TemporaryDirectory() as tmp, \
             patch("agent_app.artifacts.SPECTROGRAM_DIR", Path(tmp) / "spec"), \
             patch("agent_app.artifacts.DETECTION_DIR", Path(tmp) / "det"), \
             patch("agent_app.vision._get_detector", return_value=fake_det):
            aid = save_spectrogram(png)
            spec_file = spectrogram_path(aid)
            sidecar = {
                "artifact_id": aid,
                "filename": "RAD_BEDOUR_20261022_2200_LUKIRC_SYS001.png",
                "receiver_id": "LUKIRC_SYS001",
                "receiver_name": "Kirchberg",
                "recording_start_utc": "2026-10-22T22:00:00Z",
                "recording_end_utc": "2026-10-22T22:05:00Z",
                "image_source_url": "https://example.invalid/kirchberg.png",
                "source_url": "https://brams.aeronomie.be/v4/file.php",
                "sha256": hashlib.sha256(png).hexdigest(),
            }
            spec_file.with_suffix(".json").write_text(json.dumps(sidecar))

            reused = _detect_meteors("Kirchberg", START, END, artifact_id=aid)
            self.assertEqual(reused["status"], "ok")
            self.assertTrue(reused["reused_artifact"])
            self.assertFalse(reused["downloaded"])
            self.assertEqual(reused["total_meteor_count"], 1)
            self.assertFalse(reused["coverage_complete"])
            self.assertIsNone(reused["available_slots_in_window"])
            with self._patch_fresh_sdk(), patch("agent_app.brams_sdk._archive_files", return_value=self._make_fake_stream(png, 12)):
                full = _detect_meteors("Kirchberg", START, END, max_slots=12, artifact_id=aid)
            self.assertEqual(full["inspected_slots_count"], 12)
            self.assertTrue(full["coverage_complete"])

            mismatch = _detect_meteors("BEHUMA", START, END, artifact_id=aid)
            self.assertEqual(mismatch["status"], "artifact_mismatch")

            spec_file.write_bytes(make_borderless_rmz_png())
            tampered = _detect_meteors("Kirchberg", START, END, artifact_id=aid)
            self.assertEqual(tampered["status"], "artifact_changed")

    async def test_validation_and_setup_required_handling(self):
        for kwargs in [
            {"search": "", "start_utc": START, "end_utc": END},
            {"search": "Kirchberg", "start_utc": START, "end_utc": END, "max_slots": 0},
            {"search": "Kirchberg", "start_utc": START, "end_utc": END, "max_slots": 13},
            {"search": "Kirchberg", "start_utc": START, "end_utc": START + timedelta(hours=3), "max_slots": 4},
            {"search": "Kirchberg", "start_utc": START, "end_utc": END, "confidence_threshold": 0.01},
        ]:
            with self.assertRaises(ModelRetry):
                await detect_meteors(**kwargs)

        missing_status = vision_setup_status("/nonexistent/yolo11n_brams.onnx")
        self.assertFalse(missing_status["ready"])
        self.assertEqual(missing_status["code"], "weights_missing")

        with patch("agent_app.brams_tools.pybrams_setup_status", return_value=READY_PYBRAMS), \
             patch("agent_app.brams_tools.vision_setup_status", return_value=missing_status):
            res = await detect_meteors("Kirchberg", START, END)
        self.assertEqual(res["status"], "setup_required")
        self.assertFalse(res["analysis_performed"])

    async def test_end_to_end_agent_turn_and_gradio_image_attachment(self):
        from app import build_ui

        class DetectFakeModel(FakeModel):
            def respond(self, payload):
                self.requests.append(payload)
                last = payload["messages"][-1]
                if last["role"] != "tool":
                    return 200, {
                        "id": "det_1",
                        "object": "chat.completion",
                        "created": 1793181600,
                        "model": payload["model"],
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": "call_det_1",
                                    "type": "function",
                                    "function": {
                                        "name": "detect_meteors",
                                        "arguments": json.dumps({
                                            "search": "Kirchberg",
                                            "start_utc": "2026-10-22T22:00:00Z",
                                            "end_utc": "2026-10-22T23:00:00Z",
                                            "max_slots": 1,
                                            "direct_answer": True,
                                        }),
                                    },
                                }],
                            },
                            "finish_reason": "tool_calls",
                        }],
                        "usage": {"prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32},
                    }
                return 200, {
                    "id": "det_2",
                    "object": "chat.completion",
                    "created": 1793181601,
                    "model": payload["model"],
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Yes — 1 meteor echo was detected in the Kirchberg spectrogram at 22:01:15Z.",
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 40, "completion_tokens": 18, "total_tokens": 58},
                }

        fake = DetectFakeModel()
        with tempfile.TemporaryDirectory() as tmp, \
             patch("agent_app.artifacts.DETECTION_DIR", Path(tmp)):
            overlay_id = save_detection_image(make_framed_brams_png(200, 140))
            tool_payload = {
                "status": "ok",
                "downloaded": True,
                "reused_artifact": False,
                "analysis_performed": True,
                "receiver_id": "LUKIRC_SYS001",
                "receiver_name": "Kirchberg",
                "source_url": "https://brams.aeronomie.be/v4/file.php",
                "image_source_url": "https://brams.aeronomie.be/downloader.php?type=png",
                "available_slots_in_window": 12,
                "inspected_slots_count": 1,
                "uninspected_slots_count": 11,
                "slots_with_meteors": 1,
                "meteor_detected": True,
                "total_meteor_count": 1,
                "high_confidence_count": 1,
                "max_confidence": 0.91,
                "recording_start_utc": "2026-10-22T22:00:00Z",
                "recording_end_utc": "2026-10-22T22:05:00Z",
                "filename": "RAD_BEDOUR_20261022_2200_LUKIRC_SYS001.png",
                "model_name": "yolo11n_brams",
                "model_architecture": "YOLOv11-n",
                "backend": "onnxruntime",
                "confidence_threshold": 0.25,
                "annotated_artifacts": [{
                    "artifact_id": overlay_id,
                    "filename": "RAD_BEDOUR_20261022_2200_LUKIRC_SYS001.png",
                    "recording_start_utc": "2026-10-22T22:00:00Z",
                    "recording_end_utc": "2026-10-22T22:05:00Z",
                    "meteor_count": 1,
                    "label": "Meteor detection · Kirchberg · 2026-10-22T22:00:00Z (1 meteor(s))",
                }],
                "slots": [{
                    "filename": "RAD_BEDOUR_20261022_2200_LUKIRC_SYS001.png",
                    "recording_start_utc": "2026-10-22T22:00:00Z",
                    "recording_end_utc": "2026-10-22T22:05:00Z",
                    "input_kind": "archive_png",
                    "meteor_detected": True,
                    "meteor_count": 1,
                    "high_confidence_count": 1,
                    "max_confidence": 0.91,
                    "detections": [{
                        "echo_index": 1,
                        "confidence": 0.91,
                        "confidence_tier": "high",
                        "time_offset_seconds": 75.0,
                        "time_span_seconds": [72.0, 78.0],
                        "bbox_norm_xywh": [0.25, 0.45, 0.02, 0.3],
                        "aspect_ratio_hw": 10.3,
                        "area_percent": 0.6,
                        "estimated_echo_utc": "2026-10-22T22:01:15Z",
                    }],
                    "detections_truncated": False,
                    "annotated_artifact_id": overlay_id,
                    "inference_ms": 14.2,
                }],
                "caveat": "Automated spectrogram detection (yolo11n_brams, conf >= 0.25).",
            }

            with patch("agent_app.brams_tools.pybrams_setup_status", return_value=READY_PYBRAMS), \
                 patch("agent_app.brams_tools.vision_setup_status", return_value=READY_VISION), \
                 patch("agent_app.brams_tools._detect_meteors", return_value=tool_payload):
                reply = await ask(
                    "Check at a spectrogram of Kirchberg station from 22:00 to 23:00 UTC on 22/10/2026 if there's a meteor or not.",
                    Session(),
                    Settings(),
                    transport=fake.transport(),
                )

            self.assertEqual(reply.images, [])
            self.assertEqual(len(reply.detections), 1)
            # A single analyzed slot cannot shortcut a request for the whole
            # hour when eleven available slots remain uninspected.
            self.assertEqual(len(fake.requests), 2)
            self.assertEqual(reply.requests, 2)
            self.assertIn("1 meteor echo", reply.text)
            self.assertIn("Coverage is incomplete", reply.text)
            self.assertIn("1 of 12 available recordings", reply.text)
            self.assertNotIn("Verified tool results", reply.text)
            self.assertNotIn("| Echo #", reply.text)
            self.assertTrue(any(event.get("name") == "detect_meteors" and event["event"] == "tool_result" for event in reply.trace))
            self.assertEqual(len(reply.session.messages[-2].parts), 1)
            self.assertIn("Coverage is incomplete", str(reply.session.messages[-1].parts))
            # An explicit request for explanation must go back to the model.
            with patch("agent_app.brams_tools.pybrams_setup_status", return_value=READY_PYBRAMS), \
                 patch("agent_app.brams_tools.vision_setup_status", return_value=READY_VISION), \
                 patch("agent_app.brams_tools._detect_meteors", return_value=tool_payload):
                detailed_fake = DetectFakeModel()
                detailed = await ask("Explain the meteor detection and confidence", Session(), Settings(),
                                     transport=detailed_fake.transport())
            self.assertEqual(len(detailed_fake.requests), 2)
            self.assertNotIn("Verified tool results", detailed.text)
            # Follow-up prompts retain valid call/return pairs and detector context.
            history_fake = FakeModel()
            follow_up = await ask("HISTORY", reply.session, Settings(), transport=history_fake.transport())
            messages = history_fake.requests[-1]["messages"]
            detector_return = next(msg for msg in messages if msg.get("role") == "tool")
            self.assertNotIn("bbox_norm_xywh", detector_return["content"])
            self.assertTrue(any("bbox_norm_xywh" in str(getattr(part, "content", ""))
                                for message in follow_up.session.messages for part in message.parts))
            detailed_history_fake = FakeModel()
            await ask("Show detailed echo times and confidence", follow_up.session, Settings(),
                      transport=detailed_history_fake.transport())
            self.assertIn("bbox_norm_xywh", json.dumps(detailed_history_fake.requests[-1]))
            calls = {call["id"] for msg in messages for call in msg.get("tool_calls", [])}
            returns = {msg["tool_call_id"] for msg in messages if msg["role"] == "tool"}
            self.assertEqual(calls, returns)
            # Ensure local filesystem directory never leaked into LLM payloads
            self.assertNotIn(tmp, json.dumps(fake.requests))

            summary = evidence_summary(reply.trace)
            self.assertIn("1 of 12", summary)
            self.assertIn("confidence threshold: 0.25", summary)

            ui = build_ui(Settings())
            callback = next(fn.fn for fn in ui.fns.values() if fn.fn.__name__ == "respond")
            with patch("app.ask", new=AsyncMock(return_value=reply)):
                outputs = [item async for item in callback("Check Kirchberg", [], Session(), True)]
            transcript = outputs[-1][0]
            self.assertIsInstance(transcript[-1]["content"], gr.Image)
            chatbot = next(c for c in ui.blocks.values() if isinstance(c, gr.Chatbot))
            serialized = chatbot.postprocess(transcript)
            self.assertEqual(serialized.root[-1].content[0].component, "image")

    def test_real_onnx_model_inference_smoke(self):
        status = vision_setup_status()
        if not status["ready"]:
            self.skipTest("Trained ONNX weights not present")
        self.assertEqual(status["backend"], "onnxruntime")
        with tempfile.TemporaryDirectory() as tmp, \
             patch("agent_app.artifacts.DETECTION_DIR", Path(tmp)):
            res = analyze_spectrogram_bytes(
                make_framed_brams_png(864, 595),
                recording_start=START,
                recording_end=START + timedelta(minutes=5),
                receiver_label="SmokeTest",
                save_overlay=True,
            )
            self.assertEqual(res["backend"], "onnxruntime")
            self.assertEqual(res["model_name"], "yolo11n_brams")
            self.assertIsNotNone(detection_path(res["annotated_artifact_id"]))


if __name__ == "__main__":
    unittest.main()
