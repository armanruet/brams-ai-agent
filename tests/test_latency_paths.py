"""Regressions for bounded model context and safe media shortcuts."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from agent_app.config import Settings
from agent_app.service import (
    MODEL_HISTORY_MAX_CHARS, Session, _bounded_model_history, _compact_detection_history,
    _group_cost, _simple_echo_details, ask,
)
from tests.fake_model import FakeModel


class ToolModel(FakeModel):
    def __init__(self, tool, arguments):
        super().__init__()
        self.tool = tool
        self.arguments = arguments

    def respond(self, payload):
        self.requests.append(payload)
        last = payload["messages"][-1]
        if last["role"] != "tool":
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_test", "type": "function", "function": {
                    "name": self.tool, "arguments": json.dumps(self.arguments)}}]}
            finish = "tool_calls"
        else:
            message = {"role": "assistant", "content": "The requested data is ready."}
            finish = "stop"
        return 200, {"id": "fixture", "object": "chat.completion", "created": 0,
                     "model": payload["model"], "choices": [{"index": 0, "message": message,
                     "finish_reason": finish}], "usage": {"prompt_tokens": 12,
                     "completion_tokens": 8, "total_tokens": 20}}


def detection_data():
    rows = [{"echo_index": i, "estimated_echo_utc": f"2026-10-22T22:00:{i:02d}Z",
             "time_offset_seconds": i, "confidence": .8, "confidence_tier": "high",
             "bbox_norm_xywh": [0.1, 0.2, 0.3, 0.4]} for i in range(1, 26)]
    return {"status": "ok", "analysis_performed": True, "receiver_name": "Kirchberg",
            "receiver_id": "LUKIRC_SYS001", "recording_start_utc": "2026-10-22T22:00:00Z",
            "recording_end_utc": "2026-10-22T22:05:00Z", "meteor_detected": True,
            "total_meteor_count": 25, "inspected_slots_count": 1, "available_slots_in_window": 1,
            "coverage_complete": True, "slots": [{"meteor_count": 25,
            "recording_start_utc": "2026-10-22T22:00:00Z", "detections": rows,
            "detections_truncated": False}]}


class LatencyPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_current_tool_result_is_capped_but_all_details_remain_available(self):
        fake = ToolModel("detect_meteors", {"search": "Kirchberg", "start_utc": "2026-10-22T22:00:00Z",
                                             "end_utc": "2026-10-22T22:05:00Z", "direct_answer": False})
        result = detection_data()
        with patch("agent_app.brams_tools.pybrams_setup_status", return_value={"ready": True, "version": "test"}), \
             patch("agent_app.brams_tools.vision_setup_status", return_value={"ready": True}), \
             patch("agent_app.brams_tools._detect_meteors", return_value=result):
            reply = await ask("Explain the meteor detection and confidence", Session(), Settings(), transport=fake.transport())
        self.assertEqual(len(fake.requests), 2)
        visible = json.loads(next(m["content"] for m in fake.requests[1]["messages"] if m["role"] == "tool"))
        self.assertEqual(len(visible["slots"][0]["detections"]), 3)
        self.assertTrue(visible["slots"][0]["detections_truncated"])
        returned = next(p.content for m in reply.session.messages for p in m.parts if isinstance(p, ToolReturnPart))
        self.assertEqual(len(returned["slots"][0]["detections"]), 25)
        self.assertEqual(len(next(e["result"] for e in reply.trace if e.get("event") == "tool_result")["slots"][0]["detections"]), 25)
        def no_request(request):
            raise AssertionError("A straightforward detail list should use verified saved evidence")
        followup = await ask("List each detected echo time and confidence", reply.session, Settings(),
                             transport=httpx.MockTransport(no_request))
        self.assertEqual(followup.requests, 0)
        self.assertIn("2026-10-22T22:00:25Z", followup.text)
        self.assertEqual(followup.text.count(" — confidence "), 25)

    async def test_simple_detection_with_image_is_not_treated_as_extra_task(self):
        args = {"search": "Kirchberg", "start_utc": "2026-10-22T22:00:00Z",
                "end_utc": "2026-10-22T22:05:00Z", "direct_answer": True}
        data = {**detection_data(), "annotated_artifacts": [{"artifact_id": "a" * 32}]}
        with patch("agent_app.brams_tools.pybrams_setup_status", return_value={"ready": True, "version": "test"}), \
             patch("agent_app.brams_tools.vision_setup_status", return_value={"ready": True}), \
             patch("agent_app.brams_tools._detect_meteors", return_value=data), \
             patch("agent_app.service.detection_path", return_value=Path("/tmp/verified.png")):
            fake = ToolModel("detect_meteors", args)
            reply = await ask("Detect meteor echoes for Kirchberg on 2026-10-22 from 22:00 to 22:05 UTC and show the image.",
                              Session(), Settings(), transport=fake.transport())
        self.assertEqual(reply.requests, 1)
        self.assertEqual(len(reply.detections), 1)

    async def test_simple_wav_shortcut_and_compound_request(self):
        args = {"search": "Kirchberg", "start_utc": "2026-10-22T22:00:00Z",
                "end_utc": "2026-10-22T22:05:00Z"}
        data = {"status": "ok", "downloaded": True, "artifact_id": "a" * 32,
                "receiver_id": "LUKIRC_SYS001", "receiver_name": "Kirchberg",
                "recording_start_utc": args["start_utc"], "recording_end_utc": args["end_utc"],
                "filename": "recording.wav", "wav_source_url": "https://example.invalid/file.wav"}
        with patch("agent_app.brams_tools.pybrams_setup_status", return_value={"ready": True, "version": "test"}), \
             patch("agent_app.brams_tools._load_wav", return_value=data), \
             patch("agent_app.service.wav_path", return_value=Path("/tmp/verified.wav")):
            fake = ToolModel("get_wav", args)
            reply = await ask("Play the WAV for Kirchberg from 22:00 to 22:05 UTC on 2026-10-22", Session(), Settings(), transport=fake.transport())
            self.assertEqual(len(fake.requests), 1)
            self.assertEqual(reply.requests, 1)
            self.assertEqual(len(reply.audio), 1)
            self.assertIn("unchanged original", reply.text)
            fake = ToolModel("get_wav", args)
            await ask("Play the WAV and explain the frequency", Session(), Settings(), transport=fake.transport())
            self.assertGreaterEqual(len(fake.requests), 2)
            # A request covering an hour cannot be silently reduced to one file.
            hour_args = {**args, "end_utc": "2026-10-22T23:00:00Z"}
            fake = ToolModel("get_wav", hour_args)
            await ask("Play a WAV recording from 22:00 to 23:00 UTC on 2026-10-22", Session(), Settings(), transport=fake.transport())
            self.assertGreaterEqual(len(fake.requests), 2)
            # Relative dates need a verified current clock before shortcutting.
            fake = ToolModel("get_wav", args)
            await ask("Play a WAV recording yesterday from 22:00 to 22:05 UTC", Session(), Settings(), transport=fake.transport())
            self.assertGreaterEqual(len(fake.requests), 2)

    async def test_direct_answer_identifies_replayed_evidence(self):
        args = {"search": "Kirchberg", "start_utc": "2026-10-22T22:00:00Z",
                "end_utc": "2026-10-22T22:05:00Z"}
        data = {"status": "ok", "downloaded": True, "artifact_id": "a" * 32,
                "receiver_id": "LUKIRC_SYS001", "receiver_name": "Kirchberg",
                "recording_start_utc": args["start_utc"], "recording_end_utc": args["end_utc"],
                "benchmark_provenance": {"mode": "snapshot_replay"}}
        with patch("agent_app.brams_tools.pybrams_setup_status", return_value={"ready": True, "version": "test"}), \
             patch("agent_app.brams_tools._load_wav", return_value=data), \
             patch("agent_app.service.wav_path", return_value=Path("/tmp/verified.wav")):
            reply = await ask("Play the WAV for Kirchberg from 22:00 to 22:05 UTC on 2026-10-22",
                              Session(), Settings(), transport=ToolModel("get_wav", args).transport())
        self.assertIn("Benchmark replay", reply.text)
        self.assertIn("no live archive query", reply.text)

    def test_history_budget_keeps_whole_tool_transactions(self):
        messages = []
        for i in range(4):
            messages += [ModelRequest(parts=[UserPromptPart(f"question {i}")]),
                         ModelResponse(parts=[ToolCallPart("get_wav", {"search": "x"}, f"call{i}")]),
                         ModelRequest(parts=[ToolReturnPart("get_wav", {"payload": "x" * 300}, f"call{i}")]),
                         ModelResponse(parts=[TextPart(f"answer {i}")])]
        selected = _bounded_model_history(messages, max_chars=1200)
        calls = {p.tool_call_id for m in selected for p in m.parts if isinstance(p, ToolCallPart)}
        returns = {p.tool_call_id for m in selected for p in m.parts if isinstance(p, ToolReturnPart)}
        self.assertTrue(calls)
        self.assertEqual(calls, returns)
        self.assertLess(len(selected), len(messages))
        self.assertEqual(len(messages), 16)

    def test_details_do_not_reuse_older_scan_after_another_data_result(self):
        history = [ModelRequest(parts=[ToolReturnPart("detect_meteors", detection_data(), "det")]),
                   ModelRequest(parts=[ToolReturnPart("get_spectrogram", {"status": "ok"}, "spec")])]
        self.assertIsNone(_simple_echo_details("List each detected echo time and confidence", history))
        self.assertIsNone(_simple_echo_details("List each detected echo time on 2026-10-23", history))

    def test_detailed_detection_history_retains_bounded_rows_beyond_3k_chars(self):
        single = detection_data()
        self.assertGreater(len(json.dumps(single)), 3000)
        compacted = _compact_detection_history([ModelRequest(parts=[ToolReturnPart("detect_meteors", single, "d1")])], detailed=True)
        slot0 = compacted[0].parts[0].content["slots"][0]
        self.assertEqual(len(slot0["detections"]), 25)
        self.assertNotIn("bbox_norm_xywh", slot0["detections"][0])

        multi = detection_data()
        multi["slots"] = [{**multi["slots"][0], "recording_start_utc": f"2026-10-22T22:{i * 5:02d}:00Z"} for i in range(12)]
        self.assertGreater(len(json.dumps(multi)), 40000)
        compacted_multi = _compact_detection_history([ModelRequest(parts=[ToolReturnPart("detect_meteors", multi, "d12")])], detailed=True)
        kept = [len(slot["detections"]) for slot in compacted_multi[0].parts[0].content["slots"]]
        self.assertTrue(all(count >= 1 for count in kept))
        self.assertLessEqual(len(json.dumps(compacted_multi[0].parts[0].content)), MODEL_HISTORY_MAX_CHARS)

    def test_bounded_model_history_truncates_oversized_latest_turn_instead_of_dropping(self):
        messages = [
            ModelRequest(parts=[UserPromptPart("List all stations")]),
            ModelResponse(parts=[ToolCallPart("list_receivers", {"search": ""}, "big1")]),
            ModelRequest(parts=[ToolReturnPart("list_receivers", {
                "status": "ok", "receiver_id": "BEHUMA_SYS001", "blob": "x" * 15000,
            }, "big1")]),
            ModelResponse(parts=[TextPart("Stations listed.")]),
        ]
        selected = _bounded_model_history(messages, max_chars=MODEL_HISTORY_MAX_CHARS)
        self.assertEqual(len(selected), 4)
        self.assertLessEqual(_group_cost(selected), MODEL_HISTORY_MAX_CHARS)
        calls = {p.tool_call_id for m in selected for p in m.parts if isinstance(p, ToolCallPart)}
        returns = {p.tool_call_id for m in selected for p in m.parts if isinstance(p, ToolReturnPart)}
        self.assertEqual(calls, returns)
        ret = next(p.content for m in selected for p in m.parts if isinstance(p, ToolReturnPart))
        self.assertTrue(ret.get("history_truncated"))
        self.assertEqual(ret.get("receiver_id"), "BEHUMA_SYS001")


if __name__ == "__main__":
    unittest.main()
