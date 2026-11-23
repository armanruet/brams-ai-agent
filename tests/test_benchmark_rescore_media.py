"""Offline rescoring must use recorded execution outcomes, not current files."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent_app import benchmark


class RescoreMediaTests(unittest.TestCase):
    def rescore(self, original_score, expected_media):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); original = root / "original"; original.mkdir()
            benchmark.write_json(original / "manifest.json", {"config": {"models": []}, "repetitions": 1})
            turn = {"required_calls": [], "facts": [], "require_source": False,
                    "kind": "recall", "expected_media": expected_media}
            benchmark.write_json(original / "cases.json", {"cases": [{"id": "F01", "turns": [turn]}]})
            row = {"model_id": "fixture", "case_id": "F01", "category": "media", "repetition": 1,
                   "turn": 1, "status": "completed", "elapsed_seconds": 1, "prompt": "fixture",
                   "model_answer": None, "displayed_answer": "fixture", "trace": [], "score": original_score}
            (original / "turns.jsonl").write_text(json.dumps(row) + "\n")
            before = (original / "turns.jsonl").read_bytes()
            with patch("agent_app.benchmark.ask", side_effect=AssertionError("No model calls in rescore")):
                benchmark.rescore(original, root / "out")
            self.assertEqual((original / "turns.jsonl").read_bytes(), before)
            return json.loads((root / "out" / "turns.jsonl").read_text())

    def test_preserves_recorded_direct_answer_and_media_after_files_removed(self):
        original = {"direct_answer_completion": True, "audio_attachments": 1,
                    "image_attachments": 2, "detection_attachments": 3}
        row = self.rescore(original, ["audio", "image", "detection"])
        self.assertEqual(row["original_score"], original)
        self.assertTrue(row["score"]["direct_answer_completion"])
        self.assertEqual([row["score"][name] for name in
                          ("audio_attachments", "image_attachments", "detection_attachments")], [1, 2, 3])
        self.assertTrue(row["score"]["media_delivery_pass"])

    def test_explicit_recorded_zero_is_failure_and_direct_false_is_preserved(self):
        row = self.rescore({"direct_answer_completion": False, "audio_attachments": 0}, ["audio"])
        self.assertIs(row["score"]["direct_answer_completion"], False)
        self.assertIs(row["score"]["media_delivery_pass"], False)
        self.assertIsNone(row["score"]["image_attachments"])
        self.assertIsNone(row["score"]["detection_attachments"])

    def test_missing_or_invalid_original_counts_are_unknown(self):
        row = self.rescore({"audio_attachments": True, "image_attachments": -1}, ["audio", "image"])
        self.assertIsNone(row["score"]["direct_answer_completion"])
        self.assertIsNone(row["score"]["audio_attachments"])
        self.assertIsNone(row["score"]["image_attachments"])
        self.assertIsNone(row["score"]["media_delivery_pass"])

    def test_rescore_recomputes_controlled_fixture_exercised(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); original = root / "original"; original.mkdir()
            benchmark.write_json(original / "manifest.json", {
                "config": {"models": [{"id": "fixture", "model": "fixture"}]},
                "selected_models": ["fixture"], "repetitions": 1, "mode": "live",
                "case_ids": ["F_EX", "F_MISS", "F_PLAIN"],
                "case_evidence_modes": {"F_EX": "controlled_fixture", "F_MISS": "controlled_fixture", "F_PLAIN": "live"},
            })
            turn = {"required_calls": [], "facts": [], "require_source": False, "kind": "recall", "expected_media": []}
            benchmark.write_json(original / "cases.json", {"cases": [
                {"id": "F_EX", "controlled_fixture": "availability_zero", "turns": [turn]},
                {"id": "F_MISS", "controlled_fixture": "availability_zero", "turns": [turn]},
                {"id": "F_PLAIN", "turns": [turn]},
            ]})
            rows = [
                {"model_id": "fixture", "case_id": "F_EX", "category": "controlled_availability",
                 "evidence_mode": "controlled_fixture", "repetition": 1, "turn": 1, "status": "completed",
                 "elapsed_seconds": 1, "prompt": "p", "model_answer": "a", "displayed_answer": "a",
                 "trace": [{"event": "tool_result", "name": "check_availability", "result": {
                     "benchmark_fixture": "availability_zero", "status": "ok"}}],
                 "score": {"controlled_fixture_exercised": True}},
                {"model_id": "fixture", "case_id": "F_MISS", "category": "controlled_availability",
                 "evidence_mode": "controlled_fixture", "repetition": 1, "turn": 1, "status": "completed",
                 "elapsed_seconds": 1, "prompt": "p", "model_answer": "a", "displayed_answer": "a",
                 "trace": [], "score": {"controlled_fixture_exercised": False}},
                {"model_id": "fixture", "case_id": "F_PLAIN", "category": "general",
                 "evidence_mode": "live", "repetition": 1, "turn": 1, "status": "completed",
                 "elapsed_seconds": 1, "prompt": "p", "model_answer": "a", "displayed_answer": "a",
                 "trace": [], "score": {}},
            ]
            (original / "turns.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            with patch("agent_app.benchmark.ask", side_effect=AssertionError("No model calls in rescore")):
                benchmark.rescore(original, root / "out")
            rescored = [json.loads(line) for line in (root / "out" / "turns.jsonl").read_text().splitlines() if line.strip()]
            self.assertIs(rescored[0]["score"]["controlled_fixture_exercised"], True)
            self.assertIs(rescored[1]["score"]["controlled_fixture_exercised"], False)
            self.assertIsNone(rescored[2]["score"]["controlled_fixture_exercised"])
            summary = benchmark.read_json(root / "out" / "summary.json")
            self.assertEqual(summary["models"][0]["controlled_fixture_exercised_turns"], 1)

    def test_snapshot_preflight_requires_and_accepts_media_files(self):
        spectrum_id, overlay_id = "a" * 32, "b" * 32
        cases = [{"id": "F01", "turns": [{"required_calls": [
            {"tool": "get_spectrogram", "arguments": {}},
            {"tool": "detect_meteors", "arguments": {}}],
            "expected_media": ["image", "detection"]}]}]
        results = {
            "get_spectrogram": {"status": "ok", "downloaded": True, "artifact_id": spectrum_id},
            "detect_meteors": {"status": "ok", "analysis_performed": True,
                               "annotated_artifacts": [{"artifact_id": overlay_id}]},
        }
        lookup = SimpleNamespace(result=lambda name, args: results[name])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("agent_app.artifacts.SPECTROGRAM_DIR", root / "spec"), \
                 patch("agent_app.artifacts.DETECTION_DIR", root / "det"):
                with self.assertRaisesRegex(ValueError, "image.*No model requests were sent"):
                    benchmark._validate_snapshot_media(cases, lookup)
                (root / "spec").mkdir(); (root / "spec" / f"{spectrum_id}.png").write_bytes(b"fixture")
                with self.assertRaisesRegex(ValueError, "detection.*No model requests were sent"):
                    benchmark._validate_snapshot_media(cases, lookup)
                (root / "det").mkdir(); (root / "det" / f"{overlay_id}.png").write_bytes(b"fixture")
                benchmark._validate_snapshot_media(cases, lookup)

    def test_snapshot_preflight_requires_original_and_preview_audio(self):
        artifact = "c" * 32
        cases = [{"id": "F02", "turns": [{"required_calls": [
            {"tool": "get_wav", "arguments": {}}]}]}]
        lookup = SimpleNamespace(result=lambda name, args: {"status": "ok", "downloaded": True, "artifact_id": artifact})
        with tempfile.TemporaryDirectory() as temp, patch("agent_app.artifacts.WAV_DIR", Path(temp)):
            directory = Path(temp) / artifact; directory.mkdir()
            (directory / "original.wav").write_bytes(b"original")
            with self.assertRaisesRegex(ValueError, "audio.*No model requests were sent"):
                benchmark._validate_snapshot_media(cases, lookup)
            (directory / "preview.wav").write_bytes(b"preview")
            benchmark._validate_snapshot_media(cases, lookup)


class SnapshotPreflightIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_snapshot_media_fails_before_model_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            arguments = {"search": "BEHUMA_SYS001", "start_utc": "2023-12-14T21:15:00Z",
                         "end_utc": "2023-12-14T21:20:00Z"}
            cases = [{"id": "F01", "turns": [{"required_calls": [
                {"tool": "detect_meteors", "arguments": arguments}], "expected_media": ["detection"]}]}]
            case_path = root / "cases.json"
            benchmark.write_json(case_path, {"cases": cases})
            calls = benchmark.reference_calls(cases)
            entries = [{**call, "result": ({"status": "ok", "analysis_performed": True,
                                             "annotated_artifacts": [{"artifact_id": "d" * 32}]}
                                            if call["tool"] == "detect_meteors" else {"status": "ok"})}
                       for call in calls]
            snapshot = root / "reference.json"
            benchmark.write_json(snapshot, {"kind": "real_tool_capture", "schema_version": 1,
                                            "cases_sha256": benchmark.digest(case_path), "entries": entries})
            args = SimpleNamespace(mode="snapshot", snapshot=snapshot, cases=case_path, include_public=False,
                                   output=root / "out", clock=None)
            with patch("agent_app.artifacts.DETECTION_DIR", root / "missing"), \
                 patch("agent_app.benchmark.ask", new=AsyncMock()) as model, \
                 patch("agent_app.benchmark.warmup", new=AsyncMock()) as warmup:
                with self.assertRaisesRegex(ValueError, "Snapshot fixture unavailable.*No model requests were sent"):
                    await benchmark.run(args, {}, [], cases)
            model.assert_not_called()
            warmup.assert_not_called()
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
