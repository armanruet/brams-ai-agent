from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent_app.brams_sdk import _load_wav
from tests.test_brams_sdk import SDK_READY, catalog, CODE
from tests.test_wav import wav_bytes

START = datetime(2026, 10, 22, tzinfo=timezone.utc)
END = START + timedelta(hours=1)


@unittest.skipUnless(SDK_READY, "Requires PyBRAMS on Python 3.13+")
class WavSDKTests(unittest.TestCase):
    def test_sdk_save_raw_downloads_unchanged_wav_without_processing(self):
        from pybrams import File, Receiver, Location
        from agent_app.artifacts import wav_path

        raw_receiver = {**vars(catalog()[CODE]), "location_url": "https://example.invalid/location.php"}
        raw_file = dict(year=2026, month=10, day=22, hours=0, minutes=0,
            samplerate=8000, pps_count=0, duration=300, precise_start=int(START.timestamp() * 1e6),
            precise_end=int((START + timedelta(minutes=5)).timestamp() * 1e6),
            size=100, receiver_code=CODE, location_code="TESTKIR",
            location_url="https://example.invalid/location.php", receiver_url="https://example.invalid/receiver.php",
            wav_url="https://example.invalid/test.wav", wav_name="test.wav",
            png_url="https://example.invalid/test.png", png_name="test.png")
        def api_response(endpoint, payload=None):
            return [raw_receiver] if endpoint.endswith("receiver.php") else [raw_file]
        with tempfile.TemporaryDirectory() as directory, \
             patch("agent_app.artifacts.WAV_DIR", Path(directory)), \
             patch("pybrams.brams.fetch.api.request", side_effect=api_response), \
             patch("pybrams.utils.http.post", return_value=SimpleNamespace(json=lambda: [raw_file])), \
             patch("pybrams.utils.http.get", return_value=SimpleNamespace(content=wav_bytes())) as http_get, \
             patch.object(Receiver, "get", return_value=catalog()), \
             patch.object(Location, "get", return_value=SimpleNamespace()), \
             patch.object(File, "load") as load, patch.object(File, "process") as process:
            result = _load_wav("Kirchberg", START, END)
            self.assertEqual(wav_path(result["artifact_id"]).read_bytes(), wav_bytes())
            self.assertTrue(wav_path(result["artifact_id"], preview=True).is_file())
        http_get.assert_called_once_with("https://example.invalid/test.wav")
        load.assert_not_called()
        process.assert_not_called()
        self.assertEqual(result["status"], "ok")

    def test_invalid_audio_is_not_exposed_as_an_attachment(self):
        from pybrams import Receiver, File
        def save_raw(directory):
            (directory / "bad.wav").write_bytes(b"<html>server error</html>")
        recording = SimpleNamespace(receiver_code=CODE, start=START, end=END,
                                    wav_url="https://example.invalid/bad.wav", wav_name="bad.wav", save_raw=save_raw)
        with tempfile.TemporaryDirectory() as directory, patch("agent_app.artifacts.WAV_DIR", Path(directory)), \
             patch.object(Receiver, "get_all", return_value=catalog()), \
             patch("agent_app.brams_sdk._archive_files", return_value=(row for row in [recording])):
            with self.assertRaises(ValueError):
                _load_wav("Kirchberg", START, END)
            self.assertEqual(list(Path(directory).iterdir()), [])
