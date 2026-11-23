from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent_app.brams_sdk import _load_spectrogram
from tests.test_brams_sdk import SDK_READY, catalog, CODE


START = datetime(2026, 10, 22, tzinfo=timezone.utc)
END = START + timedelta(hours=1)


@unittest.skipUnless(SDK_READY, "Requires PyBRAMS on Python 3.13+")
class SpectrogramSDKTests(unittest.TestCase):
    def test_real_sdk_fetches_png_and_never_loads_wav(self):
        from PIL import Image
        from pybrams import File, Receiver, Location
        from agent_app.artifacts import spectrogram_path

        data = BytesIO()
        Image.new("RGB", (8, 8), "black").save(data, format="PNG")
        raw_receiver = {**vars(catalog()[CODE]), "location_url": "https://example.invalid/location.php"}
        raw_file = dict(year=2026, month=10, day=22, hours=0, minutes=0,
            samplerate=6048, pps_count=0, duration=300,
            precise_start=int(START.timestamp() * 1e6),
            precise_end=int((START + timedelta(minutes=5)).timestamp() * 1e6),
            size=100, receiver_code=CODE, location_code="TESTKIR",
            location_url="https://example.invalid/location.php", receiver_url="https://example.invalid/receiver.php",
            wav_url="https://example.invalid/test.wav", wav_name="test.wav",
            png_url="https://example.invalid/test.png", png_name="test.png")
        def api_response(endpoint, payload=None):
            if endpoint.endswith("receiver.php"):
                return [raw_receiver]
            self.assertTrue(endpoint.endswith("file.php"))
            self.assertEqual(payload["receiver_code[]"], [CODE])
            return [raw_file]
        with tempfile.TemporaryDirectory() as directory, \
             patch("agent_app.artifacts.SPECTROGRAM_DIR", Path(directory)), \
             patch("pybrams.brams.fetch.api.request", side_effect=api_response), \
             patch("pybrams.utils.http.post", return_value=SimpleNamespace(json=lambda: [raw_file])), \
             patch("pybrams.utils.http.get", return_value=SimpleNamespace(content=data.getvalue())) as http_get, \
             patch.object(Receiver, "get", return_value=catalog()), \
             patch.object(Location, "get", return_value=SimpleNamespace()), \
             patch.object(File, "load") as wav_load:
            result = _load_spectrogram("Kirchberg", START, END)
            self.assertTrue(spectrogram_path(result["artifact_id"]).is_file())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["recording_start_utc"], "2026-10-22T00:00:00Z")
        http_get.assert_called_once_with("https://example.invalid/test.png")
        wav_load.assert_not_called()

    def test_ambiguous_receiver_does_not_fetch_files(self):
        from pybrams import Receiver, File
        receivers = catalog()
        receivers["TESTKIR_SYS002"] = SimpleNamespace(**{**vars(receivers[CODE]), "receiver_code": "TESTKIR_SYS002"})
        with patch.object(Receiver, "get_all", return_value=receivers), \
             patch.object(File, "stream") as stream:
            result = _load_spectrogram("Kirchberg", START, END)
        stream.assert_not_called()
        self.assertEqual(result["status"], "receiver_ambiguous")

    def test_non_overlapping_records_are_not_displayed(self):
        from pybrams import Receiver, File
        wrong = SimpleNamespace(receiver_code=CODE, start=END, end=END + timedelta(minutes=5), png_url="unused")
        with patch.object(Receiver, "get_all", return_value=catalog()), \
             patch("agent_app.brams_sdk._archive_files", return_value=(record for record in [wrong])):
            result = _load_spectrogram("Kirchberg", START, END)
        self.assertFalse(result["downloaded"])
        self.assertEqual(result["status"], "no_spectrogram_found")

    def test_http_failure_is_not_reported_as_an_empty_archive(self):
        import asyncio
        from pybrams import Receiver
        from pybrams.utils.http import HttpRequestError
        from agent_app.brams_tools import get_spectrogram
        with patch.object(Receiver, "get_all", return_value=catalog()), \
             patch("pybrams.utils.http.post", side_effect=HttpRequestError("HTTP 404", status_code=404)):
            result = asyncio.run(get_spectrogram("Kirchberg", START, END))
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["downloaded"])
