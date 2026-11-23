"""TOOL_CHECKER questions with synthetic, explicitly non-live SDK/LLM fixtures."""

from dataclasses import fields
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pydantic_ai import ModelRetry

from agent_app import brams_tools
from agent_app.brams_sdk import _load_wav, _load_spectrogram
from agent_app.config import Settings
from agent_app.recovery import evidence_summary
from agent_app.science import _load_recordings, _load_beacon, _load_psd, receiver_coordinates
from agent_app.service import Session, ask
from agent_app.tools import convert_local_time
from tests.fake_model import FakeModel
from tests.test_brams_sdk import SDK_READY

START = datetime(2023, 12, 14, 21, 15, tzinfo=timezone.utc)
END = START + timedelta(minutes=5)
CODE = "BEHUMA_SYS001"
FILENAME = "RAD_BEDOUR_20231214_2115_BEHUMA_SYS001.wav"
SOURCE = "https://example.invalid/" + FILENAME
READY = {"ready": True, "version": "0.1.7"}


def catalog(count=6):
    return {f"BEHUMA_SYS{i:03d}": SimpleNamespace(receiver_code=f"BEHUMA_SYS{i:03d}",
            location_code="BEHUMA", name=f"Humain {i}", start="2015-04-23", end=None, antenna=i)
            for i in range(1, count+1)}


def metadata_entry(slot=START):
    precise_start = int(slot.timestamp()*1e6)+809161
    return dict(year=slot.year, month=slot.month, day=slot.day, hours=slot.hour, minutes=slot.minute,
                samplerate=5512.5, pps_count=300, duration=300226576,
                precise_start=precise_start, precise_end=precise_start+300226576, size=3315484,
                receiver_code=CODE, location_code="BEHUMA", location_url="https://example.invalid/location.php",
                receiver_url="https://example.invalid/receiver.php", wav_url=SOURCE,
                wav_name=FILENAME.replace("2115", slot.strftime("%H%M")),
                png_url=SOURCE.replace(".wav", ".png"), png_name=FILENAME.replace(".wav", ".png"))


def scientific_wav(amplitude=1):
    """Two known tones, real BRA1/BRA2 chunks, precise fractional samplerate."""
    import numpy as np
    from pybrams.brams.wavfile import Metadata, WavFile
    from pybrams.brams.formats.wav import Wav
    from pybrams.signal.processing.series import Series
    from pybrams.signal.processing.pps import PPS

    values = {field.name: 0 for field in fields(Metadata)}
    values.update(version=1, samplerate=5512.5, lo_freq=49969000, start_us=1702588500809161,
                  pps_count=3, beacon_freq=49970000, ant_lat=50.1925017, ant_long=5.2546467,
                  ant_alt=327.5, beacon_code="BEDOUR", observer_code="BEHUMA", station_code="BEHUMA",
                  description="Synthetic regression fixture, not an archive recording")
    t = np.arange(11025)/5512.5
    data = (amplitude*32768*(.02*np.sin(2*np.pi*850*t)+.2*np.sin(2*np.pi*1505*t))).astype(np.int16)
    pps = PPS(np.array([0, 5512, 11024], dtype=np.uint64),
              np.array([1702588500809161, 1702588501809161, 1702588502809161], dtype=np.uint64))
    return Wav.write(WavFile(Metadata(**values), Series(data, 5512.5, dtype=np.int16), pps))


def recording(data):
    def save_raw(directory):
        (directory/FILENAME).write_bytes(data)
    return SimpleNamespace(receiver_code=CODE, start=START+timedelta(microseconds=809161), end=END,
                           wav_name=FILENAME, wav_url=SOURCE, save_raw=save_raw)


class TimeCheckerTests(unittest.TestCase):
    def test_D07_correct_brussels_conversion_without_invented_day_rollover(self):
        value = convert_local_time("2023-12-15T23:30:00", "Europe/Brussels")
        self.assertEqual(value["utc_time"], "2023-12-15T22:30:00Z")
        self.assertEqual(value["utc_offset"], "+01:00")
        self.assertEqual(convert_local_time("2023-12-15T00:30:00")["utc_time"], "2023-12-14T23:30:00Z")
        # Common timezone aliases and accidental trailing Z are normalized cleanly.
        self.assertEqual(convert_local_time("2023-12-15T23:30:00Z", "CET")["utc_time"], "2023-12-15T22:30:00Z")
        self.assertEqual(convert_local_time("2026-07-15T23:30:00", "Brussels time")["utc_offset"], "+02:00")

    def test_interval_conversion_in_single_call_and_ordering_validation(self):
        window = convert_local_time("2023-12-15T23:00:00", "Europe/Brussels", end_local_datetime="2023-12-16T00:00:00")
        self.assertEqual(window["start_utc"], "2023-12-15T22:00:00Z")
        self.assertEqual(window["end_utc"], "2023-12-15T23:00:00Z")
        self.assertEqual(window["duration_minutes"], 60.0)
        with self.assertRaises(ModelRetry):
            convert_local_time("2023-12-15T23:00:00", "Europe/Brussels", end_local_datetime="2023-12-15T22:00:00")
        summary = evidence_summary([{"event": "tool_result", "name": "convert_local_time", "result": window}])
        self.assertIn("2023-12-15T22:00:00Z", summary)
        self.assertIn("2023-12-15T23:00:00Z", summary)
        self.assertIn("Europe/Brussels", summary)

    def test_dst_gap_and_ambiguous_time_require_clarification(self):
        for local in ["2026-03-29T02:30:00", "2026-10-25T02:30:00"]:
            with self.assertRaises(ModelRetry):
                convert_local_time(local)
        self.assertNotEqual(convert_local_time("2026-10-25T02:30:00", fold=0)["utc_time"],
                            convert_local_time("2026-10-25T02:30:00", fold=1)["utc_time"])


@unittest.skipUnless(SDK_READY, "PyBRAMS requires Python 3.13+")
class SDKCheckerTests(unittest.TestCase):
    def test_A02_C06_real_location_conversion_and_single_lookup_per_site(self):
        from pybrams import Location
        from agent_app.brams_sdk import filter_receivers
        location = Location("BEHUMA", "Humain", "active", 5.254884, 50.192434, .294, "")
        with patch.object(Location, "get", return_value=location) as lookup:
            rows = receiver_coordinates(filter_receivers(catalog(), "BEHUMA"))
        lookup.assert_called_once_with("BEHUMA")
        self.assertEqual(rows[0]["coordinates"]["latitude_degrees"], 50.192434)
        self.assertEqual(rows[0]["coordinates"]["longitude_degrees"], 5.254884)
        self.assertEqual(rows[0]["coordinates"]["altitude_m"], 294)

    def test_A03_B06_F03_emitter_frequency_wavelength_and_nonzero_distance(self):
        from pybrams import Location, Receiver
        from pybrams.brams.emitter import Emitter
        emitter_location = Location("BEDOUR", "Dourbes", "active", 4.588487, 50.097569, .225, "")
        receiver_location = Location("BEHUMA", "Humain", "active", 5.254884, 50.192434, .294, "")
        emitter = SimpleNamespace(emitter_code="BEDOUR", name="Dourbes", frequency=49970000,
                                  power=150, polarization="RHCP", location=emitter_location, location_code="BEDOUR")
        with patch.object(Emitter, "get", return_value=emitter), \
             patch.object(Receiver, "get_all", return_value=catalog()), \
             patch.object(Location, "get", return_value=receiver_location):
            result = _load_beacon("BEDOUR", "BEHUMA")
        self.assertAlmostEqual(result["wavelength_m"], 5.9994488293)
        self.assertGreater(result["distance_km"], 48)
        self.assertLess(result["distance_km"], 49)
        self.assertEqual(result["coordinates"]["altitude_m"], 225)

    def test_A06_A09_D08_metadata_counts_pagination_boundary_and_microseconds(self):
        from pybrams import File, Receiver
        entries = [metadata_entry(START+timedelta(minutes=i*5)) for i in range(-1, 6)]
        with patch.object(Receiver, "get_all", return_value=catalog()), \
             patch("pybrams.brams.fetch.api.request", return_value=entries) as request, \
             patch.object(File, "load") as load:
            result = _load_recordings("BEHUMA", START, START+timedelta(minutes=30), 3, 0)
        load.assert_not_called()
        self.assertEqual(request.call_args.args[1]["receiver_code[]"], [CODE])
        self.assertEqual(result["count"], 7)
        self.assertEqual(len(result["recordings"]), 3)
        self.assertTrue(result["has_more"])
        self.assertTrue(result["recordings"][0]["begins_before_requested_window"])
        row = result["recordings"][1]
        self.assertEqual(row["recording_start_utc"], "2023-12-14T21:15:00.809161Z")
        self.assertEqual(row["duration_seconds"], 300.226576)
        self.assertEqual((row["size_bytes"], row["sample_rate_hz"]), (3315484, 5512.5))

    def test_D04_unknown_receiver_never_queries_file_endpoint(self):
        from pybrams import Receiver
        with patch.object(Receiver, "get_all", return_value=catalog()), \
             patch("pybrams.brams.fetch.api.request") as request:
            result = _load_recordings("BEFAKE", START, END, 10, 0)
        request.assert_not_called()
        self.assertEqual(result["status"], "receiver_not_found")

    def test_A10_exact_slot_preferred_over_previous_overlapping_file(self):
        from pybrams import File, Receiver
        from tests.test_spectrogram import png_bytes
        previous = SimpleNamespace(receiver_code=CODE, start=START-timedelta(minutes=5),
            end=START+timedelta(seconds=1), png_url="https://example.invalid/2110.png", png_name="2110.png",
            spectrogram=png_bytes())
        selected = SimpleNamespace(receiver_code=CODE, start=START+timedelta(seconds=.809161), end=END,
            png_url="https://example.invalid/2115.png", png_name="2115.png", spectrogram=png_bytes())
        with tempfile.TemporaryDirectory() as directory, \
             patch("agent_app.artifacts.SPECTROGRAM_DIR", Path(directory)), \
             patch.object(Receiver, "get_all", return_value=catalog()), \
             patch("agent_app.brams_sdk._archive_files", return_value=(row for row in [previous, selected])):
            result = _load_spectrogram("BEHUMA", START, END)
        self.assertEqual(result["filename"], "2115.png")

    def download_and_measure(self, amplitude=1):
        from pybrams import File, Receiver
        data = scientific_wav(amplitude)
        with patch.object(Receiver, "get_all", return_value=catalog()), \
             patch("agent_app.brams_sdk._archive_files", side_effect=lambda *a, **k: (row for row in [recording(data)])):
            result = _load_psd("BEHUMA", START, END, 800, 900, 3)
        return result, data

    def test_A11_A15_C04_C08_real_header_and_fft_not_player_preview(self):
        from agent_app.artifacts import wav_path
        with tempfile.TemporaryDirectory() as directory, patch("agent_app.artifacts.WAV_DIR", Path(directory)):
            result, data = self.download_and_measure()
            self.assertEqual(wav_path(result["artifact_id"]).read_bytes(), data)
            self.assertEqual(result["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(result["size_bytes"], len(data))
            self.assertEqual(result["sample_rate_hz"], 5512.5)
            self.assertEqual(result["container_sample_rate_hz"], 5512)
            self.assertEqual(result["analyzed_duration_seconds"], 2)
            self.assertAlmostEqual(result["noise_psd"], 3e-6, delta=3e-8)
            self.assertAlmostEqual(result["calibrator_psd"], .005, delta=5e-5)
            self.assertEqual(result["calibrator_frequency_hz"], 1505)
            self.assertEqual(result["header_summary"]["nominal_beacon_minus_lo_hz"], 1000)
            self.assertEqual(result["header_summary"]["antenna_coordinates"]["altitude_m"], 327.5)
            self.assertEqual(result["header_summary"]["state_raw"], 0)
            # Peak-normalized previews would give the same PSD for both files.
            weaker, _ = self.download_and_measure(.5)
            self.assertAlmostEqual(weaker["noise_psd"]/result["noise_psd"], .25, delta=.002)

    def test_C05_reuse_verified_original_without_second_download(self):
        from agent_app.artifacts import wav_path
        with tempfile.TemporaryDirectory() as directory, patch("agent_app.artifacts.WAV_DIR", Path(directory)):
            original, _ = self.download_and_measure()
            with patch("agent_app.brams_sdk._load_wav") as download:
                reused = _load_psd(CODE, START, END, 800, 900, 3, original["artifact_id"])
                wrong = _load_psd("BEFAKE_SYS001", START, END, 800, 900, 3, original["artifact_id"])
                path = wav_path(original["artifact_id"])
                path.write_bytes(path.read_bytes()+b"changed")
                changed = _load_psd(CODE, START, END, 800, 900, 3, original["artifact_id"])
            download.assert_not_called()
            self.assertEqual(reused["noise_psd"], original["noise_psd"])
            self.assertFalse(reused["downloaded"])
            self.assertTrue(reused["reused_original_artifact"])
            self.assertEqual(wrong["status"], "artifact_mismatch")
            self.assertEqual(changed["status"], "artifact_changed")

    def test_bad_nyquist_band_is_explicit_without_nan_measurements(self):
        with tempfile.TemporaryDirectory() as directory, patch("agent_app.artifacts.WAV_DIR", Path(directory)):
            original, _ = self.download_and_measure()
            result = _load_psd(CODE, START, END, 2800, 2900, 3, original["artifact_id"])
        self.assertEqual(result["status"], "invalid_frequency_band")
        self.assertNotIn("noise_psd", result)


class ToolCheckerValidationTests(unittest.IsolatedAsyncioTestCase):
    def test_question_bank_extracts_the_document_questions(self):
        from agent_app.tool_checker import questions
        bank = questions(Path(__file__).resolve().parent.parent / "TOOL_CHECKER.md")
        self.assertEqual(len(bank), 54)
        self.assertEqual(bank["A02"], "Give me the exact coordinates of station BEHUMA.")
        self.assertIn("Brussels", bank["D07"])

    async def test_A01_inventory_is_counted_but_never_dumped_in_full(self):
        with patch.object(brams_tools, "pybrams_setup_status", return_value=READY), \
             patch.object(brams_tools, "_load_catalog", return_value=(catalog(74), SOURCE)):
            result = await brams_tools.list_receivers()
        self.assertEqual(result["count"], 74)
        self.assertEqual(len(result["receivers"]), 20)
        self.assertEqual(result["multi_receiver_sites"], {"BEHUMA": 74})
        self.assertTrue(result["has_more"])

    async def test_E01_E04_invalid_or_bulk_calls_stop_before_io(self):
        with patch.object(brams_tools, "_load_recordings") as recordings, \
             patch.object(brams_tools, "_load_psd") as psd:
            for call in [brams_tools.get_recordings("BEHUMA", START, START+timedelta(days=2)),
                         brams_tools.get_recordings("BEHUMA", START, END, limit=289),
                         brams_tools.get_psd("BEHUMA", START, END, noise_low_hz=float("nan")),
                         brams_tools.get_psd("", START, END)]:
                with self.assertRaises(ModelRetry):
                    await call
        recordings.assert_not_called()
        psd.assert_not_called()

    async def test_download_or_api_errors_remain_unknown_not_no_data(self):
        with patch.object(brams_tools, "pybrams_setup_status", return_value=READY), \
             patch.object(brams_tools, "_load_psd", side_effect=RuntimeError("fixture")):
            result = await brams_tools.get_psd("BEHUMA", START, END)
        self.assertEqual(result["status"], "error")
        self.assertNotIn("noise_psd", result)

    async def test_A02_protocol_arguments_and_verified_chat_output(self):
        class CoordinateModel(FakeModel):
            def respond(self, payload):
                # Reuse the deterministic two-response protocol fixture.
                copied = dict(payload)
                copied["messages"] = [dict(message) for message in payload["messages"]]
                for message in copied["messages"]:
                    if message["role"] == "user":
                        message["content"] = "Which receivers are available at Ecotron?"
                status, body = super().respond(copied)
                message = body["choices"][0]["message"]
                if message.get("tool_calls"):
                    message["tool_calls"][0]["function"]["arguments"] = '{"search":"BEHUMA","detail":"coordinates"}'
                else:
                    message["content"] = "Here are the requested site coordinates."
                return status, body
        rows = [{"receiver_id": CODE, "location_code": "BEHUMA", "name": "Humain",
                 "coordinates": {"latitude_degrees": 50.192434, "longitude_degrees": 5.254884, "altitude_m": 294},
                 "coordinates_source_url": "https://example.invalid/location.php?location_code=BEHUMA"}]
        with patch.object(brams_tools, "pybrams_setup_status", return_value=READY), \
             patch.object(brams_tools, "_load_catalog", return_value=(catalog(), SOURCE)), \
             patch.object(brams_tools, "receiver_coordinates", return_value=rows):
            reply = await ask("Give me the exact coordinates of station BEHUMA.", Session(), Settings(),
                              transport=CoordinateModel().transport())
        self.assertEqual(reply.trace[0]["name"], "list_receivers")
        self.assertEqual(reply.trace[0]["arguments"], {"search": "BEHUMA", "detail": "coordinates"})
        for expected in ["50.192434", "5.254884", "294 m", "[Location registry]", "Verified tool results"]:
            self.assertIn(expected, reply.text)

    def test_fallback_psd_retains_labels_units_bands_and_source(self):
        data = dict(status="ok", filename=FILENAME, receiver_id=CODE, recording_start_utc="start",
                    recording_end_utc="end", noise_band_hz=[800, 900], noise_psd=2.368e-6,
                    calibrator_band_hz=[1502.13, 1508.13], calibrator_psd=3.485e-4,
                    calibrator_frequency_hz=1505.13, analyzed_duration_seconds=300.226576,
                    units="normalized ADC amplitude squared per Hz", method="PyBRAMS Series.psd",
                    analysis_input="original WAV", wav_source_url=SOURCE)
        text = evidence_summary([{"event": "tool_result", "name": "get_psd", "result": data}])
        for expected in ["Noise", "Calibrator", "800–900", "1502.13–1508.13", "ADC", "Original WAV", SOURCE]:
            self.assertIn(expected, text)

    @unittest.skipUnless(SDK_READY, "PyBRAMS requires Python 3.13+")
    async def test_A11_full_protocol_to_sdk_decoding_fft_and_gradio_text(self):
        from pybrams import File, Receiver
        from agent_app.tool_checker import questions
        class PSDModel(FakeModel):
            def respond(self, payload):
                copied = dict(payload)
                copied["messages"] = [dict(message) for message in payload["messages"]]
                for message in copied["messages"]:
                    if message["role"] == "user":
                        message["content"] = "DURATION"
                status, body = super().respond(copied)
                message = body["choices"][0]["message"]
                if message.get("tool_calls"):
                    import json
                    message["tool_calls"][0]["function"] = {"name": "get_psd", "arguments": json.dumps({
                        "search": "BEHUMA", "start_utc": START.isoformat(), "end_utc": END.isoformat()})}
                else:
                    message["content"] = "The scientific tool completed the measurements."
                return status, body
        data = scientific_wav()
        model = PSDModel()
        with tempfile.TemporaryDirectory() as directory, \
             patch("agent_app.artifacts.WAV_DIR", Path(directory)), \
             patch.object(Receiver, "get_all", return_value=catalog()), \
             patch("agent_app.brams_sdk._archive_files", side_effect=lambda *a, **k: (row for row in [recording(data)])):
            prompt = questions(Path(__file__).resolve().parent.parent / "TOOL_CHECKER.md")["A11"]
            reply = await ask(prompt, Session(), Settings(), transport=model.transport())
        self.assertEqual(reply.trace[0]["name"], "get_psd")
        self.assertAlmostEqual(reply.trace[1]["result"]["noise_psd"], 3e-6, delta=3e-8)
        self.assertEqual(reply.requests, 2)
        for required in ["Verified tool results", "Noise", "Calibrator", "800–900", "original WAV", SOURCE]:
            self.assertIn(required, reply.text)
