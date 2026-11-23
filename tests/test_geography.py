"""Geographic filtering, exact recording evidence, and user-facing responses."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from pydantic_ai import ModelRetry
from agent_app.artifacts import save_spectrogram
from agent_app.brams_tools import query_geographic_receivers
from agent_app.config import Settings
from agent_app.geography import _recording_active_ids, query_belgian_region, select_region
from agent_app.recovery import geographic_summary
from agent_app.service import Session, ask
from tests.fake_model import FakeModel

START = datetime(2026, 11, 8, 6, tzinfo=timezone.utc)
END = START + timedelta(minutes=5)


def rec(code, name, site):
    return SimpleNamespace(receiver_code=code, name=name, location_code=site,
                           antenna=1, start="2020-01-01", end="")


CATALOG = {
    "BEHUMA_SYS001": rec("BEHUMA_SYS001", "Humain 1", "BEHUMA"),
    "BEHUMA_SYS002": rec("BEHUMA_SYS002", "Humain 2", "BEHUMA"),
    "BEGENT_SYS001": rec("BEGENT_SYS001", "Gent", "BEGENT"),
    "FRMAUB_SYS001": rec("FRMAUB_SYS001", "Maubeuge", "FRMAUB"),
    "BELONZ_SYS001": rec("BELONZ_SYS001", "Lonzée", "BELONZ"),
    "BEMISS_SYS001": rec("BEMISS_SYS001", "Unknown", "BEMISS"),
}
LOCATIONS = [
    {"location_code": "BEHUMA", "name": "Humain", "latitude": 50.192434, "longitude": 5.254884},
    {"location_code": "BEGENT", "name": "Gent", "latitude": 51.02325, "longitude": 3.710111},
    {"location_code": "FRMAUB", "name": "Maubeuge", "latitude": 50.277938, "longitude": 3.986977},
    {"location_code": "BELONZ", "name": "Lonzée", "latitude": 4.724757, "longitude": 50.551532},
]


def entry(rid, slot):
    end = slot + timedelta(minutes=5)
    return {"receiver_code": rid, "year": slot.year, "month": slot.month, "day": slot.day,
            "hours": slot.hour, "minutes": slot.minute,
            "precise_start": int(slot.timestamp()*1e6), "precise_end": int(end.timestamp()*1e6)}


def fake_sdk(file_response):
    @contextmanager
    def context():
        def request(endpoint, payload=None):
            return LOCATIONS if endpoint == "location" else file_response
        yield SimpleNamespace(get_all=lambda: CATALOG), SimpleNamespace(base_url="https://example.invalid/", request=request), SimpleNamespace(get=lambda section, key: section.split('.')[-1])
    return patch("agent_app.geography._fresh_sdk", context)


class GeographicTests(unittest.IsolatedAsyncioTestCase):
    def test_polygon_country_split_sites_and_unknown_coordinates(self):
        from agent_app.brams_sdk import filter_receivers
        rows = filter_receivers(CATALOG, "")
        locations = {row["location_code"]: row for row in LOCATIONS}
        south = select_region(rows, locations, "south")
        self.assertEqual([site["name"] for site in south["sites"]], ["Humain"])
        self.assertEqual(len(south["sites"][0]["receivers"]), 2)
        self.assertEqual(south["missing_coordinate_sites"], 1)
        self.assertEqual(south["suspect_coordinate_sites"], 1)
        north = select_region(rows, locations, "north")
        self.assertEqual([site["name"] for site in north["sites"]], ["Gent"])
        self.assertFalse(any(site["name"] == "Maubeuge" for site in south["sites"]))
        explicit = select_region(rows, locations, "south", 50.3)
        self.assertEqual(explicit["definition_type"], "user-specified")
        self.assertEqual(explicit["cutoff_degrees"], 50.3)

    def test_exact_interval_and_boundary_previous_slot(self):
        prior = entry("BEHUMA_SYS001", START - timedelta(minutes=5))
        prior["precise_end"] += 400_000
        valid = entry("BEHUMA_SYS002", START)
        bad = {"receiver_code": "BEGENT_SYS001"}
        found, unknown = _recording_active_ids([prior, valid, bad], list(CATALOG), START, END)
        self.assertEqual(found, {"BEHUMA_SYS002"})
        self.assertEqual(unknown, {"BEGENT_SYS001"})

    def test_active_counts_sites_not_antennas_and_404_unknown(self):
        with fake_sdk([entry("BEHUMA_SYS001", START), entry("BEHUMA_SYS002", START)]):
            result = query_belgian_region("south", "active_count", START, END)
        self.assertEqual(result["active_site_count"], 1)
        self.assertEqual(result["active_receiver_count"], 2)
        self.assertTrue(result["recording_availability_checked"])
        self.assertIn("had a recording listed", geographic_summary(result))
        with fake_sdk(None):
            result = query_belgian_region("south", "active_count", START, END)
        self.assertEqual(result["status"], "availability_unknown")
        self.assertFalse(result["recording_availability_checked"])

    def test_spectrogram_uses_only_matching_region_and_recording(self):
        blob = BytesIO(); Image.new("RGB", (8, 8)).save(blob, format="PNG")
        with tempfile.TemporaryDirectory() as tmp, patch("agent_app.artifacts.SPECTROGRAM_DIR", Path(tmp)):
            aid = save_spectrogram(blob.getvalue())
            image = {"status": "ok", "downloaded": True, "artifact_id": aid,
                     "receiver_name": "Humain", "receiver_id": "BEHUMA_SYS001",
                     "recording_start_utc": "2026-11-08T06:00:00Z", "recording_end_utc": "2026-11-08T06:05:00Z"}
            with fake_sdk([entry("BEHUMA_SYS001", START), entry("BEGENT_SYS001", START)]), \
                 patch("agent_app.geography._load_spectrogram", return_value=image) as loader:
                result = query_belgian_region("south", "spectrogram", START, END)
            loader.assert_called_once_with("BEHUMA_SYS001", START, END)
            self.assertEqual(result["artifact_id"], aid)
            self.assertEqual(result["action"], "spectrogram")

    def test_spectrogram_rejects_previous_boundary_slot(self):
        previous = {"status": "ok", "downloaded": True, "artifact_id": "a" * 32,
                    "receiver_id": "BEHUMA_SYS001", "recording_start_utc": "2026-11-08T05:55:00.3Z",
                    "recording_end_utc": "2026-11-08T06:00:00.3Z"}
        with fake_sdk([entry("BEHUMA_SYS001", START)]), \
             patch("agent_app.geography._load_spectrogram", return_value=previous):
            result = query_belgian_region("south", "spectrogram", START, END)
        self.assertEqual(result["status"], "spectrogram_unavailable")
        self.assertNotIn("artifact_id", result)

    async def test_tool_errors_are_not_zero(self):
        with patch("agent_app.brams_tools.pybrams_setup_status", return_value={"ready": True, "version": "0.1.7"}), \
             patch("agent_app.geography.query_belgian_region", side_effect=RuntimeError):
            result = await query_geographic_receivers("Belgium", "east", "active_count", START, END)
        self.assertEqual(result["status"], "error")
        self.assertIn("unknown", result["message"])
        with self.assertRaises(ModelRetry):
            await query_geographic_receivers("Belgium", "east", "active_count", START, START)
        self.assertEqual((await query_geographic_receivers("France", "east", "count"))["status"], "unsupported_country")

    async def test_count_direct_answer_and_compact_trace(self):
        class GeoModel(FakeModel):
            def respond(self, payload):
                self.requests.append(payload)
                return 200, {"id": "geo", "object": "chat.completion", "created": 1,
                    "model": payload["model"], "choices": [{"index": 0, "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "geo1", "type": "function",
                    "function": {"name": "query_geographic_receivers", "arguments": json.dumps({"country": "Belgium",
                    "direction": "south", "action": "count"})}}]}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}}
        fake = GeoModel()
        with fake_sdk([]), patch("agent_app.brams_tools.pybrams_setup_status", return_value={"ready": True, "version": "0.1.7"}):
            reply = await ask("How many stations are there in the southern part of Belgium?", Session(), Settings(), transport=fake.transport())
        self.assertEqual(reply.requests, 1)
        self.assertIn("**1 station**", reply.text)
        self.assertIn("2 receivers", reply.text)
        self.assertIn("geographic midpoint", reply.text)
        self.assertIn("could not be classified", reply.text)
        self.assertNotIn("Verified tool results", reply.text)
        self.assertEqual(reply.trace[-1]["result"]["site_count"], 1)

    async def test_geographic_query_caches_catalog_and_locations_within_ttl(self):
        from types import SimpleNamespace
        from pybrams import Receiver
        from agent_app import brams_sdk

        brams_sdk.clear_catalog_cache()
        self.addCleanup(brams_sdk.clear_catalog_cache)
        clock = [50.0]
        catalog = {
            "BEBRUS_SYS001": SimpleNamespace(receiver_code="BEBRUS_SYS001", name="Brussels", location_code="BEBRUS", antenna="SYS001", start=None, end=None),
            "BEHUMA_SYS001": SimpleNamespace(receiver_code="BEHUMA_SYS001", name="Humain", location_code="BEHUMA", antenna="SYS001", start=None, end=None),
        }
        locations = [
            {"location_code": "BEBRUS", "name": "Brussels", "latitude": 50.85, "longitude": 4.35, "altitude": 0.1},
            {"location_code": "BEHUMA", "name": "Humain", "latitude": 50.19, "longitude": 5.25, "altitude": 0.3},
        ]
        with patch("agent_app.brams_sdk.time.monotonic", side_effect=lambda: clock[0]), \
             patch("agent_app.brams_sdk._is_cacheable_sdk", return_value=True), \
             patch("agent_app.brams_tools.pybrams_setup_status", return_value={"ready": True, "version": "0.1.7"}), \
             patch.object(Receiver, "get_all", return_value=catalog) as get_all, \
             patch("pybrams.brams.fetch.api.request", return_value=locations) as api_request:
            first = await query_geographic_receivers("Belgium", "south", "count")
            clock[0] = 80.0
            second = await query_geographic_receivers("Belgium", "north", "count")
            clock[0] = 200.0
            expired = await query_geographic_receivers("Belgium", "north", "count")

        self.assertEqual(get_all.call_count, 2)
        self.assertEqual(api_request.call_count, 2)
        self.assertEqual(first["status"], "ok")
        self.assertFalse(first["catalog_cache"]["catalog_cached"])
        self.assertFalse(first["catalog_cache"]["locations_cached"])
        self.assertTrue(second["catalog_cache"]["catalog_cached"])
        self.assertTrue(second["catalog_cache"]["locations_cached"])
        self.assertEqual(second["catalog_cache"]["catalog_age_seconds"], 30.0)
        self.assertEqual(second["catalog_cache"]["locations_age_seconds"], 30.0)
        self.assertFalse(expired["catalog_cache"]["catalog_cached"])
        self.assertFalse(expired["catalog_cache"]["locations_cached"])
