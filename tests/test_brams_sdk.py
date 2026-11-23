"""SDK tests can run without the Gradio/LLM dependencies installed."""
from datetime import datetime, timedelta, timezone
from importlib.util import find_spec
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from agent_app.brams_sdk import _load_availability

SDK_READY = sys.version_info >= (3, 13) and find_spec("pybrams") is not None
START = datetime(2026, 10, 22, tzinfo=timezone.utc)
END = START + timedelta(hours=1)
CODE = "TESTKIR_SYS001"


def catalog():
    return {CODE: SimpleNamespace(receiver_code=CODE, name="Kirchberg",
            start="2020-01-01", end=None, antenna=1, location_code="TESTKIR")}


@unittest.skipUnless(SDK_READY, "Requires PyBRAMS on Python 3.13+")
class NativeAvailabilityTests(unittest.TestCase):
    def query(self, values, start=START, end=END):
        from pybrams import Receiver
        with patch.object(Receiver, "get_all", return_value=catalog()), \
             patch("agent_app.brams_sdk._raw_availability", return_value={CODE: values}) as call:
            result = _load_availability("Kirchberg", start, end)
        return result, call

    def test_hourly_full_partial_and_zero(self):
        for value, coverage, percent in [(1, "full", 100), (0.5, "partial", 50), (0, "none", 0),
                                          ("0.75", "partial", 75)]:
            result, _ = self.query({START: value})
            row = result["receivers"][0]
            self.assertEqual(row["coverage"], coverage)
            self.assertEqual(row["availability_percent"], percent)

    def test_subhour_bounds_are_aligned_and_partial_is_unknown(self):
        result, call = self.query({START: 0.5}, START + timedelta(minutes=15), START + timedelta(minutes=30))
        self.assertEqual(call.call_args.args[2:], (START, END))
        self.assertEqual(result["receivers"][0]["coverage"], "unknown")
        self.assertIsNone(result["receivers"][0]["availability_percent"])

    def test_missing_or_malformed_metrics_are_unknown(self):
        for value in [None, "invalid", -1, 2, float("nan"), True]:
            result, _ = self.query({START: value})
            self.assertEqual(result["receivers"][0]["coverage"], "unknown")

    def test_empty_response_is_not_claimed_as_no_recordings(self):
        from pybrams import Receiver
        with patch.object(Receiver, "get_all", return_value=catalog()), \
             patch("agent_app.brams_sdk._raw_availability", return_value={}):
            result = _load_availability("Kirchberg", START, END)
        self.assertEqual(result["status"], "no_availability_information")
        self.assertFalse(result["recording_availability_checked"])

    def test_unknown_receiver_does_not_query_files(self):
        from pybrams import Receiver
        with patch.object(Receiver, "get_all", return_value=catalog()), \
             patch("agent_app.brams_sdk._raw_availability") as call:
            result = _load_availability("no-such-receiver", START, END)
        call.assert_not_called()
        self.assertEqual(result["status"], "receiver_not_found")

    def test_multiple_hours_and_receivers_are_reported_separately(self):
        from pybrams import Receiver
        receivers = catalog()
        second = CODE.replace("001", "002")
        receivers[second] = SimpleNamespace(**{**vars(receivers[CODE]), "receiver_code": second})
        with patch.object(Receiver, "get_all", return_value=receivers), \
             patch("agent_app.brams_sdk._raw_availability", return_value={
                 CODE: {START: 1, END: 0}, second: {START: 1, END: 1}}):
            result = _load_availability("Kirchberg", START, END + timedelta(hours=1))
        self.assertEqual(result["count"], 2)
        self.assertEqual([row["coverage"] for row in result["receivers"]], ["partial", "full"])
        self.assertEqual([row["availability_percent"] for row in result["receivers"]], [50, 100])

    def test_sdk_configuration_restored_after_failure(self):
        from pybrams import Receiver
        from pybrams.utils import Config
        keys = [("pybrams.utils.cache", "use"), ("pybrams.utils.http", "timeout"),
                ("pybrams.utils.http", "max_retries")]
        before = [Config.get(*key) for key in keys]
        with patch.object(Receiver, "get_all", return_value=catalog()), \
             patch("agent_app.brams_sdk._raw_availability", side_effect=RuntimeError("fixture failure")):
            with self.assertRaises(RuntimeError):
                _load_availability("Kirchberg", START, END)
        self.assertEqual(before, [Config.get(*key) for key in keys])

    def test_network_query_returns_compact_summaries_for_all_receivers(self):
        from pybrams import Receiver
        receivers = {}
        raw = {}
        for index in range(40):
            code = f"TEST{index:02d}_SYS001"
            receivers[code] = SimpleNamespace(**{**vars(catalog()[CODE]), "receiver_code": code})
            raw[code] = {START: 1 if index < 30 else 0}
        with patch.object(Receiver, "get_all", return_value=receivers), \
             patch("agent_app.brams_sdk._raw_availability", return_value=raw) as query:
            result = _load_availability("", START, END)
        query.assert_called_once()
        self.assertEqual(result["count"], 40)
        self.assertEqual(result["receivers_with_recordings"], 30)
        self.assertEqual(set(result["receivers"][0]),
                         {"receiver_id", "name", "coverage", "availability_percent"})

    def test_actual_sdk_parsing_and_config_restoration(self):
        from pybrams.utils import Config
        raw_receiver = vars(catalog()[CODE]).copy()
        raw_receiver["location_url"] = "https://example.invalid/fixture/location.php"
        before = Config.get("pybrams.utils.cache", "use")
        response = SimpleNamespace(json=lambda: [{"receiver_code": CODE,
                                                  "datetime": "2026-10-22 00:00:00",
                                                  "availability": 0.75}])
        with patch("pybrams.brams.fetch.api.request", return_value=[raw_receiver]), \
             patch("pybrams.utils.http.post", return_value=response) as post, \
             patch.object(Config, "save") as save:
            result = _load_availability("Kirchberg", START, END)
        self.assertTrue(post.call_args.args[1]["availability"])
        self.assertEqual((post.call_args.args[1]["from"], post.call_args.args[1]["to"]), (START, END))
        self.assertEqual(result["receivers"][0]["availability_percent"], 75)
        self.assertEqual(before, Config.get("pybrams.utils.cache", "use"))
        save.assert_not_called()
