from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent_app.brams_tools import (
    _load_catalog, filter_receivers, list_receivers, pybrams_setup_status,
)
from agent_app.config import Settings
from agent_app.service import Session, ask
from tests.fake_model import FakeModel


READY = {"ready": True, "version": "0.1.7"}
SOURCE = "https://example.invalid/fixture/receiver.php"


def catalog_fixture():
    return {
        "BEECON_SYS002": SimpleNamespace(receiver_code="BEECON_SYS002", name="Ecotron 2",
            start="2014-01-01", end=None, antenna=1, location_code="BEECON"),
        "BEECON_SYS001": SimpleNamespace(receiver_code="BEECON_SYS001", name="Ecotron 1",
            start="2014-01-01", end=None, antenna=1, location_code="BEECON"),
        "BESAFF_SYS001": SimpleNamespace(receiver_code="BESAFF_SYS001", name="Saffraanberg",
            start="2014-01-01", end=None, antenna=1, location_code="BESAFF"),
    }


class ReceiverTests(unittest.IsolatedAsyncioTestCase):
    def test_filter_matches_names_codes_case_and_spacing(self):
        catalog = catalog_fixture()
        for query in ["Ecotron", "  ECOTRON  ", "BEECON"]:
            rows = filter_receivers(catalog, query)
            self.assertEqual([row["receiver_id"] for row in rows],
                             ["BEECON_SYS001", "BEECON_SYS002"])
        self.assertEqual(len(filter_receivers(catalog, "BEECON_SYS002")), 1)
        self.assertEqual(len(filter_receivers(catalog, "")), 3)

    async def test_catalog_result_does_not_claim_recording_availability(self):
        with patch("agent_app.brams_tools.pybrams_setup_status", return_value=READY), \
             patch("agent_app.brams_tools._load_catalog", return_value=(catalog_fixture(), SOURCE)):
            result = await list_receivers("Ecotron")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["count"], 2)
        self.assertFalse(result["recording_availability_checked"])
        self.assertEqual(result["scope"], "receiver_catalog")
        self.assertEqual(result["source_url"], SOURCE)

    async def test_empty_matches_are_different_from_network_failure(self):
        with patch("agent_app.brams_tools.pybrams_setup_status", return_value=READY):
            with patch("agent_app.brams_tools._load_catalog", return_value=(catalog_fixture(), SOURCE)):
                empty = await list_receivers("no-such-location")
            with patch("agent_app.brams_tools._load_catalog", side_effect=RuntimeError("fixture error")):
                failed = await list_receivers("Ecotron")
        self.assertEqual(empty["status"], "ok")
        self.assertEqual(empty["count"], 0)
        self.assertEqual(failed["status"], "error")
        self.assertIsNone(failed["count"])

    async def test_missing_setup_does_not_query_the_network(self):
        setup = {"ready": False, "code": "dependency_missing", "message": "Install PyBRAMS."}
        with patch("agent_app.brams_tools.pybrams_setup_status", return_value=setup), \
             patch("agent_app.brams_tools._load_catalog") as loader:
            result = await list_receivers("Ecotron")
        loader.assert_not_called()
        self.assertEqual(result["status"], "setup_required")
        self.assertIsNone(result["count"])

    def test_python_version_requirement_is_actionable(self):
        with patch("agent_app.brams_tools.sys.version_info", (3, 12, 0)):
            status = pybrams_setup_status()
        self.assertFalse(status["ready"])
        self.assertIn("3.13", status["message"])

    def test_missing_package_requirement_is_actionable(self):
        with patch("agent_app.brams_tools.sys.version_info", (3, 13, 0)), \
             patch("agent_app.brams_tools.version", side_effect=PackageNotFoundError):
            status = pybrams_setup_status()
        self.assertFalse(status["ready"])
        self.assertIn("requirements-brams.txt", status["message"])

    async def test_ecotron_question_executes_registered_tool_without_time_arguments(self):
        fake = FakeModel()
        with patch("agent_app.brams_tools.pybrams_setup_status", return_value=READY), \
             patch("agent_app.brams_tools._load_catalog", return_value=(catalog_fixture(), SOURCE)):
            reply = await ask("Which receivers are available at Ecotron?", Session(), Settings(),
                              transport=fake.transport())
        self.assertEqual(reply.trace[0]["name"], "list_receivers")
        self.assertEqual(reply.trace[0]["arguments"], {"search": "Ecotron"})
        self.assertEqual(reply.trace[-1]["result"]["count"], 2)
        self.assertIn("BEECON_SYS001", reply.text)


@unittest.skipUnless(pybrams_setup_status()["ready"], "Install requirements-brams.txt on Python 3.13+")
class NativeSDKTests(unittest.TestCase):
    def test_actual_sdk_parses_catalog_and_restores_config(self):
        from pybrams.utils import Config
        fields = [
            ("pybrams.utils.cache", "use"),
            ("pybrams.utils.http", "timeout"),
            ("pybrams.utils.http", "max_retries"),
        ]
        before = {key: Config.get(*key) for key in fields}
        response = []
        for receiver in catalog_fixture().values():
            raw = vars(receiver).copy()
            raw["location_url"] = "https://example.invalid/fixture/location.php"
            response.append(raw)
        with patch("pybrams.brams.fetch.api.request", return_value=response) as request, \
             patch.object(Config, "save") as save:
            catalog, source = _load_catalog()
            _load_catalog()
        self.assertEqual(len(catalog), 3)
        self.assertEqual(request.call_count, 2)
        self.assertTrue(source.endswith("/receiver.php"))
        self.assertEqual(before, {key: Config.get(*key) for key in fields})
        save.assert_not_called()

    def test_actual_sdk_config_is_restored_after_api_failure(self):
        from pybrams.utils import Config
        before = Config.get("pybrams.utils.cache", "use")
        with patch("pybrams.brams.fetch.api.request", side_effect=RuntimeError("fixture failure")):
            with self.assertRaises(RuntimeError):
                _load_catalog()
        self.assertEqual(before, Config.get("pybrams.utils.cache", "use"))


class ReceiverCatalogCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_receivers_caches_catalog_within_ttl_and_resets_on_expiry_or_clear(self):
        from types import SimpleNamespace
        from pybrams import Receiver
        from agent_app import brams_sdk

        brams_sdk.clear_catalog_cache()
        self.addCleanup(brams_sdk.clear_catalog_cache)
        clock = [100.0]
        sample = {
            "BEHUMA_SYS001": SimpleNamespace(
                receiver_code="BEHUMA_SYS001",
                name="Humain",
                location_code="BEHUMA",
                antenna="SYS001",
                start="2020-01-01",
                end=None,
            )
        }
        with patch("agent_app.brams_sdk.time.monotonic", side_effect=lambda: clock[0]), \
             patch("agent_app.brams_sdk._is_cacheable_sdk", return_value=True), \
             patch.object(Receiver, "get_all", return_value=sample) as get_all:
            first = await list_receivers()
            clock[0] = 115.0
            second = await list_receivers()
            clock[0] = 225.0
            expired = await list_receivers()
            brams_sdk.clear_catalog_cache()
            after_clear = await list_receivers()

        self.assertEqual(get_all.call_count, 3)
        self.assertEqual(first["count"], 1)
        self.assertFalse(first["catalog_cache"]["cached"])
        self.assertTrue(second["catalog_cache"]["cached"])
        self.assertEqual(second["catalog_cache"]["age_seconds"], 15.0)
        self.assertEqual(second["catalog_cache"]["ttl_seconds"], brams_sdk.CATALOG_CACHE_TTL_SECONDS)
        self.assertIn("always fetched live", second["catalog_cache"]["freshness_policy"])
        self.assertFalse(expired["catalog_cache"]["cached"])
        self.assertFalse(after_clear["catalog_cache"]["cached"])

    async def test_mocked_sdk_bypasses_ttl_cache(self):
        from pybrams import Receiver
        from agent_app import brams_sdk

        brams_sdk.clear_catalog_cache()
        self.addCleanup(brams_sdk.clear_catalog_cache)
        with patch.object(Receiver, "get_all", return_value={}) as get_all:
            first = await list_receivers()
            second = await list_receivers()

        self.assertEqual(get_all.call_count, 2)
        self.assertFalse(first["catalog_cache"]["cached"])
        self.assertFalse(second["catalog_cache"]["cached"])
