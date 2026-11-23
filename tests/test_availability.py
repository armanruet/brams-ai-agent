from datetime import datetime, timedelta, timezone
import json
import unittest
from unittest.mock import patch

from pydantic_ai import ModelRetry

from agent_app.brams_tools import check_availability
from agent_app.config import Settings
from agent_app.service import ask, Session
from tests.fake_model import FakeModel, KIRCHBERG_PROMPT


START = datetime(2026, 10, 22, tzinfo=timezone.utc)
END = START + timedelta(hours=1)
READY = {"ready": True, "version": "0.1.7"}
CODE = "TESTKIR_SYS001"


class AvailabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_question_executes_registered_tool_and_preserves_history(self):
        result = {"status": "ok", "recording_availability_checked": True,
                  "receivers": [{"receiver_id": CODE, "coverage": "partial"}],
                  "source_url": "https://example.invalid/fixture/file.php"}
        fake = FakeModel()
        with patch("agent_app.brams_tools.pybrams_setup_status", return_value=READY), \
             patch("agent_app.brams_tools._load_availability", return_value=result) as load:
            reply = await ask(KIRCHBERG_PROMPT, Session(), Settings(), transport=fake.transport())
        load.assert_called_once_with("Kirchberg", START, END)
        self.assertEqual(reply.requests, 2)
        self.assertEqual(reply.trace[0]["name"], "check_availability")
        self.assertEqual(reply.trace[-1]["result"]["start_utc"], "2026-10-22T00:00:00Z")
        self.assertIn('"role": "tool"', json.dumps(fake.requests[-1]["messages"]))
        self.assertIn(CODE, reply.text)

    async def test_invalid_range_does_not_query_sdk(self):
        with patch("agent_app.brams_tools._load_availability") as load:
            for start, end, name in [(END, START, "Kirchberg"),
                                     (START, START + timedelta(days=8), "Kirchberg")]:
                with self.assertRaises(ModelRetry):
                    await check_availability(name, start, end)
        load.assert_not_called()

    async def test_failure_is_unknown_not_unavailable(self):
        with patch("agent_app.brams_tools.pybrams_setup_status", return_value=READY), \
             patch("agent_app.brams_tools._load_availability", side_effect=RuntimeError):
            result = await check_availability("Kirchberg", START, END)
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["recording_availability_checked"])
        self.assertIsNone(result["count"])

    async def test_setup_failure_does_not_query_network(self):
        with patch("agent_app.brams_tools.pybrams_setup_status", return_value={
                "ready": False, "code": "python_version", "message": "Use Python 3.13+"}), \
             patch("agent_app.brams_tools._load_availability") as load:
            result = await check_availability("Kirchberg", START, END)
        load.assert_not_called()
        self.assertEqual(result["status"], "setup_required")

    async def test_explicit_offset_is_converted_not_relabelled(self):
        offset = timezone(timedelta(hours=2))
        with patch("agent_app.brams_tools.pybrams_setup_status", return_value=READY), \
             patch("agent_app.brams_tools._load_availability", return_value={"status": "ok"}) as load:
            result = await check_availability("Kirchberg", START.astimezone(offset), END.astimezone(offset))
        load.assert_called_once_with("Kirchberg", START, END)
        self.assertEqual(result["start_utc"], "2026-10-22T00:00:00Z")

    async def test_network_wide_query_needs_no_receiver(self):
        with patch("agent_app.brams_tools.pybrams_setup_status", return_value=READY), \
             patch("agent_app.brams_tools._load_availability", return_value={"status": "ok"}) as load:
            result = await check_availability("", START, END)
        load.assert_called_once_with("", START, END)
        self.assertEqual(result["status"], "ok")


