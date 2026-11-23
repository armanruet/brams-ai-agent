"""Focused regressions for previously misrouted and misreported cases."""
import unittest
from unittest.mock import patch
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from agent_app.grounding import (apply_grounding_guard, required_domains, safe_unfulfilled_response,
                                 _requested_receiver_names, _matches_explicit_scope,
                                 requires_tool_or_history, deterministic_limit_response)
from agent_app.config import Settings
from agent_app.service import Session, ask
from tests.fake_model import FakeModel


class FailedCaseRegressions(unittest.TestCase):
    def test_receiver_and_beacon_scope(self):
        b11 = "How was BEHUMA_SYS001's clock timing in the 21:15 UTC recording on 2023-12-14? Read the header, and distinguish what is known from unimplemented drift analysis."
        b06 = "How far is BEHUMA_SYS001 from the BEDOUR beacon? State the calculation method, coordinate sources and units."
        self.assertEqual(_requested_receiver_names(b11), [])
        self.assertEqual(_requested_receiver_names(b06), [])
        wav = {"status": "ok", "receiver_id": "BEHUMA_SYS001", "start_utc": "2023-12-14T21:15:00Z", "end_utc": "2023-12-14T21:20:00Z"}
        beacon = {"status": "ok", "emitter_code": "BEDOUR", "receiver_id": "BEHUMA_SYS001"}
        self.assertFalse(requires_tool_or_history(b11, [{"event": "tool_result", "name": "get_wav", "result": wav}]))
        self.assertTrue(_matches_explicit_scope(b06, beacon))
        self.assertFalse(_matches_explicit_scope(b06, {**beacon, "receiver_id": "BEBILZ_SYS001"}))
        self.assertFalse(_matches_explicit_scope(b06, {**beacon, "emitter_code": "BEOTHR"}))
        self.assertIn("humain", _requested_receiver_names("Show a recording from Humain station."))

    def test_unfulfilled_requests_are_bounded_before_model_calls(self):
        prompts = ["Clean BEHUMA's 21:15 recording on 2023-12-14 UTC. What changed?",
                   "Is there an optical observation matching a BEHUMA radio echo at 21:16:20 UTC on 2023-12-14?",
                   "Was the 21:15 UTC BEHUMA recording on 2023-12-14 affected by airplanes?",
                   "Download every recording from the whole network for 2023-12-14.",
                   "Render spectrograms for all 289 BEHUMA files on 2023-12-14.",
                   "Scan every hour of 2024 to find all meteor echoes.",
                   "Quickly check all receivers from 2013 to now for availability.",
                   "What's the PSD of receiver BEDOUR at 21:15?"]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertIsNotNone(deterministic_limit_response(prompt))
        self.assertIn("beacon", deterministic_limit_response("What's the PSD of receiver BEDOUR at 21:15?"))
        self.assertIsNone(deterministic_limit_response("Show a spectrogram and check optical matches"))
        self.assertIsNone(deterministic_limit_response("Was the recording affected by airplanes and tell me how many stations are active?"))
        self.assertIsNone(deterministic_limit_response("What is the PSD at 21:15?", has_history=True))

    def test_failed_domain_classification(self):
        cases = {
            "Where is the BEDOUR beacon and on what frequency does it transmit?": {"get_beacon"},
            "Compare the sample rates of BEHUMA_SYS001 and BEBILZ_SYS001 recordings": {"get_recordings"},
            "How large is the BEHUMA_SYS001 recording? Give bytes.": {"get_recordings"},
            "Compare likely hardware types using their recordings": {"get_recordings"},
            "What frequency did BEHUMA_SYS001 listen at in its recording? Give LO, beacon and nominal beacon-minus-LO frequencies.": {"get_wav"},
            "Was the GPS clock healthy during the recording?": {"get_wav"},
            "How was clock timing in the recording? Read the header.": {"get_wav"},
            "Get the original BEHUMA_SYS001 recording": {"get_wav"},
            "Where is the beacon and where is the receiver?": {"get_beacon", "list_receivers"},
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(required_domains(prompt), expected)

    def test_clarification_and_refusal_do_not_excuse_fabrication_or_compound_request(self):
        self.assertTrue(safe_unfulfilled_response("What's the PSD of BEDOUR at 21:15?", "What date and receiver should I use?"))
        self.assertTrue(safe_unfulfilled_response("Scan every hour of 2024 to find all meteor echoes.", "I cannot scan that much at once. Please choose a shorter interval."))
        self.assertFalse(safe_unfulfilled_response("What's the PSD of BEDOUR at 21:15?", "The PSD is 25 dB. What date?"))
        self.assertFalse(safe_unfulfilled_response("What's the PSD of BEDOUR at 21:15?", "The PSD is 25. What date?"))
        self.assertFalse(safe_unfulfilled_response("Show a spectrogram and check optical matches", "Optical matching is not supported."))

    def test_zero_rows_are_not_described_as_missing(self):
        row = {"receiver_id": "LUKIRC_SYS001", "name": "Kirchberg", "coverage": "none", "availability_percent": 0.0}
        data = {"status": "ok", "recording_availability_checked": True, "receivers": [row]}
        trace = [{"event": "tool_result", "name": "check_availability", "result": data}]
        for bad in ("No hourly rows were returned.", "There were no usable rows."):
            text, issues = apply_grounding_guard("Was Kirchberg available?", bad, trace)
            self.assertIn("explicit_zero_reported_as_missing_rows", [i["code"] for i in issues])
            self.assertNotIn(bad, text)
        _, issues = apply_grounding_guard("Was Kirchberg available?", "The archive returned an explicit zero.", trace)
        self.assertNotIn("explicit_zero_reported_as_missing_rows", [i["code"] for i in issues])


class HistoryPrefixRegression(unittest.IsolatedAsyncioTestCase):
    async def test_unsupported_aircraft_question_avoids_model_and_preserves_history(self):
        fake = FakeModel()
        prompt = "Was the 21:15 UTC BEHUMA recording on 2023-12-14 affected by airplanes?"
        reply = await ask(prompt, Session(), Settings(), transport=fake.transport())
        self.assertEqual(reply.requests, 0)
        self.assertFalse(fake.requests)
        self.assertIn("can’t determine", reply.text)
        self.assertTrue(any(isinstance(p, UserPromptPart) and p.content == prompt
                            for m in reply.session.messages for p in m.parts))

    async def test_trim_notice_keeps_system_prefix_and_original_user_message(self):
        fake = FakeModel()
        old = [ModelRequest(parts=[UserPromptPart("old user")]), ModelResponse(parts=[TextPart("old answer")])]
        with patch("agent_app.service._bounded_model_history", return_value=[]):
            trimmed = await ask("Hello", Session(messages=old), Settings(), transport=fake.transport())
        await ask("Hello", Session(), Settings(), transport=fake.transport())
        first, second = fake.requests
        self.assertEqual([m for m in first["messages"] if m["role"] == "system"],
                         [m for m in second["messages"] if m["role"] == "system"])
        self.assertIn("Some earlier conversation was omitted", str(first["messages"]))
        self.assertNotIn("Some earlier conversation was omitted", str(trimmed.session.messages))
        self.assertTrue(any(isinstance(p, UserPromptPart) and p.content == "Hello"
                            for m in trimmed.session.messages for p in m.parts))


if __name__ == "__main__":
    unittest.main()
