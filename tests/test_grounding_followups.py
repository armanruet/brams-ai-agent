"""Grounding distinguishes a general explanation from a scoped detection."""

import unittest

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from agent_app.config import Settings
from agent_app.grounding import required_domains, requires_tool_or_history
from agent_app.service import Session, ask
from tests.fake_model import FakeModel


DETECTION = {
    "status": "ok", "analysis_performed": True,
    "receiver_id": "BEHUMA_SYS001", "receiver_name": "Humain",
    "start_utc": "2023-12-14T21:15:00Z", "end_utc": "2023-12-14T21:20:00Z",
    "total_meteor_count": 2,
}


def result(name, data):
    return ModelRequest(parts=[ToolReturnPart(tool_name=name, content=data)])


class GroundingFollowupTests(unittest.TestCase):
    def test_network_descriptor_is_not_treated_as_a_named_station(self):
        south = {"status": "ok", "country": "Belgium", "direction": "south",
                 "action": "count", "site_count": 12}
        trace = [{"event": "tool_result", "name": "query_geographic_receivers", "result": south}]
        for question in ("How many BRAMS station sites are in the southern part of Belgium?",
                         "How many Belgian stations are in southern Belgium?",
                         "How many southern station sites are in Belgium?"):
            with self.subTest(question=question):
                self.assertFalse(requires_tool_or_history(question, trace))
                north_trace = [{**trace[0], "result": {**south, "direction": "north"}}]
                self.assertTrue(requires_tool_or_history(question, north_trace))

    def test_named_station_still_needs_matching_evidence(self):
        question = "Show the Kirchberg station spectrogram."
        for data in ({"status": "ok"}, {"status": "ok", "receiver_name": "Humain"}):
            with self.subTest(data=data):
                trace = [{"event": "tool_result", "name": "get_spectrogram", "result": data}]
                self.assertTrue(requires_tool_or_history(question, trace))
        trace = [{"event": "tool_result", "name": "get_spectrogram", "result": {
            "status": "ok", "receiver_name": "Kirchberg"}}]
        self.assertFalse(requires_tool_or_history(question, trace))

    def test_concept_questions_do_not_require_detection(self):
        for question in ("Briefly explain what a meteor echo is.",
                         "What are meteor echoes?", "How does meteor echo detection work?",
                         "What is a meteor echo in a spectrogram?", "WHAT IS A METEOR ECHO?"):
            with self.subTest(question=question):
                self.assertNotIn("detect_meteors", required_domains(question))
                self.assertFalse(requires_tool_or_history(question, []))

    def test_actual_detection_requests_still_require_the_correct_tool(self):
        for question in ("Are there meteors in this spectrogram?",
                         "Check the spectrogram for meteor echoes.",
                         "Was there a meteor near 21:16 UTC on 2023-12-14 at BEHUMA?",
                         "Run meteor detection for Kirchberg yesterday.",
                         "List each detected echo time and confidence."):
            with self.subTest(question=question):
                self.assertIn("detect_meteors", required_domains(question))
                self.assertTrue(requires_tool_or_history(question, []))

    def test_compound_concept_and_spectrogram_requires_spectrogram(self):
        question = "What is a meteor echo and show a spectrogram?"
        self.assertEqual(required_domains(question), {"get_spectrogram"})
        self.assertTrue(requires_tool_or_history(question, []))

    def test_implicit_followup_uses_recent_successful_detection(self):
        history = [result("detect_meteors", DETECTION)]
        for prompt in ("List each detected echo time and confidence.",
                       "Show the detection results.",
                       "What were the echo times?"):
            with self.subTest(prompt=prompt):
                self.assertTrue(requires_tool_or_history(prompt, []))
                self.assertFalse(requires_tool_or_history(prompt, [], history))

    def test_new_scope_does_not_reuse_old_detection(self):
        history = [result("detect_meteors", DETECTION)]
        for prompt in ("List detected echoes from Kirchberg.",
                       "List the detected echoes at BEHUMA_SYS001 on 2023-12-15.",
                       "List detected echoes at 22:00.",
                       "Show the detection results for yesterday."):
            with self.subTest(prompt=prompt):
                self.assertTrue(requires_tool_or_history(prompt, [], history))

    def test_later_failed_detection_or_new_recording_blocks_stale_evidence(self):
        question = "List each detected echo time and confidence."
        older = result("detect_meteors", DETECTION)
        for later in (result("detect_meteors", {"status": "error", "analysis_performed": False}),
                      result("get_spectrogram", {"status": "ok", "receiver_id": "LUKIRC_SYS001"}),
                      result("list_receivers", {"status": "ok", "receivers": []})):
            with self.subTest(tool=later.parts[0].tool_name):
                self.assertTrue(requires_tool_or_history(question, [], [older, later]))

    def test_explicit_followup_does_not_reuse_an_old_failed_or_unrelated_scan(self):
        question = "Show the times from that detection."
        old = result("detect_meteors", DETECTION)
        self.assertFalse(requires_tool_or_history(question, [], [old]))
        self.assertTrue(requires_tool_or_history(question, [], [old, result("detect_meteors", {
            "status": "error", "analysis_performed": False,
        })]))


class GroundingServiceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_general_question_does_not_trigger_a_grounding_retry(self):
        fake = FakeModel()
        reply = await ask("Briefly explain what a meteor echo is.", Session(), Settings(), transport=fake.transport())
        self.assertEqual(reply.requests, 1)
        self.assertFalse(any(event.get("event") == "grounding_retry" for event in reply.trace))
        self.assertNotEqual(reply.warning, "The requested evidence was incomplete.")

    async def test_implicit_detection_followup_does_not_trigger_retry(self):
        fake = FakeModel()
        session = Session(messages=[
            ModelRequest(parts=[UserPromptPart(content="Detect meteor echoes in the recording.")]),
            ModelResponse(parts=[ToolCallPart(tool_name="detect_meteors", args={}, tool_call_id="detection_1")]),
            ModelRequest(parts=[ToolReturnPart(tool_name="detect_meteors", content=DETECTION,
                                               tool_call_id="detection_1")]),
            ModelResponse(parts=[TextPart(content="Two candidate echoes were detected.")]),
        ])
        reply = await ask("List each detected echo time and confidence.", session, Settings(), transport=fake.transport())
        self.assertEqual(reply.requests, 1)
        self.assertFalse(any(event.get("event") == "grounding_retry" for event in reply.trace))
        self.assertNotEqual(reply.warning, "The requested evidence was incomplete.")
