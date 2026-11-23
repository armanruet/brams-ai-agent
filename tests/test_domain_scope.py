"""Phase B regression tests: BRAMS domain scoping, negative controls, and ASSISTANT_SCOPE toggle."""

import os
import unittest
from unittest.mock import patch

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolReturnPart, UserPromptPart

from agent_app.config import Settings, load_settings
from agent_app.grounding import deterministic_limit_response, required_domains
from agent_app.harness import BASE_INSTRUCTIONS, DOMAIN_BASE_INSTRUCTIONS
from agent_app.service import Session, ask
from tests.fake_model import FakeModel


class DomainScopeTests(unittest.IsolatedAsyncioTestCase):
    def test_high_confidence_off_domain_prompts_are_refused(self):
        off_domain = [
            "Write a poem about cats.",
            "Give me a recipe for pasta.",
            "What is the capital of Japan?",
            "Write a Python script to scrape Amazon.",
            "Is Python 3.14 available?",
            "Are concert tickets available tonight?",
            "Give me medical advice for a headache.",
        ]
        for prompt in off_domain:
            with self.subTest(prompt=prompt):
                response = deterministic_limit_response(prompt, assistant_scope="domain")
                self.assertIsNotNone(response)
                self.assertIn("BRAMS", response)

    def test_negative_controls_and_unfamiliar_science_are_never_blocked(self):
        allowed = [
            ("Briefly explain what a radio meteor echo is.", False),
            ("What is forward scatter?", False),
            ("Explain the Hann window.", False),
            # Unfamiliar phrasing whose terms are not in _DOMAIN_CUE_RE must never be blocked.
            ("Explain how Welch's method reduces variance.", False),
            ("What causes tropospheric ducting?", False),
            ("Hi", False),
            ("What can you do?", False),
            ("Reply with one short greeting.", False),
            ("why?", True),
            ("Why is it higher?", True),
        ]
        for prompt, has_history in allowed:
            with self.subTest(prompt=prompt):
                self.assertIsNone(
                    deterministic_limit_response(prompt, has_history=has_history, assistant_scope="domain")
                )

    def test_bare_available_off_domain_does_not_require_check_availability(self):
        self.assertEqual(required_domains("Is Python 3.14 available?"), set())
        self.assertEqual(required_domains("Are concert tickets available tonight?"), set())
        self.assertEqual(required_domains("Was Kirchberg available?"), {"check_availability"})
        self.assertEqual(
            required_domains("Check BEHUMA_SYS001 availability on 2023-12-14 from 21:00 to 22:00 UTC."),
            {"check_availability"},
        )

    async def test_assistant_scope_toggle_controls_prompt_and_deterministic_refusal(self):
        fake_domain = FakeModel()
        reply_domain = await ask("Write a poem about cats.", Session(), Settings(assistant_scope="domain"),
                                 transport=fake_domain.transport())
        self.assertEqual(reply_domain.requests, 0)
        self.assertEqual(fake_domain.requests, [])
        self.assertIn("BRAMS", reply_domain.text)

        fake_general = FakeModel()
        reply_general = await ask("Write a poem about cats.", Session(), Settings(assistant_scope="general"),
                                  transport=fake_general.transport())
        self.assertEqual(reply_general.requests, 1)
        system_general = " ".join(m["content"] for m in fake_general.requests[0]["messages"] if m["role"] == "system")
        self.assertIn(BASE_INSTRUCTIONS.splitlines()[0], system_general)
        self.assertNotIn("domain-scoped assistant", system_general)

        fake_science = FakeModel()
        reply_science = await ask("Explain the Hann window.", Session(), Settings(assistant_scope="domain"),
                                  transport=fake_science.transport())
        self.assertEqual(reply_science.requests, 1)
        system_domain = " ".join(m["content"] for m in fake_science.requests[0]["messages"] if m["role"] == "system")
        self.assertIn(DOMAIN_BASE_INSTRUCTIONS.splitlines()[0], system_domain)

    async def test_followup_why_after_psd_passes_through_in_domain_scope(self):
        fake = FakeModel()
        prior = [
            ModelRequest(parts=[UserPromptPart("Measure noise PSD for BEHUMA_SYS001 at 21:15 UTC on 2023-12-14.")]),
            ModelRequest(parts=[ToolReturnPart(tool_name="get_psd", content={"status": "ok", "noise_psd": 1.2e-5}, tool_call_id="p1")]),
            ModelResponse(parts=[TextPart("The noise PSD was 1.2e-5 ADC²/Hz.")]),
        ]
        reply = await ask("why?", Session(messages=prior), Settings(assistant_scope="domain"),
                          transport=fake.transport())
        self.assertEqual(reply.requests, 1)

    def test_env_loading_and_validation_for_assistant_scope(self):
        with patch.dict(os.environ, {"ASSISTANT_SCOPE": "general"}, clear=False):
            self.assertEqual(load_settings().assistant_scope, "general")
        with patch.dict(os.environ, {"ASSISTANT_SCOPE": "domain"}, clear=False):
            self.assertEqual(load_settings().assistant_scope, "domain")
        with self.assertRaises(ValueError):
            Settings(assistant_scope="invalid")


if __name__ == "__main__":
    unittest.main()
