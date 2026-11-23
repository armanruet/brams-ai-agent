"""Hosted/local switching, authenticated routing, session privacy and UI callbacks.

All HTTP responses are fixtures. No external service or paid token is used.
"""

import json
import unittest
from unittest.mock import AsyncMock, patch

import gradio as gr
import httpx
from pydantic_ai.exceptions import ModelHTTPError

from agent_app.config import Settings
from agent_app.connections import ConnectionProfiles, OPENAI_BASE_URL, apply_profile
from agent_app.diagnostics import check_server
from agent_app.service import Reply, Session, ask, friendly_error
from tests.fake_model import FakeModel


KEY = "fixture-secret-never-a-real-key"
HOST = "https://fixture-provider.invalid/api/v1"


class ConnectionProfileTests(unittest.TestCase):
    def setUp(self):
        self.local = Settings(api_key="private-local-fixture")
        self.initial = ConnectionProfiles.initial(self.local)

    def test_apply_hosted_and_restore_local_without_overwriting_defaults(self):
        hosted = apply_profile(self.initial, "compatible", HOST, "fixture-model", KEY)
        self.assertEqual(hosted.active.base_url, HOST)
        self.assertEqual(hosted.active.api_key, KEY)
        self.assertEqual(self.initial.active, self.local)
        restored = apply_profile(hosted, "local", "ignored", "ignored", "ignored")
        self.assertEqual(restored.active, self.local)
        self.assertEqual(restored.saved_hosted["compatible"].api_key, KEY)
        again = apply_profile(restored, "compatible", HOST+"/", "fixture-model-2", "")
        self.assertEqual(again.active.api_key, KEY)
        self.assertEqual(again.active.model, "fixture-model-2")

    def test_never_forward_saved_key_to_new_host_path_or_provider(self):
        hosted = apply_profile(self.initial, "compatible", HOST, "fixture", KEY)
        for url in ["https://other.invalid/v1", HOST+"/another-base"]:
            with self.assertRaisesRegex(ValueError, "Enter an API key"):
                apply_profile(hosted, "compatible", url, "fixture", "")
        with self.assertRaisesRegex(ValueError, "Enter an API key"):
            apply_profile(hosted, "openai", OPENAI_BASE_URL, "fixture", "")
        self.assertEqual(hosted.active.base_url, HOST)

    def test_keys_are_session_scoped_and_hidden_in_repr(self):
        first = apply_profile(self.initial, "compatible", HOST, "fixture", KEY)
        second = ConnectionProfiles.initial(self.local)
        self.assertEqual(second.saved_hosted, {})
        self.assertNotIn(KEY, repr(first))
        self.assertNotIn(KEY, repr(first.active))

    def test_reject_empty_fields_unsafe_credentials_and_non_https_remote(self):
        for url, model, key in [
            (HOST, "", KEY), (HOST, "fixture", ""), (HOST, "fixture", None),
            ("http://remote.invalid/v1", "fixture", KEY),
            ("https://user:pass@host.invalid/v1", "fixture", KEY),
            (HOST+"?api_key=bad", "fixture", KEY), (HOST+"/chat/completions", "fixture", KEY),
        ]:
            with self.subTest(url=url, model=model):
                with self.assertRaises(ValueError):
                    apply_profile(self.initial, "compatible", url, model, key)

    def test_other_compatible_base_paths_and_loopback_no_key(self):
        custom = apply_profile(self.initial, "compatible", "https://fixture.invalid/v1beta/openai", "fixture", KEY)
        self.assertEqual(custom.active.base_url, "https://fixture.invalid/v1beta/openai")
        local = apply_profile(self.initial, "compatible", "http://localhost:8081/api", "fixture", "")
        self.assertEqual(local.active.api_key, "local-not-needed")

    def test_openai_mode_pins_the_official_destination(self):
        with self.assertRaisesRegex(ValueError, "OpenAI mode"):
            apply_profile(self.initial, "openai", HOST, "fixture", KEY)
        official = apply_profile(self.initial, "openai", OPENAI_BASE_URL, "fixture", KEY)
        self.assertTrue(official.active.uses_completion_tokens)
        self.assertFalse(self.local.uses_completion_tokens)


class HostedProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_hosted_tool_loop_routes_key_model_and_token_parameter(self):
        for provider, url, parameter in [("openai", OPENAI_BASE_URL, "max_completion_tokens"),
                                         ("compatible", HOST, "max_tokens")]:
            with self.subTest(provider=provider):
                fake = FakeModel()
                requests = []
                def handler(request):
                    requests.append(request)
                    # Host/path assertions are independent of FakeModel's fixed path.
                    self.assertEqual(str(request.url), url+"/chat/completions")
                    self.assertEqual(request.headers["authorization"], "Bearer "+KEY)
                    return httpx.Response(200, json=fake.respond(json.loads(request.content))[1])
                settings = Settings(provider=provider, base_url=url, model="fixture-hosted-model", api_key=KEY)
                reply = await ask("CLOCK", Session(), settings, transport=httpx.MockTransport(handler))
                self.assertEqual(len(requests), 2)
                self.assertEqual(fake.requests[0]["model"], "fixture-hosted-model")
                self.assertEqual(fake.requests[0][parameter], 4096)
                other = "max_tokens" if parameter == "max_completion_tokens" else "max_completion_tokens"
                self.assertNotIn(other, fake.requests[0])
                self.assertEqual(reply.trace[-1]["name"], "get_current_utc_time")
                self.assertIn("utc_time", reply.text)
                self.assertNotIn(KEY, json.dumps(fake.requests))
                self.assertNotIn(KEY, json.dumps(reply.trace))

    async def test_compatible_can_explicitly_use_modern_token_parameter(self):
        fake = FakeModel()
        settings = Settings(provider="compatible", base_url="https://fixture.invalid/v1", model="fixture",
                            api_key=KEY, token_parameter="max_completion_tokens")
        await ask("Hello", Session(), settings, False, transport=fake.transport())
        self.assertEqual(fake.requests[0]["max_completion_tokens"], 4096)
        self.assertNotIn("max_tokens", fake.requests[0])

    async def test_official_model_profile_does_not_send_legacy_local_flags(self):
        fake = FakeModel()
        settings = Settings(provider="openai", base_url=OPENAI_BASE_URL, model="gpt-4.1", api_key=KEY)
        await ask("CLOCK", Session(), settings, transport=fake.transport())
        definitions = fake.requests[0]["tools"]
        # Optional-argument tools can be non-strict; the SDK decides per schema.
        wav = next(tool for tool in definitions if tool["function"]["name"] == "get_wav")
        self.assertTrue(wav["function"]["strict"])
        self.assertTrue(all(tool["function"]["parameters"]["additionalProperties"] is False for tool in definitions))

    async def test_check_server_uses_active_profile_without_disclosing_key(self):
        settings = Settings(provider="compatible", base_url=HOST, model="fixture", api_key=KEY)
        def handler(request):
            self.assertEqual(str(request.url), HOST+"/models")
            self.assertEqual(request.headers["authorization"], "Bearer "+KEY)
            return httpx.Response(200, json={"data": [{"id": "fixture"}]})
        report = await check_server(settings, transport=httpx.MockTransport(handler))
        self.assertTrue(report["selected_model_found"])
        self.assertEqual(report["provider"], "compatible")
        self.assertNotIn(KEY, json.dumps(report))

    def test_auth_quota_errors_are_useful_and_hide_provider_bodies(self):
        settings = Settings(provider="compatible", base_url=HOST, model="fixture", api_key=KEY)
        for status, expected in [(401, "Authentication"), (403, "Access denied"), (429, "quota"), (400, "token-limit")]:
            result = friendly_error(ModelHTTPError(status_code=status, model_name="fixture", body=KEY), settings)
            self.assertIn(expected, result)
            self.assertNotIn(KEY, result)


class ConnectionUITests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        from app import build_ui
        cls.local = Settings(api_key="private-local-fixture")
        cls.ui = build_ui(cls.local)
        cls.callbacks = {function.fn.__name__: function.fn for function in cls.ui.fns.values()}
        cls.initial = ConnectionProfiles.initial(cls.local)

    def test_gradio_config_never_contains_initial_or_saved_credentials(self):
        config = json.dumps(self.ui.get_config_file(), default=str)
        self.assertNotIn(self.local.api_key, config)
        self.assertNotIn(KEY, config)
        password = next(component for component in self.ui.blocks.values()
                        if isinstance(component, gr.Textbox) and component.label == "API key")
        self.assertEqual(password.type, "password")
        self.assertEqual(password.value, "")
        model_label = next(component for component in self.ui.blocks.values()
                           if getattr(component, "elem_id", None) == "monitor-current-model")
        self.assertIn(self.local.model, model_label.value)
        self.assertFalse(any(getattr(component, "elem_id", None) == "active-connection"
                             for component in self.ui.blocks.values()))

    def test_model_settings_toggle_and_close_follow_open_state(self):
        opened, state = self.callbacks["toggle_connection"](False)
        self.assertTrue(opened["open"])
        self.assertTrue(state)
        closed, state = self.callbacks["toggle_connection"](state)
        self.assertFalse(closed["open"])
        self.assertFalse(state)
        self.assertEqual(self.callbacks["close_connection"]()[1], False)

    def test_individual_copy_and_delete_controls_are_enabled(self):
        chatbot = next(component for component in self.ui.blocks.values()
                       if getattr(component, "elem_id", None) == "conversation")
        self.assertEqual(chatbot.buttons, ["copy"])
        self.assertFalse(chatbot.group_consecutive_messages)
        self.assertIn("delete_message", self.callbacks)

    def test_apply_clears_chat_history_trace_and_password_then_returns_private_state(self):
        applied = self.callbacks["apply_connection"]("compatible", HOST, "fixture", KEY, "auto", self.initial)
        self.assertEqual(applied[0].active.api_key, KEY)
        self.assertEqual(applied[1], [])
        self.assertIsInstance(applied[2], Session)
        self.assertEqual(applied[2].messages, [])
        self.assertEqual(applied[5], [])
        self.assertEqual(applied[6], "")
        self.assertIs(applied[7], False)  # Apply synchronizes the drawer's toggle state.
        self.assertEqual(applied[-1]["open"], False)  # Successful apply returns to the workspace.
        self.assertIn("fixture", applied[-2])  # Monitor follows the applied session model.
        self.assertNotIn(KEY, str(applied[3:]))
        # Invalid edits keep prior active state/history rather than partly applying.
        invalid = self.callbacks["apply_connection"]("compatible", HOST, "", KEY, "auto", applied[0])
        self.assertEqual(invalid[0], gr.skip())
        self.assertEqual(invalid[7], gr.skip())
        self.assertIn("Model ID", invalid[8])
        self.assertEqual(invalid[-1], gr.skip())  # Keep the drawer open to correct invalid fields.

    def test_draft_switch_does_not_apply_or_return_a_saved_key_to_browser(self):
        hosted = apply_profile(self.initial, "compatible", HOST, "fixture", KEY)
        changed = self.callbacks["change_profile"]("compatible", hosted)
        self.assertEqual(changed[2]["value"], "")
        self.assertEqual(hosted.active.model, "fixture")
        local_draft = self.callbacks["change_profile"]("local", hosted)
        self.assertEqual(local_draft[0]["value"], self.local.base_url)
        self.assertEqual(hosted.active.base_url, HOST)

    def test_local_model_selection_applies_and_is_retained(self):
        applied = self.callbacks["apply_connection"]("local", "ignored", "local-second-model", "", "auto", self.initial)
        self.assertEqual(applied[0].active.model, "local-second-model")
        self.assertEqual(applied[0].local.model, "local-second-model")
        self.assertEqual(applied[0].active.base_url, self.local.base_url)

    async def test_refresh_local_models_preserves_selection_and_failure(self):
        refresh = self.callbacks["refresh_local_models"]
        with patch("app.check_server", new=AsyncMock(return_value={"available_model_ids": ["first", "second"]})):
            update, status = await refresh(self.initial, "second")
        self.assertEqual(update["value"], "second")
        self.assertEqual(update["choices"], ["first", "second"])
        self.assertIn("2 local model", status)
        with patch("app.check_server", new=AsyncMock(side_effect=RuntimeError("offline"))):
            update, status = await refresh(self.initial, "second")
        self.assertEqual(update, gr.skip())
        self.assertIn("offline", status)

    def test_public_openai_endpoint_uses_official_profile(self):
        applied = self.callbacks["apply_connection"]("compatible", OPENAI_BASE_URL, "fixture", KEY, "auto", self.initial)
        self.assertEqual(applied[0].active.provider, "openai")
        self.assertTrue(applied[0].active.uses_completion_tokens)

    async def test_chat_uses_active_session_settings_and_redacts_output_metrics(self):
        profiles = apply_profile(self.initial, "compatible", HOST, "fixture", KEY)
        reply = Reply("Answer "+KEY, Session(), [{"event": "tool_call", "name": "fixture", "result": KEY}],
                      1.25, 2, 100, 25)
        with patch("app.ask", new=AsyncMock(return_value=reply)) as request:
            result = [item async for item in self.callbacks["respond"]("Hello", [], Session(), True, profiles)]
        self.assertIs(request.call_args.args[2], profiles.active)
        self.assertIn("fixture", result[-1][3])
        self.assertNotIn(KEY, str(result[-1][0]))
        self.assertNotIn(KEY, str(result[-1][4]))
        self.assertEqual(self.local.base_url, "http://127.0.0.1:8080/v1")

    async def test_standalone_tool_test_uses_fresh_history_and_active_profile(self):
        profiles = apply_profile(self.initial, "compatible", HOST, "fixture", KEY)
        reply = Reply("It is now UTC", Session(), [{"event": "tool_result", "name": "get_current_utc_time"}],
                      1, 2, 100, 25)
        with patch("app.ask", new=AsyncMock(return_value=reply)) as request:
            status, report = await self.callbacks["test_tool_connection"](profiles)
        self.assertEqual(request.call_args.args[1].messages, [])
        self.assertIs(request.call_args.args[2], profiles.active)
        self.assertTrue(report["clock_tool_executed"])
        self.assertIn("executed", status)
        self.assertNotIn(KEY, json.dumps(report))

    async def test_ui_check_server_uses_active_connection(self):
        profiles = apply_profile(self.initial, "compatible", HOST, "fixture", KEY)
        with patch("app.check_server", new=AsyncMock(return_value={"selected_model_found": True})) as request:
            status, _ = await self.callbacks["server_status"](profiles)
        self.assertIs(request.call_args.args[0], profiles.active)
        self.assertIn("configured model found", status)
