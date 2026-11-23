"""Safety regressions for direct answers and retained detector detail."""
import unittest
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import httpx
from pydantic_ai.agent import ModelRequestNode
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from agent_app.config import Settings
from agent_app.service import (Session, _direct_media, _scope_verified_trace,
                               _shortcut_time_verified, _simple_echo_details, ask)
from agent_app.tools import convert_local_time
from tests.test_latency_paths import ToolModel, detection_data


class ShortcutSafetyTests(unittest.TestCase):
    def setUp(self):
        self.args = {'search': 'Kirchberg', 'start_utc': '2026-10-22T22:00:00Z',
                     'end_utc': '2026-10-22T22:05:00Z', 'direct_answer': True}
        self.result = {'status': 'ok', 'query': 'Kirchberg', 'receiver_name': 'Kirchberg',
                       'receiver_id': 'LUKIRC_SYS001', 'start_utc': self.args['start_utc'],
                       'end_utc': self.args['end_utc'], 'downloaded': True, 'artifact_id': 'a'*32}

    def direct(self, name, args, data, prompt):
        prior = [ModelResponse(parts=[ToolCallPart(name, args, 'c')])]
        node = ModelRequestNode(ModelRequest(parts=[ToolReturnPart(name, data, 'c')]))
        with patch('agent_app.service.spectrogram_path', return_value=Path('/tmp/test.png')), \
             patch('agent_app.service.detection_path', return_value=Path('/tmp/detection.png')):
            return _direct_media(node, prior, prompt)

    def test_compound_location_question_does_not_shortcut(self):
        prompt = ('Show the spectrogram for Kirchberg station on 2026-10-22 from '
                  '22:00 to 22:05 UTC and where is the station?')
        self.assertIsNone(self.direct('get_spectrogram', self.args, self.result, prompt))

    def test_yes_no_availability_is_a_single_request(self):
        prompt = ('Check if a spectrogram is available in Kirchberg on 2026-10-22 '
                  'from 22:00 to 22:05 UTC or not')
        self.assertIsNotNone(self.direct('get_spectrogram', self.args, self.result, prompt))

    def test_wrong_station_cannot_shortcut_or_attach_as_matching_evidence(self):
        data = {**self.result, 'query': 'Humain', 'receiver_name': 'Humain', 'receiver_id': 'BEHUMA_SYS001'}
        prompt = 'Show the spectrogram for Kirchberg on 2026-10-22 from 22:00 to 22:05 UTC.'
        self.assertIsNone(self.direct('get_spectrogram', self.args, data, prompt))
        events = [{'event': 'tool_result', 'name': 'get_spectrogram', 'result': data}]
        self.assertEqual(_scope_verified_trace(prompt, events), [])
        self.assertEqual(len(_scope_verified_trace(prompt, [{'event': 'tool_result', 'name': 'get_spectrogram', 'result': self.result}])), 1)

    def test_relative_hour_range_must_match_pm(self):
        clock = [{'event': 'tool_result', 'name': 'get_current_utc_time',
                  'result': {'utc_time': '2026-11-14T12:00:00Z'}}]
        prompt = 'Detect meteors at Kirchberg station from 10-11 pm yesterday.'
        bad = {'query': 'Kirchberg', 'start_utc': '2026-11-13T09:00:00Z', 'end_utc': '2026-11-13T10:00:00Z'}
        good = {'query': 'Kirchberg', 'start_utc': '2026-11-13T22:00:00Z', 'end_utc': '2026-11-13T23:00:00Z'}
        self.assertFalse(_shortcut_time_verified(prompt, clock, bad))
        self.assertTrue(_shortcut_time_verified(prompt, clock, good))

    def test_non_europe_iana_timezone_requires_conversion(self):
        prompt = 'Show the spectrogram from 2026-10-22 10:00 to 10:05 America/New_York.'
        conversion = convert_local_time('2026-10-22T10:00:00', 'America/New_York',
                                        end_local_datetime='2026-10-22T10:05:00')
        data = {'start_utc': conversion['start_utc'], 'end_utc': conversion['end_utc']}
        self.assertFalse(_shortcut_time_verified(prompt, [], data))
        trace = [{'event': 'tool_result', 'name': 'convert_local_time', 'result': conversion}]
        self.assertTrue(_shortcut_time_verified(prompt, trace, data))
        other = 'Show 2026-10-22 10:00 to 10:05 America/Argentina/Buenos_Aires.'
        self.assertFalse(_shortcut_time_verified(other, [], data))

    def test_iso_offsets_are_checked_without_guessing(self):
        prompt = ('Show 2026-10-22T22:00+02:00 to 2026-10-22T22:05+02:00 spectrogram.')
        good = {'start_utc': '2026-10-22T20:00:00Z', 'end_utc': '2026-10-22T20:05:00Z'}
        bad = {'start_utc': '2026-10-22T22:00:00Z', 'end_utc': '2026-10-22T22:05:00Z'}
        self.assertTrue(_shortcut_time_verified(prompt, [], good))
        self.assertFalse(_shortcut_time_verified(prompt, [], bad))
        self.assertFalse(_shortcut_time_verified('Show 22:00 to 22:05 UTC+02:00.', [], good))

    def test_available_but_uninspected_slots_do_not_shortcut(self):
        args = {**self.args, 'end_utc': '2026-10-22T23:00:00Z', 'max_slots': 1}
        data = {**detection_data(), **self.result, 'end_utc': args['end_utc'],
                'analysis_performed': True, 'annotated_artifacts': [{'artifact_id': 'a'*32}],
                'available_slots_in_window': 12, 'inspected_slots_count': 1,
                'uninspected_slots_count': 11, 'coverage_complete': False}
        prompt = 'Detect meteor echoes for Kirchberg on 2026-10-22 from 22:00 to 23:00 UTC.'
        self.assertIsNone(self.direct('detect_meteors', args, data, prompt))
        # A slot already attempted and failed is not an untouched available slot.
        data.update(available_slots_in_window=2, uninspected_slots_count=1,
                    slot_errors=[{'stage': 'inference'}])
        self.assertIsNotNone(self.direct('detect_meteors', args, data, prompt))


class IndexedEchoTests(unittest.IsolatedAsyncioTestCase):
    def history(self, data):
        return [ModelRequest(parts=[UserPromptPart('Detect this recording')]),
                ModelResponse(parts=[ToolCallPart('detect_meteors', {}, 'scan')]),
                ModelRequest(parts=[ToolReturnPart('detect_meteors', data, 'scan')]),
                ModelResponse(parts=[TextPart('Detection complete')])]

    async def test_indexed_echo_uses_retained_result_without_model_call(self):
        messages = self.history(detection_data())
        def forbidden(request):
            raise AssertionError('Indexed detail should not call the model')
        reply = await ask('What was the confidence for echo 13?', Session(messages), Settings(),
                          transport=httpx.MockTransport(forbidden))
        self.assertEqual(reply.requests, 0)
        self.assertIn('echo #13', reply.text)
        self.assertIn('0.80', reply.text)
        self.assertIn('2026-10-22T22:00:13Z', reply.text)
        followup = await ask("What is echo 14's confidence?", reply.session, Settings(),
                             transport=httpx.MockTransport(forbidden))
        self.assertEqual(followup.requests, 0)
        self.assertIn('echo #14', followup.text)

    def test_missing_ambiguous_and_stale_results_are_not_guessed(self):
        data = detection_data()
        messages = self.history(data)
        missing = _simple_echo_details('What was the confidence for echo 99?', messages)
        self.assertIn("couldn't find", missing)
        data['slots'].append(dict(data['slots'][0]))
        ambiguous = _simple_echo_details('What was the confidence for echo 13?', self.history(data))
        self.assertIn('more than one recording', ambiguous)
        stale = messages + [ModelRequest(parts=[UserPromptPart('Show another recording')]),
                            ModelRequest(parts=[ToolReturnPart('get_spectrogram', {'status': 'ok'}, 'other')])]
        self.assertIsNone(_simple_echo_details('What was the confidence for echo 13?', stale))

    async def test_wrong_station_image_is_not_attached_even_after_recovery(self):
        calls = 0
        async def respond(request):
            nonlocal calls
            calls += 1
            payload = json.loads(request.content)
            if calls == 1:
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': 'wrong', 'type': 'function', 'function': {'name': 'get_spectrogram',
                    'arguments': json.dumps({'search': 'Humain', 'start_utc': '2026-10-22T22:00:00Z',
                                              'end_utc': '2026-10-22T22:05:00Z', 'direct_answer': True})}}]}
                finish = 'tool_calls'
            else:
                message = {'role': 'assistant', 'content': 'Here is the Kirchberg spectrogram.'}
                finish = 'stop'
            return httpx.Response(200, json={'id': 'fixture', 'object': 'chat.completion',
                'created': 0, 'model': payload['model'], 'choices': [{'index': 0,
                'message': message, 'finish_reason': finish}], 'usage': {'prompt_tokens': 10,
                'completion_tokens': 5, 'total_tokens': 15}})
        data = {'status': 'ok', 'query': 'Humain', 'receiver_name': 'Humain',
                'receiver_id': 'BEHUMA_SYS001', 'start_utc': '2026-10-22T22:00:00Z',
                'end_utc': '2026-10-22T22:05:00Z', 'downloaded': True, 'artifact_id': 'a'*32}
        with patch('agent_app.brams_tools.pybrams_setup_status', return_value={'ready': True, 'version': 'test'}), \
             patch('agent_app.brams_tools._load_spectrogram', return_value=data), \
             patch('agent_app.service.spectrogram_path', return_value=Path('/tmp/verified.png')):
            reply = await ask('Show the spectrogram for Kirchberg on 2026-10-22 from 22:00 to 22:05 UTC.',
                              Session(), Settings(), transport=httpx.MockTransport(respond))
        self.assertEqual(reply.images, [])
        self.assertIn('couldn’t verify', reply.text)
        self.assertNotIn('Humain', reply.text)
        self.assertEqual(calls, 3)

    async def test_out_of_scope_metadata_is_not_appended_to_valid_answer(self):
        async def get_spectrogram(search: str, start_utc: str, end_utc: str, direct_answer: bool = False) -> dict:
            """Get the verified spectrogram."""
            return {'status': 'ok', 'query': 'Kirchberg', 'receiver_name': 'Kirchberg',
                    'receiver_id': 'LUKIRC_SYS001', 'start_utc': start_utc, 'end_utc': end_utc,
                    'downloaded': True, 'artifact_id': 'a'*32}
        async def get_recordings(search: str, start_utc: str, end_utc: str) -> dict:
            """List recording metadata."""
            return {'status': 'ok', 'query': 'Humain', 'receiver_name': 'Humain',
                    'receiver_id': 'BEHUMA_SYS001', 'start_utc': start_utc, 'end_utc': end_utc,
                    'recordings': [], 'count': 0, 'source_url': 'https://example.invalid/archive'}
        calls = 0
        async def respond(request):
            nonlocal calls
            calls += 1
            payload = json.loads(request.content)
            arguments = {'start_utc': '2026-10-22T22:00:00Z', 'end_utc': '2026-10-22T22:05:00Z'}
            if calls <= 2:
                name = 'get_spectrogram' if calls == 1 else 'get_recordings'
                arguments['search'] = 'Kirchberg' if calls == 1 else 'Humain'
                if calls == 1:
                    arguments['direct_answer'] = False
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': f'call{calls}',
                    'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}]}
                finish = 'tool_calls'
            else:
                message = {'role': 'assistant', 'content': 'The requested spectrogram is displayed.'}
                finish = 'stop'
            return httpx.Response(200, json={'id': 'fixture', 'object': 'chat.completion', 'created': 0,
                'model': payload['model'], 'choices': [{'index': 0, 'message': message,
                'finish_reason': finish}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}})
        with patch('agent_app.service.spectrogram_path', return_value=Path('/tmp/verified.png')):
            reply = await ask('Show the spectrogram for Kirchberg on 2026-10-22 from 22:00 to 22:05 UTC.',
                              Session(), Settings(), transport=httpx.MockTransport(respond),
                              tool_functions=[get_spectrogram, get_recordings])
        self.assertEqual(calls, 3)
        self.assertNotIn('BEHUMA_SYS001', reply.text)
        self.assertNotIn('Humain', reply.text)
        self.assertNotIn('Verified tool results', reply.text)
        self.assertEqual(len(reply.images), 1)

    async def test_incomplete_hour_gets_explicit_coverage_even_if_model_omits_it(self):
        args = {'search': 'Kirchberg', 'start_utc': '2026-10-22T22:00:00Z',
                'end_utc': '2026-10-22T23:00:00Z', 'max_slots': 1, 'direct_answer': True}
        data = {**detection_data(), 'query': 'Kirchberg', 'start_utc': args['start_utc'],
                'end_utc': args['end_utc'], 'available_slots_in_window': 12,
                'inspected_slots_count': 1, 'uninspected_slots_count': 11,
                'coverage_complete': False, 'annotated_artifacts': [{'artifact_id': 'a'*32}]}
        fake = ToolModel('detect_meteors', args)
        with patch('agent_app.brams_tools.pybrams_setup_status', return_value={'ready': True, 'version': 'test'}), \
             patch('agent_app.brams_tools.vision_setup_status', return_value={'ready': True}), \
             patch('agent_app.brams_tools._detect_meteors', return_value=data), \
             patch('agent_app.service.detection_path', return_value=Path('/tmp/detection.png')):
            reply = await ask('Detect meteor echoes for Kirchberg on 2026-10-22 from 22:00 to 23:00 UTC.',
                              Session(), Settings(), transport=fake.transport())
        self.assertGreaterEqual(reply.requests, 2)
        self.assertIn('Coverage is incomplete', reply.text)
        self.assertIn('1 of 12 available recordings', reply.text)
        final = reply.session.messages[-1]
        self.assertIn('Coverage is incomplete', next(p.content for p in final.parts if isinstance(p, TextPart)))


if __name__ == '__main__': unittest.main()
