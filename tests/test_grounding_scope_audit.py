"""Grounding keeps unknown, receiver and time scope tied to actual evidence."""
import json
import unittest
from unittest.mock import patch
from pydantic_ai.messages import ModelResponse, TextPart

from agent_app.config import Settings
from agent_app.grounding import _matches_explicit_scope, _requested_receiver_names, apply_grounding_guard, check_unbacked_claims, required_domains, requires_tool_or_history
from agent_app.service import Session, ask
from tests.fake_model import FakeModel, KIRCHBERG_PROMPT


class GroundingScopeTests(unittest.TestCase):
    def test_utc_hyphenated_range_after_date_is_not_an_iso_offset(self):
        prompt = "Fetch BEHUMA_SYS001 spectrogram on 2023-12-14 21:15-21:20 UTC."
        result = {"receiver_id": "BEHUMA_SYS001", "start_utc": "2023-12-14T21:15:00Z",
                  "end_utc": "2023-12-14T21:20:00Z"}
        self.assertTrue(_matches_explicit_scope(prompt, result))
        self.assertFalse(_matches_explicit_scope(prompt, {**result, "start_utc": "2023-12-14T20:15:00Z"}))

    def test_scientific_original_is_not_interpreted_as_station(self):
        prompt = ("Get the original WAV for BEHUMA_SYS001, 2023-12-14 21:15-21:20 UTC. "
                  "Explain how the preview differs from the scientific original.")
        result = {"receiver_id": "BEHUMA_SYS001", "start_utc": "2023-12-14T21:15:00Z",
                  "end_utc": "2023-12-14T21:20:00Z"}
        self.assertTrue(_matches_explicit_scope(prompt, result))

    def test_units_and_catalog_terms_are_not_station_names(self):
        for prompt in ("Show at most five example receiver IDs in the catalog.",
                       "Give coordinates in decimal degrees and altitude in metres.",
                       "List duration in seconds and sample rate in Hz.",
                       "Find receiver BEFAKE_SYS001 in the BRAMS receiver registry."):
            with self.subTest(prompt=prompt):
                self.assertEqual(_requested_receiver_names(prompt), [])

    def test_recording_metadata_requires_recordings_not_availability_or_audio(self):
        for prompt in ("List BEHUMA_SYS001 recordings for 2023-12-14 21:15-21:45 UTC.",
                       "List recording metadata for BEHUMA_SYS001. Do not download audio."):
            with self.subTest(prompt=prompt):
                self.assertEqual(required_domains(prompt), {"get_recordings"})
        self.assertEqual(required_domains("Check recording availability for BEHUMA_SYS001."),
                         {"check_availability"})
        self.assertEqual(required_domains("Using that returned header, give the LO and beacon frequencies in Hz. Do not download it again."),
                         {"get_wav"})

    def test_unknown_availability_is_not_zero_but_explicit_zero_is_allowed(self):
        unknown = {'status': 'ok', 'receivers': [{'receiver_id': 'LUKIRC_SYS001', 'name': 'Kirchberg', 'coverage': 'unknown'}],
                   'coverage_counts': {'unknown': 1}, 'start_utc': '2026-10-22T00:00:00Z', 'end_utc': '2026-10-22T01:00:00Z'}
        event = {'event': 'tool_result', 'name': 'check_availability', 'result': unknown}
        text, issues = apply_grounding_guard('Was Kirchberg available?', 'There were 0 recordings at LUKIRC_SYS001.', [event])
        self.assertIn('unknown_availability_reported_as_zero', [i['code'] for i in issues])
        self.assertNotIn('0 recordings', text)
        self.assertIn('unknown', text.lower())
        known = {**unknown, 'coverage_counts': {'unknown': 0},
                 'receivers': [{**unknown['receivers'][0], 'coverage': 'none', 'availability_percent': 0}]}
        self.assertNotIn('unknown_availability_reported_as_zero', [i['code'] for i in check_unbacked_claims(
            'Was Kirchberg available?', 'There were 0 recordings at LUKIRC_SYS001.', [{**event, 'result': known}])])

    def test_explicit_zero_is_not_unknown_but_other_uptime_or_receiver_can_be(self):
        data = {'status': 'ok', 'recording_availability_checked': True, 'receivers': [
            {'receiver_id': 'LUKIRC_SYS001', 'name': 'Kirchberg', 'coverage': 'none', 'availability_percent': 0}],
            'start_utc': '2026-10-22T00:00:00Z', 'end_utc': '2026-10-22T01:00:00Z'}
        trace = [{'event': 'tool_result', 'name': 'check_availability', 'result': data}]
        wrong = 'Coverage is reported as UNKNOWN-equivalent "none" for the hour.'
        text, issues = apply_grounding_guard('Was Kirchberg available?', wrong, trace)
        self.assertIn('known_zero_reported_as_unknown', [issue['code'] for issue in issues])
        self.assertNotIn('UNKNOWN-equivalent', text)
        self.assertIn('none', text.lower())
        self.assertIn('0%', text)
        alternative = 'The availability is UNKNOWN-to-zero (a clean zero in the fixture).'
        self.assertIn('known_zero_reported_as_unknown',
                      [issue['code'] for issue in check_unbacked_claims('Was Kirchberg available?', alternative, trace)])
        for correct in ('Archive coverage is none. Current station uptime is unknown.',
                        'Archive coverage is none. Current station availability is unknown.',
                        'Kirchberg has no recordings; Humain has unknown coverage.',
                        'BEHUMA_SYS001 has unknown coverage.',
                        'Coverage for BEHUMA_SYS001 is unknown.',
                        'The coverage is not unknown: it is none.'):
            with self.subTest(text=correct):
                self.assertNotIn('known_zero_reported_as_unknown', [issue['code'] for issue in
                                 check_unbacked_claims('Was Kirchberg available?', correct, trace)])
        self.assertIn('known_zero_reported_as_unknown', [issue['code'] for issue in
                      check_unbacked_claims('Was Kirchberg available?', 'Coverage for LUKIRC_SYS001 is unknown.', trace)])

    def test_mixed_known_zero_and_unknown_is_scoped(self):
        data = {'status': 'ok', 'coverage_counts': {'unknown': 1, 'none': 1}, 'receivers': [
            {'receiver_id': 'LUKIRC_SYS001', 'name': 'Kirchberg', 'coverage': 'none'},
            {'receiver_id': 'BEHUMA_SYS001', 'name': 'Humain', 'coverage': 'unknown'}]}
        trace = [{'event': 'tool_result', 'name': 'check_availability', 'result': data}]
        self.assertFalse(any(i['code'] == 'unknown_availability_reported_as_zero' for i in
                         check_unbacked_claims('Check both stations.', 'LUKIRC_SYS001: 0 recordings.', trace)))
        self.assertTrue(any(i['code'] == 'unknown_availability_reported_as_zero' for i in
                        check_unbacked_claims('Check both stations.', 'BEHUMA_SYS001: 0 recordings.', trace)))
        self.assertTrue(any(i['code'] == 'unknown_availability_reported_as_zero' for i in
                        check_unbacked_claims('Check both stations.', 'There were 0 recordings.', trace)))
        self.assertFalse(any(i['code'] == 'unknown_availability_reported_as_zero' for i in
                         check_unbacked_claims('Check both stations.', 'Kirchberg had zero recordings.', trace)))
        self.assertTrue(any(i['code'] == 'unknown_availability_reported_as_zero' for i in
                        check_unbacked_claims('Check both stations.', 'Humain had no recordings.', trace)))

    def test_receiver_name_scope_and_alias(self):
        prompt = 'Show the spectrogram for Kirchberg on 2026-10-22 from 22:00 to 22:05 UTC.'
        base = {'status': 'ok', 'start_utc': '2026-10-22T22:00:00Z', 'end_utc': '2026-10-22T22:05:00Z'}
        self.assertFalse(_matches_explicit_scope(prompt, {**base, 'receiver_name': 'Humain', 'receiver_id': 'BEHUMA_SYS001'}))
        self.assertTrue(_matches_explicit_scope(prompt, {**base, 'receiver_name': 'Kirchberg', 'receiver_id': 'LUKIRC_SYS001'}))
        self.assertFalse(_matches_explicit_scope(prompt, {**base, 'query': 'Kirchberg', 'receiver_name': 'Humain'}))
        self.assertTrue(_matches_explicit_scope('Show a spectrogram for Ecotron 1.', {'receiver_name': 'Ecotron 1'}))
        self.assertFalse(_matches_explicit_scope('Show a spectrogram for Ecotron 1.', {'receiver_name': 'Ecotron 2'}))
        self.assertTrue(_matches_explicit_scope('Show LUKIRC_SYS001 from 22:00 to 22:05 UTC on 2026-10-22.',
                        {**base, 'receiver_id': 'LUKIRC_SYS001'}))
        self.assertTrue(_matches_explicit_scope(KIRCHBERG_PROMPT,
                        {'status': 'ok', 'query': 'Kirchberg', 'start_utc': '2026-10-22T00:00:00Z',
                         'end_utc': '2026-10-22T01:00:00Z', 'receivers': [{'receiver_id': 'TESTKIR_SYS001'}]}))
        self.assertTrue(_matches_explicit_scope('Give me the exact coordinates of station BEHUMA.',
                        {'status': 'ok', 'receivers': [{'receiver_id': 'BEHUMA_SYS001', 'name': 'Humain'}]}))
        self.assertTrue(_matches_explicit_scope("What's the noise level in BEHUMA's 21:15 recording (2023-12-14)?",
                        {'status': 'ok', 'receiver_id': 'BEHUMA_SYS001', 'receiver_name': 'Humain',
                         'start_utc': '2023-12-14T21:15:00Z', 'end_utc': '2023-12-14T21:20:00Z'}))

    def test_range_and_offset_bounds(self):
        day = {'status': 'ok', 'receiver_name': 'Kirchberg', 'start_utc': '2026-11-13T22:00:00Z', 'end_utc': '2026-11-13T23:00:00Z'}
        prompt = 'Detect meteors at Kirchberg station from 10-11 pm yesterday.'
        self.assertTrue(_matches_explicit_scope(prompt, day))
        self.assertFalse(_matches_explicit_scope(prompt, {**day, 'start_utc': '2026-11-13T09:00:00Z', 'end_utc': '2026-11-13T10:00:00Z'}))
        bare = 'Detect meteors at Kirchberg station yesterday, 10–11 pm.'
        self.assertTrue(_matches_explicit_scope(bare, day))
        self.assertFalse(_matches_explicit_scope(bare, {**day, 'start_utc': '2026-11-13T10:00:00Z', 'end_utc': '2026-11-13T11:00:00Z'}))
        self.assertFalse(_matches_explicit_scope('Detect meteors at Kirchberg station from 10 to 11 yesterday.', day))
        self.assertTrue(_matches_explicit_scope('Detect meteors at Kirchberg station from 22 to 23 yesterday.', day))
        iso = 'Show the spectrogram for Kirchberg station from 2026-10-22T22:00+02:00 to 2026-10-22T22:05+02:00.'
        actual = {'status': 'ok', 'receiver_name': 'Kirchberg', 'start_utc': '2026-10-22T20:00:00Z', 'end_utc': '2026-10-22T20:05:00Z'}
        self.assertTrue(_matches_explicit_scope(iso, actual))
        self.assertFalse(_matches_explicit_scope(iso, {**actual, 'start_utc': '2026-10-22T22:00:00Z'}))
        ny = 'Show the spectrogram at Kirchberg station from 18:00 to 18:05 America/New_York on 2026-10-22.'
        self.assertFalse(_matches_explicit_scope(ny, actual))
        self.assertTrue(_matches_explicit_scope(ny, {**actual, 'start_utc': '2026-10-22T22:00:00Z', 'end_utc': '2026-10-22T22:05:00Z'}))

    def test_textual_dates_and_local_zone_match_real_requested_day(self):
        data = {'status': 'ok', 'receiver_name': 'Kirchberg', 'start_utc': '2026-10-22T22:00:00Z', 'end_utc': '2026-10-22T23:00:00Z'}
        for date in ('22 October 2026', '22 Oct 2026', 'October 22, 2026', 'Oct 22 2026'):
            prompt = f'Detect meteors at Kirchberg station on {date} from 22:00 to 23:00 UTC.'
            with self.subTest(date=date):
                self.assertTrue(_matches_explicit_scope(prompt, data))
                self.assertFalse(_matches_explicit_scope(prompt, {**data, 'start_utc': '2026-10-23T22:00:00Z', 'end_utc': '2026-10-23T23:00:00Z'}))
        self.assertFalse(_matches_explicit_scope('Detect meteors at Kirchberg station on 32 October 2026 from 22:00 to 23:00 UTC.', data))
        local = 'Detect meteors at Kirchberg station on 22 October 2026 from 22:00 to 23:00 America/New_York.'
        self.assertTrue(_matches_explicit_scope(local, {**data, 'start_utc': '2026-10-23T02:00:00Z', 'end_utc': '2026-10-23T03:00:00Z'}))
        self.assertFalse(_matches_explicit_scope(local, data))
        offset = 'Detect meteors at Kirchberg station on October 22, 2026 from 22:00 to 23:00 UTC+02:00.'
        self.assertTrue(_matches_explicit_scope(offset, {**data, 'start_utc': '2026-10-22T20:00:00Z', 'end_utc': '2026-10-22T21:00:00Z'}))
        overnight = 'Detect meteors at Kirchberg station on 22 October 2026 from 23:00 to 00:05 UTC.'
        self.assertTrue(_matches_explicit_scope(overnight, {**data, 'start_utc': '2026-10-22T23:00:00Z', 'end_utc': '2026-10-23T00:05:00Z'}))
        self.assertFalse(_matches_explicit_scope(overnight, {**data, 'start_utc': '2026-10-21T23:00:00Z', 'end_utc': '2026-10-22T00:05:00Z'}))

    def test_multiple_receiver_ids_require_each_relevant_result(self):
        prompt = 'Compare noise level for BEHUMA_SYS001 and LUKIRC_SYS001 from 22:00 to 22:05 UTC on 22 October 2026.'
        base = {'status': 'ok', 'start_utc': '2026-10-22T22:00:00Z', 'end_utc': '2026-10-22T22:05:00Z'}
        humain = {**base, 'receiver_id': 'BEHUMA_SYS001'}
        kirchberg = {**base, 'receiver_id': 'LUKIRC_SYS001'}
        unrelated = {**base, 'receiver_id': 'BEECON_SYS001', 'source_url': 'https://example.invalid/LUKIRC_SYS001'}
        event = lambda data: {'event': 'tool_result', 'name': 'get_psd', 'result': data}
        self.assertFalse(_matches_explicit_scope(prompt, humain))  # A shortcut must not accept only one.
        self.assertTrue(_matches_explicit_scope(prompt, humain, allow_partial_receivers=True))
        self.assertFalse(_matches_explicit_scope(prompt, unrelated, allow_partial_receivers=True))
        self.assertTrue(requires_tool_or_history(prompt, [event(humain)]))
        self.assertFalse(requires_tool_or_history(prompt, [event(humain), event(kirchberg)]))
        self.assertTrue(requires_tool_or_history(prompt, [event(humain), event(unrelated)]))
        spoofed = {**humain, 'source_url': 'https://example.invalid/LUKIRC_SYS001'}
        self.assertTrue(requires_tool_or_history(prompt, [event(spoofed)]))

    def test_named_receiver_comparison_requires_both_names(self):
        prompt = 'Compare noise level for Kirchberg and Humain from 22:00 to 22:05 UTC on 22 October 2026.'
        base = {'status': 'ok', 'start_utc': '2026-10-22T22:00:00Z', 'end_utc': '2026-10-22T22:05:00Z'}
        kirchberg = {**base, 'receiver_name': 'Kirchberg', 'receiver_id': 'LUKIRC_SYS001'}
        humain = {**base, 'receiver_name': 'Humain', 'receiver_id': 'BEHUMA_SYS001'}
        wrong = {**base, 'receiver_name': 'Ecotron', 'receiver_id': 'BEECON_SYS001', 'source_url': 'https://example.invalid/Humain'}
        event = lambda data: {'event': 'tool_result', 'name': 'get_psd', 'result': data}
        self.assertFalse(_matches_explicit_scope(prompt, kirchberg))
        self.assertTrue(_matches_explicit_scope(prompt, kirchberg, allow_partial_receivers=True))
        self.assertTrue(requires_tool_or_history(prompt, [event(kirchberg)]))
        self.assertTrue(requires_tool_or_history(prompt, [event(kirchberg), event(wrong)]))
        self.assertFalse(requires_tool_or_history(prompt, [event(kirchberg), event(humain)]))

    def test_geographic_spectrogram_does_not_treat_prepositions_as_station_names(self):
        prompt = 'Show a spectrogram from a station in northern Belgium on 2023-12-14 from 21:00 to 21:05 UTC.'
        data = {'status': 'ok', 'action': 'spectrogram', 'direction': 'north', 'receiver_name': 'Ecotron',
                'start_utc': '2023-12-14T21:00:00Z', 'end_utc': '2023-12-14T21:05:00Z'}
        self.assertFalse(requires_tool_or_history(prompt, [{'event': 'tool_result', 'name': 'query_geographic_receivers', 'result': data}]))

    def test_backticked_verified_source_does_not_become_invented_url(self):
        source = 'https://brams.aeronomie.be/v4/file.php'
        trace = [{'event': 'tool_result', 'name': 'check_availability', 'result': {'status': 'ok', 'source_url': source}}]
        text, issues = apply_grounding_guard('Check availability.', f'Source: `{source}`', trace)
        self.assertNotIn('invented_url', [issue['code'] for issue in issues])
        self.assertIn(source, text)

    def test_where_is_station_requires_coordinates_evidence(self):
        prompt = 'Show the spectrogram for Kirchberg station from 22:00 to 22:05 and where is the station?'
        self.assertEqual(required_domains(prompt), {'get_spectrogram', 'list_receivers'})
        spec = {'event': 'tool_result', 'name': 'get_spectrogram', 'result': {'status': 'ok', 'receiver_name': 'Kirchberg',
                'start_utc': '2026-10-22T22:00:00Z', 'end_utc': '2026-10-22T22:05:00Z'}}
        self.assertTrue(requires_tool_or_history(prompt, [spec]))
        summary = {'event': 'tool_result', 'name': 'list_receivers', 'result': {'status': 'ok', 'detail': 'summary',
                   'receivers': [{'name': 'Kirchberg'}]}}
        self.assertTrue(requires_tool_or_history(prompt, [spec, summary]))
        summary['result']['detail'] = 'coordinates'
        self.assertFalse(requires_tool_or_history(prompt, [spec, summary]))


class FalseZeroModel(FakeModel):
    def respond(self, payload):
        self.requests.append(payload)
        if payload['messages'][-1]['role'] != 'tool':
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'avail', 'type': 'function',
                'function': {'name': 'check_availability', 'arguments': json.dumps({'search': 'Kirchberg',
                'start_utc': '2026-10-22T00:00:00Z', 'end_utc': '2026-10-22T01:00:00Z'})}}]}
            finish = 'tool_calls'
        else:
            message = {'role': 'assistant', 'content': 'There were 0 recordings at LUKIRC_SYS001.'}
            finish = 'stop'
        return 200, {'id': 'fixture', 'object': 'chat.completion', 'created': 0, 'model': payload['model'],
                     'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
                     'usage': {'prompt_tokens': 12, 'completion_tokens': 8, 'total_tokens': 20}}


class GroundingScopeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_correction_replaces_visible_and_history_text(self):
        unknown = {'status': 'ok', 'recording_availability_checked': True, 'coverage_counts': {'unknown': 1},
                   'receivers': [{'receiver_id': 'LUKIRC_SYS001', 'name': 'Kirchberg', 'coverage': 'unknown'}]}
        with patch('agent_app.brams_tools.pybrams_setup_status', return_value={'ready': True, 'version': 'test'}), \
             patch('agent_app.brams_tools._load_availability', return_value=unknown):
            reply = await ask('Is Kirchberg station available from 00:00 to 01:00 UTC on 2026-10-22?',
                              Session(), Settings(), transport=FalseZeroModel().transport())
        self.assertNotIn('0 recordings', reply.text)
        self.assertIn('unknown', reply.text.lower())
        final = next(p.content for m in reversed(reply.session.messages) if isinstance(m, ModelResponse)
                     for p in m.parts if isinstance(p, TextPart))
        self.assertNotIn('0 recordings', final)
        self.assertIn('unknown', final.lower())

    async def test_explicit_zero_mislabeled_unknown_is_replaced_in_reply_and_history(self):
        class FalseUnknownModel(FalseZeroModel):
            def respond(self, payload):
                status, body = super().respond(payload)
                message = body['choices'][0]['message']
                if message.get('content'):
                    message['content'] = 'Coverage is reported as UNKNOWN-equivalent "none" for the hour.'
                return status, body

        known = {'status': 'ok', 'recording_availability_checked': True,
                 'receivers': [{'receiver_id': 'LUKIRC_SYS001', 'name': 'Kirchberg',
                                'coverage': 'none', 'availability_percent': 0}]}
        with patch('agent_app.brams_tools.pybrams_setup_status', return_value={'ready': True, 'version': 'test'}), \
             patch('agent_app.brams_tools._load_availability', return_value=known):
            reply = await ask('Is Kirchberg station available from 00:00 to 01:00 UTC on 2026-10-22?',
                              Session(), Settings(), transport=FalseUnknownModel().transport())
        self.assertNotIn('UNKNOWN-equivalent', reply.text)
        self.assertIn('none', reply.text.lower())
        self.assertIn('0%', reply.text)
        final = next(p.content for m in reversed(reply.session.messages) if isinstance(m, ModelResponse)
                     for p in m.parts if isinstance(p, TextPart))
        self.assertNotIn('UNKNOWN-equivalent', final)
        self.assertIn('none', final.lower())

    async def test_two_receiver_comparison_retains_both_results_without_retry(self):
        class ComparisonModel(FakeModel):
            def respond(self, payload):
                self.requests.append(payload)
                count = sum(item['role'] == 'tool' for item in payload['messages'])
                if count < 2:
                    name = ['BEHUMA_SYS001', 'LUKIRC_SYS001'][count]
                    args = {'search': name, 'start_utc': '2026-10-22T22:00:00Z', 'end_utc': '2026-10-22T22:05:00Z'}
                    message = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': f'psd_{count}', 'type': 'function',
                               'function': {'name': 'get_psd', 'arguments': json.dumps(args)}}]}
                    finish = 'tool_calls'
                else:
                    message = {'role': 'assistant', 'content': 'Both requested noise measurements are available below.'}
                    finish = 'stop'
                return 200, {'id': 'fixture', 'object': 'chat.completion', 'created': 0, 'model': payload['model'],
                             'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
                             'usage': {'prompt_tokens': 12, 'completion_tokens': 8, 'total_tokens': 20}}

        def psd(search, *args):
            return {'status': 'ok', 'receiver_id': search, 'filename': search + '.wav',
                    'recording_start_utc': '2026-10-22T22:00:00Z', 'recording_end_utc': '2026-10-22T22:05:00Z',
                    'noise_band_hz': [800, 900], 'noise_psd': 0.01, 'calibrator_band_hz': [1497, 1503],
                    'calibrator_psd': 0.02, 'calibrator_frequency_hz': 1500, 'analyzed_duration_seconds': 300,
                    'units': 'normalized ADC amplitude squared per Hz', 'method': 'fixture', 'analysis_input': 'original WAV',
                    'wav_source_url': 'https://example.invalid/' + search}
        model = ComparisonModel()
        with patch('agent_app.brams_tools.pybrams_setup_status', return_value={'ready': True, 'version': 'test'}), \
             patch('agent_app.brams_tools._load_psd', side_effect=psd):
            reply = await ask('Compare noise level for BEHUMA_SYS001 and LUKIRC_SYS001 from 22:00 to 22:05 UTC on 22 October 2026.',
                              Session(), Settings(), transport=model.transport())
        self.assertEqual(reply.requests, 3)
        self.assertFalse(any(event.get('event') == 'grounding_retry' for event in reply.trace))
        self.assertEqual({event['result']['receiver_id'] for event in reply.trace if event.get('event') == 'tool_result'},
                         {'BEHUMA_SYS001', 'LUKIRC_SYS001'})
        self.assertIn('BEHUMA_SYS001', reply.text)
        self.assertIn('LUKIRC_SYS001', reply.text)

    def test_contradiction_replacement_excludes_out_of_scope_tool_result(self):
        from agent_app.grounding import apply_grounding_guard
        trace = [
            {'event': 'tool_result', 'name': 'check_availability', 'result': {
                'status': 'ok', 'recording_availability_checked': True,
                'receivers': [{'receiver_id': 'BEGENT_SYS001', 'name': 'Gent', 'coverage': 'full', 'availability_percent': 100}]}},
            {'event': 'tool_result', 'name': 'check_availability', 'result': {
                'status': 'ok', 'recording_availability_checked': True,
                'receivers': [{'receiver_id': 'BEHUMA_SYS001', 'name': 'Humain', 'coverage': 'unknown', 'availability_percent': None}]}},
        ]
        cleaned, issues = apply_grounding_guard(
            'Check BEHUMA_SYS001 availability on 2023-12-14 from 21:00 to 22:00 UTC.',
            'There were 0 recordings at BEHUMA_SYS001.', trace,
        )
        self.assertTrue(issues)
        self.assertIn('BEHUMA_SYS001', cleaned)
        self.assertNotIn('BEGENT_SYS001', cleaned)

    def test_negated_do_not_query_recordings_does_not_require_availability(self):
        prompt = ('Convert 2026-10-25 01:30 to 03:30 Brussels local time to UTC and report the '
                  'UTC start, UTC end, and duration in minutes. Do not query recordings.')
        self.assertEqual(required_domains(prompt), set())
        time_trace = [{'event': 'tool_result', 'name': 'convert_local_time', 'result': {
            'start_utc': '2026-10-24T23:30:00Z', 'end_utc': '2026-10-25T02:30:00Z', 'duration_minutes': 180}}]
        self.assertFalse(requires_tool_or_history(prompt, time_trace))


if __name__ == '__main__': unittest.main()
