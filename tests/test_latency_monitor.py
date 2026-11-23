"""Token cache observations are truthful when a provider supplies them."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
import httpx
from pydantic_ai.messages import ModelResponse
from pydantic_ai.usage import RequestUsage
from agent_app.presentation import activity_view
from agent_app.progress import ObservedModel, RunProgress


class LatencyMonitorTests(unittest.IsolatedAsyncioTestCase):
    async def test_server_phase_timings_and_zero_cache_are_recorded(self):
        progress = RunProgress()
        step = await progress.begin('model', 'Model request 1')
        response = httpx.Response(200, request=httpx.Request('POST', 'http://localhost/v1/chat/completions'),
            json={'timings': {'prompt_ms': 1200, 'predicted_ms': 2300},
                  'usage': {'prompt_tokens': 500, 'prompt_tokens_details': {'cached_tokens': 0}}})
        await progress.observe_model_http(response)
        await progress.finish(step)
        rendered = activity_view(progress.snapshot())
        self.assertIn('0 cached', rendered)
        self.assertIn('500 newly processed', rendered)
        self.assertIn('Prompt processing: 1.20s', rendered)
        self.assertIn('Generation: 2.30s', rendered)

    async def test_cached_tokens_are_shown_per_request(self):
        progress = RunProgress()
        wrapped = SimpleNamespace(request=AsyncMock(return_value=ModelResponse(
            parts=[], usage=RequestUsage(input_tokens=100, output_tokens=10, cache_read_tokens=80))))
        observer = object.__new__(ObservedModel)
        observer.wrapped = wrapped
        observer.progress = progress
        await observer.request([], {}, None)
        step = progress.snapshot()[0]
        self.assertEqual(step['cached_input_tokens'], 80)
        self.assertEqual(step['uncached_input_tokens'], 20)
        self.assertIn('80 cached', activity_view([step]))

    async def test_unreported_cache_is_not_claimed(self):
        progress = RunProgress()
        wrapped = SimpleNamespace(request=AsyncMock(return_value=ModelResponse(
            parts=[], usage=RequestUsage(input_tokens=10, output_tokens=1))))
        observer = object.__new__(ObservedModel)
        observer.wrapped = wrapped
        observer.progress = progress
        await observer.request([], {}, None)
        self.assertNotIn('cached_input_tokens', progress.snapshot()[0])


if __name__ == '__main__': unittest.main()
