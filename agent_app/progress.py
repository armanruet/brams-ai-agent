"""Per-run observations of actual model requests and validated tool executions."""

import inspect
import json
import logging
from time import perf_counter

from pydantic_ai.models.wrapper import WrapperModel

from .connections import redact


LOGGER = logging.getLogger(__name__)


class RunProgress:
    def __init__(self, callback=None, api_key=""):
        self.callback = callback
        self.api_key = api_key
        self.started = perf_counter()
        self.steps = []

    async def publish(self, step):
        if self.callback is None:
            return
        snapshot = redact(json.loads(json.dumps(step, default=str)), self.api_key)
        try:
            result = self.callback(snapshot)
            if inspect.isawaitable(result):
                await result
        except Exception as error:
            # Display observers must never change the outcome of a tool/model call.
            LOGGER.warning("Progress observer failed: %s", type(error).__name__)

    async def begin(self, kind, name, arguments=None):
        step = {"id": len(self.steps)+1, "kind": kind, "name": name,
                "status": "running", "started_at_seconds": perf_counter()-self.started,
                "duration_seconds": None}
        if arguments is not None:
            step["arguments"] = arguments
        self.steps.append(step)
        await self.publish(step)
        return step

    async def finish(self, step, status="completed", **details):
        step.update(status=status, duration_seconds=max(0, perf_counter()-self.started-step["started_at_seconds"]), **details)
        await self.publish(step)

    async def observe_model_http(self, response):
        """Capture optional local-server phase timings, without recording response content."""
        if not response.request.url.path.endswith("/chat/completions") or response.status_code != 200:
            return
        try:
            await response.aread()
            data = response.json()
            timing = data.get("timings") or {}
            step = next((item for item in reversed(self.steps)
                         if item["kind"] == "model" and item["status"] == "running"), None)
            if step is None:
                return
            for source, target in (("prompt_ms", "prompt_seconds"),
                                   ("predicted_ms", "generation_seconds")):
                value = timing.get(source)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value < 1e12:
                    step[target] = value / 1000.0
            cached = (data.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens")
            total = data.get("usage", {}).get("prompt_tokens")
            if isinstance(cached, int) and isinstance(total, int) and 0 <= cached <= total:
                step.update(cached_input_tokens=cached, uncached_input_tokens=total - cached)
        except (AttributeError, TypeError, ValueError):
            # Optional diagnostics must never make a model call fail.
            return

    def snapshot(self):
        return redact(json.loads(json.dumps(self.steps, default=str)), self.api_key)


class ObservedModel(WrapperModel):
    def __init__(self, wrapped, progress):
        super().__init__(wrapped)
        self.progress = progress

    async def request(self, messages, model_settings, model_request_parameters):
        index = 1+sum(step["kind"] == "model" for step in self.progress.steps)
        step = await self.progress.begin("model", f"Model request {index}")
        try:
            response = await self.wrapped.request(messages, model_settings, model_request_parameters)
        except BaseException as error:
            await self.progress.finish(step, "cancelled" if not isinstance(error, Exception) else "error",
                                       error_type=type(error).__name__)
            raise
        details = {"input_tokens": response.usage.input_tokens,
                   "output_tokens": response.usage.output_tokens, "finish_reason": response.finish_reason}
        cached = getattr(response.usage, "cache_read_tokens", 0)
        # A zero default also means the provider did not report caching. Only
        # advertise a cached/uncached split when the provider reported reuse.
        if isinstance(cached, int) and 0 < cached <= response.usage.input_tokens:
            details.update(cached_input_tokens=cached,
                           uncached_input_tokens=response.usage.input_tokens - cached)
        await self.progress.finish(step, **details)
        return response
