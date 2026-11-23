"""Deterministic protocol fixture, intentionally confined to the test directory."""

import json

import httpx

from agent_app.config import DEFAULT_MODEL

KIRCHBERG_PROMPT = ("Can you check if Kirchberg receiver is available from "
                    "22/10/2026 00:00:00 to 22/10/2026 01:00:00 or not?")
SPECTROGRAM_PROMPT = "Can you please show me 1 spectrogram in this hour?"
WAV_PROMPT = "Can you play one WAV recording from this hour?"


class FakeModel:
    def __init__(self):
        self.requests = []

    def respond(self, payload):
        self.requests.append(payload)
        messages = payload["messages"]
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        latest_user = user_messages[-1]
        last = messages[-1]
        called = [m for m in messages if m["role"] == "assistant" and m.get("tool_calls")]
        call = None
        text = "Hello from the protocol test fixture."
        finish = "stop"
        if latest_user == "HTTP_ERROR":
            return 503, {"error": {"message": "fixture unavailable", "type": "server_error"}}
        if latest_user == "LENGTH":
            text, finish = "Incomplete reply", "length"
        elif latest_user == "HISTORY":
            text = "Earlier user messages: " + " | ".join(user_messages[:-1])
        elif latest_user in {"CLOCK", "LOOP", "BAD_ARGS", "UNKNOWN_TOOL", "DURATION",
                            "Which receivers are available at Ecotron?", KIRCHBERG_PROMPT,
                            SPECTROGRAM_PROMPT, WAV_PROMPT, "WAV_LOOP", "WAV_LENGTH", "DISTINCT_LOOP"}:
            if latest_user in {"LOOP", "WAV_LOOP", "DISTINCT_LOOP"} or last["role"] != "tool":
                name, args = "get_current_utc_time", {}
                if latest_user in {"BAD_ARGS", "DURATION"}:
                    name = "inspect_time_window"
                    args = {"start_utc": "2026-10-28T11:00:00+01:00",
                            "end_utc": "2026-10-28T12:30:00+01:00"}
                    if latest_user == "BAD_ARGS":
                        args["start_utc"] = "2026-10-28T11:00:00"
                elif latest_user == "UNKNOWN_TOOL":
                    name = "unregistered_tool"
                elif latest_user == "Which receivers are available at Ecotron?":
                    name, args = "list_receivers", {"search": "Ecotron"}
                elif latest_user == KIRCHBERG_PROMPT:
                    name, args = "check_availability", {"search": "Kirchberg",
                        "start_utc": "2026-10-22T00:00:00Z", "end_utc": "2026-10-22T01:00:00Z"}
                elif latest_user == SPECTROGRAM_PROMPT:
                    name, args = "get_spectrogram", {"search": "Kirchberg",
                        "start_utc": "2026-10-22T00:00:00Z", "end_utc": "2026-10-22T01:00:00Z"}
                elif latest_user in {WAV_PROMPT, "WAV_LOOP", "WAV_LENGTH"}:
                    name, args = "get_wav", {"search": "Kirchberg",
                        "start_utc": "2026-10-22T00:00:00Z", "end_utc": "2026-10-22T01:00:00Z"}
                elif latest_user == "DISTINCT_LOOP":
                    name, args = "inspect_time_window", {
                        "start_utc": "2026-10-22T00:00:00Z",
                        "end_utc": f"2026-10-22T{len(self.requests):02d}:00:00Z"}
                call = (name, args)
            elif latest_user in {"BAD_ARGS", "UNKNOWN_TOOL"} and len(called) == 1:
                call = ("get_current_utc_time", {})
            else:
                text = "The tool returned: " + last["content"]
                if latest_user == "WAV_LENGTH":
                    finish = "length"
        message = {"role": "assistant", "content": text,
                   "reasoning_content": "Synthetic fixture field; not a model reasoning trace."}
        if call:
            name, arguments = call
            message["content"] = None
            message["tool_calls"] = [{
                "id": "call_" + str(len(self.requests)), "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }]
            finish = "tool_calls"
        return 200, {
            "id": "fixture_completion", "object": "chat.completion", "created": 1793181600,
            "model": DEFAULT_MODEL,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        }

    def handler(self, request):
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": DEFAULT_MODEL}]})
        if request.method == "POST" and request.url.path == "/v1/chat/completions":
            status, body = self.respond(json.loads(request.content))
            return httpx.Response(status, json=body)
        return httpx.Response(404, json={"error": "unexpected fixture URL"})

    def transport(self):
        return httpx.MockTransport(self.handler)
