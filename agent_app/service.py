"""Keep the framework's tool history separate from the visible Gradio transcript."""

import asyncio
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import math
import re
import threading
from time import perf_counter
from typing import Any

import httpx
from openai import APIConnectionError, APITimeoutError
from pydantic_ai import ModelRetry, capture_run_messages
from pydantic_ai.agent import CallToolsNode, ModelRequestNode
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart,
    RetryPromptPart, ToolCallPart, ToolReturnPart,
)
from pydantic_ai.usage import RunUsage, UsageLimits

from .cancellation import OperationCancelled, cancellation_scope, check_cancelled
from .config import DEFAULT_LOCAL_CONTEXT_SIZE, Settings
from .artifacts import detection_path, spectrogram_path, wav_path
from .grounding import _matches_explicit_scope, _requested_clock_range, apply_grounding_guard, requires_tool_or_history, safe_unfulfilled_response, deterministic_limit_response
from .harness import open_agent
from .recovery import complete_history, evidence_summary, meteor_summary, spectrogram_summary, geographic_summary, wav_summary, _format_recording_period
from .tool_guard import RepeatedToolCall
from .progress import RunProgress


@dataclass
class Session:
    """Server-side, per-browser-session state. Never accept this from a text box."""
    messages: list[ModelMessage] = field(default_factory=list)
    successful_turns: int = 0
    ui_messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Reply:
    text: str
    session: Session
    trace: list[dict[str, Any]]
    elapsed_seconds: float
    requests: int
    input_tokens: int
    output_tokens: int
    images: list[str] = field(default_factory=list)
    audio: list[dict[str, str]] = field(default_factory=list)
    warning: str | None = None
    timings: list[dict[str, Any]] = field(default_factory=list)
    model_text: str | None = None
    detections: list[dict[str, str]] = field(default_factory=list)


class IncompleteResponseError(RuntimeError):
    pass


class ContextLengthExceededError(RuntimeError):
    """Raised when a prompt/turn exceeds the active model's served context window."""

    def __init__(
        self,
        message: str = "The request exceeded the model's available context window.",
        *,
        estimated_prompt_tokens: int,
        context_size: int | None,
        requested_output_tokens: int = 0,
    ) -> None:
        super().__init__(message)
        self.estimated_prompt_tokens = estimated_prompt_tokens
        self.context_size = context_size
        self.requested_output_tokens = requested_output_tokens
        self.total_estimated_tokens = estimated_prompt_tokens + requested_output_tokens


def wav_attachments(trace):
    audio = []
    seen = set()
    for event in trace:
        result = event.get("result", {})
        if (event.get("event") == "tool_result" and event.get("name") == "get_wav"
                and isinstance(result, dict) and result.get("status") == "ok"
                and result.get("downloaded") is True):
            original = wav_path(result.get("artifact_id"))
            preview = wav_path(result.get("artifact_id"), preview=True)
            if original and preview and str(original) not in seen:
                rname = result.get("receiver_name")
                rid = result.get("receiver_id")
                station = f"{rname} ({rid})" if rname and rid and rname != rid else (rname or rid or "")
                period = (
                    _format_recording_period(result.get("recording_start_utc"), result.get("recording_end_utc"))
                    if result.get("recording_start_utc") and result.get("recording_end_utc")
                    else ""
                )
                label_parts = ["Listening preview (normalized PCM16)"]
                if station:
                    label_parts.append(str(station))
                if period:
                    label_parts.append(period)
                audio.append({
                    "original": str(original),
                    "preview": str(preview),
                    "filename": result.get("filename", "Original BRAMS WAV"),
                    "label": " · ".join(label_parts),
                })
                seen.add(str(original))
    return audio


def spectrogram_images(trace: list[dict]) -> list[str]:
    images = []
    for event in trace:
        result = event.get("result", {})
        if (event.get("event") == "tool_result" and event.get("name") in {"get_spectrogram", "query_geographic_receivers"}
                and isinstance(result, dict) and result.get("status") == "ok"
                and result.get("downloaded") is True):
            path = spectrogram_path(result.get("artifact_id"))
            if path and str(path) not in images:
                images.append(str(path))
    return images


def detection_images(trace: list[dict]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for event in trace:
        result = event.get("result", {})
        if (event.get("event") == "tool_result" and event.get("name") == "detect_meteors"
                and isinstance(result, dict) and result.get("status") == "ok"
                and result.get("analysis_performed") is True):
            for entry in result.get("annotated_artifacts", []):
                if not isinstance(entry, dict):
                    continue
                path = detection_path(entry.get("artifact_id", ""))
                if path and str(path) not in seen:
                    items.append({
                        "path": str(path),
                        "label": str(entry.get("label") or "Meteor detection spectrogram"),
                    })
                    seen.add(str(path))
    return items


def _scope_verified_trace(prompt, trace):
    """Keep diagnostic trace intact but never present out-of-scope evidence."""
    verified = []
    for event in trace:
        if event.get("event") == "tool_result" and isinstance(event.get("result"), dict):
            # Multi-receiver requests produce one result per receiver; keep
            # individually relevant evidence while the grounding guard checks
            # that all requested receivers are covered in aggregate.
            if not _matches_explicit_scope(prompt, event["result"], allow_partial_receivers=True):
                continue
        verified.append(event)
    return verified


def _detection_coverage_notice(trace):
    """Describe incomplete latest scan even if the model omitted that caveat."""
    event = next((item for item in reversed(trace)
                  if item.get("event") == "tool_result" and item.get("name") == "detect_meteors"
                  and isinstance(item.get("result"), dict) and item["result"].get("analysis_performed") is True), None)
    if event is None:
        return ""
    result = event["result"]
    remaining = result.get("uninspected_slots_count")
    if not ((isinstance(remaining, int) and not isinstance(remaining, bool) and remaining > 0)
            or result.get("coverage_complete") is False):
        return ""
    inspected, available = result.get("inspected_slots_count"), result.get("available_slots_in_window")
    if isinstance(inspected, int) and isinstance(available, int):
        return (f"**Coverage is incomplete:** {inspected} of {available} available recordings were analyzed; "
                "the result does not establish meteor activity across the full requested window.")
    return ("**Coverage is incomplete:** the result applies only to the recordings analyzed; "
            "meteor activity across the full requested window is unknown.")


def tool_trace(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """Show tool actions and validation feedback, without model reasoning text."""
    trace = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                try:
                    arguments = part.args_as_dict()
                except (ValueError, TypeError):
                    arguments = {"unparsed_arguments": str(part.args)}
                trace.append({
                    "event": "tool_call", "name": part.tool_name,
                    "call_id": part.tool_call_id, "arguments": arguments,
                })
            elif isinstance(part, ToolReturnPart):
                if part.outcome == "success":
                    trace.append({
                        "event": "tool_result", "name": part.tool_name,
                        "call_id": part.tool_call_id, "result": part.content,
                    })
                else:
                    trace.append({
                        "event": "validation_retry" if part.outcome == "retried" else "tool_error",
                        "name": part.tool_name, "call_id": part.tool_call_id,
                        "outcome": part.outcome, "feedback": part.content,
                        "retry": asdict(part.retry) if part.retry is not None else None,
                    })
            elif isinstance(part, RetryPromptPart):
                trace.append({
                    "event": "validation_retry", "name": part.tool_name,
                    "call_id": part.tool_call_id, "feedback": part.content,
                })
    # Error details can contain datetime objects; normalize for gr.JSON.
    return json.loads(json.dumps(trace, default=str))


def _detailed_detection_request(prompt: str) -> bool:
    # Do not shortcut requests that ask the model to do more than give the verdict.
    return bool(re.search(
        r"\b(detail(?:s|ed)?|list|each|every echo|times? of|when|confidence|"
        r"explain|why|how(?!\s+many\b)|compar(?:e|ison)|summari[sz]e|table|other|also|"
        r"noise|calibrat\w*|psd|audio|wav|frequency|coordinates|receiver distance)\b",
        prompt, re.IGNORECASE,
    ))


def _compact_detection_result(result, *, rows_per_slot=0):
    """Keep detector totals and provenance while limiting individual rows sent to the model."""
    keep = {"status", "analysis_performed", "query", "receiver_id", "receiver_name", "source_url",
            "image_source_url", "start_utc", "end_utc", "recording_start_utc", "recording_end_utc",
            "artifact_id", "annotated_artifacts", "available_slots_in_window", "inspected_slots_count",
            "uninspected_slots_count", "coverage_complete", "expected_slots_in_window", "slot_errors",
            "meteor_detected", "total_meteor_count", "high_confidence_count", "max_confidence",
            "detection_limit_reached",
            "confidence_threshold", "model_name", "message", "error_code", "caveat",
            "uncovered_slot_starts_utc", "overlay_errors"}
    slot_keep = {"filename", "recording_start_utc", "recording_end_utc", "meteor_count",
                 "meteor_detected", "max_confidence", "annotated_artifact_id", "detection_limit_reached",
                 "high_confidence_count", "input_kind"}
    compact = {key: value for key, value in result.items() if key in keep}
    if isinstance(result.get("slots"), list):
        slots = []
        for slot in result["slots"]:
            if not isinstance(slot, dict):
                continue
            item = {key: value for key, value in slot.items() if key in slot_keep}
            rows = slot.get("detections")
            if isinstance(rows, list):
                row_keys = {"echo_index", "estimated_echo_utc", "time_offset_seconds", "confidence", "confidence_tier"}
                item["detections"] = [{key: value for key, value in row.items() if key in row_keys}
                                      for row in rows[:rows_per_slot] if isinstance(row, dict)]
                item["retained_detections_count"] = len(item["detections"])
                item["detections_truncated"] = bool(slot.get("detections_truncated") or len(rows) > len(item["detections"]))
            slots.append(item)
        compact["slots"] = slots
    if result.get("analysis_performed") is True:
        compact["individual_echo_details"] = "Only the displayed sample is included; complete retained details are available on request."
    return compact


def _compact_detailed_detection(result, *, max_chars=5000):
    """Retain bounded compact echo rows for a detailed follow-up even on large scans."""
    if len(json.dumps(result, default=str)) <= 3000:
        return result
    slots = result.get("slots") if isinstance(result.get("slots"), list) else []
    max_rows = max((len(slot.get("detections", [])) for slot in slots
                    if isinstance(slot, dict) and isinstance(slot.get("detections"), list)), default=0)
    if max_rows <= 0:
        return _compact_detection_result(result)
    for candidate in dict.fromkeys([max_rows, 20, 10, 5, 3, 2, 1]):
        if candidate > max_rows:
            continue
        compacted = _compact_detection_result(result, rows_per_slot=candidate)
        if len(json.dumps(compacted, default=str)) <= max_chars:
            return compacted
    one_per_slot = _compact_detection_result(result, rows_per_slot=1)
    if len(json.dumps(one_per_slot, default=str)) <= max(max_chars, MODEL_HISTORY_MAX_CHARS - 2000):
        return one_per_slot
    return _compact_detection_result(result)


def _compact_detection_history(messages, *, detailed=False):
    """Compact the copy sent to the model; full evidence stays in the session."""
    latest = None
    if detailed:
        latest = next((part for message in reversed(messages) for part in reversed(message.parts)
                       if isinstance(part, ToolReturnPart) and part.tool_name == "detect_meteors"
                       and isinstance(part.content, dict) and part.content.get("status") == "ok"), None)
    for message in messages:
        for part in message.parts:
            if not isinstance(part, ToolReturnPart) or part.tool_name != "detect_meteors" or not isinstance(part.content, dict):
                continue
            # Keep compact rows for the latest scan on a detailed follow-up,
            # while stripping older scans so prompt growth stays bounded.
            if part is latest:
                part.content = _compact_detailed_detection(part.content)
                continue
            part.content = _compact_detection_result(part.content)
    return messages


# This is a conservative character budget, not an exact model-token limit.  The
# system prompt/tool schemas and the current user request have separate costs.
MODEL_HISTORY_MAX_CHARS = 12000
INTRA_TURN_TOOL_MAX_CHARS = 8000
SYSTEM_TOOLS_ESTIMATED_TOKENS = 3800
SYSTEM_BASE_ESTIMATED_TOKENS = 250
CONTEXT_SAFETY_MARGIN_TOKENS = 512
MIN_OUTPUT_TOKENS = 256
_CONTEXT_CACHE_TTL_SECONDS = 60.0
_PROBED_CONTEXT_CACHE: dict[tuple[str, str], tuple[float, int | None]] = {}
_CONTEXT_OVERFLOW_RE = re.compile(
    r"exceed_context_size_error|"
    r"exceeds?\s+(?:the\s+)?(?:available\s+)?context\s+(?:size|window|length)|"
    r"context_length_exceeded|maximum\s+context\s+length|"
    r"context\s+window|prompt\s+is\s+too\s+long|too\s+many\s+tokens",
    re.I,
)
_HISTORY_OMITTED_NOTE = (
    "\n\n[Context note: Some earlier conversation was omitted to fit the available context. "
    "Do not guess missing station, time or results; ask for clarification if needed.]"
)


def _estimate_text_tokens(text: str) -> int:
    return math.ceil(len(text) / 3.5) if text else 0


def _estimate_request_tokens(
    history: list[ModelMessage],
    prompt_text: str,
    instructions_suffix: str,
    enable_tools: bool,
) -> int:
    base = SYSTEM_TOOLS_ESTIMATED_TOKENS if enable_tools else SYSTEM_BASE_ESTIMATED_TOKENS
    history_chars = sum(_group_cost([message]) for message in history) if history else 0
    return (
        base
        + _estimate_text_tokens(instructions_suffix)
        + _estimate_text_tokens(prompt_text)
        + math.ceil(history_chars / 3.5)
    )


async def _resolve_context_size(
    settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None,
) -> int | None:
    if settings.context_size is not None:
        return settings.context_size
    if settings.provider != "local":
        return None
    if transport is not None:
        return DEFAULT_LOCAL_CONTEXT_SIZE
    key = (settings.base_url, settings.model)
    now = perf_counter()
    cached = _PROBED_CONTEXT_CACHE.get(key)
    if cached is not None and (now - cached[0]) < _CONTEXT_CACHE_TTL_SECONDS:
        return cached[1] or DEFAULT_LOCAL_CONTEXT_SIZE
    from .diagnostics import probe_local_context_size
    probed = await probe_local_context_size(settings)
    _PROBED_CONTEXT_CACHE[key] = (now, probed)
    return probed or DEFAULT_LOCAL_CONTEXT_SIZE


def _plan_output_budget(
    estimated_prompt_tokens: int,
    target_max_tokens: int,
    context_size: int | None,
) -> int:
    if context_size is None:
        return target_max_tokens
    remaining = context_size - estimated_prompt_tokens - CONTEXT_SAFETY_MARGIN_TOKENS
    if remaining < MIN_OUTPUT_TOKENS:
        raise ContextLengthExceededError(
            estimated_prompt_tokens=estimated_prompt_tokens,
            context_size=context_size,
            requested_output_tokens=target_max_tokens,
        )
    return min(target_max_tokens, remaining)


def _is_context_overflow_error(error: Exception) -> bool:
    if not isinstance(error, ModelHTTPError) or error.status_code != 400:
        return False
    detail = f"{error} {getattr(error, 'body', '')}"
    return bool(_CONTEXT_OVERFLOW_RE.search(detail))


def _group_cost(group):
    # ModelRequest.instructions holds the static system instructions on
    # every request object, but those are sent once as the system prefix.
    # Count message parts rather than repeatedly charging that prefix.
    return sum(64 + sum(len(str(part)) for part in message.parts) for message in group)


def _shrink_tool_content(content, budget):
    """Compact an oversized tool return while keeping provenance and paired tool structure."""
    if not isinstance(content, dict):
        text = str(content)
        return text[:max(0, budget - 20)] + "...[truncated]" if len(text) > budget else content
    if isinstance(content.get("slots"), list):
        content = _compact_detection_result(content, rows_per_slot=0)
    compact = deepcopy(content)
    if len(json.dumps(compact, default=str)) <= budget:
        return compact
    for key, value in list(compact.items()):
        if isinstance(value, list) and len(value) > 3:
            compact[key] = value[:3]
            compact["history_truncated"] = True
    if len(json.dumps(compact, default=str)) <= budget:
        return compact
    keep_keys = (
        "status", "query", "receiver_id", "receiver_name", "start_utc", "end_utc",
        "recording_start_utc", "recording_end_utc", "source", "source_url",
        "image_source_url", "wav_source_url", "artifact_id", "count", "site_count",
        "active_site_count", "total_meteor_count", "inspected_slots_count",
        "available_slots_in_window", "coverage_complete", "recording_availability_checked",
        "analysis_performed", "scope", "action", "direction", "message", "error_code", "caveat",
    )
    trimmed = {key: compact[key] for key in keep_keys if key in compact}
    for key in ("message", "caveat", "source", "query"):
        if isinstance(trimmed.get(key), str) and len(trimmed[key]) > 300:
            trimmed[key] = trimmed[key][:280] + "...[truncated]"
    trimmed["history_truncated"] = True
    if len(json.dumps(trimmed, default=str)) <= budget:
        return trimmed
    minimal = {key: trimmed[key] for key in ("status", "receiver_id", "start_utc", "end_utc") if key in trimmed}
    minimal["history_truncated"] = True
    return minimal


def _shrink_history_group(group, max_chars):
    """Fit the latest turn into budget instead of dropping all prior context."""
    shrunk = deepcopy(group)
    if _group_cost(shrunk) <= max_chars:
        return shrunk
    tool_return_count = sum(1 for message in shrunk for part in message.parts if isinstance(part, ToolReturnPart))
    per_tool_budget = max(512, max_chars // max(2, tool_return_count + 1))
    for message in shrunk:
        for part in message.parts:
            if isinstance(part, ToolReturnPart) and len(str(part)) > per_tool_budget:
                part.content = _shrink_tool_content(part.content, per_tool_budget)
    if _group_cost(shrunk) <= max_chars:
        return shrunk
    overflow = _group_cost(shrunk) - max_chars
    for message in shrunk:
        for part in message.parts:
            if overflow <= 0:
                break
            if isinstance(part, (TextPart, UserPromptPart)) and isinstance(part.content, str) and len(part.content) > 256:
                cut = min(overflow + 32, len(part.content) - 128)
                if cut > 0:
                    part.content = part.content[:len(part.content) - cut] + "\n...[truncated]"
                    overflow = _group_cost(shrunk) - max_chars
    return shrunk


def _bounded_model_history(messages, *, max_chars=MODEL_HISTORY_MAX_CHARS):
    """Keep a suffix of whole user turns, including each tool call and its return."""
    groups = []
    for message in messages:
        if any(isinstance(part, UserPromptPart) for part in message.parts) or not groups:
            groups.append([])
        groups[-1].append(message)
    selected, used = [], 0
    for group in reversed(groups):
        cost = _group_cost(group)
        if used + cost > max_chars:
            if not selected:
                fitted = _shrink_history_group(group, max_chars)
                if _group_cost(fitted) <= max_chars:
                    selected.append(fitted)
            break
        selected.append(group)
        used += cost
    return [message for group in reversed(selected) for message in group]


def _limit_current_tool_results(node, full_results, *, max_chars=INTRA_TURN_TOOL_MAX_CHARS):
    """Save full tool payloads, then cap oversized returns before the next model call."""
    if not isinstance(node, ModelRequestNode):
        return
    for part in node.request.parts:
        if not isinstance(part, ToolReturnPart):
            continue
        if part.tool_name == "detect_meteors" and isinstance(part.content, dict):
            full_results[part.tool_call_id] = deepcopy(part.content)
            part.content = _compact_detection_result(part.content, rows_per_slot=3)
        elif len(json.dumps(part.content, default=str)) > max_chars:
            full_results[part.tool_call_id] = deepcopy(part.content)
            part.content = _shrink_tool_content(part.content, max_chars)


_limit_current_detection_results = _limit_current_tool_results


def _restore_tool_results(messages, full_results):
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolReturnPart) and part.tool_call_id in full_results:
                part.content = deepcopy(full_results[part.tool_call_id])


_restore_detection_results = _restore_tool_results


def _latest_detection_for_details(messages):
    """Return only the immediately preceding successful data result."""
    # Only the most recent non-clock tool result is eligible; otherwise an
    # implicit follow-up could accidentally use a different earlier recording.
    data = None
    intervening_user_turn = False
    verified_detail_turns = 0
    for message in reversed(messages):
        if (isinstance(message, ModelResponse) and isinstance(message.metadata, dict)
                and message.metadata.get("direct_detection_details") is True):
            verified_detail_turns += 1
        for part in reversed(message.parts):
            if isinstance(part, UserPromptPart):
                if verified_detail_turns:
                    verified_detail_turns -= 1
                else:
                    intervening_user_turn = True
                continue
            if not isinstance(part, ToolReturnPart) or part.tool_name in {"get_current_utc_time", "convert_local_time", "inspect_time_window"}:
                continue
            if (not intervening_user_turn and part.tool_name == "detect_meteors" and part.outcome == "success" and isinstance(part.content, dict)
                    and part.content.get("status") == "ok" and part.content.get("analysis_performed") is True):
                data = part.content
            break
        else:
            continue
        break
    return data if isinstance(data, dict) and isinstance(data.get("slots"), list) else None


def _indexed_echo_details(prompt, messages):
    """Resolve a narrowly phrased indexed echo query from retained evidence."""
    match = re.fullmatch(
        r"\s*(?:(?:what\s+(?:was|is)|show\s+me|give\s+me)\s+(?:the\s+)?"
        r"(?:confidence|time|timestamp|details?)\s+(?:for|of)\s+(?:the\s+)?"
        r"(?:detected\s+)?(?:meteor\s+)?echo\s*#?\s*(\d+)|"
        r"(?:what\s+(?:was|is)\s+)?(?:the\s+)?(?:meteor\s+)?echo\s*#?\s*(\d+)"
        r"(?:'s|’s)\s+(?:confidence|time|timestamp|details?))\s*[?.!]*\s*",
        prompt, re.I,
    )
    if not match:
        return None
    number = int(match.group(1) or match.group(2))
    if number < 1:
        return None
    data = _latest_detection_for_details(messages)
    if data is None:
        return None
    matches = [(slot, row) for slot in data["slots"] if isinstance(slot, dict)
               for row in (slot.get("detections") if isinstance(slot.get("detections"), list) else []) if isinstance(row, dict)
               and row.get("echo_index") == number]
    if len(matches) > 1:
        return (f"Echo #{number} occurs in more than one recording. "
                "Please specify which recording or time window you mean.")
    if not matches:
        return (f"I couldn't find retained details for echo #{number} in the latest detection. "
                "The detector or stored detail limit may have omitted it.")
    slot, row = matches[0]
    when = row.get("estimated_echo_utc") or "time unavailable"
    confidence = row.get("confidence")
    confidence_text = f"{confidence:.2f}" if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else "unavailable"
    station = data.get("receiver_name") or data.get("receiver_id") or "the selected station"
    period = _format_recording_period(slot.get("recording_start_utc") or data.get("recording_start_utc"),
                                      slot.get("recording_end_utc") or data.get("recording_end_utc"))
    text = (f"For **echo #{number}** at **{station}** ({period}), the estimated time is "
            f"**{when}** and confidence is **{confidence_text}**. "
            "The time and detection are automated estimates.")
    if data.get("coverage_complete") is False:
        text += " Coverage of the requested interval was incomplete."
    provenance = data.get("benchmark_provenance", {})
    if isinstance(provenance, dict) and provenance.get("mode") == "snapshot_replay":
        text = "Benchmark replay: this result was captured earlier; no new analysis occurred.\n\n" + text
    return text


def _simple_echo_details(prompt, messages):
    """Answer narrow list or indexed requests from the most recent detection."""
    indexed = _indexed_echo_details(prompt, messages)
    if indexed is not None:
        return indexed
    words = re.findall(r"[a-z]+", prompt.casefold())
    allowed = {"please", "can", "could", "you", "i", "me", "list", "show", "give", "provide",
               "what", "are", "were", "the", "a", "all", "each", "every", "of", "for", "this",
               "that", "these", "those", "their", "with", "and", "detected", "detection", "meteor",
               "meteors", "echo", "echoes", "time", "times", "timestamp", "timestamps", "confidence",
               "confidences", "values", "value", "details", "detail"}
    if (not words or not set(words) <= allowed or re.search(r"\d", prompt)
            or not set(words) & {"list", "show", "give", "provide", "what"}
            or not set(words) & {"list", "each", "every", "all"}
            or not set(words) & {"meteor", "meteors", "echo", "echoes"}
            or not set(words) & {"time", "times", "timestamp", "timestamps", "confidence", "confidences", "details", "detail"}):
        return None
    data = _latest_detection_for_details(messages)
    if data is None:
        return None
    count = data.get("total_meteor_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        return None
    station = data.get("receiver_name") or data.get("receiver_id") or "the selected station"
    period = _format_recording_period(data.get("recording_start_utc"), data.get("recording_end_utc"))
    lines = [f"Estimated echo details for **{station}**, **{period}** ({count} candidate echo(es) in the analyzed recordings):"]
    provenance = data.get("benchmark_provenance", {})
    if isinstance(provenance, dict) and provenance.get("mode") == "snapshot_replay":
        lines.insert(0, "Benchmark replay: these results were captured earlier; no new analysis occurred in this turn.")
    retained = 0
    omitted = False
    for slot in data["slots"]:
        if not isinstance(slot, dict):
            continue
        rows = slot.get("detections", [])
        if not isinstance(rows, list):
            rows = []
        rows = [row for row in rows if isinstance(row, dict)]
        retained += len(rows)
        expected = slot.get("meteor_count")
        omitted = omitted or bool(slot.get("detections_truncated")) or (isinstance(expected, int) and len(rows) < expected)
        for row in rows:
            when = row.get("estimated_echo_utc") or "time unavailable"
            conf = row.get("confidence")
            confidence = f"{conf:.2f}" if isinstance(conf, (int, float)) else "unavailable"
            lines.append(f"- **{when}** — confidence {confidence}")
    if not retained:
        lines.append("No individual echo details were retained for these analyzed recordings.")
    if omitted:
        lines.append(f"Only {retained} of {count} individual details were retained; the list is incomplete.")
    if data.get("detection_limit_reached"):
        lines.append("The detector reached its output limit, so the count may be incomplete.")
    if data.get("coverage_complete") is False or data.get("available_slots_in_window") is None:
        lines.append("Coverage of the requested interval is incomplete or unknown; these results cover only the analyzed recordings.")
    lines.append("Times are estimates from automated detection; false detections and missed echoes are possible.")
    return "\n\n".join(lines[:1] + ["\n".join(lines[1:])])


def _detailed_wav_request(prompt: str) -> bool:
    """Return True when the user asks for technical WAV/BRA1 header metadata rather than simple playback."""
    return bool(re.search(
        r"\b(header|bra1|lo\b|local oscillator|beacon|nominal|offset|pps|gps|"
        r"antenna|coord\w*|lat\w*|lon\w*|alt\w*|sha-?256|checksum|hash|"
        r"sample\s*rate|samplerate|sample\s*count|start_us|state|config\w*|"
        r"metadata|explain|why|compar(?:e|ison)|psd|noise|calibrat\w*)\b",
        prompt, re.IGNORECASE,
    ))


def _simple_wav_request(prompt: str) -> bool:
    """Accept only a single, plainly requested WAV playback or download."""
    return bool(
        re.search(r"\b(?:play|listen|hear|download)\b", prompt, re.I)
        and re.search(r"\b(?:wav|audio|recording)\b", prompt, re.I)
        and not re.search(
            r"\b(?:and|also|plus|all|every|multiple|several|entire|without|not|"
            r"how|why|explain|compare|meteor|echo|detect\w*|analy[sz]\w*|spectrogram|frequency)\b",
            prompt, re.I,
        )
        and not _detailed_wav_request(prompt)
    )


def _direct_media(node, messages, prompt, scope_prompt=None):
    if not isinstance(node, ModelRequestNode):
        return None
    results = [part for part in node.request.parts if isinstance(part, ToolReturnPart)]
    if len(results) != 1 or results[0].tool_name not in {
        "detect_meteors", "get_spectrogram", "query_geographic_receivers", "get_wav"
    }:
        return None
    result = results[0]
    tool_name = result.tool_name
    # “Available or not” is one yes/no question, not a second operation.
    compound_prompt = re.sub(r"\bor\s+not\b", "", prompt, flags=re.I)
    if tool_name == "detect_meteors":
        # An annotated image is already part of a simple detection result.
        # Only remove a final, standalone image request; other extra tasks
        # continue through the normal model path.
        compound_prompt = re.sub(
            r"\b(?:and|also|plus)\s+(?:show|display)\s+(?:me\s+)?(?:the\s+|an?\s+)?"
            r"(?:annotated\s+)?(?:image|spectrogram)\s*[.!?]*\s*$", "", compound_prompt, flags=re.I)
    # Only a final, explicit request for the annotated image above is known to
    # be fulfilled by the detection result itself. Treat all other conjunctions
    # and extra questions conservatively as potentially compound requests.
    if re.search(r"\b(?:and|also|plus|then|or)\b", compound_prompt, re.I) or re.search(
        r"\?\s*\S|[.!]\s+(?:where|what|which|how|when|who|why)\b", compound_prompt, re.I
    ):
        return None
    if tool_name == "get_wav" and not _simple_wav_request(prompt):
        return None
    if tool_name != "get_wav" and _detailed_detection_request(compound_prompt):
        return None
    if tool_name == "query_geographic_receivers" and re.search(
        r"\b(which|names?|list|compare|explain|why|details?|each|all of them)\b|\band\s+(?:what|how|show|find|provide|give|check)\b", prompt, re.IGNORECASE
    ):
        return None
    if tool_name == "get_spectrogram" and re.search(r"\b(meteor|echo|detect\w*|analy[sz]\w*)\b", prompt, re.IGNORECASE):
        return None
    if result.outcome != "success" or not isinstance(result.content, dict):
        return None
    if not _matches_explicit_scope(scope_prompt or prompt, result.content):
        return None
    prospective = tool_trace([*messages, node.request])
    if requires_tool_or_history(prompt, prospective):
        return None
    if not _shortcut_time_verified(prompt, prospective, result.content):
        return None
    if result.content.get("status") != "ok":
        return None
    if tool_name == "query_geographic_receivers":
        action = result.content.get("action")
        if action == "count" and not isinstance(result.content.get("site_count"), int):
            return None
        if action == "active_count" and (result.content.get("recording_availability_checked") is not True
                                           or not isinstance(result.content.get("active_site_count"), int)):
            return None
        if action not in {"count", "active_count", "spectrogram"}:
            return None
    if tool_name == "detect_meteors":
        if result.content.get("analysis_performed") is not True or not detection_images(
            [{"event": "tool_result", "name": tool_name, "result": result.content}]
        ):
            return None
        # Allow a partial answer when slots are absent or a specific slot failed,
        # but do not shortcut while available slots remain unattempted.
        remaining = result.content.get("uninspected_slots_count")
        failures = result.content.get("slot_errors")
        failure_count = len(failures) if isinstance(failures, list) else 0
        if isinstance(remaining, int) and not isinstance(remaining, bool) and remaining > failure_count:
            return None
        if remaining is None and result.content.get("coverage_complete") is False:
            return None
    elif tool_name == "get_wav":
        if result.content.get("downloaded") is not True or not wav_attachments(
            [{"event": "tool_result", "name": tool_name, "result": result.content}]
        ):
            return None
        # A single WAV does not establish that a longer requested window was
        # fully retrieved. Leave longer-window explanation to the agent.
        try:
            from datetime import datetime
            start = datetime.fromisoformat(result.content["start_utc"].replace("Z", "+00:00"))
            end = datetime.fromisoformat(result.content["end_utc"].replace("Z", "+00:00"))
            if not start.tzinfo or not end.tzinfo or not 0 < (end - start).total_seconds() <= 300:
                return None
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
    elif tool_name == "get_spectrogram" or result.content.get("action") == "spectrogram":
        if result.content.get("downloaded") is not True or not spectrogram_path(result.content.get("artifact_id")):
            return None
    # Earlier unrelated tool work may need an integrated answer from the model.
    safe_prior_tools = {"get_current_utc_time", "inspect_time_window", "convert_local_time",
                        "check_availability", "list_receivers", "get_spectrogram", "detect_meteors",
                        "query_geographic_receivers", "get_wav"}
    if tool_name == "get_spectrogram":
        safe_prior_tools.remove("detect_meteors")
    if any(isinstance(part, ToolReturnPart) and part.tool_name not in safe_prior_tools
           for message in messages for part in message.parts):
        return None
    calls = [part for message in messages if isinstance(message, ModelResponse)
             for part in message.parts if isinstance(part, ToolCallPart)
             and part.tool_call_id == result.tool_call_id and part.tool_name == tool_name]
    if len(calls) != 1:
        return None
    try:
        answer_directly = (True if tool_name == "get_wav" else
                           calls[0].args_as_dict().get("direct_answer", tool_name == "query_geographic_receivers"))
        return (tool_name, result.content) if answer_directly is True else None
    except (ValueError, TypeError):
        return None


def _shortcut_time_verified(prompt, trace, result):
    """Fast answers require matching clock and timezone evidence when applicable."""
    from datetime import datetime, timedelta, timezone
    from .tools import _resolve_timezone

    relative = re.search(r"\b(yesterday|today|tomorrow|last night|this morning|this hour|last hour|\d+ hours? ago)\b", prompt, re.I)
    # Explicit time expressions, including short 12-hour ranges and ISO
    # offsets, must match the tool's UTC interval. Do not infer clock times.
    if not _matches_explicit_scope(prompt, result):
        return False
    # Fixed numeric timezone expressions outside ISO timestamps are not
    # supported by convert_local_time; avoid guessing their UTC conversion.
    if re.search(r"\b(?:UTC|GMT)\s*[+-]\s*\d", prompt, re.I):
        return False
    zone_match = re.search(
        r"\b([A-Za-z_]+(?:/[A-Za-z_+\-]+){1,2}|CET|CEST|Brussels(?: time)?|Luxembourg time)\b",
        prompt, re.I,
    )
    zone_name = zone_match.group(1) if zone_match else None
    if zone_name and zone_name.casefold() == "luxembourg time":
        zone_name = "Europe/Luxembourg"
    clock = [e.get("result", {}) for e in trace if e.get("event") == "tool_result" and e.get("name") == "get_current_utc_time"]
    conversions = [e.get("result", {}) for e in trace if e.get("event") == "tool_result" and e.get("name") == "convert_local_time"]
    if relative and not clock:
        return False
    if zone_name and not conversions:
        return False
    try:
        start_raw, end_raw = result.get("start_utc"), result.get("end_utc")
        has_explicit_clock = re.search(
            r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(?:am|pm)\b|"
            r"\b\d{1,2}\s*(?:-|–|—|to)\s*\d{1,2}\s*(?:am|pm)\b", prompt, re.I)
        if (relative or zone_name or has_explicit_clock) and (not start_raw or not end_raw):
            return False
        start = datetime.fromisoformat(start_raw.replace("Z", "+00:00")).astimezone(timezone.utc) if start_raw else None
        end = datetime.fromisoformat(end_raw.replace("Z", "+00:00")).astimezone(timezone.utc) if end_raw else None
        if zone_name:
            zone, resolved = _resolve_timezone(zone_name)
            bounds = []
            for item in conversions:
                if item.get("timezone", "").casefold() != resolved.casefold():
                    continue
                bounds.extend(item.get(k) for k in ("utc_time", "start_utc", "end_utc") if item.get(k))
            normalized = {datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc) for value in bounds}
            if start not in normalized or end not in normalized:
                return False
        else:
            zone = timezone.utc
        if relative:
            word = relative.group(1).casefold()
            if word not in {"yesterday", "today", "tomorrow"}:
                return False
            now = datetime.fromisoformat(clock[-1]["utc_time"].replace("Z", "+00:00")).astimezone(zone)
            expected = now.date() + timedelta(days={"yesterday": -1, "today": 0, "tomorrow": 1}[word])
            if start.astimezone(zone).date() != expected:
                return False
    except (KeyError, ValueError, TypeError, AttributeError, ModelRetry):
        return False
    return True


def _expand_explicit_detection_window(node, prompt, trace):
    """Complete a clearly requested bounded window before running the tool.

    The model may select max_slots=1 for an hour. Expand only when the user
    explicitly supplied a verifiable time range and did not ask for one file.
    The returned event preserves what the model asked for and what will run.
    """
    from datetime import datetime, timedelta, timezone

    if not isinstance(node, CallToolsNode) or not re.search(r"\b(?:meteors?|echo(?:es)?)\b", prompt, re.I):
        return None
    # Do not widen an explicitly bounded, negated or conditional request.
    # A single image requested as the output is different from asking to scan
    # only a single recording; the latter must stay within the requested cap.
    if re.search(r"\b(?:only|just|at\s+most|no\s+more\s+than|maximum|max|limit|"
                 r"stop\s+after|do\s+not|don['’]?t|never|without|skip|except|unless|but)\b", prompt, re.I):
        return None
    if re.search(r"\b(?:first\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|"
                 r"eleven|twelve|single|few|several|some|\d+)\s+(?:available\s+)?(?:5[- ]minute\s+)?"
                 r"(?:recordings?|spectrograms?|slots?|files?)\b", prompt, re.I):
        return None
    if re.search(r"\b(?:this|that|the\s+(?:previous|earlier))\s+(?:image|spectrogram|recording|file)\b|"
                 r"\b(?:image|spectrogram|recording|file)\s+above\b", prompt, re.I):
        return None
    clocks = _requested_clock_range(prompt)
    iso_bounds = re.findall(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
                            r"(?:Z|[+-]\d{2}:?\d{2})", prompt, re.I)
    if not clocks and len(iso_bounds) < 2:
        return None
    calls = [part for part in node.model_response.parts if isinstance(part, ToolCallPart)]
    if len(calls) != 1 or calls[0].tool_name != "detect_meteors":
        return None
    call = calls[0]
    try:
        args = call.args_as_dict()
        if args.get("artifact_id"):
            return None
        start = datetime.fromisoformat(str(args["start_utc"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(args["end_utc"]).replace("Z", "+00:00"))
        if not start.tzinfo or not end.tzinfo:
            return None
        start, end = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
        if not timedelta(0) < end - start <= timedelta(minutes=90):
            return None
        # Include a pending status so _receiver_identity_ids checks args["search"]
        # for explicit station codes (e.g. BEHUMA_SYS001) before tool execution.
        requested = {"status": "pending", "query": args.get("search", ""), "start_utc": start.isoformat(), "end_utc": end.isoformat()}
        if not _matches_explicit_scope(prompt, requested) or not _shortcut_time_verified(prompt, trace, requested):
            return None
        cursor = start.replace(minute=(start.minute // 5) * 5, second=0, microsecond=0)
        expected = 0
        while cursor < end and expected <= 12:
            expected += 1
            cursor += timedelta(minutes=5)
        previous = args.get("max_slots", 1)
        if isinstance(previous, bool) or not isinstance(previous, int) or not 1 <= previous < expected <= 12:
            return None
        call.args = {**args, "max_slots": expected}
        return {"event": "orchestration_adjustment", "name": "detect_meteors", "call_id": call.tool_call_id,
                "model_max_slots": previous, "executed_max_slots": expected,
                "reason": "Expanded the verified requested window to cover its five-minute slots."}
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _insert_orchestration_adjustments(trace, adjustments):
    """Record any actual argument expansion alongside its tool call."""
    result = []
    inserted = {event.get("call_id") for event in trace if event.get("event") == "orchestration_adjustment"}
    for event in trace:
        result.append(event)
        call_id = event.get("call_id")
        if event.get("event") == "tool_call" and call_id not in inserted:
            result.extend(item for item in adjustments if item.get("call_id") == call_id)
            inserted.add(call_id)
    return result


def _followup_scope_prompt(prompt, history):
    """Retain the user's last explicit time scope for a short media follow-up."""
    short_media = (r"\s*(?:(?:please\s+)?(?:show|display|scan|detect|analy[sz]e|check|run|do|give)"
                   r"(?:\s+(?:me|the|this|that|a|an|it|please|spectrogram|image|recording|"
                   r"meteor|meteors|echo|echoes|detection|wav|audio|on|for|of))*|"
                   r"(?:meteor\s+)?echo(?:es)?(?:\s+please)?)[.!?]*\s*")
    if not re.fullmatch(short_media, prompt, re.I):
        return prompt
    if re.search(r"\b(?:yesterday|today|tomorrow|instead|different|from|between|at)\b|\d{1,2}:\d{2}|\d{4}-\d{2}-\d{2}", prompt, re.I):
        return prompt
    for message in reversed(history):
        for part in reversed(getattr(message, "parts", ())):
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                prior = part.content
                if _requested_clock_range(prior) not in (None, ()):
                    return prior
                if not re.fullmatch(short_media, prior, re.I):
                    return prompt
    return prompt


async def ask(
    prompt: str,
    session: Session,
    settings: Settings,
    enable_tools: bool = True,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    on_event=None,
    tool_functions=None,
    instructions_suffix="",
    cancel_event: threading.Event | None = None,
) -> Reply:
    if not prompt.strip():
        raise ValueError("Enter a message first.")
    check_cancelled(cancel_event)
    started = perf_counter()
    limited = deterministic_limit_response(
        prompt,
        has_history=bool(session.messages),
        assistant_scope=settings.assistant_scope,
        enable_tools=enable_tools,
    )
    if limited is not None:
        messages = [*session.messages, ModelRequest(parts=[UserPromptPart(prompt.strip())]),
                    ModelResponse(parts=[TextPart(limited)], finish_reason="stop",
                                  metadata={"direct_limit_response": True})]
        return Reply(text=limited, session=Session(messages, session.successful_turns + 1),
                     trace=[{"event": "direct_limit_response"}], elapsed_seconds=perf_counter() - started,
                     requests=0, input_tokens=0, output_tokens=0)
    scope_prompt = _followup_scope_prompt(prompt, session.messages)
    detail_text = _simple_echo_details(prompt, session.messages) if enable_tools else None
    if detail_text is not None:
        messages = [*session.messages, ModelRequest(parts=[UserPromptPart(prompt.strip())]),
                    ModelResponse(parts=[TextPart(detail_text)], finish_reason="stop",
                                  metadata={"direct_detection_details": True})]
        return Reply(text=detail_text, session=Session(messages, session.successful_turns + 1),
                     trace=[{"event": "history_evidence", "name": "detect_meteors"}],
                     elapsed_seconds=perf_counter() - started, requests=0, input_tokens=0, output_tokens=0)
    usage = RunUsage()
    warning = None
    result = None
    model_text = None
    context_size = await _resolve_context_size(settings, transport=transport)
    needs_detection_detail = bool(re.search(r"\b(?:meteor|echo|echoes|detection)\b", prompt, re.I)
                                  and _detailed_detection_request(prompt))
    compacted_session = _compact_detection_history(deepcopy(session.messages), detailed=needs_detection_detail)
    history = _bounded_model_history(compacted_session)
    history_omitted = len(history) < len(session.messages)
    captured = []
    retry_trace = []
    orchestration_adjustments = []
    full_tool_results = {}
    active_prompt = prompt.strip()
    if history_omitted:
        active_prompt += _HISTORY_OMITTED_NOTE
    if context_size is not None:
        estimated_prompt = _estimate_request_tokens(history, active_prompt, instructions_suffix, enable_tools)
        if estimated_prompt + settings.max_tokens + CONTEXT_SAFETY_MARGIN_TOKENS > context_size and history:
            base_without_history = _estimate_request_tokens(
                [], prompt.strip() + _HISTORY_OMITTED_NOTE, instructions_suffix, enable_tools,
            )
            allowed_history_tokens = context_size - settings.max_tokens - CONTEXT_SAFETY_MARGIN_TOKENS - base_without_history
            if allowed_history_tokens >= 256:
                history = _bounded_model_history(compacted_session, max_chars=min(MODEL_HISTORY_MAX_CHARS, int(allowed_history_tokens * 3.5)))
            else:
                history = []
            if len(history) < len(session.messages):
                history_omitted = True
                if _HISTORY_OMITTED_NOTE not in active_prompt:
                    active_prompt = prompt.strip() + _HISTORY_OMITTED_NOTE
            estimated_prompt = _estimate_request_tokens(history, active_prompt, instructions_suffix, enable_tools)
        try:
            _plan_output_budget(estimated_prompt, settings.max_tokens, context_size)
            _plan_output_budget(estimated_prompt, settings.recovery_max_tokens, context_size)
        except ContextLengthExceededError as overflow:
            overflow.tool_trace = []
            overflow.timings = []
            overflow.elapsed_seconds = perf_counter() - started
            raise
    initial_history_length = len(history)
    model_prompt = active_prompt
    retry_instructions = None
    retry_kind: str | None = None
    progress = RunProgress(on_event, settings.api_key)
    with cancellation_scope(cancel_event):
        async with open_agent(settings, enable_tools, transport=transport, progress=progress,
                              tool_functions=tool_functions, instructions_suffix=instructions_suffix) as agent:
            for attempt in range(2):
                check_cancelled(cancel_event)
                error = None
                current_est = _estimate_request_tokens(history, active_prompt or "", instructions_suffix, enable_tools)
                target_max = settings.recovery_max_tokens if (attempt > 0 and retry_kind == "generation_retry") else settings.max_tokens
                try:
                    attempt_max_tokens = _plan_output_budget(current_est, target_max, context_size)
                except ContextLengthExceededError as overflow:
                    overflow.tool_trace = retry_trace
                    overflow.timings = progress.snapshot()
                    overflow.elapsed_seconds = perf_counter() - started
                    raise
                with capture_run_messages() as captured:
                    try:
                        direct_result = None
                        direct_messages = None
                        async with agent.iter(
                            active_prompt, message_history=history,
                            instructions=retry_instructions,
                            usage_limits=UsageLimits(request_limit=settings.request_limit,
                                                     tool_calls_limit=settings.tool_calls_limit),
                            usage=usage,
                            timeout=max(0.1, settings.timeout_seconds - (perf_counter() - started)),
                            model_settings={"max_tokens": attempt_max_tokens},
                        ) as run:
                            async for node in run:
                                check_cancelled(cancel_event)
                                adjustment = _expand_explicit_detection_window(
                                    node, prompt, tool_trace(run.all_messages()[len(history):]))
                                if adjustment is not None:
                                    orchestration_adjustments.append(adjustment)
                                direct_result = _direct_media(node, run.all_messages()[len(history):], prompt, scope_prompt)
                                if direct_result is not None:
                                    # The next tool-return request has not yet been appended to history.
                                    direct_messages = [*run.all_messages(), node.request]
                                    break
                                _limit_current_tool_results(node, full_tool_results)
                            check_cancelled(cancel_event)
                            result = run.result

                    except Exception as failure:
                        error = failure
                check_cancelled(cancel_event)
                # The model only saw a bounded sample. Restore original tool
                # evidence for tracing, grounding, recovery and future follow-ups.
                _restore_tool_results(captured, full_tool_results)
                if direct_messages is not None:
                    _restore_tool_results(direct_messages, full_tool_results)
                if result is not None:
                    _restore_tool_results(result.all_messages(), full_tool_results)
                if error is None and direct_result is not None and direct_messages is not None:
                    direct_tool, direct_data = direct_result
                    trace = _insert_orchestration_adjustments(
                        retry_trace + tool_trace(direct_messages[len(history):]), orchestration_adjustments)
                    time_events = [event for event in trace if event.get("event") == "tool_result"
                                   and event.get("name") in {"convert_local_time", "get_current_utc_time"}]
                    renderers = {"detect_meteors": meteor_summary, "get_spectrogram": spectrogram_summary,
                                 "query_geographic_receivers": geographic_summary, "get_wav": wav_summary}
                    text = renderers[direct_tool](direct_data, time_events=time_events)
                    provenance = direct_data.get("benchmark_provenance", {})
                    if isinstance(provenance, dict) and provenance.get("mode") == "snapshot_replay":
                        text = ("Benchmark replay: this result was captured earlier; no live archive "
                                "query or new analysis occurred in this turn.\n\n" + text)
                    metadata = {"direct_detection_summary": True} if direct_tool == "detect_meteors" else {"direct_" + direct_tool + "_summary": True}
                    all_messages = [*direct_messages, ModelResponse(parts=[TextPart(text)], finish_reason="stop",
                                                                   metadata=metadata)]
                    break
                new_messages = captured[len(history):]
                trace = _insert_orchestration_adjustments(
                    retry_trace + tool_trace(new_messages), orchestration_adjustments)
                responses = [message for message in new_messages if isinstance(message, ModelResponse)]
                truncated = bool(responses and responses[-1].finish_reason == "length")
                if error is None and not truncated and result and result.output.strip():
                    text = result.output.strip()
                    missing_evidence = (enable_tools and requires_tool_or_history(prompt, trace, session.messages)
                                        and not safe_unfulfilled_response(prompt, text))
                    if (missing_evidence and attempt == 0
                            and perf_counter() - started < settings.timeout_seconds):
                        retry_trace = trace + [{"event": "grounding_retry",
                                                "reason": "factual BRAMS query answered without calling any tool"}]
                        # Preserve completed tool calls for the retry without keeping
                        # the unverified first answer as conversational fact.
                        history = _compact_detection_history(deepcopy(result.all_messages()), detailed=needs_detection_detail)
                        for message in reversed(history):
                            if isinstance(message, ModelResponse) and not any(isinstance(p, ToolCallPart) for p in message.parts):
                                message.parts = [p for p in message.parts if not isinstance(p, TextPart)] + [
                                    TextPart("The previous answer was not fully verified; further checking is required.")]
                                break
                        # The original user message is already in history. A second
                        # copy would survive deletion of the visible user bubble.
                        active_prompt = None
                        retry_kind = "grounding_retry"
                        retry_instructions = (
                            "Call the appropriate BRAMS or time tool first; do not answer factual "
                            "station, availability, or archive questions without tool results."
                        )
                        continue
                    model_text = text  # Kept separately from Python-added evidence for benchmarking.
                    details = [event for event in _scope_verified_trace(prompt, trace) if event.get("event") == "tool_result" and (
                        event.get("name") in {"get_beacon", "get_recordings", "get_psd", "convert_local_time"}
                        or (event.get("name") == "list_receivers" and event.get("result", {}).get("detail") == "coordinates")
                        or (event.get("name") == "get_wav" and event.get("result", {}).get("header_summary")
                            and _detailed_wav_request(prompt)))]
                    verified = evidence_summary(details)
                    if verified:
                        text += "\n\n---\n**Verified tool results**\n\n" + verified
                    text, grounding_issues = apply_grounding_guard(
                        prompt, text, trace, session.messages, already_verified_events=details,
                    )
                    if grounding_issues:
                        trace.append({"event": "grounding_guard", "issues": grounding_issues})
                        if warning is None:
                            warning = "Unverified or conflicting claims in the model output were corrected using tool results."
                    if missing_evidence:
                        text = "I couldn’t verify all requested BRAMS information. Please try again or narrow the request."
                        summary = evidence_summary(_scope_verified_trace(prompt, trace))
                        if summary:
                            text += "\n\n" + summary
                        warning = "The requested evidence was incomplete."
                        trace.append({"event": "grounding_guard", "issues": [{"code": "missing_relevant_evidence"}]})
                    coverage_notice = _detection_coverage_notice(_scope_verified_trace(prompt, trace))
                    if coverage_notice:
                        text += "\n\n" + coverage_notice
                    all_messages = deepcopy(result.all_messages())
                    if grounding_issues or missing_evidence or coverage_notice:
                        # Preserve tool-call protocol and raw model_text, but future turns
                        # should see the corrected answer shown to the user.
                        for message in reversed(all_messages):
                            if isinstance(message, ModelResponse) and not any(isinstance(p, ToolCallPart) for p in message.parts):
                                message.parts = [p for p in message.parts if not isinstance(p, TextPart)] + [TextPart(text)]
                                break
                    if attempt and warning is None:
                        warning = (
                            "Tool execution was triggered after an ungrounded initial response."
                            if any(e.get("event") == "grounding_retry" for e in trace)
                            else "Earlier conversation was omitted to fit the model context window."
                            if any(e.get("event") == "context_retry" for e in trace)
                            else "A longer response budget was needed; recovery succeeded."
                        )
                    break
                if error is None:
                    error = IncompleteResponseError("The model did not finish its response.")
                context_overflow = _is_context_overflow_error(error)
                summary = evidence_summary(_scope_verified_trace(scope_prompt, trace), meteor_detail=_detailed_detection_request(prompt))
                recoverable = isinstance(error, (RepeatedToolCall, UsageLimitExceeded, IncompleteResponseError,
                                                 TimeoutError, APITimeoutError, ModelHTTPError,
                                                 UnexpectedModelBehavior, APIConnectionError, httpx.TimeoutException))
                if summary and recoverable:
                    reason = ("a repeated tool call" if isinstance(error, RepeatedToolCall)
                              else "the request limit" if isinstance(error, UsageLimitExceeded)
                              else "reaching the model's context limit" if context_overflow
                              else "the response token limit" if truncated else "an incomplete model response")
                    warning = f"Model stopped after {reason}; showing completed tool results."
                    incomplete = requires_tool_or_history(prompt, trace, session.messages)
                    text = summary + ("\n\nThe requested information could not be verified; "
                                      "please try again." if incomplete else
                                      "\n\nOnly the completed operations above are confirmed.")
                    all_messages = complete_history(captured)
                    all_messages.append(ModelResponse(parts=[TextPart(text)], finish_reason="stop",
                                                      metadata={"tool_result_fallback": True}))
                    trace.append({"event": "recovery", "reason": reason, "message": warning})
                    break
                if (context_overflow and attempt == 0 and history
                        and not any(event["event"] == "tool_result" for event in trace)
                        and perf_counter() - started < settings.timeout_seconds):
                    history = []
                    initial_history_length = 0
                    history_omitted = True
                    active_prompt = prompt.strip() + _HISTORY_OMITTED_NOTE
                    model_prompt = active_prompt
                    retry_kind = "context_retry"
                    retry_trace = trace + [{"event": "context_retry",
                                            "reason": "request exceeded model context size; retrying without earlier history"}]
                    continue
                # A single larger-budget retry is safe only before any tool ran.
                # Never repeat side effects or downloads to regenerate an explanation.
                if (truncated and not any(event["event"] == "tool_result" for event in trace)
                        and attempt == 0 and settings.recovery_max_tokens > settings.max_tokens
                        and perf_counter() - started < settings.timeout_seconds):
                    try:
                        planned_recovery = _plan_output_budget(current_est, settings.recovery_max_tokens, context_size)
                    except ContextLengthExceededError:
                        planned_recovery = attempt_max_tokens
                    if planned_recovery > attempt_max_tokens:
                        retry_kind = "generation_retry"
                        retry_trace = trace + [{"event": "generation_retry", "reason": "truncated before any tool completed",
                                                "max_tokens": planned_recovery}]
                        continue
                if context_overflow:
                    error = ContextLengthExceededError(
                        estimated_prompt_tokens=current_est,
                        context_size=context_size,
                        requested_output_tokens=attempt_max_tokens,
                    )
                error.tool_trace = trace
                error.timings = progress.snapshot()
                error.elapsed_seconds = perf_counter()-started
                raise error
    # A follow-up may omit the time; never show media for a conflicting earlier
    # user-requested interval, even if the model generated a confident answer.
    media_tools = {"get_spectrogram", "detect_meteors", "get_wav"}
    mismatched_media = [event for event in trace if event.get("event") == "tool_result"
                        and event.get("name") in media_tools and isinstance(event.get("result"), dict)
                        and not _matches_explicit_scope(scope_prompt, event["result"])]
    if mismatched_media:
        text = ("I couldn’t verify the requested recording: the retrieved result does not match "
                "the requested station or time. Please try again with the station and UTC interval.")
        warning = "A result did not match the requested station or time."
        trace.append({"event": "grounding_guard", "issues": [{"code": "media_scope_mismatch"}]})
        all_messages = deepcopy(all_messages)
        mismatched_ids = {event.get("call_id") for event in mismatched_media}
        for message in all_messages:
            for part in getattr(message, "parts", ()):
                if isinstance(part, ToolReturnPart) and part.tool_call_id in mismatched_ids:
                    part.content = {"status": "error", "message": "Result rejected: station or time did not match the user's request."}
        for message in reversed(all_messages):
            if isinstance(message, ModelResponse) and not any(isinstance(p, ToolCallPart) for p in message.parts):
                message.parts = [p for p in message.parts if not isinstance(p, TextPart)] + [TextPart(text)]
                break
    # The request may use compact history; retain original detailed tool results
    # in the session so a later explicit request can still access them.
    if history_omitted:
        for message in all_messages[initial_history_length:]:
            for part in getattr(message, "parts", ()):
                if isinstance(part, UserPromptPart) and part.content == model_prompt:
                    part.content = prompt.strip()
    all_messages = [*session.messages, *all_messages[initial_history_length:]]
    return Reply(
        text=text,
        session=Session(all_messages, session.successful_turns + 1),
        trace=trace,
        elapsed_seconds=perf_counter() - started,
        requests=usage.requests,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        images=spectrogram_images(_scope_verified_trace(scope_prompt, trace)),
        detections=detection_images(_scope_verified_trace(scope_prompt, trace)),
        audio=wav_attachments(_scope_verified_trace(scope_prompt, trace)),
        warning=warning,
        timings=progress.snapshot(),
        model_text=model_text,
    )


def friendly_error(error: BaseException, settings: Settings) -> str:
    if isinstance(error, (OperationCancelled, asyncio.CancelledError)):
        return "Request cancelled; earlier conversation preserved."
    if isinstance(error, ContextLengthExceededError):
        if error.context_size is not None:
            estimate = f" (≈{error.total_estimated_tokens:,} / {error.context_size:,} tokens, estimated)"
        else:
            estimate = f" (≈{error.total_estimated_tokens:,} tokens, estimated)"
        return (
            f"This request exceeded the active model's context window{estimate}. "
            "Your earlier conversation is preserved. Start a New chat, shorten the prompt or narrow the "
            "time/station window, or switch to a larger-context model in Model settings."
        )
    if _is_context_overflow_error(error):
        return (
            "This request exceeded the active model's context window. Your earlier conversation is preserved. "
            "Start a New chat, shorten the prompt or narrow the time/station window, or switch to a larger-context "
            "model in Model settings."
        )
    if isinstance(error, (TimeoutError, APITimeoutError, httpx.TimeoutException)):
        if settings.provider != "local":
            return "The API request timed out. Check the endpoint and provider status, then retry."
        return (
            "The model request timed out. Model autoload can take time; check the llama.cpp "
            "terminal or increase LLAMA_TIMEOUT_SECONDS in .env."
        )
    if isinstance(error, (APIConnectionError, httpx.ConnectError)):
        if settings.provider != "local":
            return f"Cannot connect to {settings.base_url}. Check the API URL and network connection."
        return (
            f"Cannot connect to {settings.base_url}. Start llama.cpp and run Gradio on the "
            "same computer, or update LLAMA_BASE_URL in .env."
        )
    if isinstance(error, ModelHTTPError):
        if error.status_code == 401:
            return "Authentication failed (HTTP 401). Check the API key in Connection and apply it again."
        if error.status_code == 403:
            return "Access denied (HTTP 403). Check the key's permissions and access to the selected model."
        if error.status_code == 429:
            return "API quota or rate limit reached (HTTP 429). Check provider usage/billing or retry later."
        if settings.provider != "local":
            return (f"The API returned HTTP {error.status_code}. Check the exact model ID, Chat Completions/tool support, "
                    "and token-limit parameter under Connection. Some providers do not offer /v1/models.")
        return (
            f"The model server returned HTTP {error.status_code}. Check the llama.cpp logs "
            "and confirm LLAMA_MODEL matches an ID from /v1/models."
        )
    if isinstance(error, UsageLimitExceeded):
        return "The model used its request/tool budget before completing the operation. Your earlier conversation is preserved; try a more specific receiver or interval. See Tool activity for this attempt."
    if isinstance(error, IncompleteResponseError):
        return "The model could not finish within its response budget. No completed tool result was available to show. Your earlier conversation is preserved; try a shorter request."
    # Do not expose API keys or a Python stack trace in the browser.
    detail = str(error).replace(settings.api_key, "[redacted]")
    return f"{type(error).__name__}: {detail[:700]}"
