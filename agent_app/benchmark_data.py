"""Frozen real-tool evidence and transparent, deliberately limited benchmark checks."""

from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
import inspect
import json
import re

from pydantic import TypeAdapter

from .harness import registered_tools
from .tool_validation import validate_tool_arguments
from .benchmark_values import at_path, fact_value, value_mentioned
from .benchmark_audit import scientific_flags
from .brams_sdk import normalize_search


TOOLS = {function.__name__: function for function in registered_tools()}
SCORER_VERSION = 7


def bound_arguments(name, arguments):
    signature = inspect.signature(TOOLS[name])
    bound = signature.bind(**arguments)
    bound.apply_defaults()
    return {key: TypeAdapter(signature.parameters[key].annotation).validate_python(value)
            for key, value in bound.arguments.items()}


def canonical(name, arguments):
    values = bound_arguments(name, arguments)
    values.pop("direct_answer", None)
    for optional, default in (("end_local_datetime", None), ("end_fold", None), ("timezone_hint", "UTC")):
        if optional in values and (optional not in arguments or values[optional] == default):
            values.pop(optional, None)
    for key, value in values.items():
        if isinstance(value, datetime):
            values[key] = value.astimezone(timezone.utc).isoformat()
        elif name == "convert_local_time" and key in {"local_datetime", "end_local_datetime"} and isinstance(value, str):
            # Equivalent ISO local times must match regardless of whether the
            # model included seconds. Keep offsets rather than guessing a zone.
            try:
                values[key] = datetime.fromisoformat(value).isoformat()
            except ValueError:
                pass
        elif key in {"search", "receiver_search", "emitter_code"}:
            values[key] = " ".join(value.casefold().split())
            if values[key] == "*":
                values[key] = ""
    return name, json.dumps(values, sort_keys=True, default=str)


def reference_calls(cases):
    """Deduplicate gold queries, then add bounded discovery queries for planning."""
    calls = [call for case in cases for turn in case["turns"]
             for call in [*turn["required_calls"], *turn.get("support_calls", []),
                          *(call for route in turn.get("acceptable_call_sets", []) for call in route)]]
    searches = {call["arguments"].get("search") for call in calls} - {None}
    calls += [{"tool": "list_receivers", "arguments": {"search": search}}
              for search in sorted(searches)]
    calls += [{"tool": "get_current_utc_time", "arguments": {}}]
    unique = {}
    for call in calls:
        unique[canonical(call["tool"], call["arguments"])] = call
    return list(unique.values())


class SnapshotTools:
    """Same public signatures; unmatched queries never escape to the live archive."""

    def __init__(self, snapshot):
        self.captured_at = snapshot.get("captured_at_utc")
        self.entries = snapshot["entries"]
        self.lookup = {canonical(row["tool"], row["arguments"]): row["result"] for row in self.entries}
        self.misses = []
        fields = ("receiver_id", "name", "location_code", "antenna", "catalog_start", "catalog_end")
        self.catalog = None
        self.catalog_time = self.captured_at
        self.catalog_source = next((row["result"].get("source_url") for row in self.entries
                                    if row["tool"] == "list_receivers" and row["result"].get("source_url")), None)
        self.coordinates = {receiver["receiver_id"]: {key: receiver[key] for key in ("coordinates", "coordinates_source_url") if key in receiver} for row in self.entries
                            if row["tool"] == "list_receivers" for receiver in row["result"].get("receivers", [])
                            if "receiver_id" in receiver and "coordinates" in receiver}
        for row in self.entries:
            result = row["result"]
            receivers = result.get("receivers", [])
            if (row["tool"] in {"list_receivers", "check_availability"} and not row["arguments"].get("search")
                    and result.get("status") == "ok" and result.get("count") == len(receivers)
                    and receivers and all(set(fields) <= receiver.keys() for receiver in receivers)):
                self.catalog = [{key: receiver[key] for key in fields} for receiver in receivers]
                self.catalog_time = result.get("retrieved_at_utc", self.captured_at)
                break
        self.semantic_lookup = {self.semantic_key(row["tool"], row["arguments"]): row["result"] for row in self.entries}

    def catalog_rows(self, search, primary=False):
        if self.catalog is None:
            return None
        if primary and any(row["receiver_id"].casefold() == search.strip().casefold() + "_sys001" for row in self.catalog):
            search += "_SYS001"
        terms = normalize_search(search).split()
        return sorted([row for row in self.catalog if all(term in normalize_search(
            f"{row['receiver_id']} {row['name']} {row['location_code']}") for term in terms)], key=lambda row: row["receiver_id"])

    def semantic_arguments(self, name, arguments):
        def normalize_receiver(values):
            field = "search" if "search" in values else "receiver_search" if "receiver_search" in values else None
            if field and values[field]:
                receivers = self.catalog_rows(values[field], primary=name in {
                    "get_wav", "get_spectrogram", "get_psd", "get_recordings", "get_beacon", "detect_meteors"})
                if receivers:
                    values[field] = [row["receiver_id"] for row in receivers]
            return values

        values = normalize_receiver(json.loads(canonical(name, arguments)[1]))
        if name == "detect_meteors" and values.get("artifact_id") is None:
            # A larger slot limit is equivalent only when the frozen result
            # proves the entire window contained and inspected one slot.
            for row in self.entries:
                if row["tool"] != name:
                    continue
                captured = normalize_receiver(json.loads(canonical(name, row["arguments"])[1]))
                result = row["result"]
                if (result.get("status") != "ok" or result.get("coverage_complete") is not True
                        or result.get("available_slots_in_window") != 1
                        or result.get("inspected_slots_count") != 1
                        or result.get("expected_slots_in_window") != 1):
                    continue
                if any(captured.get(key) != value for key, value in values.items()
                       if key not in {"max_slots", "confidence_threshold"}):
                    continue
                threshold = values.get("confidence_threshold")
                if threshold is not None and threshold != result.get("confidence_threshold"):
                    continue
                values["max_slots"] = captured["max_slots"]
                break
        if name == "detect_meteors" and values.get("confidence_threshold") is None:
            # An omitted threshold means the configured default at capture time.
            # Equate it only with the actual threshold recorded by a matching
            # default-threshold capture, never a guessed global default.
            candidates = []
            for row in self.entries:
                if row["tool"] != name:
                    continue
                captured = normalize_receiver(json.loads(canonical(name, row["arguments"])[1]))
                if captured.get("confidence_threshold") is not None:
                    continue
                if any(captured.get(key) != value for key, value in values.items()
                       if key != "confidence_threshold"):
                    continue
                threshold = row["result"].get("confidence_threshold")
                if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
                    candidates.append(threshold)
            if candidates and len(set(candidates)) == 1:
                values["confidence_threshold"] = candidates[0]
        return values

    def semantic_key(self, name, arguments):
        return name, json.dumps(self.semantic_arguments(name, arguments), sort_keys=True)

    def replayed(self, result, name):
        result = deepcopy(result)
        if self.captured_at and name not in {"get_current_utc_time", "inspect_time_window", "convert_local_time"}:
            result["benchmark_provenance"] = {"mode": "snapshot_replay", "captured_at_utc": result.get("retrieved_at_utc", self.captured_at),
                                               "live_query_made": False, "analysis_recomputed": False}
            for key in ("download_origin", "input_origin"):
                if key in result:
                    result["captured_" + key] = result[key]
                    result[key] = "Previously captured original artifact/result; no fresh download or analysis in this benchmark turn."
        return result

    def result(self, name, arguments):
        validate_tool_arguments(name, bound_arguments(name, arguments))
        key = canonical(name, arguments)
        if key in self.lookup:
            return self.replayed(self.lookup[key], name)
        equivalent = self.semantic_lookup.get(self.semantic_key(name, arguments))
        if equivalent is not None:
            return self.replayed(equivalent, name)
        if name == "list_receivers" and self.catalog is not None and self.catalog_source:
            values = bound_arguments(name, arguments)
            rows = self.catalog_rows(values["search"])
            if values["detail"] == "summary" or all(row["receiver_id"] in self.coordinates for row in rows):
                if values["detail"] == "coordinates":
                    rows = [{**row, **self.coordinates[row["receiver_id"]]} for row in rows]
                locations = {row["location_code"] for row in rows}
                counts = {code: sum(row["location_code"] == code for row in rows) for code in sorted(locations)}
                result = {"status": "ok", "query": values["search"], "count": len(rows), "location_count": len(locations),
                          "receivers": rows[values["offset"]:values["offset"]+values["limit"]], "detail": values["detail"],
                          "offset": values["offset"], "limit": values["limit"],
                          "has_more": values["offset"]+values["limit"] < len(rows),
                          "multi_receiver_sites": {key: count for key, count in counts.items() if count > 1},
                          "source_url": self.catalog_source, "source": "Python filter of complete captured BRAMS receiver catalog",
                          "retrieved_at_utc": self.catalog_time, "scope": "receiver_catalog", "recording_availability_checked": False,
                          "derived_from_captured_catalog": True}
                return self.replayed(result, name)
        if name in {"inspect_time_window", "convert_local_time"}:
            # Pure calculations can accept auxiliary intervals without archive access.
            return TOOLS[name](**bound_arguments(name, arguments))
        values = json.loads(key[1])
        # The production file tools use SYS001 for an explicit site code.
        if name in {"get_wav", "get_spectrogram", "get_psd", "get_recordings"}:
            search = values["search"]
            if re.fullmatch(r"[a-z]{6}", search):
                values["search"] += "_sys001"
            if name == "get_psd" and values.get("artifact_id"):
                artifact = values["artifact_id"]
                for row in self.entries:
                    if row["tool"] != "get_wav" or row["result"].get("artifact_id") != artifact:
                        continue
                    wav = json.loads(canonical("get_wav", row["arguments"])[1])
                    if all(values[field] == wav[field] for field in ("search", "start_utc", "end_utc")):
                        values["artifact_id"] = None
            exact = self.lookup.get((name, json.dumps(values, sort_keys=True)))
            if exact is not None:
                return self.replayed(exact, name)
        # Smaller pages are an equivalent presentation of the captured page.
        if name in {"list_receivers", "get_recordings"} and 1 <= values["limit"] <= 20 and values["offset"] >= 0:
            for row in self.entries:
                if row["tool"] != name:
                    continue
                saved = json.loads(canonical(name, row["arguments"])[1])
                if any(values[field] != saved[field] for field in values if field not in {"limit", "offset"}):
                    continue
                field = "receivers" if name == "list_receivers" else "recordings"
                result = deepcopy(row["result"])
                first = values["offset"] - saved["offset"]
                last = first + values["limit"]
                rows = result.get(field, [])
                if first >= 0 and (last <= len(rows) or result.get("has_more") is False):
                    result.update({field: rows[first:last], "offset": values["offset"],
                                   "limit": values["limit"],
                                   "has_more": values["offset"] + values["limit"] < result.get("count", 0)})
                    return self.replayed(result, name)
        self.misses.append({"tool": name, "arguments": arguments})
        return {"status": "error", "error_code": "benchmark_unmatched_call",
                "message": "This query is outside the frozen benchmark reference. No live query was made; do not invent the result."}

    def functions(self):
        def wrap(function):
            @wraps(function)
            async def replay(**kwargs):
                return self.result(function.__name__, kwargs)
            return replay
        return [wrap(function) for function in TOOLS.values()]


def call_matches(expected, actual, reference=None):
    if expected["tool"] != actual.get("name"):
        return False
    try:
        wanted = json.loads(canonical(expected["tool"], expected["arguments"])[1])
        got = json.loads(canonical(actual["name"], actual["arguments"])[1])
        if reference is not None:
            wanted = reference.semantic_arguments(expected["tool"], expected["arguments"])
            got = reference.semantic_arguments(actual["name"], actual["arguments"])
        # Ignore pagination unless the test explicitly specifies it.
        fields = set(wanted) - ({"limit", "offset", "direct_answer", "end_local_datetime", "timezone_hint"} - set(expected["arguments"]))
        if expected["tool"] == "get_beacon" and not expected["arguments"].get("receiver_search"):
            fields.discard("receiver_search")
        if expected["tool"] in {"get_wav", "get_spectrogram", "get_psd", "get_recordings"}:
            for values in (wanted, got):
                if isinstance(values["search"], str) and re.fullmatch(r"[a-z]{6}", values["search"]):
                    values["search"] += "_sys001"
        # Valid reuse is checked by the replay result, rather than by opaque ID equality.
        if expected["tool"] == "get_psd" and "artifact_id" not in expected["arguments"]:
            fields.discard("artifact_id")
        return all(got[field] == wanted[field] for field in fields)
    except (ValueError, TypeError, KeyError):
        return False


def evidence_strings(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if (isinstance(child, str) and ("source" in key or key == "filename")
                    and (child.startswith(("https://", "http://")) or key == "filename")):
                yield child
            yield from evidence_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from evidence_strings(child)


def score_turn(turn, reply, previous_trace=(), *, reference=None):
    trace = reply.trace
    calls = [row for row in trace if row["event"] == "tool_call"]
    results = [row for row in trace if row["event"] == "tool_result"]
    all_results = [row for row in [*previous_trace, *trace] if row["event"] == "tool_result"]
    adjustments = {row.get("call_id"): row for row in trace
                   if row.get("event") == "orchestration_adjustment" and row.get("call_id")}
    model_calls = deepcopy(calls)
    for actual in model_calls:
        adjustment = adjustments.get(actual.get("call_id"))
        if adjustment and actual.get("name") == "detect_meteors" and isinstance(actual.get("arguments"), dict):
            actual["arguments"]["max_slots"] = adjustment.get("model_max_slots")

    def completed_call(expected, actual_calls=calls):
        for actual in actual_calls:
            if not call_matches(expected, actual, reference):
                continue
            for returned in results:
                if (returned["name"] == actual["name"] and returned.get("call_id") == actual.get("call_id")
                        and isinstance(returned["result"], dict)
                        and returned["result"].get("status", "ok") in expected.get("expected_statuses", ["ok"])):
                    return True
        return False
    matched = [completed_call(expected) for expected in turn["required_calls"]]
    routes = [turn["required_calls"], *turn.get("acceptable_call_sets", [])]
    route_pass = any(route and all(completed_call(call) for call in route) for route in routes)
    model_route_pass = any(route and all(completed_call(call, model_calls) for call in route) for route in routes)
    errors = [row for row in results if isinstance(row["result"], dict)
              and row["result"].get("status") in {"error", "setup_required"}]
    def expected_error(returned):
        return any(
            returned.get("call_id") == actual.get("call_id")
            and returned.get("name") == actual.get("name")
            and returned["result"].get("status") in expected.get("expected_statuses", [])
            and call_matches(expected, actual, reference)
            for route in routes for expected in route for actual in calls
        )
    unexpected_errors = [row for row in errors if not expected_error(row)]
    text = reply.model_text or ""
    visible_text = reply.text or ""
    # Keep model and displayed-answer grounding checks distinct. This also
    # exposes contradictions caught by the runtime guard without equating a
    # deterministic correction with a correct native model answer.
    from .grounding import check_unbacked_claims
    review_trace = [*previous_trace, *trace]
    model_grounding_flags = check_unbacked_claims(turn.get("prompt", ""), text, review_trace)
    visible_grounding_flags = check_unbacked_claims(turn.get("prompt", ""), visible_text, review_trace)
    sources = set(source for row in all_results for source in evidence_strings(row["result"]))
    facts = []
    for fact in turn["facts"]:
        matching = [row["result"] for row in all_results if row["name"] == fact["tool"]]
        try:
            if reference is not None and "arguments" in fact:
                value = fact_value(reference.result(fact["tool"], fact["arguments"]), fact)
            else:
                value = fact_value(matching[-1], fact)
            options = {key: fact[key] for key in ("unit", "rel_tol", "abs_tol", "datetime_tolerance_seconds") if key in fact}
            if "unit" not in options:
                for suffix, unit in (("_hz", "Hz"), ("_seconds", "s"), ("_minutes", "min"),
                                     ("_bytes", "bytes"), ("_km", "km"), ("_m", "m"), ("_w", "W")):
                    if fact["path"].endswith(suffix):
                        options["unit"] = unit
                        break
            present = value_mentioned(value, text, **options)
            facts.append({**fact, "expected": value, "mentioned": present,
                          "visible_mentioned": value_mentioned(value, visible_text, **options)})
        except (IndexError, KeyError, TypeError):
            facts.append({**fact, "expected": None, "mentioned": False, "reference_missing": True})
    forbidden = [row for row in calls if row["name"] in turn.get("forbidden_tools", [])]
    # Explicit expectations allow the current suite to grade meteor overlays and
    # geographic spectrograms without changing historical case definitions.
    expected_media = turn.get("expected_media")
    if expected_media is None:
        expected_media = ["audio" if call["tool"] == "get_wav" else "image"
                          for call in turn["required_calls"]
                          if call["tool"] in {"get_wav", "get_spectrogram"}]
    if set(expected_media) - {"audio", "image", "detection"}:
        raise ValueError("Unknown expected media type in benchmark case.")
    delivered_media = {"audio": reply.audio, "image": reply.images, "detection": reply.detections}
    return {
        "scorer_version": SCORER_VERSION,
        "tool_plan_pass": route_pass if any(routes) else None,
        "model_tool_plan_pass": model_route_pass if any(routes) else None,
        "orchestration_adjustments": len(adjustments),
        "strict_tool_plan_pass": (all(matched) and not unexpected_errors) if matched else None,
        "required_call_matches": matched,
        "native_completion": reply.model_text is not None,
        "direct_answer_completion": any(
            str(key).startswith("direct_") and str(key).endswith(("_summary", "_details")) and value is True
            for message in getattr(reply.session, "messages", ())[-1:]
            for key, value in (getattr(message, "metadata", None) or {}).items()
        ) if getattr(reply, "session", None) is not None else False,
        "fallback": any(row["event"] == "recovery" for row in trace),
        "generation_retries": sum(row["event"] == "generation_retry" for row in trace),
        "validation_retries": sum(row["event"] == "validation_retry" for row in trace),
        "tool_errors": len(errors),
        "unexpected_tool_errors": len(unexpected_errors),
        "forbidden_call_attempts": len(forbidden),
        "tool_call_count": len(calls),
        "call_budget_exceeded": len(calls) > turn.get("max_tool_calls", 1000000),
        "media_delivery_pass": (all(bool(delivered_media[name]) for name in expected_media)
                                if expected_media else None),
        "review_rubric": turn.get("review_rubric", []),
        "scientific_review_flags": scientific_flags(text, all_results, turn),
        "visible_scientific_review_flags": scientific_flags(visible_text, all_results, turn),
        "model_grounding_review_flags": model_grounding_flags,
        "visible_grounding_review_flags": visible_grounding_flags,
        "snapshot_misses": sum(row["result"].get("error_code") == "benchmark_unmatched_call" for row in errors),
        "source_mentioned": any(source in text for source in sources) if turn["require_source"] else None,
        "visible_source_mentioned": any(source in visible_text for source in sources) if turn["require_source"] else None,
        "fact_presence_heuristic": facts,
        "visible_fact_presence_heuristic": [{**fact, "mentioned": fact.get("visible_mentioned", False)} for fact in facts],
        "unexpected_calls": len(calls) if turn["kind"] in {"clarification", "unsupported", "recall"} else None,
        "media_expected": bool(expected_media),
        "audio_attachments": len(reply.audio), "image_attachments": len(reply.images),
        "detection_attachments": len(reply.detections),
        "manual_review_required": True,
    }
