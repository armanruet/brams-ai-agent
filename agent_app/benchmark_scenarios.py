"""Controlled negative scenarios with production tool schemas and no archive I/O.

These are explicitly synthetic interventions, never real-tool reference captures.
The catalog seed is an actual registry row; empty catalogs and file/service states
are controlled conditions. No result can claim a successful download.
"""

from copy import deepcopy
from datetime import timedelta
from functools import wraps
import inspect

from .brams_sdk import normalize_search
from .tool_validation import validate_tool_arguments


FIXTURES = {"no_candidates", "no_png", "archive_error", "availability_missing", "availability_zero", "availability_mixed", "detection_partial"}
TIME_TOOLS = {"get_current_utc_time", "inspect_time_window", "convert_local_time"}


def validate_fixture(case):
    fixture = case.get("controlled_fixture")
    if fixture is None:
        return
    if fixture not in FIXTURES:
        raise ValueError(f"Unknown controlled fixture for {case['id']}.")
    seed = case.get("fixture_catalog_seed", {})
    if fixture != "no_candidates" and (not seed.get("receiver_id") or not seed.get("coordinates_source_url")):
        raise ValueError(f"Controlled fixture {case['id']} needs a provenance-bearing registry seed.")


def controlled_tools(functions, case):
    """Keep every schema; intercept every non-time tool so fixtures cannot escape."""
    validate_fixture(case)
    fixture = case.get("controlled_fixture")
    if fixture is None:
        return functions

    def wrap(function):
        if function.__name__ in TIME_TOOLS:
            return function
        signature = inspect.signature(function)

        @wraps(function)
        async def respond(**kwargs):
            bound = signature.bind(**kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            validate_tool_arguments(function.__name__, arguments)
            base = {"benchmark_fixture": fixture, "evidence_mode": "controlled_fixture",
                    "live_query_made": False, "downloaded": False,
                    "source": "Controlled benchmark intervention, not a live archive observation",
                    "source_url": f"urn:brams:benchmark:controlled:{fixture}"}
            if arguments.get("search"):
                base["query"] = arguments["search"]
            for field in ("start_utc", "end_utc", "direction", "action", "country"):
                value = arguments.get(field)
                if value is not None:
                    base[field] = value.isoformat() if hasattr(value, "isoformat") else value
            name = function.__name__
            if name == "list_receivers":
                rows = [] if fixture == "no_candidates" else [deepcopy(case["fixture_catalog_seed"])]
                terms = normalize_search(arguments["search"]).split()
                rows = [row for row in rows if all(term in normalize_search(
                    f"{row['receiver_id']} {row.get('name', '')} {row.get('location_code', '')}") for term in terms)]
                count = len(rows)
                if arguments["detail"] != "coordinates":
                    rows = [{key: value for key, value in row.items() if not key.startswith("coordinates")} for row in rows]
                start, limit = arguments["offset"], arguments["limit"]
                return {**base, "status": "ok", "query": arguments["search"],
                        "receivers": rows[start:start+limit], "count": count,
                        "detail": arguments["detail"], "offset": start, "limit": limit,
                        "has_more": start+limit < count,
                        "scope": "controlled_catalog_subset", "recording_availability_checked": False,
                        "note": "Catalog scope is this controlled scenario only. The retained registry coordinates keep their original source; country polygon membership was not checked."}
            if fixture == "archive_error":
                return {**base, "status": "error", "error_code": "controlled_archive_failure",
                        "recording_availability_checked": False,
                        "message": "Controlled service failure. Availability is unknown; this is not evidence of no recordings."}
            if fixture.startswith("availability_") and name == "check_availability":
                coverage = "none" if fixture == "availability_zero" else "unknown"
                row = {"receiver_id": case["fixture_catalog_seed"]["receiver_id"],
                       "name": case["fixture_catalog_seed"]["name"], "coverage": coverage,
                       "availability_percent": 0.0 if fixture == "availability_zero" else None}
                if fixture == "availability_mixed":
                    row["known_recordings_in_queried_hours"] = True
                return {**base, "status": "ok", "scope": "hourly_recording_availability",
                        "receivers": [row], "count": 1, "recording_availability_checked": True,
                        "message": {"availability_missing": "Controlled missing hourly value; coverage is unknown.",
                                    "availability_zero": "Controlled explicit zero; no coverage for this hour.",
                                    "availability_mixed": "Controlled observed positive hour plus a missing hour; whole interval coverage is unknown."}[fixture]}
            if fixture == "detection_partial" and name == "detect_meteors":
                return {**base, "status": "ok", "analysis_performed": True,
                        "receiver_id": case["fixture_catalog_seed"]["receiver_id"],
                        "receiver_name": case["fixture_catalog_seed"]["name"],
                        "recording_start_utc": base.get("start_utc"),
                        "recording_end_utc": (arguments["start_utc"] + timedelta(minutes=5)).isoformat(),
                        "meteor_detected": True, "total_meteor_count": 2,
                        "inspected_slots_count": 1, "available_slots_in_window": 2,
                        "uninspected_slots_count": 1, "coverage_complete": False,
                        "slot_errors": [{"stage": "inference", "error_type": "ControlledError"}],
                        "annotated_artifacts": [], "slots": [],
                        "message": "Controlled partial detection, not an archive observation."}
            if fixture.startswith("availability_") or fixture == "detection_partial":
                return {**base, "status": "unknown", "recording_availability_checked": False,
                        "message": "This controlled scenario has no evidence for this operation."}
            if fixture == "no_candidates":
                return {**base, "status": "receiver_not_found", "receivers": [], "count": 0,
                        "recording_availability_checked": False,
                        "message": "No matching receiver in the controlled catalog. No archive request was made."}
            if name == "query_geographic_receivers" and arguments.get("action") == "spectrogram":
                return {**base, "status": "no_spectrogram_found", "recording_availability_checked": False,
                        "message": "No archived PNG was returned in this controlled scenario."}
            if name == "get_spectrogram":
                return {**base, "status": "no_spectrogram_found", "query": arguments["search"],
                        "start_utc": arguments["start_utc"].isoformat(),
                        "end_utc": arguments["end_utc"].isoformat(),
                        "message": "No archived PNG was returned in this controlled scenario. WAV/recording availability was not established."}
            return {**base, "status": "unknown", "recording_availability_checked": False,
                    "message": "This controlled PNG-missing scenario supplies no evidence about this other operation. Do not infer no WAVs or no recordings."}
        return respond
    return [wrap(fn) for fn in functions]


def fixture_exercised(case, trace):
    """Reaching a condition is distinct from responding to it correctly."""
    fixture = case.get("controlled_fixture")
    if fixture is None:
        return None
    for event in trace:
        result = event.get("result", {})
        if event.get("event") != "tool_result" or not isinstance(result, dict) or result.get("benchmark_fixture") != fixture:
            continue
        if fixture == "no_candidates" and event.get("name") in {"list_receivers", "query_geographic_receivers"} and result.get("count") == 0:
            return True
        if fixture == "no_png" and event.get("name") in {"get_spectrogram", "query_geographic_receivers"} and result.get("status") == "no_spectrogram_found":
            return True
        if fixture == "archive_error" and result.get("status") == "error":
            return True
        if fixture.startswith("availability_") and event.get("name") == "check_availability" and result.get("status") == "ok":
            return True
        if fixture == "detection_partial" and event.get("name") == "detect_meteors" and result.get("coverage_complete") is False:
            return True
    return False
