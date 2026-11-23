"""Validated receiver catalog and recording availability tools using PyBRAMS."""

import asyncio
from datetime import timezone
from importlib.metadata import PackageNotFoundError, version
import logging
import sys
import math
from typing import Literal

from pydantic import AwareDatetime
from pydantic_ai import ModelRetry
from .config import load_settings
from .tools import inspect_time_window
from .vision import vision_setup_status

from .tools import get_current_utc_time
from .tool_validation import validate_tool_arguments
from .brams_sdk import (
    CATALOG_CACHE_TTL_SECONDS,
    _detect_meteors,
    _load_availability,
    _load_catalog,
    _load_catalog_with_meta,
    _load_spectrogram,
    _load_wav,
    filter_receivers,
    pop_thread_cache_events,
)
from .science import receiver_coordinates, _load_beacon, _load_recordings, _load_psd

_DEFAULT_LOAD_CATALOG = _load_catalog


LOGGER = logging.getLogger(__name__)


def pybrams_setup_status() -> dict:
    if sys.version_info < (3, 13):
        return {
            "ready": False, "code": "python_version",
            "message": "Receiver lookup needs Python 3.13+ and requirements-brams.txt.",
        }
    try:
        installed = version("pybrams")
    except PackageNotFoundError:
        return {
            "ready": False, "code": "dependency_missing",
            "message": "Install receiver lookup with: python -m pip install -r requirements-brams.txt",
        }
    return {
        "ready": True, "version": installed, "code": "ready",
            "message": "PyBRAMS tools: receivers and coordinates, beacons, recording metadata, availability, spectrograms, meteor echo detection, WAV playback/download and noise/calibrator PSD. BRAMS times default to UTC.",
    }


async def check_availability(search: str, start_utc: AwareDatetime, end_utc: AwareDatetime) -> dict:
    """Check hourly archived recording coverage (not live uptime); missing rows mean unknown.

    Args:
        search: Receiver name/ID (empty = all stations).
        start_utc: Inclusive ISO UTC/offset start.
        end_utc: Exclusive ISO UTC/offset end.
    """
    window = validate_tool_arguments("check_availability", locals())
    search = "" if search.strip() == "*" else search.strip()
    base = {"query": search, **window, "scope": "hourly_recording_availability",
            "resolution": "one hour", "interval_convention": "start inclusive, end exclusive",
            "recording_availability_checked": False,
            "note": "Hourly archive metrics do not measure live station uptime. Partial-hour requests expand to surrounding hours; mixed hourly coverage cannot establish exact sub-hour coverage."}
    setup = pybrams_setup_status()
    if not setup["ready"]:
        return {**base, "status": "setup_required", "error_code": setup["code"],
                "message": setup["message"], "count": None, "receivers": []}
    try:
        result = await asyncio.to_thread(_load_availability, search,
                                        start_utc.astimezone(timezone.utc),
                                        end_utc.astimezone(timezone.utc))
        return {**base, **result, "source": "PyBRAMS receiver catalog + BRAMS hourly availability archive",
                "sdk_version": setup["version"],
                "retrieved_at_utc": get_current_utc_time()["utc_time"]}
    except Exception as error:
        LOGGER.warning("PyBRAMS availability lookup failed: %s", type(error).__name__)
        return {**base, "status": "error", "error_code": "availability_lookup_failed",
                "message": "The BRAMS query failed. Recording availability is unknown, not unavailable.",
                "exception_type": type(error).__name__, "count": None, "receivers": []}


async def get_spectrogram(search: str, start_utc: AwareDatetime, end_utc: AwareDatetime,
                          direct_answer: bool = False) -> dict:
    """Retrieve one archived spectrogram PNG without pixel analysis.

    Args:
        search: Receiver name/ID.
        start_utc: Inclusive ISO UTC/offset start.
        end_utc: Exclusive ISO UTC/offset end (<=24h).
        direct_answer: True only if one PNG satisfies request.
    """
    window = validate_tool_arguments("get_spectrogram", locals())
    base = {**window, "query": search, "scope": "archived_spectrogram",
            "downloaded": False, "image_analysis_performed": False}
    setup = pybrams_setup_status()
    if not setup["ready"]:
        return {**base, "status": "setup_required", "error_code": setup["code"],
                "message": setup["message"]}
    try:
        result = await asyncio.to_thread(_load_spectrogram, search,
                                        start_utc.astimezone(timezone.utc),
                                        end_utc.astimezone(timezone.utc))
        return {**base, **result, "source": "PyBRAMS receiver catalog + BRAMS file archive + File.spectrogram",
                "retrieved_at_utc": get_current_utc_time()["utc_time"]}
    except Exception as error:
        LOGGER.warning("PyBRAMS spectrogram lookup failed: %s", type(error).__name__)
        return {**base, "status": "error", "error_code": "spectrogram_download_failed",
                "message": "The archive query or PNG download failed. No spectrogram can be displayed.",
                "exception_type": type(error).__name__}


async def get_wav(search: str, start_utc: AwareDatetime, end_utc: AwareDatetime) -> dict:
    """Retrieve one original WAV with normalized playback preview.

    Args:
        search: Unique receiver name/ID.
        start_utc: Inclusive ISO UTC/offset start.
        end_utc: Exclusive ISO UTC/offset end (<=24h).
    """
    window = validate_tool_arguments("get_wav", locals())
    base = {**window, "query": search, "scope": "archived_wav", "downloaded": False}
    setup = pybrams_setup_status()
    if not setup["ready"]:
        return {**base, "status": "setup_required", "error_code": setup["code"], "message": setup["message"]}
    try:
        result = await asyncio.to_thread(_load_wav, search, start_utc.astimezone(timezone.utc),
                                        end_utc.astimezone(timezone.utc))
        return {**base, **result, "source": "PyBRAMS receiver catalog + BRAMS file archive + File.save_raw",
                "retrieved_at_utc": get_current_utc_time()["utc_time"]}
    except Exception as error:
        LOGGER.warning("PyBRAMS WAV download failed: %s", type(error).__name__)
        return {**base, "status": "error", "error_code": "wav_download_failed",
                "message": "The archive query or WAV download failed; no recording can be attached.",
                "exception_type": type(error).__name__}


async def list_receivers(search: str = "", detail: Literal["summary", "coordinates"] = "summary",
                         limit: int = 20, offset: int = 0) -> dict:
    """Find catalog receivers, sites, or coordinates (does not prove activity).

    Args:
        search: Name or code (empty = all).
        detail: summary or coordinates (deg/m).
        limit: Page size 1-20.
        offset: Nonnegative page offset.
    """
    validate_tool_arguments("list_receivers", locals())
    setup = pybrams_setup_status()
    if not setup["ready"]:
        return {
            "status": "setup_required", "error_code": setup["code"],
            "message": setup["message"], "receivers": [], "count": None,
            "recording_availability_checked": False,
        }
    try:
        if _load_catalog is not _DEFAULT_LOAD_CATALOG:
            catalog, source_url = await asyncio.to_thread(_load_catalog)
            catalog_cache = {
                "cached": False,
                "ttl_seconds": CATALOG_CACHE_TTL_SECONDS,
                "age_seconds": 0.0,
                "retrieved_at_utc": None,
                "source_kind": "receiver_catalog",
                "freshness_policy": "Direct/mocked catalog loader used without TTL cache.",
            }
        else:
            catalog, source_url, catalog_cache = await asyncio.to_thread(
                _load_catalog_with_meta, CATALOG_CACHE_TTL_SECONDS
            )
        rows = filter_receivers(catalog, search)
        locations = {}
        for row in rows:
            locations[row["location_code"]] = locations.get(row["location_code"], 0) + 1
        page = rows[offset:offset+limit]
        if detail == "coordinates":
            page = await asyncio.to_thread(receiver_coordinates, page)
        return {
            "status": "ok", "query": search, "count": len(rows),
            "receivers": page, "detail": detail, "offset": offset, "limit": limit,
            "has_more": offset+limit < len(rows), "location_count": len(locations),
            "multi_receiver_sites": {code: count for code, count in locations.items() if count > 1},
            "source": "PyBRAMS Receiver.get_all + Location.get" if detail == "coordinates" else "PyBRAMS Receiver.get_all",
            "source_url": source_url, "sdk_version": setup["version"],
            "retrieved_at_utc": get_current_utc_time()["utc_time"],
            "catalog_cache": catalog_cache,
            "scope": "receiver_catalog", "recording_availability_checked": False,
            "note": "Catalog entries do not establish current uptime or recording coverage.",
        }
    except Exception as error:
        LOGGER.warning("PyBRAMS receiver lookup failed: %s", type(error).__name__)
        return {"status": "error", "error_code": "receiver_lookup_failed",
                "message": "The PyBRAMS receiver catalog query failed. This is not evidence that no receivers exist.",
                "exception_type": type(error).__name__, "count": None, "receivers": [],
                "recording_availability_checked": False}


async def _scientific_query(loader, scope, *arguments):
    setup = pybrams_setup_status()
    if not setup["ready"]:
        return {"status": "setup_required", "scope": scope, "message": setup["message"]}
    try:
        result = await asyncio.to_thread(loader, *arguments)
        return {**result, "scope": scope, "sdk_version": setup["version"],
                "retrieved_at_utc": get_current_utc_time()["utc_time"]}
    except Exception as error:
        LOGGER.warning("PyBRAMS %s failed: %s", scope, type(error).__name__)
        return {"status": "error", "scope": scope, "exception_type": type(error).__name__,
                "message": "The metadata/download/calculation failed; the requested values are unknown. Do not invent a measurement or interpret this as no data."}


async def get_beacon(emitter_code: str = "BEDOUR", receiver_search: str = "") -> dict:
    """Look up transmitter metadata or surface distance to a receiver.

    Args:
        emitter_code: Transmitter code (e.g. BEDOUR).
        receiver_search: Optional receiver/site for distance.
    """
    return await _scientific_query(_load_beacon, "emitter_registry", emitter_code.strip().upper(), receiver_search)


async def get_recordings(search: str, start_utc: AwareDatetime, end_utc: AwareDatetime,
                         limit: int = 10, offset: int = 0) -> dict:
    """List recording metadata; preceding boundary files are labelled.

    Args:
        search: Receiver ID/name (site selects SYS001).
        start_utc: Inclusive ISO UTC/offset start.
        end_utc: Exclusive ISO UTC/offset end (<=24h).
        limit: Page size 1-20.
        offset: Nonnegative page offset.
    """
    window = validate_tool_arguments("get_recordings", locals())
    return {**window, "query": search, **await _scientific_query(_load_recordings, "recording_metadata",
            search, start_utc.astimezone(timezone.utc), end_utc.astimezone(timezone.utc), limit, offset)}


async def get_psd(search: str, start_utc: AwareDatetime, end_utc: AwareDatetime,
                  noise_low_hz: float = 800, noise_high_hz: float = 900,
                  calibrator_half_width_hz: float = 3, artifact_id: str | None = None) -> dict:
    """Measure noise and calibrator PSD on one original WAV (never normalized preview).

    Args:
        search: Unique receiver ID/name (site selects SYS001).
        start_utc: Inclusive ISO UTC/offset start.
        end_utc: Exclusive ISO UTC/offset end (<=24h).
        noise_low_hz: Noise band lower Hz (>=0).
        noise_high_hz: Noise band upper Hz (>low).
        calibrator_half_width_hz: Positive half-width Hz (<=100).
        artifact_id: Optional prior original WAV artifact ID.
    """
    window = validate_tool_arguments("get_psd", locals())
    return {**window, "query": search, **await _scientific_query(_load_psd, "waveform_psd",
            search, start_utc.astimezone(timezone.utc), end_utc.astimezone(timezone.utc),
            noise_low_hz, noise_high_hz, calibrator_half_width_hz, artifact_id)}


async def detect_meteors(
    search: str,
    start_utc: AwareDatetime,
    end_utc: AwareDatetime,
    max_slots: int = 1,
    confidence_threshold: float | None = None,
    artifact_id: str | None = None,
    direct_answer: bool = False,
) -> dict:
    """Detect meteor echo candidates in spectrograms and create annotated images.

    Args:
        search: Unique receiver name or ID.
        start_utc: Inclusive ISO UTC/offset start.
        end_utc: Exclusive ISO UTC/offset end (<=24h for one slot, <=1.5h for multiple slots).
        max_slots: Five-minute recordings to inspect, 1–12; use 1 per recording, 12 per hour.
        confidence_threshold: Score in [0.1, 0.95] (default 0.25).
        artifact_id: Optional prior spectrogram/WAV artifact ID.
        direct_answer: True only for a simple detection and annotated image; false for extra tasks or details.
    """
    window = inspect_time_window(start_utc, end_utc)
    if not search.strip():
        raise ModelRetry("Specify a receiver name or ID before running meteor detection.")
    if not isinstance(max_slots, int) or isinstance(max_slots, bool) or not 1 <= max_slots <= 12:
        raise ModelRetry("Use max_slots between 1 and 12.")
    if max_slots == 1 and window["duration_hours"] > 24:
        raise ModelRetry("Use at most 24 hours when inspecting 1 spectrogram (max_slots=1).")
    if max_slots > 1 and window["duration_hours"] > 1.5:
        raise ModelRetry("Multi-slot meteor scans (max_slots > 1) are bounded to at most 1.5 hours.")
    if confidence_threshold is not None and (
        isinstance(confidence_threshold, bool)
        or not math.isfinite(confidence_threshold)
        or not 0.1 <= confidence_threshold <= 0.95
    ):
        raise ModelRetry("confidence_threshold must be between 0.1 and 0.95 (or omitted for default 0.25).")

    base = {
        **window,
        "query": search.strip(),
        "scope": "spectrogram_meteor_detection",
        "downloaded": False,
        "analysis_performed": False,
    }
    setup = pybrams_setup_status()
    if not setup["ready"]:
        return {**base, "status": "setup_required", "error_code": setup["code"], "message": setup["message"]}
    vision_status = vision_setup_status(load_settings().vision_model_path)
    if not vision_status["ready"]:
        return {
            **base,
            "status": "setup_required",
            "error_code": vision_status["code"],
            "message": vision_status["message"],
        }
    try:
        result = await asyncio.to_thread(
            _detect_meteors,
            search.strip(),
            start_utc.astimezone(timezone.utc),
            end_utc.astimezone(timezone.utc),
            max_slots,
            confidence_threshold,
            artifact_id,
        )
        return {
            **base,
            **result,
            "source": "PyBRAMS + local YOLOv11-n spectrogram detector",
            "sdk_version": setup["version"],
            "retrieved_at_utc": get_current_utc_time()["utc_time"],
        }
    except Exception as error:
        LOGGER.warning("Meteor echo detection failed: %s", type(error).__name__)
        return {
            **base,
            "status": "error",
            "error_code": "meteor_detection_failed",
            "message": "Spectrogram download or local YOLO inference failed; meteor presence is unknown, not zero.",
            "exception_type": type(error).__name__,
        }



def _summarize_cache_events(events: list[dict]) -> dict:
    by_kind = {}
    for event in events:
        kind = event.get("source_kind")
        if kind and kind not in by_kind:
            by_kind[kind] = event
    catalog = by_kind.get("receiver_catalog")
    locations = by_kind.get("location_registry")
    return {
        "ttl_seconds": CATALOG_CACHE_TTL_SECONDS,
        "catalog_cached": bool(catalog["cached"]) if catalog else False,
        "catalog_age_seconds": catalog["age_seconds"] if catalog else None,
        "catalog_retrieved_at_utc": catalog["retrieved_at_utc"] if catalog else None,
        "locations_cached": bool(locations["cached"]) if locations else False,
        "locations_age_seconds": locations["age_seconds"] if locations else None,
        "locations_retrieved_at_utc": locations["retrieved_at_utc"] if locations else None,
        "freshness_policy": (catalog or locations or {}).get("freshness_policy", ""),
    }


def _run_geographic_query(direction, action, start_utc, end_utc, boundary_degrees):
    from .geography import query_belgian_region

    pop_thread_cache_events()
    result = query_belgian_region(direction, action, start_utc, end_utc, boundary_degrees)
    events = pop_thread_cache_events()
    if events and isinstance(result, dict) and "catalog_cache" not in result:
        return {**result, "catalog_cache": _summarize_cache_events(events)}
    return result


async def query_geographic_receivers(
    country: str,
    direction: Literal["north", "south", "east", "west"],
    action: Literal["count", "active_count", "spectrogram"],
    start_utc: AwareDatetime | None = None,
    end_utc: AwareDatetime | None = None,
    boundary_degrees: float | None = None,
    direct_answer: bool = True,
) -> dict:
    """Count or select Belgian BRAMS stations by coordinates, not administrative regions.

    Args:
        country: Belgium only.
        direction: north, south, east or west.
        action: count catalog sites, active_count for exact-interval recording metadata, or spectrogram for one image.
        start_utc: Inclusive ISO start (required for active_count/spectrogram).
        end_utc: Exclusive ISO end (<=24h, required for active_count/spectrogram).
        boundary_degrees: Optional lat/lon cutoff (default midpoint).
        direct_answer: True only if this result fully answers the request.
    """
    from .geography import query_belgian_region
    if country.strip().casefold() not in {"belgium", "belgique", "belgië", "belgie"}:
        return {"status": "unsupported_country", "message": "Geographic station selection currently supports Belgium only."}
    if direction not in {"north", "south", "east", "west"} or action not in {"count", "active_count", "spectrogram"}:
        raise ModelRetry("Choose a direction and one of count, active_count or spectrogram.")
    if boundary_degrees is not None and (isinstance(boundary_degrees, bool) or not math.isfinite(boundary_degrees)):
        raise ModelRetry("The region cutoff must be a finite coordinate in degrees.")
    if action != "count":
        if start_utc is None or end_utc is None:
            raise ModelRetry("Provide the UTC start and end of the requested recording interval.")
        window = inspect_time_window(start_utc, end_utc)
        if window["duration_hours"] > 24:
            raise ModelRetry("Geographic recording queries cover at most 24 hours per request.")
    elif start_utc is not None or end_utc is not None:
        raise ModelRetry("Use active_count to check recording activity in a time window.")
    setup = pybrams_setup_status()
    if not setup["ready"]:
        return {"status": "setup_required", "message": setup["message"]}
    try:
        result = await asyncio.to_thread(
            _run_geographic_query,
            direction,
            action,
            start_utc.astimezone(timezone.utc) if start_utc else None,
            end_utc.astimezone(timezone.utc) if end_utc else None,
            boundary_degrees,
        )
        # Keep the tool result compact; all sites were counted before limiting examples.
        for key in ("sites", "active_sites"):
            if key in result:
                result[key + "_omitted"] = max(0, len(result[key]) - 5)
                result[key] = [{"name": row["name"], "location_code": row["location_code"],
                                "receivers": row["receivers"]} for row in result[key][:5]]
        result.pop("active_receiver_ids", None)
        return result
    except ValueError as error:
        if str(error).startswith("The geographic boundary"):
            raise ModelRetry(str(error)) from error
        LOGGER.warning("Geographic lookup failed: %s", type(error).__name__)
    except Exception as error:
        LOGGER.warning("Geographic lookup failed: %s", type(error).__name__)
    return {"status": "error", "message": "The geographic or archive lookup failed; the requested count is unknown."}
