"""Fresh PyBRAMS metadata access and hourly coverage calculations.

This module uses only the standard library until an SDK call is made.
"""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import tempfile
import threading
import time
import unicodedata

from .cancellation import check_cancelled, current_cancel_event, stream_download_bytes, stream_download_to_file

SDK_LOCK = threading.Lock()


def _is_real_pybrams_recording(recording) -> bool:
    if current_cancel_event() is None or not type(recording).__module__.startswith("pybrams."):
        return False
    try:
        from pybrams.utils import http as pybrams_http
        return getattr(pybrams_http.get, "__module__", "").startswith("pybrams.")
    except Exception:
        return False
CATALOG_CACHE_TTL_SECONDS = 120.0
_CATALOG_FRESHNESS_POLICY = (
    "Receiver catalog and static site coordinates may be cached in memory for up to "
    "120 seconds; hourly availability, archive files, spectrograms, WAVs, and beacon "
    "queries are always fetched live."
)
_CATALOG_CACHE_LOCK = threading.Lock()
_CATALOG_CACHE: dict | None = None
_LOCATIONS_CACHE: dict[str, dict] = {}
_THREAD_CACHE_EVENTS = threading.local()


def _utc_string(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_search(value: str) -> str:
    text = unicodedata.normalize("NFKD", value.casefold())
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split())


def filter_receivers(receivers: dict, search: str) -> list[dict]:
    terms = normalize_search(search).split()
    rows = []
    for receiver in receivers.values():
        text = normalize_search(
            f"{receiver.receiver_code} {receiver.name} {receiver.location_code}"
        )
        if all(term in text for term in terms):
            rows.append({
                "receiver_id": receiver.receiver_code,
                "name": receiver.name,
                "location_code": receiver.location_code,
                "antenna": receiver.antenna,
                "catalog_start": receiver.start,
                "catalog_end": receiver.end,
            })
    return sorted(rows, key=lambda row: row["receiver_id"])


@contextmanager
def _fresh_sdk():
    # The SDK has process-global config and in-memory caches. Serialize this
    # narrow operation, restore configuration, and never delete user caches.
    with SDK_LOCK:
        from pybrams import Receiver, Location
        from pybrams.brams.fetch import api
        from pybrams.utils import Config

        previous = {
            ("pybrams.utils.cache", "use"): Config.get("pybrams.utils.cache", "use"),
            ("pybrams.utils.http", "timeout"): Config.get("pybrams.utils.http", "timeout"),
            ("pybrams.utils.http", "max_retries"): Config.get("pybrams.utils.http", "max_retries"),
        }
        try:
            # A fresh, small metadata query; no archive downloads and no stale
            # permanent metadata cache represented as a live catalog response.
            Config.set("pybrams.utils.cache", "use", False, save_to_disk=False)
            Config.set("pybrams.utils.http", "timeout", 20, save_to_disk=False)
            Config.set("pybrams.utils.http", "max_retries", 2, save_to_disk=False)
            Receiver.get_all.cache_clear()
            Location._get_single.cache_clear()
            Location._get_multiple.cache_clear()
            yield Receiver, api, Config
        finally:
            Receiver.get_all.cache_clear()
            Location._get_single.cache_clear()
            Location._get_multiple.cache_clear()
            for (section, key), value in previous.items():
                Config.set(section, key, value, save_to_disk=False)


_ORIGINAL_FRESH_SDK = _fresh_sdk


def clear_catalog_cache() -> None:
    """Clear the bounded-TTL receiver catalog and static location registry caches."""
    global _CATALOG_CACHE
    with _CATALOG_CACHE_LOCK:
        _CATALOG_CACHE = None
        _LOCATIONS_CACHE.clear()


def _record_thread_cache_event(event: dict) -> None:
    events = getattr(_THREAD_CACHE_EVENTS, "events", None)
    if events is None:
        events = []
        _THREAD_CACHE_EVENTS.events = events
    events.append(event)


def pop_thread_cache_events() -> list[dict]:
    events = getattr(_THREAD_CACHE_EVENTS, "events", None) or []
    _THREAD_CACHE_EVENTS.events = []
    return list(events)


def _is_cacheable_sdk(Receiver, api, sdk_context=None) -> bool:
    """Bypass caching whenever unit tests mock or replace SDK callables."""
    active_ctx = _fresh_sdk if sdk_context is None else sdk_context
    if active_ctx is not _ORIGINAL_FRESH_SDK:
        return False
    if any(
        type(target).__module__.startswith("unittest.mock")
        for target in (Receiver, Receiver.get_all, api, api.request)
    ):
        return False
    return (
        getattr(Receiver, "__module__", "").startswith("pybrams")
        and getattr(Receiver.get_all, "__module__", "").startswith("pybrams")
        and getattr(api, "__name__", "").startswith("pybrams")
        and getattr(api.request, "__module__", "").startswith("pybrams")
    )


def _cache_meta(
    *,
    cached: bool,
    max_age_seconds: float,
    age_seconds: float,
    retrieved_at_utc: str,
    source_kind: str,
) -> dict:
    return {
        "cached": cached,
        "ttl_seconds": max_age_seconds,
        "age_seconds": round(max(0.0, age_seconds), 3),
        "retrieved_at_utc": retrieved_at_utc,
        "source_kind": source_kind,
        "freshness_policy": _CATALOG_FRESHNESS_POLICY,
    }


def _read_or_load_cached(
    cache_lookup,
    cache_store,
    loader,
    *,
    max_age_seconds: float,
    cacheable: bool,
    source_kind: str,
) -> tuple[object, dict]:
    if max_age_seconds > 0 and cacheable:
        now = time.monotonic()
        with _CATALOG_CACHE_LOCK:
            entry = cache_lookup()
            if entry is not None and (now - entry["monotonic"]) < max_age_seconds:
                meta = _cache_meta(
                    cached=True,
                    max_age_seconds=max_age_seconds,
                    age_seconds=now - entry["monotonic"],
                    retrieved_at_utc=entry["retrieved_at_utc"],
                    source_kind=source_kind,
                )
                _record_thread_cache_event(meta)
                return entry["value"], meta
    value = loader()
    retrieved_at = _utc_string(datetime.now(timezone.utc))
    if max_age_seconds > 0 and cacheable:
        with _CATALOG_CACHE_LOCK:
            cache_store({"value": value, "monotonic": time.monotonic(), "retrieved_at_utc": retrieved_at})
    meta = _cache_meta(
        cached=False,
        max_age_seconds=max_age_seconds,
        age_seconds=0.0,
        retrieved_at_utc=retrieved_at,
        source_kind=source_kind,
    )
    _record_thread_cache_event(meta)
    return value, meta


def _set_catalog_cache(entry: dict) -> None:
    global _CATALOG_CACHE
    _CATALOG_CACHE = entry


def _catalog_from_sdk_with_meta(
    Receiver,
    api,
    Config,
    *,
    max_age_seconds: float = CATALOG_CACHE_TTL_SECONDS,
    sdk_context=None,
) -> tuple[dict, str, dict]:
    source_url = api.base_url + Config.get("pybrams.brams.receiver", "api_endpoint")
    cacheable = _is_cacheable_sdk(Receiver, api, sdk_context=sdk_context)
    catalog, meta = _read_or_load_cached(
        lambda: _CATALOG_CACHE if _CATALOG_CACHE and _CATALOG_CACHE.get("source_url") == source_url else None,
        lambda entry: _set_catalog_cache({**entry, "source_url": source_url}),
        Receiver.get_all,
        max_age_seconds=max_age_seconds,
        cacheable=cacheable,
        source_kind="receiver_catalog",
    )
    return (dict(catalog) if isinstance(catalog, dict) else catalog), source_url, meta


def _locations_from_sdk_with_meta(
    Receiver,
    api,
    endpoint: str,
    *,
    max_age_seconds: float = CATALOG_CACHE_TTL_SECONDS,
    sdk_context=None,
):
    cache_key = api.base_url + endpoint
    cacheable = _is_cacheable_sdk(Receiver, api, sdk_context=sdk_context)
    locations, meta = _read_or_load_cached(
        lambda: _LOCATIONS_CACHE.get(cache_key),
        lambda entry: _LOCATIONS_CACHE.__setitem__(cache_key, entry),
        lambda: api.request(endpoint),
        max_age_seconds=max_age_seconds,
        cacheable=cacheable,
        source_kind="location_registry",
    )
    if isinstance(locations, list):
        return list(locations), meta
    if isinstance(locations, dict):
        return dict(locations), meta
    return locations, meta


def _load_catalog_with_meta(max_age_seconds: float = CATALOG_CACHE_TTL_SECONDS) -> tuple[dict, str, dict]:
    with _fresh_sdk() as (Receiver, api, Config):
        return _catalog_from_sdk_with_meta(
            Receiver,
            api,
            Config,
            max_age_seconds=max_age_seconds,
            sdk_context=_fresh_sdk,
        )


def _load_catalog(max_age_seconds: float = 0.0) -> tuple[dict, str]:
    catalog, source_url, _ = _load_catalog_with_meta(max_age_seconds=max_age_seconds)
    return catalog, source_url


def recording_receivers(catalog: dict, search: str) -> list[dict]:
    """An exact site code selects its primary antenna; site names stay ambiguous."""
    primary = search.strip().upper() + "_SYS001"
    return filter_receivers(catalog, primary if primary in catalog else search)


def _has_recording_url(recording, url_attribute: str) -> bool:
    if url_attribute == "any":
        return bool(getattr(recording, "png_url", None) or getattr(recording, "wav_url", None))
    return bool(getattr(recording, url_attribute, None))


def _archive_files(api, Config, start, end, receiver_id):
    """Read a file listing without PyBRAMS conflating HTTP 404 with an empty list."""
    from pybrams.brams.file import _process_file
    from pybrams.utils.http import post

    endpoint = Config.get("pybrams.brams.file", "api_endpoint")
    response = post(api.base_url + endpoint, {
        "from": start, "to": end, "receiver_code[]": [receiver_id],
    }).json()
    if not isinstance(response, list):
        raise ValueError("Unexpected response from the BRAMS file archive.")
    for metadata in response:
        yield _process_file(metadata, False, False, False, False)[1]


def _nominal_slot(recording):
    if all(hasattr(recording, key) for key in ("year", "month", "day", "hours", "minutes")):
        return datetime(recording.year, recording.month, recording.day,
                        recording.hours, recording.minutes, tzinfo=timezone.utc)
    return recording.start.replace(minute=(recording.start.minute // 5) * 5, second=0, microsecond=0)


def _select_recordings(files, start, end, receiver_id, url_attribute="png_url", max_slots: int = 1):
    # API listings can include the previous five-minute file because its actual
    # duration slightly overlaps the requested slot. Prefer slots at/after preferred.
    preferred = start.replace(minute=(start.minute // 5) * 5, second=0, microsecond=0)
    preferred_rows = []
    for recording in files:
        if (recording.receiver_code != receiver_id or recording.start >= end
                or recording.end <= start or not _has_recording_url(recording, url_attribute)):
            continue
        slot = _nominal_slot(recording)
        if slot >= preferred:
            preferred_rows.append(recording)
    candidates = sorted(preferred_rows, key=_nominal_slot)
    return candidates[:max(1, int(max_slots))], len(candidates)


def _select_recording(files, start, end, receiver_id, url_attribute):
    # Stop streaming immediately on the first preferred match for single-file tools.
    preferred = start.replace(minute=(start.minute // 5) * 5, second=0, microsecond=0)
    for recording in files:
        if (recording.receiver_code != receiver_id or recording.start >= end
                or recording.end <= start or not getattr(recording, url_attribute, None)):
            continue
        slot = _nominal_slot(recording)
        if slot >= preferred:
            return recording
    return None


def _load_spectrogram(search: str, start: datetime, end: datetime) -> dict:
    import hashlib
    import json
    from .artifacts import save_spectrogram, spectrogram_path

    check_cancelled()
    with _fresh_sdk() as (Receiver, api, Config):
        catalog, _, _ = _catalog_from_sdk_with_meta(Receiver, api, Config, sdk_context=_fresh_sdk)
        rows = recording_receivers(catalog, search)
        source = api.base_url + Config.get("pybrams.brams.file", "api_endpoint")
        base = {"source_url": source, "receivers": rows, "downloaded": False}
        if not rows:
            return {**base, "status": "receiver_not_found"}
        if len(rows) > 1:
            return {**base, "status": "receiver_ambiguous",
                    "message": "Choose one of the returned receiver IDs before downloading."}
        row = rows[0]
        # Sequential streaming creates only the selected File object. Loading
        # WAV data and running signal processing are explicitly disabled.
        files = _archive_files(api, Config, start, end, row["receiver_id"])
        try:
            recording = _select_recording(files, start, end, row["receiver_id"], "png_url")
            if recording is None:
                return {**base, "status": "no_spectrogram_found",
                        "message": "No spectrogram was listed for the selected receiver and interval."}
            png_name = recording.png_name
            rec_start = recording.start
            rec_end = recording.end
            png_url = recording.png_url
            use_stream = _is_real_pybrams_recording(recording) and bool(png_url)
            png_bytes = None if use_stream else recording.spectrogram
        finally:
            files.close()

    # Run streamed media download outside SDK_LOCK so cancellation or slow archive I/O
    # never blocks other metadata queries or touches PyBRAMS disk caches.
    check_cancelled()
    if use_stream:
        png_bytes = stream_download_bytes(png_url)
    check_cancelled()
    artifact_id = save_spectrogram(png_bytes)
    result = {**base, "status": "ok", "downloaded": True,
              "artifact_id": artifact_id, "filename": png_name,
              "receiver_id": row["receiver_id"], "receiver_name": row["name"],
              "recording_start_utc": _utc_string(rec_start),
              "recording_end_utc": _utc_string(rec_end),
              "image_source_url": png_url,
              "note": "One archived PNG overlapping the requested interval. The full recording image is displayed without cropping. No WAV was downloaded; the model has not inspected image pixels."}
    saved_path = spectrogram_path(artifact_id)
    if saved_path is not None:
        manifest = {**result, "sha256": hashlib.sha256(png_bytes).hexdigest()}
        saved_path.with_suffix(".json").write_text(json.dumps(manifest, default=str))
    return result


def _load_wav(search: str, start: datetime, end: datetime) -> dict:
    from .artifacts import save_wav_file

    check_cancelled()
    with _fresh_sdk() as (Receiver, api, Config):
        catalog, _, _ = _catalog_from_sdk_with_meta(Receiver, api, Config, sdk_context=_fresh_sdk)
        rows = recording_receivers(catalog, search)
        base = {"source_url": api.base_url + Config.get("pybrams.brams.file", "api_endpoint"),
                "receivers": rows, "downloaded": False}
        if not rows:
            return {**base, "status": "receiver_not_found"}
        if len(rows) > 1:
            return {**base, "status": "receiver_ambiguous",
                    "message": "Choose one of the returned receiver IDs before downloading."}
        row = rows[0]
        files = _archive_files(api, Config, start, end, row["receiver_id"])
        try:
            recording = _select_recording(files, start, end, row["receiver_id"], "wav_url")
            if recording is None:
                return {**base, "status": "no_wav_found", "message": "No WAV was listed for this receiver and interval."}
            filename = recording.wav_name
            if not filename or Path(filename).name != filename or not filename.lower().endswith(".wav"):
                raise ValueError("Invalid WAV filename in archive metadata.")
            wav_url = recording.wav_url
            rec_start = recording.start
            rec_end = recording.end
            use_stream = _is_real_pybrams_recording(recording) and bool(wav_url)
            downloaded_temp = None
            if not use_stream:
                downloaded_temp = tempfile.TemporaryDirectory(prefix="brams-wav-")
                try:
                    recording.save_raw(Path(downloaded_temp.name))
                except BaseException:
                    downloaded_temp.cleanup()
                    raise
        finally:
            files.close()

    # Stream WAV downloads outside SDK_LOCK so cancellation aborts sockets
    # immediately and never leaves partial files in .artifacts/.
    check_cancelled()
    temp_ctx = downloaded_temp if downloaded_temp is not None else tempfile.TemporaryDirectory(prefix="brams-wav-")
    with temp_ctx as temporary:
        target = Path(temporary) / filename
        if use_stream:
            stream_download_to_file(wav_url, target)
        check_cancelled()
        artifact = save_wav_file(target)
        from .science import wav_header
        artifact.update(wav_header(target))
        if artifact.get("header_summary"):
            actual_rate = artifact["header_summary"]["samplerate_hz"]
            if not math.isfinite(actual_rate) or actual_rate <= 0:
                raise ValueError("Invalid BRA1 sample rate")
            artifact["container_sample_rate_hz"] = artifact["sample_rate_hz"]
            artifact["sample_rate_hz"] = actual_rate
            artifact["duration_seconds"] = artifact["sample_count"] / actual_rate
            if actual_rate != artifact["container_sample_rate_hz"]:
                artifact["preview_note"] += " The player uses the integer WAV-container sample rate; scientific duration and PSD use the precise BRA1 sample rate."
    result = {**base, **artifact, "status": "ok", "downloaded": True,
            "filename": filename, "receiver_id": row["receiver_id"],
            "download_origin": "fresh archive download; SDK caches bypassed",
            "receiver_name": row["name"], "wav_source_url": wav_url,
            "recording_start_utc": _utc_string(rec_start),
            "recording_end_utc": _utc_string(rec_end),
            "note": "One complete original WAV overlapping the requested interval; not cropped. No scientific interpretation was performed."}
    import json
    from .artifacts import wav_path
    (wav_path(artifact["artifact_id"]).parent / "source.json").write_text(json.dumps(result, default=str))
    return result


def _raw_availability(api, Config, start: datetime, end: datetime) -> dict:
    """Read hourly rows before the SDK fills absent receiver-hours with zero."""
    from pybrams.utils.http import post

    endpoint = Config.get("pybrams.brams.file", "api_endpoint")
    response = post(api.base_url + endpoint, {
        "from": start, "to": end, "availability": True,
    }).json()
    if not isinstance(response, list):
        raise ValueError("Unexpected response from the BRAMS availability archive.")
    raw = {}
    for entry in response:
        if not isinstance(entry, dict):
            raise ValueError("Malformed BRAMS availability row.")
        receiver = entry.get("receiver_code")
        stamp = entry.get("datetime")
        if not isinstance(receiver, str) or not isinstance(stamp, str):
            raise ValueError("BRAMS availability row is missing its receiver or time.")
        try:
            hour = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("BRAMS availability row has an invalid time.") from exc
        hour = hour.replace(tzinfo=timezone.utc) if hour.tzinfo is None else hour.astimezone(timezone.utc)
        values = raw.setdefault(receiver, {})
        value = entry.get("availability")
        # Conflicting duplicate rows do not establish a trustworthy value.
        if hour in values and values[hour] != value:
            values[hour] = None
        else:
            values[hour] = value
    return raw


def _load_availability(search: str, start: datetime, end: datetime) -> dict:

    with _fresh_sdk() as (Receiver, api, Config):
        catalog, _, _ = _catalog_from_sdk_with_meta(Receiver, api, Config, sdk_context=_fresh_sdk)
        rows = filter_receivers(catalog, search)
        source = api.base_url + Config.get("pybrams.brams.file", "api_endpoint")
        if not rows:
            return {"status": "receiver_not_found", "receivers": [], "count": 0,
                    "source_url": api.base_url + Config.get("pybrams.brams.receiver", "api_endpoint"),
                    "recording_availability_checked": False}
        # SDK 0.1.7 does not actually round the bounds despite calling replace().
        # Align them here or its hourly keys can silently become all-zero.
        hour_start = start.replace(minute=0, second=0, microsecond=0)
        hour_end = end.replace(minute=0, second=0, microsecond=0)
        if hour_end < end:
            hour_end += timedelta(hours=1)
        raw = _raw_availability(api, Config, hour_start, hour_end)
        if not raw:
            return {"status": "no_availability_information", "receivers": rows,
                    "count": len(rows), "source_url": source,
                    "recording_availability_checked": False,
                    "message": "BRAMS returned no availability information; coverage is unknown."}
        aligned = start == hour_start and end == hour_end
        known_recording_receivers = 0
        for row in rows:
            buckets = []
            hour = hour_start
            while hour < hour_end:
                metric = raw.get(row["receiver_id"], {}).get(hour)
                if isinstance(metric, str):
                    try:
                        metric = float(metric)
                    except ValueError:
                        metric = None
                if (isinstance(metric, bool) or not isinstance(metric, (int, float))
                        or not math.isfinite(metric) or not 0 <= metric <= 1):
                    metric = None
                buckets.append({"start_utc": _utc_string(hour),
                                "end_utc": _utc_string(hour + timedelta(hours=1)),
                                "availability_fraction": metric})
                hour += timedelta(hours=1)
            metrics = [bucket["availability_fraction"] for bucket in buckets]
            complete = all(value is not None for value in metrics)
            coverage = "unknown"
            if complete:
                if all(value == 0 for value in metrics):
                    coverage = "none"
                elif all(value == 1 for value in metrics):
                    coverage = "full"
                elif aligned:
                    coverage = "partial"
            known_positive = any(value is not None and value > 0 for value in metrics)
            if coverage in {"full", "partial"} or (aligned and known_positive):
                known_recording_receivers += 1
            row.update({"coverage": coverage, "hours": buckets,
                        "availability_percent": round(100 * sum(metrics) / len(metrics), 4)
                        if complete and aligned else None})
            if not complete and known_positive:
                row["known_recordings_in_queried_hours"] = True
        # Network-wide queries must not send hundreds of hourly buckets and
        # catalog fields per receiver back into the model's context.
        if len(rows) > 1:
            rows = [{key: row[key] for key in ("receiver_id", "name", "coverage", "availability_percent",
                                                "known_recordings_in_queried_hours") if key in row}
                    for row in rows]
        return {"status": "ok", "receivers": rows, "count": len(rows),
                "coverage_counts": {kind: sum(row["coverage"] == kind for row in rows)
                                    for kind in ("full", "partial", "none", "unknown")},
                "receivers_with_recordings": known_recording_receivers,
                "receivers_with_recordings_is_lower_bound": any(row["coverage"] == "unknown" for row in rows),
                "source_url": source, "recording_availability_checked": True,
                "queried_start_utc": _utc_string(hour_start),
                "queried_end_utc": _utc_string(hour_end),
                "requested_whole_hours": aligned,
                "missing_hour_policy": "Absent hourly archive rows are unknown, not zero."}


def _cap_slot_detections(detections: list[dict], cap: int) -> tuple[list[dict], bool]:
    cleaned = [{k: v for k, v in item.items() if k != "image_xyxy"} for item in detections]
    if len(cleaned) <= cap:
        return cleaned, False
    top = sorted(cleaned, key=lambda d: (-d["confidence"], d["echo_index"]))[:cap]
    top.sort(key=lambda d: d["echo_index"])
    return top, True


def _matches_artifact_receiver(manifest: dict, search: str) -> bool:
    query_norm = normalize_search(search)
    if not query_norm:
        return False
    rid = str(manifest.get("receiver_id", ""))
    rname = str(manifest.get("receiver_name", ""))
    if rid.casefold() in {search.strip().casefold(), f"{search.strip().casefold()}_sys001"}:
        return True
    haystack = normalize_search(f"{rid} {rname}")
    return all(term in haystack for term in query_norm.split())


def _fetch_recording_spectrogram_bytes(recording, input_source: str) -> tuple[bytes, str, str | None]:
    from .vision import render_rmz_from_wav_path

    check_cancelled()
    use_stream = _is_real_pybrams_recording(recording)
    if input_source != "rmz_from_wav" and getattr(recording, "png_url", None):
        try:
            png_bytes = stream_download_bytes(recording.png_url) if use_stream else recording.spectrogram
            check_cancelled()
            return png_bytes, "archive_png", recording.png_url
        except Exception:
            if input_source == "archive_png" or not getattr(recording, "wav_url", None):
                raise
    if not getattr(recording, "wav_url", None):
        raise ValueError("No spectrogram or WAV URL available for this recording.")
    filename = recording.wav_name
    if not filename or Path(filename).name != filename or not filename.lower().endswith(".wav"):
        raise ValueError("Invalid WAV filename in archive metadata.")
    with tempfile.TemporaryDirectory(prefix="brams-det-wav-") as temporary:
        target = Path(temporary) / filename
        if use_stream:
            stream_download_to_file(recording.wav_url, target)
        else:
            recording.save_raw(Path(temporary))
        check_cancelled()
        png_bytes, _ = render_rmz_from_wav_path(target)
    return png_bytes, "rmz_from_wav", recording.wav_url


def _summarize_detection_slots(
    slot_records: list[dict],
    *,
    receiver_id: str,
    receiver_name: str,
    source_url: str,
    available_slots: int | None,
    reused_artifact: bool,
    conf_threshold: float,
    slot_errors: list[dict] | None = None,
) -> dict:
    from .vision import create_annotated_overlay

    if len(slot_records) == 1:
        overlay_indices = {0}
    else:
        positive = [idx for idx, rec in enumerate(slot_records) if rec["analysis"]["meteor_count"] > 0]
        if positive:
            ranked = sorted(
                positive,
                key=lambda idx: (
                    -slot_records[idx]["analysis"]["meteor_count"],
                    -(slot_records[idx]["analysis"]["max_confidence"] or 0.0),
                    idx,
                ),
            )[:3]
            overlay_indices = set(ranked)
        else:
            overlay_indices = {0}

    annotated_artifacts: list[dict] = []
    slot_summaries: list[dict] = []
    overlay_errors: list[dict] = []

    for idx, rec in enumerate(slot_records):
        check_cancelled()
        analysis = rec["analysis"]
        overlay_id = None
        if idx in overlay_indices:
            try:
                overlay_id = create_annotated_overlay(
                    rec["png_bytes"],
                    analysis["detections"],
                    recording_start=rec["start_dt"],
                    recording_end=rec["end_dt"],
                    receiver_label=f"{receiver_name} ({receiver_id})" if receiver_name else receiver_id,
                    model_name=analysis["model_name"],
                    conf_threshold=conf_threshold,
                )
            except Exception as exc:
                overlay_errors.append({"filename": rec["filename"],
                                       "recording_start_utc": _utc_string(rec["start_dt"]),
                                       "stage": "overlay", "error_type": type(exc).__name__})
        if overlay_id is not None:
            verdict_short = (
                f"{'At least ' if analysis.get('detection_limit_reached') else ''}{analysis['meteor_count']} meteor echo{'es' if analysis['meteor_count'] != 1 else ''}"
                if analysis["meteor_count"]
                else "No meteors detected"
            )
            annotated_artifacts.append({
                "artifact_id": overlay_id,
                "filename": rec["filename"],
                "recording_start_utc": _utc_string(rec["start_dt"]),
                "recording_end_utc": _utc_string(rec["end_dt"]),
                "meteor_count": analysis["meteor_count"],
                "label": (
                    f"Meteor detection · {receiver_name or receiver_id} ({receiver_id}) · "
                    f"{rec['start_dt'].astimezone(timezone.utc).strftime('%d %b %Y, %H:%M')}–"
                    f"{rec['end_dt'].astimezone(timezone.utc).strftime('%H:%M UTC')} · {verdict_short}"
                    if receiver_name and receiver_name != receiver_id
                    else f"Meteor detection · {receiver_id} · "
                    f"{rec['start_dt'].astimezone(timezone.utc).strftime('%d %b %Y, %H:%M')}–"
                    f"{rec['end_dt'].astimezone(timezone.utc).strftime('%H:%M UTC')} · {verdict_short}"
                ),
            })

        capped_dets, truncated = _cap_slot_detections(analysis["detections"], len(analysis["detections"]))
        slot_entry = {
            "filename": rec["filename"],
            "recording_start_utc": _utc_string(rec["start_dt"]),
            "recording_end_utc": _utc_string(rec["end_dt"]),
            "input_kind": rec["input_kind"],
            "meteor_detected": analysis["meteor_detected"],
            "meteor_count": analysis["meteor_count"],
            "high_confidence_count": analysis["high_confidence_count"],
            "max_confidence": analysis["max_confidence"],
            "detections": capped_dets,
            "detections_truncated": truncated,
            "retained_detections_count": len(capped_dets),
            "detection_limit_reached": analysis.get("detection_limit_reached", False),
            "annotated_artifact_id": overlay_id,
            "inference_ms": analysis["inference_ms"],
        }
        if rec.get("media_source_url"):
            key = "wav_source_url" if rec["input_kind"] == "rmz_from_wav" else "image_source_url"
            slot_entry[key] = rec["media_source_url"]
        slot_summaries.append(slot_entry)

    total_meteors = sum(s["meteor_count"] for s in slot_summaries)
    total_high = sum(s["high_confidence_count"] for s in slot_summaries)
    slots_with_meteors = sum(1 for s in slot_summaries if s["meteor_detected"])
    max_confs = [s["max_confidence"] for s in slot_summaries if s["max_confidence"] is not None]
    first_analysis = slot_records[0]["analysis"]

    result = {
        "status": "ok",
        "downloaded": not reused_artifact,
        "reused_artifact": reused_artifact,
        "analysis_performed": True,
        "receiver_id": receiver_id,
        "receiver_name": receiver_name,
        "source_url": source_url,
        "available_slots_in_window": available_slots,
        "inspected_slots_count": len(slot_summaries),
        "uninspected_slots_count": max(0, available_slots - len(slot_summaries)) if available_slots is not None else None,
        "slots_with_meteors": slots_with_meteors,
        "meteor_detected": total_meteors > 0,
        "total_meteor_count": total_meteors,
        "detection_limit_reached": any(s["detection_limit_reached"] for s in slot_summaries),
        "high_confidence_count": total_high,
        "max_confidence": round(max(max_confs), 4) if max_confs else None,
        "recording_start_utc": slot_summaries[0]["recording_start_utc"],
        "recording_end_utc": slot_summaries[-1]["recording_end_utc"],
        "filename": slot_summaries[0]["filename"] if len(slot_summaries) == 1 else None,
        "image_source_url": slot_summaries[0].get("image_source_url") if len(slot_summaries) == 1 else None,
        "model_name": first_analysis["model_name"],
        "model_architecture": first_analysis["model_architecture"],
        "backend": first_analysis["backend"],
        "confidence_threshold": conf_threshold,
        "annotated_artifacts": annotated_artifacts,
        "slots": slot_summaries,
        "caveat": (
            f"Automated spectrogram detection ({first_analysis['model_name']}, conf >= {conf_threshold:g}). "
            "Dense showers, faint underdense echoes, aircraft reflections, or RFI can cause occasional false positives or misses."
        ),
    }
    if slot_errors:
        result["slot_errors"] = slot_errors
    if overlay_errors:
        result["overlay_errors"] = overlay_errors
    return result


def _artifact_coverage(result, start, end, actual_start):
    cursor = start.replace(minute=(start.minute // 5) * 5, second=0, microsecond=0)
    inspected_slot = actual_start.replace(minute=(actual_start.minute // 5) * 5, second=0, microsecond=0)
    expected = []
    while cursor < end:
        expected.append(cursor)
        cursor += timedelta(minutes=5)
    uncovered = [slot for slot in expected if slot != inspected_slot]
    result.update(expected_slots_in_window=len(expected),
                  uncovered_slot_starts_utc=[_utc_string(slot) for slot in uncovered],
                  coverage_complete=not uncovered)
    if uncovered:
        result["available_slots_in_window"] = None
        result["uninspected_slots_count"] = None
    return result


def _artifact_overlaps_window(actual_start, actual_end, start, end):
    requested_slot = start.replace(minute=(start.minute // 5) * 5, second=0, microsecond=0)
    artifact_slot = actual_start.replace(minute=(actual_start.minute // 5) * 5, second=0, microsecond=0)
    return actual_start < end and actual_end > start and artifact_slot >= requested_slot


def _detect_meteors(
    search: str,
    start: datetime,
    end: datetime,
    max_slots: int = 1,
    confidence_threshold: float | None = None,
    artifact_id: str | None = None,
) -> dict:
    import hashlib
    import json
    from .artifacts import spectrogram_path, wav_path
    from .config import load_settings
    from .vision import analyze_spectrogram_bytes, render_rmz_from_wav_path

    settings = load_settings()
    conf = float(confidence_threshold if confidence_threshold is not None else settings.vision_conf_threshold)
    iou = float(settings.vision_iou_threshold)
    model_path = settings.vision_model_path
    input_source = settings.vision_input_source

    if artifact_id:
        spec_file = spectrogram_path(artifact_id)
        if spec_file is not None:
            sidecar = spec_file.with_suffix(".json")
            if not sidecar.is_file():
                return {"status": "artifact_not_found", "downloaded": False, "analysis_performed": False,
                        "message": "No verified spectrogram provenance is stored for this artifact."}
            manifest = json.loads(sidecar.read_text())
            png_bytes = spec_file.read_bytes()
            if hashlib.sha256(png_bytes).hexdigest() != manifest.get("sha256"):
                return {"status": "artifact_changed", "downloaded": False, "analysis_performed": False,
                        "message": "The local spectrogram differs from its verified checksum; fetch it again."}
            if not _matches_artifact_receiver(manifest, search):
                return {"status": "artifact_mismatch", "downloaded": False, "analysis_performed": False,
                        "message": "For artifact reuse pass its matching receiver and overlapping recording bounds."}
            actual_start = datetime.fromisoformat(manifest["recording_start_utc"].replace("Z", "+00:00"))
            actual_end = datetime.fromisoformat(manifest["recording_end_utc"].replace("Z", "+00:00"))
            if not _artifact_overlaps_window(actual_start, actual_end, start, end):
                return {"status": "artifact_mismatch", "downloaded": False, "analysis_performed": False,
                        "message": "The artifact does not overlap the requested window."}
            if max_slots > 1:
                return _detect_meteors(search, start, end, max_slots, confidence_threshold)
            analysis = analyze_spectrogram_bytes(
                png_bytes,
                model_path=model_path,
                conf_threshold=conf,
                iou_threshold=iou,
                recording_start=actual_start,
                recording_end=actual_end,
                receiver_label=f"{manifest.get('receiver_name', '')} ({manifest['receiver_id']})".strip(),
                save_overlay=False,
                include_image_boxes=True,
            )
            return _artifact_coverage(_summarize_detection_slots(
                [{
                    "filename": manifest.get("filename", f"{artifact_id}.png"),
                    "start_dt": actual_start,
                    "end_dt": actual_end,
                    "png_bytes": png_bytes,
                    "input_kind": "reused_spectrogram_artifact",
                    "media_source_url": manifest.get("image_source_url"),
                    "analysis": analysis,
                }],
                receiver_id=manifest["receiver_id"],
                receiver_name=manifest.get("receiver_name", ""),
                source_url=manifest.get("source_url", ""),
                available_slots=1,
                reused_artifact=True,
                conf_threshold=conf,
            ), start, end, actual_start)

        wav_file = wav_path(artifact_id)
        if wav_file is not None:
            sidecar = wav_file.parent / "source.json"
            if not sidecar.is_file():
                return {"status": "artifact_not_found", "downloaded": False, "analysis_performed": False,
                        "message": "No verified WAV provenance is stored for this artifact."}
            manifest = json.loads(sidecar.read_text())
            if (wav_file.stat().st_size > 64 * 1024 * 1024
                    or hashlib.sha256(wav_file.read_bytes()).hexdigest() != manifest.get("sha256")):
                return {"status": "artifact_changed", "downloaded": False, "analysis_performed": False,
                        "message": "The local original WAV differs from its verified checksum; fetch it again."}
            if not _matches_artifact_receiver(manifest, search):
                return {"status": "artifact_mismatch", "downloaded": False, "analysis_performed": False,
                        "message": "For artifact reuse pass its matching receiver and overlapping recording bounds."}
            actual_start = datetime.fromisoformat(manifest["recording_start_utc"].replace("Z", "+00:00"))
            actual_end = datetime.fromisoformat(manifest["recording_end_utc"].replace("Z", "+00:00"))
            if not _artifact_overlaps_window(actual_start, actual_end, start, end):
                return {"status": "artifact_mismatch", "downloaded": False, "analysis_performed": False,
                        "message": "The artifact does not overlap the requested window."}
            if max_slots > 1:
                return _detect_meteors(search, start, end, max_slots, confidence_threshold)
            png_bytes, _ = render_rmz_from_wav_path(wav_file)
            analysis = analyze_spectrogram_bytes(
                png_bytes,
                model_path=model_path,
                conf_threshold=conf,
                iou_threshold=iou,
                recording_start=actual_start,
                recording_end=actual_end,
                receiver_label=f"{manifest.get('receiver_name', '')} ({manifest['receiver_id']})".strip(),
                save_overlay=False,
                include_image_boxes=True,
            )
            return _artifact_coverage(_summarize_detection_slots(
                [{
                    "filename": manifest.get("filename", f"{artifact_id}.wav"),
                    "start_dt": actual_start,
                    "end_dt": actual_end,
                    "png_bytes": png_bytes,
                    "input_kind": "reused_wav_artifact",
                    "media_source_url": manifest.get("wav_source_url"),
                    "analysis": analysis,
                }],
                receiver_id=manifest["receiver_id"],
                receiver_name=manifest.get("receiver_name", ""),
                source_url=manifest.get("source_url", ""),
                available_slots=1,
                reused_artifact=True,
                conf_threshold=conf,
            ), start, end, actual_start)

        return {"status": "artifact_not_found", "downloaded": False, "analysis_performed": False,
                "message": "The spectrogram or WAV artifact is unavailable; omit artifact_id to fetch from the archive."}

    url_attr = (
        "wav_url"
        if input_source == "rmz_from_wav"
        else "png_url"
        if input_source == "archive_png"
        else "any"
    )

    check_cancelled()
    with _fresh_sdk() as (Receiver, api, Config):
        catalog, _, _ = _catalog_from_sdk_with_meta(Receiver, api, Config, sdk_context=_fresh_sdk)
        rows = recording_receivers(catalog, search)
        source = api.base_url + Config.get("pybrams.brams.file", "api_endpoint")
        base = {"source_url": source, "receivers": rows[:20], "downloaded": False, "analysis_performed": False}
        if not rows:
            return {**base, "status": "receiver_not_found"}
        if len(rows) > 1:
            return {**base, "status": "receiver_ambiguous",
                    "message": "Choose one of the returned receiver IDs before running meteor detection."}
        row = rows[0]
        files = _archive_files(api, Config, start, end, row["receiver_id"])
        try:
            selected, total_available = _select_recordings(
                files, start, end, row["receiver_id"], url_attr, max_slots=max_slots
            )
            if not selected:
                return {**base, "status": "no_spectrogram_found",
                        "message": "No spectrogram was listed for the selected receiver and interval."}
        finally:
            files.close()

    # Run multi-slot downloads and YOLO inference outside SDK_LOCK with
    # cooperative cancellation checkpoints between every slot and stage.
    slot_records: list[dict] = []
    slot_errors: list[dict] = []
    for recording in selected:
        check_cancelled()
        slot_name = getattr(recording, "png_name", None) or getattr(recording, "wav_name", "slot.png")
        try:
            png_bytes, input_kind, media_url = _fetch_recording_spectrogram_bytes(recording, input_source)
        except Exception as exc:
            if len(selected) == 1:
                raise
            slot_errors.append({
                "filename": slot_name,
                "recording_start_utc": _utc_string(recording.start),
                "stage": "download",
                "error_type": type(exc).__name__,
            })
            continue
        check_cancelled()
        try:
            analysis = analyze_spectrogram_bytes(
                png_bytes,
                model_path=model_path,
                conf_threshold=conf,
                iou_threshold=iou,
                recording_start=recording.start,
                recording_end=recording.end,
                receiver_label=f"{row['name']} ({row['receiver_id']})",
                save_overlay=False,
                include_image_boxes=True,
            )
        except Exception as exc:
            if len(selected) == 1:
                raise
            slot_errors.append({"filename": slot_name,
                                "recording_start_utc": _utc_string(recording.start),
                                "stage": "inference", "error_type": type(exc).__name__})
            continue
        slot_records.append({
            "filename": slot_name,
            "start_dt": recording.start,
            "end_dt": recording.end,
            "png_bytes": png_bytes,
            "input_kind": input_kind,
            "media_source_url": media_url,
            "analysis": analysis,
        })

    check_cancelled()

    if not slot_records:
        return {**base, "status": "error", "error_code": "meteor_detection_failed",
                "message": "No selected spectrogram could be analyzed; meteor presence is unknown.",
                "slot_errors": slot_errors}

    result = _summarize_detection_slots(
        slot_records,
        receiver_id=row["receiver_id"],
        receiver_name=row["name"],
        source_url=source,
        available_slots=total_available,
        reused_artifact=False,
        conf_threshold=conf,
        slot_errors=slot_errors or None,
    )
    # Archive listings are not evidence that every expected five-minute slot exists.
    # Account for gaps and failed downloads before making a window-wide claim.
    cursor = start.replace(minute=(start.minute // 5) * 5, second=0, microsecond=0)
    expected = []
    while cursor < end:
        expected.append(cursor)
        cursor += timedelta(minutes=5)
    inspected = {
        rec["start_dt"].replace(minute=(rec["start_dt"].minute // 5) * 5, second=0, microsecond=0)
        for rec in slot_records
    }
    uncovered = [slot for slot in expected if slot not in inspected]
    result["expected_slots_in_window"] = len(expected)
    result["uncovered_slot_starts_utc"] = [_utc_string(slot) for slot in uncovered]
    result["coverage_complete"] = not uncovered and not slot_errors
    return result



