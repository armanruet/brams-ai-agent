"""Two real local functions for testing the tool loop without BRAMS downloads."""

from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AwareDatetime
from pydantic_ai import ModelRetry


_TIMEZONE_ALIASES = {
    "brussels": "Europe/Brussels",
    "brussels time": "Europe/Brussels",
    "belgium": "Europe/Brussels",
    "luxembourg": "Europe/Luxembourg",
    "utc": "UTC",
    "gmt": "UTC",
}


def _resolve_timezone(name: str):
    cleaned = (name or "UTC").strip()
    if cleaned.casefold() in {"cet", "cest"}:
        hours = 1 if cleaned.casefold() == "cet" else 2
        return timezone(timedelta(hours=hours), cleaned.upper()), cleaned.upper()
    resolved = _TIMEZONE_ALIASES.get(cleaned.casefold(), cleaned)
    try:
        return ZoneInfo(resolved), resolved
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ModelRetry("Provide a valid IANA timezone, CET, CEST, or UTC.") from error


def utc_string(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def get_current_utc_time(timezone_hint: str = "UTC") -> dict[str, str]:
    """Read system clock for current or relative dates.

    Args:
        timezone_hint: Local date reference zone (default UTC).
    """
    now = datetime.now(timezone.utc)
    zone, reference = _resolve_timezone(timezone_hint)
    local = now.astimezone(zone)
    return {
        "utc_time": utc_string(now),
        "utc_date": now.strftime("%Y-%m-%d"),
        "source": "local_system_clock",
        "reference_timezone": reference,
        "local_time": local.isoformat(),
        "local_date": local.strftime("%Y-%m-%d"),
    }


def _normalize_local_iso(raw: str) -> str:
    """Strip accidental Z/offset suffixes if a model passes them to convert_local_time."""
    cleaned = raw.strip()
    if cleaned.endswith(("Z", "z")):
        cleaned = cleaned[:-1].strip()
    return cleaned


def _convert_single_local_time(local_datetime: str, zone, fold: int | None) -> datetime:
    try:
        value = datetime.fromisoformat(_normalize_local_iso(local_datetime))
    except ValueError as error:
        raise ModelRetry("Provide a valid ISO local date/time (e.g. 2023-12-15T23:30:00) and IANA timezone.") from error
    if value.tzinfo is not None or fold not in {None, 0, 1}:
        raise ModelRetry("Supply a local datetime without offset (e.g. 2023-12-15T23:30:00); fold must be 0 or 1 when needed.")
    candidates = [value.replace(tzinfo=zone, fold=index) for index in (0, 1)]
    valid = [candidate for candidate in candidates
             if candidate.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) == value]
    if not valid:
        raise ModelRetry("This local time does not exist during the DST transition. Ask for a valid time.")
    if len({candidate.utcoffset() for candidate in valid}) > 1 and fold is None:
        raise ModelRetry("This local time occurs twice during the DST transition. Ask which occurrence (fold 0 or 1).")
    return candidates[fold or 0]


def convert_local_time(
    local_datetime: str,
    timezone_name: str = "Europe/Brussels",
    fold: int | None = None,
    end_local_datetime: str | None = None,
    end_fold: int | None = None,
) -> dict:
    """Convert local wall time(s) to UTC with zoneinfo/DST.

    Args:
        local_datetime: Local ISO start without offset.
        timezone_name: IANA zone, CET, CEST, or UTC.
        fold: Repeated time occurrence: 0 first, 1 second.
        end_local_datetime: Optional local ISO end without offset.
        end_fold: Repeated end occurrence (0 or 1).
    """
    tz_clean = (timezone_name or "Europe/Brussels").strip()
    zone, resolved_tz = _resolve_timezone(tz_clean)
    selected = _convert_single_local_time(local_datetime, zone, fold)
    offset = selected.strftime("%z")
    utc_offset = f"{offset[:3]}:{offset[3:]}" if len(offset) == 5 else offset
    result = {
        "local_datetime": selected.isoformat(),
        "timezone": resolved_tz,
        "utc_offset": utc_offset,
        "utc_time": utc_string(selected),
        "source": "Fixed timezone offset" if resolved_tz in {"CET", "CEST"} else "Python zoneinfo IANA timezone database",
    }
    if end_local_datetime is not None and end_local_datetime.strip():
        end_selected = _convert_single_local_time(end_local_datetime.strip(), zone, end_fold)
        start_in_utc = selected.astimezone(timezone.utc)
        end_in_utc = end_selected.astimezone(timezone.utc)
        if end_in_utc <= start_in_utc:
            raise ModelRetry("end_local_datetime must be after local_datetime in the specified timezone.")
        result.update({
            "end_local_datetime": end_selected.isoformat(),
            "start_utc": utc_string(selected),
            "end_utc": utc_string(end_selected),
            "duration_minutes": (end_in_utc - start_in_utc).total_seconds() / 60.0,
        })
    return result


def inspect_time_window(start_utc: AwareDatetime, end_utc: AwareDatetime) -> dict:
    """Normalize ISO interval and compute duration; no BRAMS query.

    Args:
        start_utc: Inclusive ISO UTC/offset start.
        end_utc: Exclusive ISO UTC/offset end.
    """
    start = start_utc.astimezone(timezone.utc)
    end = end_utc.astimezone(timezone.utc)
    if end <= start:
        raise ModelRetry("end_utc must be after start_utc; clarify the interval with the user.")
    seconds = (end - start).total_seconds()
    return {
        "start_utc": utc_string(start_utc),
        "end_utc": utc_string(end_utc),
        "duration_seconds": seconds,
        "duration_minutes": seconds / 60,
        "duration_hours": seconds / 3600,
        "source": "local_time_calculation",
    }
