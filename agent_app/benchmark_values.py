"""Representation-aware evidence checks. Presence is never a factual verdict."""

from datetime import datetime, timezone
import math
import re
import unicodedata


SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
SCALES = {"hz": 1, "khz": 1000, "mhz": 1e6, "ghz": 1e9,
          "s": 1, "ms": .001, "µs": 1e-6, "us": 1e-6,
          "min": 60, "minutes": 60, "h": 3600, "hours": 3600,
          "b": 1, "bytes": 1, "kb": 1000, "mb": 1e6, "gb": 1e9,
          "m": 1, "km": 1000, "w": 1, "kw": 1000}
FAMILIES = {"hz": {"hz", "khz", "mhz", "ghz"}, "s": {"s", "ms", "µs", "us", "min", "minutes", "h", "hours"},
            "bytes": {"b", "bytes", "kb", "mb", "gb"}, "m": {"m", "km"}, "w": {"w", "kw"}}


def normalize(text):
    text = re.sub(r"[×x*]\s*10\s*([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
                  lambda match: "e" + match[1].translate(SUPERSCRIPTS), str(text))
    text = unicodedata.normalize("NFKC", text)
    for dash in "−‐‑‒–—":
        text = text.replace(dash, "-")
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"(?<=\d)\s+(?=e[+-]?\d)", "", text, flags=re.I)
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    text = re.sub(r"(?<=\d)[ \t]+(?=\d{3}(?:\D|$))", "", text)
    return text


def at_path(value, path):
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def fact_value(result, fact):
    selector = fact.get("select")
    if selector:
        rows = at_path(result, selector["collection"])
        selected = [row for row in rows if at_path(row, selector["field"]) == selector["equals"]]
        if len(selected) != 1:
            raise KeyError("Reference selector must identify exactly one row.")
        result = selected[0]
    value = at_path(result, fact["path"])
    if fact.get("format") == "date":
        return (value if isinstance(value, datetime) else datetime.fromisoformat(str(value))).date().isoformat()
    return value


def value_mentioned(value, text, *, unit=None, rel_tol=.005, abs_tol=0,
                    datetime_tolerance_seconds=0):
    text = normalize(text)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        pattern = r"(?<![\w.])(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)(?:\s*(GHz|MHz|kHz|Hz|minutes|hours|bytes|GB|MB|kB|µs|us|ms|s|min|h|km|m|kW|W|B)\b)?"
        for match in re.finditer(pattern, text, re.I):
            number, actual_unit = float(match[1]), (match[2] or "").casefold()
            if unit and actual_unit:
                family = next((members for members in FAMILIES.values() if unit.casefold() in members), None)
                if family and actual_unit not in family:
                    continue
                if family:
                    number *= SCALES[actual_unit] / SCALES[unit.casefold()]
            if math.isclose(number, value, rel_tol=rel_tol, abs_tol=abs_tol):
                return True
        return False
    if isinstance(value, str):
        try:
            expected = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            expected = None
        if expected is not None and expected.tzinfo is not None:
            pattern = r"(\d{4}-\d{2}-\d{2})[T\s]+(\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)\s*(UTC(?:\s*[+-]\d{2}:\d{2})?|Z|[+-]\d{2}:\d{2})(?![\w:+-])"
            for match in re.finditer(pattern, text, re.I):
                zone = match[3].upper().replace(" ", "").replace("UTC", "")
                zone = "+00:00" if zone in {"", "Z"} else zone
                try:
                    actual = datetime.fromisoformat(match[1] + "T" + match[2] + zone)
                except ValueError:
                    continue
                if abs((actual.astimezone(timezone.utc) - expected.astimezone(timezone.utc)).total_seconds()) <= datetime_tolerance_seconds:
                    return True
            return False
    return normalize(value).casefold() in text.casefold()
