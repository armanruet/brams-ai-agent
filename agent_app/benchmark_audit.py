"""Conservative claim checks for review triage, never a complete answer grader."""

import re
from .benchmark_values import normalize


def scientific_flags(text, results, turn):
    text = normalize(text)
    values = [row["result"] for row in results if isinstance(row.get("result"), dict)
              and row["result"].get("status", "ok") == "ok"]
    flags = []
    counts = {value["sample_count"] for value in values if value.get("sample_count") is not None}
    for match in re.finditer(r"(?<![\w.])(\d+)\s+samples\b", text, re.I):
        if counts and int(match[1]) not in counts and not re.search(r"\b(not|incorrect|instead of)\b", text[max(0, match.start()-20):match.start()], re.I):
            flags.append({"code": "sample_count_mismatch", "claim": match[0], "expected": sorted(counts)})
    offsets = {(value.get("header_summary") or {}).get("nominal_beacon_minus_lo_hz") for value in values}
    offsets.discard(None)
    for match in re.finditer(r"nominal\s+(-?\d+(?:\.\d+)?)\s*(kHz|Hz)\b", text, re.I):
        number = float(match[1]) * (1000 if match[2].lower() == "khz" else 1)
        if offsets and number not in offsets:
            flags.append({"code": "nominal_offset_mismatch", "claim": match[0], "expected_hz": sorted(offsets)})
    coverage = {row["receiver_id"]: row["coverage"] for value in values
                if value.get("scope") == "hourly_recording_availability"
                for row in value.get("receivers", []) if "coverage" in row}
    section = None
    for line in text.splitlines():
        heading = re.sub(r"^[#\s]+", "", line).strip().lower()
        if not line.lstrip().startswith("|"):
            if heading.startswith(("unknown coverage", "unknown (", "unknown(")):
                section = "unknown"
            elif heading.startswith(("no coverage", "no recordings")):
                section = "none"
            elif heading.startswith(("full coverage", "receivers with recordings")):
                section = "full"
            continue
        explicit = {cell.strip().lower() for cell in line.split("|")} & {"full", "none", "partial", "unknown"}
        claimed = next(iter(explicit)) if len(explicit) == 1 else section
        for receiver in re.findall(r"\b[A-Z]{6}_SYS\d{3}\b", line):
            if claimed and receiver in coverage and claimed != coverage[receiver]:
                flags.append({"code": "receiver_coverage_mismatch", "receiver_id": receiver,
                              "claimed": claimed, "expected": coverage[receiver]})
    if "delet" in turn.get("prompt", "").lower() and re.search(r"\bI(?: have|'ve)?\s+(?:successfully\s+)?deleted\b", text, re.I):
        flags.append({"code": "unsupported_deletion_claim", "expected": "No deletion tool is implemented."})
    return flags
