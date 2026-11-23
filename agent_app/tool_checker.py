"""Run selected TOOL_CHECKER questions against real PyBRAMS and/or Ornith.

Examples:
  python -m agent_app.tool_checker --card A02
  python -m agent_app.tool_checker --card A11 --chat --output psd-check.json
No synthetic fallback is used. A10/A11 download at most one PNG/WAV per run.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from .brams_tools import list_receivers, get_beacon, get_recordings, get_psd, get_spectrogram
from .config import load_settings
from .service import Session, ask
from .tools import convert_local_time


CASES = {
    "A02": ("list_receivers", {"search": "BEHUMA", "detail": "coordinates"}),
    "A03": ("get_beacon", {"emitter_code": "BEDOUR"}),
    "A07": ("get_recordings", {"search": "BEHUMA", "start_utc": "2023-12-14T21:15:00Z", "end_utc": "2023-12-14T21:45:00Z"}),
    "A10": ("get_spectrogram", {"search": "BEHUMA", "start_utc": "2023-12-14T21:15:00Z", "end_utc": "2023-12-14T21:20:00Z"}),
    "A11": ("get_psd", {"search": "BEHUMA", "start_utc": "2023-12-14T21:15:00Z", "end_utc": "2023-12-14T21:20:00Z"}),
    "D07": ("convert_local_time", {"local_datetime": "2023-12-15T23:30:00", "timezone_name": "Europe/Brussels"}),
}
FUNCTIONS = {function.__name__: function for function in (
    list_receivers, get_beacon, get_recordings, get_psd, get_spectrogram, convert_local_time)}


def questions(path: Path) -> dict[str, str]:
    content = path.read_text()
    cards = re.split(r"(?m)^### ([A-F]\d\d)[^\n]*\n", content)
    result = {}
    for index in range(1, len(cards), 2):
        match = re.search(r'\*\*(?:Prompt|Turn [12]):\*\*\s*"([^"\n]+)"', cards[index+1])
        if match:
            result[cards[index]] = match[1]
    return result


def evaluate(card: str, result: dict) -> dict:
    """Snapshot comparisons are explicit, never substituted for actual results."""
    checks = {"successful_tool_result": result.get("status", "ok") == "ok"}
    if not checks["successful_tool_result"]:
        return checks
    if card == "A02":
        rows = result.get("receivers", [])
        coordinate = rows[0].get("coordinates", {}) if rows else {}
        for key, expected in [("latitude_degrees", 50.192434), ("longitude_degrees", 5.254884), ("altitude_m", 294)]:
            checks[key] = math.isclose(coordinate.get(key, float("nan")), expected, abs_tol=1e-6)
        checks["location_provenance"] = any(row.get("coordinates_source_url") for row in rows)
    elif card == "A03":
        checks["frequency_hz"] = result.get("frequency_hz") == 49970000
        checks["wavelength_m"] = math.isclose(result.get("wavelength_m", 0), 5.9994488293, rel_tol=1e-6)
    elif card == "A07":
        checks["snapshot_count"] = result.get("count") == 7
        checks["boundary_label"] = any(row.get("begins_before_requested_window") for row in result.get("recordings", []))
    elif card == "A10":
        checks["requested_slot"] = result.get("filename", "").endswith("2115_BEHUMA_SYS001.png")
        checks["png_downloaded"] = result.get("downloaded") is True
        checks["archive_png_source"] = bool(result.get("image_source_url"))
    elif card == "A11":
        checks["snapshot_noise_psd_1pct_tolerance"] = math.isclose(result.get("noise_psd", 0), 2.368e-6, rel_tol=.01)
        checks["snapshot_calibrator_psd_1pct_tolerance"] = math.isclose(result.get("calibrator_psd", 0), 3.485e-4, rel_tol=.01)
        checks["noise_band_hz"] = result.get("noise_band_hz") == [800, 900]
        checks["analysis_performed"] = result.get("analysis_performed") is True
        checks["original_source"] = bool(result.get("wav_source_url"))
    elif card == "D07":
        checks["corrected_utc_conversion"] = result.get("utc_time") == "2023-12-15T22:30:00Z"
    return checks


async def run(args) -> dict:
    name, arguments = CASES[args.card]
    prompt = questions(Path(__file__).resolve().parent.parent / "TOOL_CHECKER.md")[args.card]
    report = {"card": args.card, "prompt": prompt, "mode": "live_ornith" if args.chat else "live_python_tool",
              "note": "Checks compare actual tool results with the document snapshot (D07 corrected). They do not certify all checker cards or model explanations."}
    if args.chat:
        reply = await ask(prompt, Session(), load_settings())
        report.update(answer=reply.text, trace=reply.trace, model_requests=reply.requests, warning=reply.warning)
        events = [event for event in reply.trace if event.get("event") == "tool_result" and event.get("name") == name]
        checks = {"expected_tool_executed": bool(events)}
        if events:
            result = events[-1]["result"]
            report["result"] = result
            checks.update(evaluate(args.card, result))
            if args.card == "D07":
                checks["recordings_queried_at_correct_utc"] = any(
                    event.get("event") == "tool_call" and event.get("name") in {"get_recordings", "check_availability", "get_wav"}
                    and event.get("arguments", {}).get("start_utc", "").startswith("2023-12-15T22:30") for event in reply.trace)
    else:
        arguments = dict(arguments)
        for key in ("start_utc", "end_utc"):
            if key in arguments:
                arguments[key] = datetime.fromisoformat(arguments[key].replace("Z", "+00:00"))
        function = FUNCTIONS[name]
        result = function(**arguments) if name == "convert_local_time" else await function(**arguments)
        report.update(result=result, tool=name)
        checks = evaluate(args.card, result)
    report.update(checks=checks, passed=all(checks.values()),
                  checked_at_utc=datetime.now(timezone.utc).isoformat())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", choices=CASES, default="A02")
    parser.add_argument("--chat", action="store_true", help="Send the document's exact question through real Ornith")
    parser.add_argument("--output", type=Path, help="Write an auditable JSON report (also printed)")
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args))
    except Exception as error:
        report = {"card": args.card, "passed": False, "error_type": type(error).__name__,
                  "message": str(error), "trace": getattr(error, "tool_trace", [])}
    encoded = json.dumps(report, indent=2, default=str)
    print(encoded)
    if args.output:
        args.output.write_text(encoded+"\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
