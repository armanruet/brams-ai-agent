"""Pure BRAMS input checks shared by live execution and benchmark replay."""

import math
from pydantic_ai import ModelRetry
from .tools import inspect_time_window


def validate_tool_arguments(name, values):
    if name == "list_receivers":
        if not 1 <= values["limit"] <= 20 or values["offset"] < 0:
            raise ModelRetry("Use limit between 1 and 20, and offset >= 0.")
        return {}
    if name not in {"check_availability", "get_spectrogram", "get_wav", "get_recordings", "get_psd"}:
        return {}
    window = inspect_time_window(values["start_utc"], values["end_utc"])
    if name == "check_availability":
        if window["duration_hours"] > 168:
            raise ModelRetry("Query at most 7 days at a time; ask for a shorter interval.")
        return window
    if name == "get_spectrogram":
        if not values["search"].strip():
            raise ModelRetry("Specify a receiver name or ID; reuse the previous receiver if known.")
        if window["duration_hours"] > 24:
            raise ModelRetry("Use at most 24 hours to select one spectrogram; ask for a shorter range.")
    elif name == "get_wav":
        if not values["search"].strip():
            raise ModelRetry("Choose a receiver name or ID before downloading audio.")
        if window["duration_hours"] > 24:
            raise ModelRetry("Use at most 24 hours to select one WAV recording.")
    elif name == "get_recordings":
        if (not values["search"].strip() or window["duration_hours"] > 24
                or not 1 <= values["limit"] <= 20 or values["offset"] < 0):
            raise ModelRetry("Supply a receiver and at most 24 hours, limit 1–20, offset >= 0.")
    elif name == "get_psd":
        low, high, width = (values[key] for key in ("noise_low_hz", "noise_high_hz", "calibrator_half_width_hz"))
        if (not values["search"].strip() or window["duration_hours"] > 24
                or not all(math.isfinite(value) for value in (low, high, width))
                or not 0 <= low < high or not 0 < width <= 100):
            raise ModelRetry("Specify one receiver, at most 24 hours, valid ordered noise frequencies and calibrator half-width 0–100 Hz.")
    return window
