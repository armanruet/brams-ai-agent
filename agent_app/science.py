"""Bounded registry, archive metadata and scientific measurements.

All values come from the SDK or the unchanged original WAV, never the player
preview. No cleaned-file cache is touched and no bulk-day downloader is used.
"""

from datetime import datetime, timezone
import math
import hashlib
from pathlib import Path

from .brams_sdk import _catalog_from_sdk_with_meta, _fresh_sdk, _utc_string, recording_receivers


def coordinates(location) -> dict:
    lon, lat, alt = location.coordinates.to_dict()["Geodetic"]
    return {"longitude_degrees": float(lon), "latitude_degrees": float(lat),
            "altitude_m": float(alt), "provenance": "location_registry",
            "note": "Site coordinates; WAV antenna coordinates can differ. SDK converts registry altitude from km to m."}


def receiver_coordinates(rows: list[dict]) -> list[dict]:
    from pybrams import Location

    with _fresh_sdk() as (_, api, Config):
        locations = {}
        for row in rows:
            code = row["location_code"]
            if code not in locations:
                locations[code] = coordinates(Location.get(code))
            row["coordinates"] = locations[code]
            row["coordinates_source_url"] = (api.base_url + Config.get("pybrams.brams.location", "api_endpoint")
                                             + "?location_code=" + code)
    return rows


def _load_beacon(code: str, receiver_search: str) -> dict:
    from pybrams.brams.emitter import Emitter

    with _fresh_sdk() as (Receiver, api, Config):
        source = api.base_url + Config.get("pybrams.brams.emitter", "api_endpoint")
        try:
            emitter = Emitter.get(code, flatten=True)
        except ValueError as error:
            if not str(error).startswith("Invalid "):
                raise
            return {"status": "emitter_not_found", "source_url": source,
                    "message": f"No emitter matched {code}."}
        position = coordinates(emitter.location)
        frequency = float(emitter.frequency)
        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError("Invalid registry transmitter frequency")
        result = {"status": "ok", "emitter_code": emitter.emitter_code,
                  "name": emitter.name, "frequency_hz": frequency,
                  "frequency_mhz": frequency / 1e6, "wavelength_m": 299792458 / frequency,
                  "power_w": emitter.power, "polarization": emitter.polarization,
                  "coordinates": position, "source_url": source,
                  "coordinates_source_url": api.base_url + Config.get("pybrams.brams.location", "api_endpoint")
                  + "?location_code=" + emitter.location_code}
        if receiver_search:
            catalog, _, _ = _catalog_from_sdk_with_meta(Receiver, api, Config, sdk_context=_fresh_sdk)
            rows = recording_receivers(catalog, receiver_search)
            if len(rows) != 1:
                result["distance_status"] = "receiver_ambiguous" if rows else "receiver_not_found"
                result["receivers"] = rows[:20]
            else:
                from pybrams import Location
                receiver_position = coordinates(Location.get(rows[0]["location_code"]))
                a, b = math.radians(position["latitude_degrees"]), math.radians(receiver_position["latitude_degrees"])
                delta_lon = math.radians(position["longitude_degrees"] - receiver_position["longitude_degrees"])
                h = math.sin((b-a)/2)**2 + math.cos(a)*math.cos(b)*math.sin(delta_lon/2)**2
                result.update(receiver_id=rows[0]["receiver_id"], receiver_coordinates=receiver_position,
                              distance_km=6371.0088 * 2 * math.asin(math.sqrt(min(1, max(0, h)))),
                              distance_method="surface great-circle haversine, mean Earth radius 6371.0088 km; altitude ignored")
        return result


def _load_recordings(search, start, end, limit, offset):
    """Use the SDK's metadata endpoint directly to avoid constructing hundreds
    of File objects (each construction also resolves receiver/location metadata).
    This is the same payload used by File.get(load=False), without WAV loading.
    """
    with _fresh_sdk() as (Receiver, api, Config):
        catalog, _, _ = _catalog_from_sdk_with_meta(Receiver, api, Config, sdk_context=_fresh_sdk)
        rows = recording_receivers(catalog, search)
        source = api.base_url + Config.get("pybrams.brams.file", "api_endpoint")
        base = {"source_url": source, "downloaded": False, "receivers": rows[:20]}
        if len(rows) != 1:
            return {**base, "status": "receiver_ambiguous" if rows else "receiver_not_found",
                    "message": "Choose one receiver ID." if rows else "No matching receiver in the catalog."}
        receiver_id = rows[0]["receiver_id"]
        payload = api.request(Config.get("pybrams.brams.file", "api_endpoint"),
                              {"from": start, "to": end, "receiver_code[]": [receiver_id]})
        if payload is None:
            raise RuntimeError("The archive did not return a valid file listing.")
        if not isinstance(payload, list):
            raise ValueError("Unexpected recording metadata response")
        recordings = []
        for entry in payload:
            if entry.get("receiver_code") != receiver_id:
                continue
            precise_start, precise_end = int(entry["precise_start"]), int(entry["precise_end"])
            rate = float(entry.get("samplerate", entry.get("sample_rate")))
            slot = datetime(int(entry["year"]), int(entry["month"]), int(entry["day"]),
                            int(entry["hours"]), int(entry["minutes"]), tzinfo=timezone.utc)
            recordings.append({"filename": entry["wav_name"], "receiver_id": receiver_id,
                               "slot_start_utc": _utc_string(slot),
                               "recording_start_utc": _utc_string(datetime.fromtimestamp(precise_start/1e6, timezone.utc)),
                               "recording_end_utc": _utc_string(datetime.fromtimestamp(precise_end/1e6, timezone.utc)),
                               "duration_seconds": (precise_end-precise_start)/1e6,
                               "sample_rate_hz": rate, "size_bytes": int(entry["size"]),
                               "pps_count": entry.get("pps_count"),
                               "wav_source_url": entry.get("wav_url"), "image_source_url": entry.get("png_url"),
                               "begins_before_requested_window": precise_start < start.timestamp()*1e6,
                               "hardware_inferred": "AR" if "BEHUMA" in receiver_id else "RSP2" if rate == 6048 else "ICOM",
                               "hardware_provenance": "SDK inference from receiver code/sample rate; not a decoded hardware header"})
        recordings.sort(key=lambda row: (row["slot_start_utc"], row["filename"]))
        return {**base, "status": "ok", "receiver_id": receiver_id, "count": len(recordings),
                "recordings": recordings[offset:offset+limit], "offset": offset, "limit": limit,
                "has_more": offset+limit < len(recordings),
                "first_filename": recordings[0]["filename"] if recordings else None,
                "last_filename": recordings[-1]["filename"] if recordings else None,
                "note": "API listing may include the preceding boundary slot. Counts describe returned metadata; no WAV downloaded."}


def wav_header(path: Path) -> dict:
    from pybrams.brams.wavfile import WavFile

    # A regular WAV remains playable without a BRA1 header. Missing metadata is
    # explicit; scientific BRAMS fields are never filled from registry guesses.
    decoded = WavFile.from_file(path, series=False, pps=False)
    metadata = decoded.metadata
    if metadata is None:
        return {"header_summary": None, "header_status": "no_brams_header"}
    beacon_frequency = float(metadata.beacon_freq)
    lo_frequency = float(metadata.lo_freq)
    return {"header_status": "ok", "header_summary": {
        "samplerate_hz": float(metadata.samplerate), "lo_frequency_hz": lo_frequency,
        "beacon_frequency_hz": beacon_frequency, "nominal_beacon_minus_lo_hz": beacon_frequency-lo_frequency,
        "start_us": int(metadata.start_us), "station_code": metadata.station_code,
        "pps_count": int(metadata.pps_count), "configuration": metadata.decode_configuration(),
        "state": metadata.decode_state(), "state_raw": metadata.state,
        "antenna_coordinates": {"longitude_degrees": metadata.ant_long,
                                "latitude_degrees": metadata.ant_lat, "altitude_m": metadata.ant_alt},
        "beacon_coordinates": {"longitude_degrees": metadata.beacon_long,
                               "latitude_degrees": metadata.beacon_lat, "altitude_m": metadata.beacon_alt},
        "provenance": "BRA1 header of original downloaded WAV; distinct from location registry",
        "gps_note": "A zero/undocumented state is not evidence of a healthy GPS fix."}}


def _load_psd(search, start, end, noise_low_hz, noise_high_hz, calibrator_half_width_hz, artifact_id=None):
    from pybrams.brams.wavfile import WavFile
    from pybrams.signal.processing.signal import Signal
    from .artifacts import wav_path
    from .brams_sdk import _load_wav

    if artifact_id:
        path = wav_path(artifact_id)
        if path is None:
            return {"status": "artifact_not_found", "message": "The original WAV artifact is unavailable; download it again."}
        # Sidecar is generated by Python, never supplied by the model.
        import json
        manifest = path.parent / "source.json"
        if not manifest.is_file():
            return {"status": "artifact_not_found", "message": "No verified recording provenance is stored for this artifact."}
        source = json.loads(manifest.read_text())
        if (path.stat().st_size > 64*1024*1024
                or hashlib.sha256(path.read_bytes()).hexdigest() != source.get("sha256")):
            return {"status": "artifact_changed", "message": "The local original differs from its verified download checksum; download it again before analysis."}
        source.update(downloaded=False, reused_original_artifact=True,
                      input_origin="reused local original from an earlier verified download")
        if source["receiver_id"].casefold() != search.strip().casefold():
            return {"status": "artifact_mismatch", "message": "For artifact reuse pass its exact receiver_id and recording bounds."}
        actual_start = datetime.fromisoformat(source["recording_start_utc"].replace("Z", "+00:00"))
        actual_end = datetime.fromisoformat(source["recording_end_utc"].replace("Z", "+00:00"))
        if actual_start >= end or actual_end <= start:
            return {"status": "artifact_mismatch", "message": "The artifact does not overlap the requested window."}
    else:
        source = _load_wav(search, start, end)
        if source.get("status") != "ok":
            return source
        path = wav_path(source["artifact_id"])
        source.update(reused_original_artifact=False, input_origin="fresh archive download; SDK caches bypassed")
    wav = WavFile.from_file(path)
    if wav.series is None or wav.metadata is None:
        return {**source, "status": "analysis_unavailable", "message": "Original WAV has no BRAMS header or signal; PSD not computed."}
    rate = float(wav.series.samplerate)
    if not 0 <= noise_low_hz < noise_high_hz <= rate/2:
        return {**source, "status": "invalid_frequency_band", "message": f"Noise band must lie within 0–{rate/2:g} Hz (Nyquist)."}
    signal = Signal(wav.series, wav.pps)
    # Exactly the signal processing called by File.process(), without rewriting
    # or deleting SDK caches. It estimates spectral frequencies and PPS only.
    signal.process(correct_pps=(rate == 6048))
    calibrator = float(signal.calibrator_frequency)
    low, high = calibrator-calibrator_half_width_hz, calibrator+calibrator_half_width_hz
    if not 0 <= low < high <= rate/2:
        return {**source, "status": "invalid_frequency_band", "message": "Calibrator band falls outside the recorded spectrum."}
    noise = float(signal.series.psd(noise_low_hz, noise_high_hz))
    calibrator_psd = float(signal.series.psd(low, high))
    if not all(math.isfinite(value) for value in (noise, calibrator_psd, calibrator)):
        raise ValueError("No finite spectral measurement")
    return {**source, "status": "ok", "analysis_performed": True,
            "noise_psd": noise, "noise_band_hz": [noise_low_hz, noise_high_hz],
            "calibrator_psd": calibrator_psd, "calibrator_frequency_hz": calibrator,
            "calibrator_band_hz": [low, high], "sample_rate_hz": rate,
            "sample_count": len(signal.series.data), "analyzed_duration_seconds": len(signal.series.data)/rate,
            "frequency_resolution_hz": rate/len(signal.series.data),
            "units": "normalized ADC amplitude squared per Hz (not calibrated W/Hz or dBm)",
            "method": "PyBRAMS Series.psd: mean(abs(Hann-windowed, amplitude-scaled one-sided FFT)**2 / 2) / (sample_rate/N); band lower inclusive, upper exclusive; PCM16 divided by 32768",
            "analysis_input": "unchanged original WAV; full recording, no peak normalization or cleaning",
            "note": "Noise/calibrator spectral measurements do not constitute meteor detection."}
