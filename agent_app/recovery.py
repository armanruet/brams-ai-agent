"""Render completed tool evidence when model generation cannot finish."""

from datetime import datetime
from pydantic_ai.messages import ModelRequest, ModelResponse, RetryPromptPart, ToolCallPart, ToolReturnPart


def complete_history(messages):
    """Drop an interrupted tail without leaving an unanswered tool call."""
    pending = set()
    safe_end = 0
    for index, message in enumerate(messages):
        if isinstance(message, ModelResponse):
            pending.update(part.tool_call_id for part in message.parts if isinstance(part, ToolCallPart))
        elif isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, (ToolReturnPart, RetryPromptPart)):
                    pending.discard(part.tool_call_id)
        if (not pending and getattr(message, "state", "complete") == "complete"
                and not (isinstance(message, ModelResponse) and message.finish_reason == "length")):
            safe_end = index + 1
    return list(messages[:safe_end])


def _format_recording_period(start, end):
    """Format actual UTC bounds without burying the useful time in microseconds."""
    try:
        begin = datetime.fromisoformat(start.replace("Z", "+00:00"))
        finish = datetime.fromisoformat(end.replace("Z", "+00:00"))
        if begin.utcoffset() is None or finish.utcoffset() is None:
            raise ValueError("Recording bounds must have a timezone")
        from datetime import timezone
        begin, finish = begin.astimezone(timezone.utc), finish.astimezone(timezone.utc)
        clock = "%H:%M:%S" if begin.second or finish.second else "%H:%M"
        first = f"{begin.day} {begin.strftime('%B %Y')}, {begin.strftime(clock)}"
        last = finish.strftime(clock) if begin.date() == finish.date() else f"{finish.day} {finish.strftime('%B %Y')}, {finish.strftime(clock)}"
        return f"{first}–{last} UTC"
    except (ValueError, TypeError, AttributeError):
        return f"{start or 'unknown start'}–{end or 'unknown end'} UTC"


def _receiver_label(data):
    name = data.get("receiver_name") or data.get("receiver_id") or "the selected station"
    receiver_id = data.get("receiver_id")
    return f"**{name}** (`{receiver_id}`)" if receiver_id and receiver_id != name else f"**{name}**"


def _format_utc_clock(timestamp):
    """Format an ISO timestamp as a concise HH:MM:SS UTC clock string."""
    try:
        from datetime import timezone
        dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).astimezone(timezone.utc)
        return dt.strftime("%H:%M:%S UTC")
    except (ValueError, TypeError, AttributeError):
        return str(timestamp or "unknown time")


def _format_slot_window(start, end):
    """Format a 5-minute slot as HH:MM–HH:MM UTC for compact inline lists."""
    try:
        from datetime import timezone
        begin = datetime.fromisoformat(str(start).replace("Z", "+00:00")).astimezone(timezone.utc)
        finish = datetime.fromisoformat(str(end).replace("Z", "+00:00")).astimezone(timezone.utc)
        clock = "%H:%M:%S" if begin.second or finish.second else "%H:%M"
        return f"{begin.strftime(clock)}–{finish.strftime(clock)} UTC"
    except (ValueError, TypeError, AttributeError):
        return _format_recording_period(start, end)


def _time_context_note(time_events=()):
    """Summarize any local-time conversions performed earlier in the same turn."""
    conversions = []
    for event in time_events or ():
        if event.get("name") != "convert_local_time" or not isinstance(event.get("result"), dict):
            continue
        res = event["result"]
        tz = res.get("timezone", "local time")
        if res.get("end_local_datetime") and res.get("end_utc"):
            conversions.append(
                f"`{res.get('local_datetime')}` → `{res.get('end_local_datetime')}` ({tz}) "
                f"to **{res.get('start_utc')} → {res.get('end_utc')}**"
            )
        elif res.get("local_datetime") and res.get("utc_time"):
            conversions.append(f"`{res['local_datetime']}` ({tz}) to **{res['utc_time']}**")
    if conversions:
        return "Time conversion (" + "; ".join(conversions) + ")."
    return None


def spectrogram_summary(data, *, time_events=()):
    period = _format_recording_period(data.get("recording_start_utc"), data.get("recording_end_utc"))
    opening = f"Here’s the archived spectrogram for {_receiver_label(data)}, recorded **{period}**."
    try:
        requested = float(data.get("duration_seconds", 0))
        start = datetime.fromisoformat(data["recording_start_utc"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(data["recording_end_utc"].replace("Z", "+00:00"))
        if requested > (end - start).total_seconds() + 2:
            opening += " This is one recording from the requested interval."
    except (KeyError, TypeError, ValueError):
        pass
    tz_note = _time_context_note(time_events)
    if tz_note:
        opening += f" {tz_note}"
    closing = (
        "This view doesn’t include meteor echo detection—ask if you’d like me to scan this "
        "recording for meteor echoes or fetch the WAV audio."
    )
    source = data.get("image_source_url") or data.get("source_url")
    if source:
        closing += f" · [View the original spectrogram]({source})"
    return f"{opening}\n\n{closing}"


def _format_hz(value) -> str:
    """Format a frequency in Hz without scientific notation."""
    if not isinstance(value, (int, float)):
        return str(value)
    if float(value).is_integer():
        return f"{int(value)} Hz"
    return f"{value:.3f}".rstrip("0").rstrip(".") + " Hz"


def wav_summary(data, *, include_header=False, time_events=()):
    """Render a conversational summary for a downloaded BRAMS WAV recording."""
    period = _format_recording_period(data.get("recording_start_utc"), data.get("recording_end_utc"))
    filename = data.get("filename") or "recording.wav"
    duration = data.get("duration_seconds")
    rate = data.get("sample_rate_hz")
    size_bytes = data.get("size_bytes")
    specs = []
    if isinstance(duration, (int, float)):
        specs.append(f"{duration:.2f} s" if duration % 1 else f"{int(duration)} s")
    if isinstance(rate, (int, float)):
        specs.append(_format_hz(rate))
    if isinstance(size_bytes, int) and size_bytes > 0:
        specs.append(f"{size_bytes / (1024 * 1024):.2f} MB ({size_bytes} bytes)" if size_bytes >= 1024 * 1024 else f"{size_bytes} bytes")
    spec_text = f" ({', '.join(specs)})" if specs else ""
    opening = (
        f"Here’s the archived WAV recording for {_receiver_label(data)}, recorded **{period}** "
        f"(`{filename}`{spec_text})."
    )
    tz_note = _time_context_note(time_events)
    if tz_note:
        opening += f" {tz_note}"
    preview_text = (
        "You can listen with the **normalized PCM16 preview player** below, or download the "
        "**unchanged original WAV** to retain the raw scientific samples and `BRA1` metadata."
    )
    if data.get("container_sample_rate_hz") and data.get("container_sample_rate_hz") != rate:
        preview_text += (
            f" The player uses the integer WAV-container rate ({_format_hz(data['container_sample_rate_hz'])}), "
            f"while scientific duration and PSD use the precise `BRA1` rate ({_format_hz(rate)})."
        )
    source = data.get("wav_source_url") or data.get("source_url")
    if source:
        preview_text += f" · [Archive source]({source})"
    paragraphs = [opening, preview_text]

    header = data.get("header_summary")
    if include_header and isinstance(header, dict):
        ant = header.get("antenna_coordinates", {})
        header_lines = [
            "**Decoded `BRA1` WAV Header & Provenance:**",
            f"- **Frequencies:** LO `{_format_hz(header.get('lo_frequency_hz'))}` · Beacon `{_format_hz(header.get('beacon_frequency_hz'))}` · Nominal offset (`beacon - LO`) `{_format_hz(header.get('nominal_beacon_minus_lo_hz'))}`",
            f"- **Timing & Station:** Station `{header.get('station_code')}` · Sample rate `{_format_hz(header.get('samplerate_hz'))}` · PPS count `{header.get('pps_count')}` · `start_us` `{header.get('start_us')}`",
            f"- **Header Antenna Coordinates:** latitude `{ant.get('latitude_degrees')}°`, longitude `{ant.get('longitude_degrees')}°`, altitude `{ant.get('altitude_m')} m` *(distinct from registry site coordinates; raw zero GPS state `{header.get('state_raw')}` does not establish a healthy fix)*",
        ]
        if data.get("sha256"):
            header_lines.append(f"- **Original SHA-256:** `{data['sha256']}`")
        paragraphs.append("\n".join(header_lines))
    return "\n\n".join(paragraphs)


def meteor_summary(data, *, time_events=()):
    """Render a conversational summary grounded in actual detector output."""
    count = data.get("total_meteor_count", 0)
    minimum = "At least " if data.get("detection_limit_reached") else ""
    verdict = f"**{minimum}{count} meteor echo{'es' if count != 1 else ''} detected**" if data.get("meteor_detected") else "**No meteor echoes detected**"
    period = _format_recording_period(data.get("recording_start_utc"), data.get("recording_end_utc"))
    inspected = data.get("inspected_slots_count", 1)
    available = data.get("available_slots_in_window", inspected)
    headline = f"{verdict} for {_receiver_label(data)} during **{period}**."
    if available is None:
        headline += f" I analyzed **{inspected} recording(s)**; availability for the rest of the window was not checked."
    elif inspected != 1 or available != 1:
        headline += f" I analyzed **{inspected} of {available} available recordings**."
    tz_note = _time_context_note(time_events)
    if tz_note:
        headline += f" {tz_note}"
    paragraphs = [headline]

    slots = data.get("slots") if isinstance(data.get("slots"), list) else []
    if data.get("meteor_detected") and slots:
        if len(slots) == 1:
            dets = slots[0].get("detections") if isinstance(slots[0].get("detections"), list) else []
            if dets:
                bullets = []
                for det in dets[:5]:
                    if not isinstance(det, dict):
                        continue
                    idx = det.get("echo_index", "?")
                    when = _format_utc_clock(det.get("estimated_echo_utc")) if det.get("estimated_echo_utc") else None
                    offset = det.get("time_offset_seconds")
                    offset_str = f" (+{offset:.1f} s)" if isinstance(offset, (int, float)) else ""
                    conf = det.get("confidence")
                    conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
                    tier = det.get("confidence_tier", "candidate")
                    time_part = f"at **{when}**{offset_str}" if when else f"offset {offset_str.strip()}"
                    bullets.append(f"- **Echo #{idx}:** {time_part} — **{tier}** confidence (`{conf_str}`)")
                additional = max(0, slots[0].get("meteor_count", len(dets)) - len(bullets))
                if additional:
                    image_note = " marked on the spectrogram below" if data.get("annotated_artifacts") else " detected"
                    bullets.append(f"- *…plus {additional} more echo(es){image_note}.*")
                if bullets:
                    paragraphs.append("\n".join(bullets))
        else:
            active_slots = [s for s in slots if isinstance(s, dict) and s.get("meteor_count", 0) > 0]
            if active_slots:
                bullets = []
                for slot in active_slots[:6]:
                    slot_window = _format_slot_window(slot.get("recording_start_utc"), slot.get("recording_end_utc"))
                    scount = slot.get("meteor_count", 0)
                    count_prefix = "at least " if slot.get("detection_limit_reached") else ""
                    hcount = slot.get("high_confidence_count", 0)
                    max_c = slot.get("max_confidence")
                    conf_note = f", max conf `{max_c:.2f}`" if isinstance(max_c, (int, float)) else ""
                    high_note = f" ({hcount} high-confidence)" if hcount else ""
                    bullets.append(
                        f"- **{slot_window}:** {count_prefix}{scount} echo{'es' if scount != 1 else ''}{high_note}{conf_note}"
                    )
                if len(active_slots) > 6:
                    bullets.append(f"- *…and {len(active_slots) - 6} more active 5-minute recording(s).*")
                paragraphs.append("Activity by 5-minute recording:\n" + "\n".join(bullets))

    notes = []
    if data.get("detection_limit_reached"):
        notes.append("The detector reached its output limit; the count may be incomplete.")
    if data.get("coverage_complete") is False or data.get("uninspected_slots_count", 0):
        notes.append("Coverage is incomplete, so this result applies only to the recordings analyzed.")
    if data.get("slot_errors"):
        notes.append(f"{len(data['slot_errors'])} recording(s) could not be analyzed.")
    if data.get("overlay_errors"):
        notes.append(f"The annotated image could not be created for {len(data['overlay_errors'])} analyzed recording(s).")
    threshold = data.get("confidence_threshold")
    threshold_text = f" (confidence threshold: {threshold:g})" if isinstance(threshold, (int, float)) else ""
    notes.append(f"This is an automated estimate{threshold_text}; false detections and missed echoes are possible.")
    source = data.get("image_source_url") or data.get("source_url")
    if source:
        notes.append(f"[BRAMS source]({source})")
    paragraphs.append(" ".join(notes))
    return "\n\n".join(paragraphs)


def geographic_summary(data, *, time_events=()):
    direction = data.get("direction", "")
    axis = data.get("axis", "latitude")
    comparison = "at or above" if data.get("comparison") == ">=" else "below"
    cutoff = data.get("cutoff_degrees")
    boundary = f"{cutoff:.7f}".rstrip("0").rstrip(".") + "°" if isinstance(cutoff, (int, float)) else "the selected cutoff"
    label = f"{direction}ern Belgium" if direction in {"north", "south"} else f"{direction}ern Belgium"
    action = data.get("action")
    if action == "spectrogram":
        lines = [spectrogram_summary(data, time_events=time_events)]
    elif action == "active_count":
        count = data.get("active_site_count", 0)
        receiver_count = data.get("active_receiver_count", 0)
        period = _format_recording_period(data.get("start_utc"), data.get("end_utc"))
        headline = (
            f"**{count} station{'s' if count != 1 else ''}** in {label} had a recording listed for **{period}** "
            f"({receiver_count} receiver{'s' if receiver_count != 1 else ''})."
        )
        tz_note = _time_context_note(time_events)
        if tz_note:
            headline += f" {tz_note}"
        lines = [headline]
        if data.get("unknown_activity_site_count"):
            lines.append(f"Activity could not be determined for {data['unknown_activity_site_count']} additional mapped station(s).")
    else:
        count, receivers = data.get("site_count", 0), data.get("receiver_count", 0)
        lines = [f"There {'are' if count != 1 else 'is'} **{count} station{'s' if count != 1 else ''}** in {label} "
                 f"in the BRAMS catalogue ({receivers} receiver{'s' if receivers != 1 else ''})."]
    kind = "For this answer, I used the geographic midpoint" if data.get("definition_type") != "user-specified" else "I used your specified boundary"
    lines.append(f"{kind}: stations inside Belgium with {axis} {comparison} {boundary}. "
                 "This is a geographic split, not an administrative boundary.")
    missing = data.get("missing_coordinate_sites", 0) + data.get("suspect_coordinate_sites", 0)
    if missing and action != "spectrogram":
        lines.append(f"{missing} site{'s' if missing != 1 else ''} across the network had missing or inconsistent coordinates and could not be classified; the count may be incomplete.")
    if action == "count":
        lines.append("The catalogue count does not indicate whether those stations were recording at a particular time.")
    source = data.get("catalog_source_url")
    geo_source = data.get("boundary_source_url")
    links = [f"[BRAMS catalogue]({source})" for source in [source] if source]
    if action == "active_count" and data.get("recording_source_url"):
        links.append(f"[BRAMS recordings]({data['recording_source_url']})")
    if geo_source:
        links.append(f"[Country boundary]({geo_source})")
    if links:
        lines.append(" · ".join(links))
    return "\n\n".join(lines)


def evidence_summary(trace, *, meteor_detail=False):
    results = []
    seen = set()
    for event in trace:
        if event.get("event") != "tool_result" or not isinstance(event.get("result"), dict):
            continue
        data = event["result"]
        name = event["name"]
        key = (name, str(data))
        if key in seen:
            continue
        seen.add(key)
        if data.get("status") not in {None, "ok"}:
            message = data.get("message") or data.get("note") or data["status"].replace("_", " ")
            if data.get("status") == "receiver_ambiguous":
                message += "\n" + "\n".join(
                    f"- {row.get('name', '')} (`{row.get('receiver_id', '')}`)"
                    for row in data.get("receivers", [])[:20])
            if name == "query_geographic_receivers":
                results.append(f"I couldn’t complete the geographic station lookup. {message}")
            else:
                results.append(f"{name}: {message}")
            continue
        if name == "check_availability":
            rows = data.get("receivers", [])
            lines = [f"Recording availability: **{data.get('start_utc')} → {data.get('end_utc')}** (UTC)."]
            if len(rows) > 1:
                matching = [row for row in rows if row.get("coverage") in {"full", "partial"}
                            or (data.get("requested_whole_hours") is True
                                and row.get("known_recordings_in_queried_hours") is True)]
                unknown = sum(row.get("coverage") == "unknown" for row in rows)
                qualifier = "At least " if unknown else ""
                lines.append(f"{qualifier}{len(matching)} of {len(rows)} receivers report recordings; {unknown} have unknown coverage.")
                rows = matching
            for row in rows[:20]:
                percentage = row.get("availability_percent")
                metric = f" ({percentage:g}%)" if isinstance(percentage, (int, float)) else ""
                lines.append(f"- {row.get('name', '')} (`{row.get('receiver_id', '')}`): {row.get('coverage', 'unknown')}{metric}")
                if row.get("known_recordings_in_queried_hours"):
                    lines.append("  Recordings are known in at least one queried hour; other hours are unknown.")
            if len(rows) > 20:
                lines.append(f"{len(rows) - 20} more receivers are listed in Tool activity.")
            lines.append("These are hourly archive metrics.")
            if data.get("requested_whole_hours") is False:
                lines.append(f"Query expanded to {data.get('queried_start_utc')} → {data.get('queried_end_utc')} UTC. Fractions refer to that expanded window, not exact requested sub-hour coverage.")
            if data.get("source_url"):
                lines.append(f"[BRAMS source]({data['source_url']})")
            results.append("\n".join(lines))
        elif name in {"get_wav", "get_spectrogram"}:
            kind = "WAV recording" if name == "get_wav" else "Spectrogram"
            if not data.get("downloaded"):
                results.append(f"No {kind.lower()} was downloaded.")
                continue
            if name == "get_spectrogram":
                results.append(spectrogram_summary(data))
                continue
            results.append(wav_summary(data, include_header=bool(data.get("header_summary"))))
        elif name == "list_receivers":
            rows = data.get("receivers", [])
            lines = [f"{data.get('count', len(rows))} matching receiver catalog entries across {data.get('location_count', '?')} sites:"]
            lines.extend(f"- {row.get('name', '')} (`{row.get('receiver_id', '')}`)" for row in rows[:20])
            sites_shown = set()
            for row in rows:
                coord = row.get("coordinates")
                if coord and row.get("location_code") not in sites_shown:
                    sites_shown.add(row.get("location_code"))
                    lines.append(f"Registry site `{row.get('location_code')}`: latitude **{coord['latitude_degrees']}°**, longitude **{coord['longitude_degrees']}°**, altitude **{coord['altitude_m']:g} m**. These are site coordinates, not WAV antenna coordinates.")
                    if row.get("coordinates_source_url"):
                        lines.append(f"[Location registry]({row['coordinates_source_url']})")
            if data.get("has_more"):
                lines.append(f"Showing {len(rows)} entries; more pages are available on request.")
            if data.get("source_url"):
                lines.append(f"[BRAMS catalog]({data['source_url']})")
            results.append("\n".join(lines))
        elif name == "get_beacon":
            coord = data["coordinates"]
            text = (f"**{data.get('name')} (`{data.get('emitter_code')}`)**: {data['frequency_hz']:g} Hz "
                    f"({data['frequency_mhz']:g} MHz), {data.get('power_w')} W, {data.get('polarization')}; "
                    f"wavelength {data['wavelength_m']:.6g} m.\n\n"
                    f"Registry: latitude {coord['latitude_degrees']}°, longitude {coord['longitude_degrees']}°, altitude {coord['altitude_m']:g} m.")
            if data.get("distance_km") is not None:
                text += f"\n\nDistance to `{data['receiver_id']}`: **{data['distance_km']:.3f} km**, {data['distance_method']}."
            if data.get("distance_status"):
                text += f"\n\nDistance unavailable: {data['distance_status']}; choose a valid unique receiver."
            text += f"\n\n[Emitter registry]({data['source_url']}) · [Location registry]({data['coordinates_source_url']})"
            results.append(text)
        elif name == "get_recordings":
            rows = data.get("recordings", [])
            lines = [f"**{data.get('count', 0)} recording metadata entries** for `{data.get('receiver_id')}`; {data.get('start_utc')} → {data.get('end_utc')} UTC. No WAV downloaded.",
                     "", "| File | Actual start (UTC) | Duration (s) | Sample rate (Hz) | Bytes |", "|---|---|---:|---:|---:|"]
            for row in rows[:10]:
                filename = row['filename'].replace('|', '\\|')
                label = " (boundary before window)" if row['begins_before_requested_window'] else ""
                lines.append(f"| `{filename}`{label} | {row['recording_start_utc']} | {row['duration_seconds']:.6f} | {row['sample_rate_hz']:g} | {row['size_bytes']} |")
            if data.get("has_more") or len(rows) > 10:
                lines.append("\nAdditional entries are available by pagination; this table shows at most 10.")
            lines.append(f"\n[Archive metadata]({data['source_url']})")
            results.append("\n".join(lines))
        elif name == "get_psd":
            text = (f"**PSD measured from `{data.get('filename')}`** (`{data.get('receiver_id')}`), "
                    f"{data.get('recording_start_utc')} → {data.get('recording_end_utc')} UTC.\n\n"
                    "| Measurement | Frequency band (Hz) | PSD |\n|---|---|---:|\n"
                    f"| Noise | {data['noise_band_hz'][0]:g}–{data['noise_band_hz'][1]:g} | {data['noise_psd']:.7g} |\n"
                    f"| Calibrator | {data['calibrator_band_hz'][0]:g}–{data['calibrator_band_hz'][1]:g} | {data['calibrator_psd']:.7g} |\n\n"
                    f"Units: **{data['units']}**. Estimated calibrator: {data['calibrator_frequency_hz']:.6g} Hz. "
                    f"Analyzed duration: {data['analyzed_duration_seconds']:.6f} s.\n\n"
                    f"Method: {data['method']}. Input: {data['analysis_input']}. "
                    f"Origin: {data.get('input_origin', 'see tool trace')}. This does not detect meteor echoes.\n\n"
                    f"[Original WAV]({data.get('wav_source_url', data.get('source_url', ''))})")
            results.append(text)
        elif name == "query_geographic_receivers":
            results.append(geographic_summary(data))
        elif name == "detect_meteors":
            if not meteor_detail:
                results.append(meteor_summary(data))
                continue
            verdict = (
                f"**{'At least ' if data.get('detection_limit_reached') else ''}{data.get('total_meteor_count', 0)} meteor echo(es) detected**"
                if data.get("meteor_detected")
                else "**No meteor echoes detected**"
            )
            inspected = data.get("inspected_slots_count", 1)
            available = data.get("available_slots_in_window", inspected)
            lines = [
                f"Meteor detection ({verdict}) for **{data.get('receiver_name') or data.get('receiver_id', '')}** "
                f"(`{data.get('receiver_id', '')}`), {data.get('recording_start_utc')} → {data.get('recording_end_utc')} UTC.",
                (f"Inspected **{inspected} of {available}** available 5-minute spectrogram slot(s) " if available is not None
                 else f"Inspected **{inspected}** 5-minute spectrogram slot(s) ")
                + f"with `{data.get('model_name', 'yolo11n_brams')}` ({data.get('backend', 'local')}, "
                f"conf >= {data.get('confidence_threshold', 0.25):g}).",
            ]
            if available is None:
                lines.append("Availability for the rest of the requested window was not checked.")
            if data.get("coverage_complete") is False:
                uncovered = data.get("uncovered_slot_starts_utc", [])
                lines.append(
                    f"Coverage is incomplete: {len(uncovered)} expected five-minute slot(s) "
                    "were missing, not inspected, or failed. The verdict applies only to analyzed recordings."
                )
            if data.get("slot_errors"):
                lines.append(f"{len(data['slot_errors'])} selected recording(s) could not be downloaded or analyzed.")
            if data.get("overlay_errors"):
                lines.append(f"Annotated images could not be created for {len(data['overlay_errors'])} analyzed recording(s); their detection results are retained.")
            if data.get("detection_limit_reached"):
                lines.append("The detector reached its output limit; the count may be incomplete.")
            slots = data.get("slots", [])
            if len(slots) == 1:
                slot = slots[0]
                if slot.get("filename"):
                    lines.append(f"File: `{slot['filename']}` ({slot.get('input_kind', 'spectrogram')}).")
                dets = slot.get("detections", [])
                if dets:
                    lines.extend([
                        "",
                        "| Echo # | Estimated UTC | Offset (s) | Confidence | Tier |",
                        "|---:|---|---:|---:|---|",
                    ])
                    for det in dets:
                        lines.append(
                            f"| {det['echo_index']} | {det.get('estimated_echo_utc', 'n/a')} | "
                            f"{det['time_offset_seconds']:.1f} | {det['confidence']:.3f} | {det['confidence_tier']} |"
                        )
                    if slot.get("detections_truncated"):
                        lines.append(f"\nRetained {len(dets)} of {slot.get('meteor_count', len(dets))} individual echo details.")
            elif len(slots) > 1:
                lines.extend([
                    "",
                    "| Slot start (UTC) | File | Meteors | High-conf | Max conf |",
                    "|---|---|---:|---:|---:|",
                ])
                for slot in slots:
                    fname = str(slot.get("filename", "")).replace("|", "\\|")
                    max_c = f"{slot['max_confidence']:.3f}" if isinstance(slot.get("max_confidence"), (int, float)) else "—"
                    lines.append(
                        f"| {slot.get('recording_start_utc')} | `{fname}` | {'≥ ' if slot.get('detection_limit_reached') else ''}{slot.get('meteor_count', 0)} | "
                        f"{slot.get('high_confidence_count', 0)} | {max_c} |"
                    )
            if data.get("caveat"):
                lines.append("\n" + str(data["caveat"]))
            source = data.get("image_source_url") or data.get("source_url")
            if source:
                lines.append(f"\n[Archive source]({source})")
            results.append("\n".join(lines))
        elif name == "convert_local_time":
            if data.get("end_local_datetime") and data.get("end_utc"):
                results.append(
                    f"{data.get('local_datetime')} → {data.get('end_local_datetime')} ({data.get('timezone')}) → "
                    f"**{data.get('start_utc')} → {data.get('end_utc')}**; Python zoneinfo."
                )
            else:
                results.append(f"{data.get('local_datetime')} ({data.get('timezone')}) → **{data.get('utc_time')}**; Python zoneinfo.")
        elif name == "get_current_utc_time":
            results.append(f"UTC time: {data.get('utc_time')} (local system clock).")
        elif name == "inspect_time_window":
            results.append(f"{data.get('start_utc')} → {data.get('end_utc')}: {data.get('duration_minutes')} minutes.")
    return "\n\n".join(results)
