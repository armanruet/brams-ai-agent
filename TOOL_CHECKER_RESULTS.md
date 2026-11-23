# Tool checker review — 29 October 2026

> **Historical review:** The measurements and test counts below describe the
> application on the review date. The current app also includes local candidate
> meteor detection and Belgian geographic station queries; this old review has
> not been rerun to validate those additions or the present live archive.

The core metadata and PSD gaps are implemented. Automated coverage below is
separate from a fresh archive/Ornith run; this is not a claim that all 54 operator
cards pass against the live model.

## Implemented changes

| Capability | Tool behavior and chat output |
|---|---|
| Site coordinates | `list_receivers(search="BEHUMA", detail="coordinates")`: latitude/longitude in degrees, altitude in metres, location-registry provenance/link. One location lookup per site. Site coordinates remain distinct from WAV antenna coordinates. |
| Inventory | Receiver/location totals, multi-receiver site counts, at most 20 returned rows and explicit pagination. Catalog entries do not imply uptime. |
| Beacon metadata | `get_beacon`: frequency in Hz/MHz, power, polarization, wavelength in metres and registry coordinates. Optional receiver distance calculated with haversine, avoiding SDK ECEF subtraction/cache issues. |
| Recording metadata | `get_recordings`: filenames, nominal slots, precise UTC bounds, duration in seconds, sample rate, byte size, PPS count and links. Full count with bounded pages; no WAV download. Uses the SDK metadata endpoint/payload directly to avoid constructing hundreds of File objects. Preceding boundary files remain labelled. Hardware inference is identified as an inference. |
| Original WAV header | Actual byte size, SHA-256, precise BRA1 sample rate, LO/beacon frequencies and nominal difference, start microseconds, station code, PPS count, decoded configuration/state and antenna/beacon coordinates. Zero/undocumented GPS state is not declared healthy. |
| Noise/calibrator PSD | `get_psd` downloads one original WAV and runs PyBRAMS signal processing and `Series.psd`. Defaults: noise 800–900 Hz; calibrator ±3 Hz around the estimated peak. Returns actual measurements, bands, precise rate/duration, FFT resolution, units, method and original source. No listening-preview normalization or meteor classification. |
| Original reuse | Python stores a provenance manifest with successful WAV downloads. PSD can reuse the opaque artifact ID with its exact receiver ID and overlapping bounds, without downloading again. Results distinguish fresh downloads from local original reuse. Older artifacts lacking the new manifest need downloading again. |
| Exact selection | File tools prefer the requested five-minute slot over a preceding file with a slight end overlap. Exact site codes select SYS001 for recordings; site listings still include all antennas. Ambiguous place names still require a receiver choice. |
| Local time | `convert_local_time` uses Python zoneinfo. DST gaps and ambiguous repeated times require clarification. |
| Gradio output | Python appends “Verified tool results” for scientific/coordinate/time queries, with readable tables, units and citations. The same evidence renderer supports token/request-limit recovery. New example questions are included. |

At the time of this review there were ten registered tools: the existing six plus
beacon metadata, recording metadata, PSD and local-time conversion. There are
now twelve, including meteor detection and Belgian geographic querying. Existing repeat-call guards,
per-message limits, conversation history and media attachments remain active.

## Verification completed

**74 tests pass: 57 existing + 17 new.** Run:

```bash
.venv-brams/bin/python -m unittest tests.test_prototype tests.test_receivers tests.test_availability tests.test_brams_sdk tests.test_spectrogram tests.test_spectrogram_sdk tests.test_wav tests.test_wav_sdk tests.test_recovery tests.test_tool_checker -q
```

The new tests cover implementation behavior from A01–A03, A06–A15, B06,
C04–C06/C08, D04/D07/D08 and bounded calls from E01/E04. This does **not** mark
entire operator cards as live PASS. Some also require live snapshot agreement,
model choices or multi-turn behavior. All 54 document questions can be extracted,
including the multi-turn headings. Protocol tests use explicitly synthetic model
responses to exercise actual Pydantic tool execution and evidence rendering.

The scientific tests create a real BRA1/BRA2 WAV with two known tones, decode it
with installed PyBRAMS 0.1.7, and measure its PSD. Assertions verify expected band
powers, byte-for-byte original preservation, checksum, fractional sample rate,
header provenance, exact-slot selection and download reuse. Halving original
amplitude gives one-quarter PSD; analyzing the normalized player preview would
fail that check. The document's reference PSD is never substituted as a result.

A separate read-only check used an existing local Kirchberg original WAV:
`.artifacts/wav/43305b91e0cb4790932efc9cafee8ec6/original.wav`.
PyBRAMS decoded its BRA1 header and processed its signal: 6048 Hz, 300 seconds,
noise PSD `1.4678386989019773e-10` in 800–900 Hz, calibrator estimate 1500 Hz,
calibrator PSD `2.9372591524099416e-08` in 1497–1503 Hz. These measurements
describe that existing file, **not** a fresh archive query or the Humain reference.

Shell connections to localhost 8080/7860 and socket binding are blocked with
`Operation not permitted`. Consequently the socket-bound Gradio integration
test was excluded. Native Chrome could open Gradio, but the observed running
process still showed the old capability text. A restart was requested before
updated live-chat verification. No fresh Ornith/BRAMS success is claimed here.

## Live checks to run after restart

Use the new Gradio examples or run:

```bash
source .venv-brams/bin/activate
python -m agent_app.tool_checker --card A02
python -m agent_app.tool_checker --card A03
python -m agent_app.tool_checker --card A07
python -m agent_app.tool_checker --card A10
python -m agent_app.tool_checker --card A11
python -m agent_app.tool_checker --card D07
```

Add `--chat` to send the document's exact question through real Ornith/Pydantic AI.
Add `--output result.json` to retain the answer, actual trace and comparison checks.
A10 downloads at most one PNG; A11 downloads at most one WAV. D07 direct mode
checks conversion only; chat mode also requires a query at the correct UTC time.
Snapshot mismatches or unavailable data fail checks instead of supplying fake
replacement values. Review the model explanation as well as the machine checks.

Useful manual PSD follow-ups: “Which is the noise floor versus the calibrator
PSD?”, “What are the units?”, and “Did you download it or reuse the original?”
Compare receivers only after measuring both in an explicit date/window.

## Corrected reference issues and remaining limits

- D07: 15 December 2023 23:30 Brussels is **15 December 22:30 UTC**, without a
  day rollover. The previous gold answer was wrong.
- A13: 49,970,000 − 49,969,000 = **1000 Hz**. A measured peak near 1005 Hz is a
  separate observation, not that arithmetic difference.
- D01: a 30-minute empty query cannot establish an empty whole night.
- `../api_catalog/samples/` and `../QUESTIONS.md` are absent from this workspace.
  Their purported reference payloads were not recreated or treated as verified.
- Installed File.availability does not perform its intended datetime rounding.
  The wrapper explicitly expands sub-hour requests to whole hours and returns
  both requested and queried bounds. A16/C07's unexpanded-window fractions are
  not directly comparable. Exact partial-hour coverage stays unknown.
- Scientific duration/PSD use the precise BRA1 rate, potentially 5512.5 Hz even
  when the WAV/player container stores an integer rate. PSD units are normalized
  ADC amplitude²/Hz, not calibrated W/Hz or dBm. Original files remain unchanged.

At the time of this review, echo detection, cleaning and merging (B01/B02/B04), optical matching (B07),
aircraft-position queries/visual image interpretation (B09), dataset export
(B10), clock-drift analysis (B11), and measured meteor classification (part of
B08) were unimplemented. Candidate meteor echo detection has since been added;
the historical review and its test counts do not validate that integration.
Cleaning, merging, optical matching, ADS-B analysis, dataset export, clock-drift
analysis and confirmed shower attribution remain outside the described tools.
They are not marked as passing based on PSD, availability or image display alone.
Multi-turn behavior and bulk-request restraint still need live-model review.
