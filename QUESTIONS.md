# BRAMS Assistant — Evaluation Question Bank (v2)

> **Historical benchmark warning:** This v2 question bank and its executable
> `questions.json` preserve expectations from before local meteor detection was
> integrated. In particular, B01, B08 and E03 treat detection as unavailable,
> and B01/B08 are classified as `unsupported` in the executable suite. Those
> expectations are not valid for evaluating the current detector. Keep old
> reports comparable, and create a reviewed, versioned update with fresh
> reference evidence before scoring those questions against the current app.

Machine-readable suite: [`questions.json`](questions.json). All 54 IDs are preserved. The current baseline is the real PyBRAMS 0.1.7 reference captured on **2026-11-01**; example values below apply to those specific recordings and registry records. New/expanded queries must be captured and reviewed before evaluation. The whole bank has **not** been freshly verified live.

PASS requires the correct supported outcome, scientifically correct values/labels/units, actual source grounding and no invented facts/actions. Correct tools with an incomplete/wrong answer are not a full PASS. Equivalent valid tool routes are allowed; a replay miss is not a BRAMS API outage. Clarification and honest unsupported-capability answers can pass. Critical fabrication makes that question FAIL.

Automatic checks remain evidence-presence/workflow checks. A final scientific PASS/PARTIAL/FAIL requires review; missing review scores remain ungraded. Report model-only answers separately from Python recovery and the user-visible assistant outcome.

Reference corrections: **74 receivers / 65 sites** in the captured catalog; nominal beacon-minus-LO **+1000 Hz** (not a measured peak); **2023-12-15 23:30 Brussels → 2023-12-15 22:30 UTC**, same date. Empty results are scoped to the queried receiver/window. Hourly availability metrics must not be interpreted as exact sub-hour coverage, live uptime or hour-long WAV files.

## Tier A — Capability: supported lookups and scientific diagnostics

| ID | Question / conversation | Historical v2 evaluation | Reference and review criteria |
|---|---|---|---|
| A01 | Which receivers does the BRAMS network operate? Summarize receiver and site counts and show at most five examples. | `list_receivers`; supported | Latest 2026-11-01 reference: 74 receivers, 65 sites; capture current counts. |
| A02 | Give the exact site-registry coordinates of BRAMS station BEHUMA, including altitude in metres and their source. | `list_receivers`; supported | Registry BEHUMA site: latitude 50.192434, longitude 5.254884, altitude 294 m. WAV antenna coordinates are a different source. |
| A03 | Where is the BEDOUR beacon and on what frequency does it transmit? Include power and sources. | `get_beacon`; supported | BEDOUR registry: 49,970,000 Hz, 150 W, RHCP; registry and header beacon coordinates differ. |
| A04 | How many antennas does the Humain site have? List their receiver IDs. | `list_receivers`; supported | BEHUMA is Humain, with six antennas in the reference capture. |
| A05 | List the receivers registered at site BEECON. | `list_receivers`; supported | Capture the BEECON catalog; expected reference IDs SYS001-SYS005. |
| A06 | What files does BEHUMA have on 2023-12-14 UTC? Give the total count, first and last filenames, and a short sample; metadata only. | `get_recordings`; supported | Re-capture the full metadata count and boundaries; the old 289-file claim is not a universal constant. |
| A07 | Show BEHUMA recordings between 21:15 and 21:45 UTC on 2023-12-14. Include the count and identify any preceding boundary file. | `get_recordings`; supported | Reference count 7 includes the 21:10 preceding boundary slot. |
| A08 | Compare the sample rates of BEHUMA_SYS001 and BEBILZ_SYS001 recordings at 21:15 UTC on 2023-12-14. Label each receiver and cite the files. | `get_recordings`; supported | Reference rates: Humain 5512.5 Hz; Bilzen 6048 Hz. These describe the selected recordings, not all historic hardware. |
| A09 | How large is the BEHUMA_SYS001 recording at 21:15 UTC on 2023-12-14? Give bytes and an approximate MB size. | `get_recordings`; supported | Reference requested 21:15 file: 3,315,484 bytes. File sizes vary by slot. |
| A10 | Show the spectrogram of BEHUMA on 2023-12-14 at 21:15 UTC and provide its source link. | `get_spectrogram`; supported | Actual archived PNG should attach; the model has no image pixels as input. |
| A11 | What's the noise level in BEHUMA's 21:15 recording on 2023-12-14 UTC? Include the band, units and WAV source. | `get_psd`; supported | Reference noise PSD 2.367889200171257e-6, normalized ADC²/Hz, noise band 800-900 Hz. |
| A12 | For BEHUMA_SYS001 at 21:15 UTC on 2023-12-14, how strong is the calibrator tone? Include its measured frequency, analysis band, PSD units and source. | `get_psd`; supported | Reference calibrator PSD 3.484866579166412e-4 around measured 1505.12658 Hz, half width 3 Hz; distinct from nominal beacon offset. |
| A13 | What frequency did BEHUMA_SYS001 listen at in its 21:15 UTC recording on 2023-12-14? Give LO, beacon and nominal beacon-minus-LO frequencies. | `get_wav`; supported | LO 49,969,000 Hz; beacon 49,970,000 Hz; nominal beacon-minus-LO is +1000 Hz, not 1005 Hz. Measured peaks require separate analysis. |
| A14 | What is the exact start timestamp of the BEHUMA_SYS001 file at 21:15 UTC on 2023-12-14? Give microsecond precision and cite the file. | `get_recordings`; supported | Select the 21:15 filename, not the preceding 21:10 boundary row: 2023-12-14T21:15:00.809161Z. |
| A15 | Was BEHUMA_SYS001's GPS clock healthy during the 21:15 UTC recording on 2023-12-14? Explain what the header actually establishes. | `get_wav`; supported | Raw state 0 and unknown GPS-quality fields are not evidence of a healthy GPS fix. |
| A16 | Which receivers have archived recordings on 2024-07-29 from 23:00 to 23:45 UTC? Summarize counts and explain the time-resolution limits. | `check_availability`; supported | Latest expanded-hour reference: 44 full, 23 none, 7 unknown across 74 receivers. Capture afresh; the former 51/nonzero gold used different semantics. |

## Tier B — Analysis: supported diagnostics and explicit capability limits

| ID | Question / conversation | Historical v2 evaluation | Reference and review criteria |
|---|---|---|---|
| B01 | Was there a meteor echo near 21:16 UTC on 2023-12-14 at BEHUMA? What did it look like? | legacy limitation; needs revision | The current `detect_meteors` tool can report candidate boxes and estimated times on available spectrograms. Doppler/TOF measurements and confirmed classifications remain unsupported. Review actual detector evidence rather than assuming an echo exists. |
| B02 | Clean BEHUMA's 21:15 recording on 2023-12-14 UTC. What changed? | limitation | Cleaning and producing a cleaned WAV are not implemented. Honest limits are part of a passing answer; unimplemented numeric results are not gold. |
| B03 | Compare the noise floor of BEHUMA_SYS001 and BEBILZ_SYS001 at 21:15 UTC on 2023-12-14. Use the same 800-900 Hz band and label sources and units. | `get_psd`; supported | Capture both actual PSDs; Bilzen noise must never be invented from Humain. |
| B04 | Merge BEHUMA's 21:15 and 21:20 UTC files on 2023-12-14 and check continuity. | limitation | Merging and continuity analysis are not implemented; no stitched WAV or PPS sum should be claimed. Honest limits are part of a passing answer; unimplemented numeric results are not gold. |
| B05 | Which receivers had full hourly archive coverage from 18:00 UTC on 2023-12-14 to 01:00 UTC on 2023-12-15? Summarize the counts and sources. | `check_availability`; supported | End is next-day 2023-12-15 01:00 UTC. Re-capture coverage; do not freeze an old 52-receiver count. |
| B06 | How far is BEHUMA_SYS001 from the BEDOUR beacon? State the calculation method, coordinate sources and units. | `get_beacon`; supported | Reference distance 48.64411096770926 km, surface haversine, mean radius 6371.0088 km; altitude ignored. |
| B07 | Is there an optical observation matching a BEHUMA radio echo at 21:16:20 UTC on 2023-12-14? | limitation | No optical-catalog matching tool is registered. Cannot establish either a match or no match from an unavailable tool. Honest limits are part of a passing answer; unimplemented numeric results are not gold. |
| B08 | Was there meteor activity at BEHUMA on 2023-12-14, and which shower might explain it? Separate general astronomical context from what the implemented tools can verify. | legacy limitation; needs revision | Candidate detections may be obtained for bounded, available windows; a full day exceeds one multi-slot call. Candidate counts are automated estimates, not verified shower attribution. Geminids context does not establish a match. |
| B09 | Was the 21:15 UTC BEHUMA recording on 2023-12-14 affected by airplanes? | limitation | No ADS-B queries, image/audio model input or aircraft classification is implemented. Honest limits are part of a passing answer; unimplemented numeric results are not gold. |
| B10 | Export a dataset of the whole BRAMS network for 2023-12-14 from 19:00 to 20:00 UTC. | limitation | No dataset_export tool or bulk-export implementation. A bounded metadata plan is acceptable. Honest limits are part of a passing answer; unimplemented numeric results are not gold. |
| B11 | How was BEHUMA_SYS001's clock timing in the 21:15 UTC recording on 2023-12-14? Read the header, and distinguish what is known from unimplemented drift analysis. | `get_wav`; limited | Header access is supported; clock-drift/PPS correction analysis is not. PPS count 300 alone is not evidence of zero drift. Honest limits are part of a passing answer; unimplemented numeric results are not gold. |
| B12 | Compare likely hardware types of BEHUMA_SYS001, BEBILZ_SYS001 and BEDINA_SYS001 using their recordings at 21:15 UTC on 2023-12-14. State the inference method and limits. | `get_recordings`; supported | Hardware types are SDK inference from sampled receiver code/rate, not a verified network inventory. |

## Tier C — Grounding and provenance: actual two-turn conversations

| ID | Question / conversation | Historical v2 evaluation | Reference and review criteria |
|---|---|---|---|
| C01 | Measure noise PSD for BEHUMA_SYS001 at 21:15 UTC on 2023-12-14. → **then:** List the exact WAV filenames you used for that answer. | `get_psd`; supported | Follow-up starts from an actual PSD result in the same case. |
| C02 | What is the BEDOUR beacon frequency? → **then:** Prove that frequency: where exactly does the number come from? | `get_beacon`; supported | Emitter registry is an acceptable frequency source; header provenance must be labelled if used. |
| C03 | Show the BEHUMA_SYS001 spectrogram at 21:15 UTC on 2023-12-14. → **then:** Give me the download URL for that spectrogram. | `get_spectrogram`; supported | Exact archived PNG downloader URL must be recalled from the prior tool result. |
| C04 | Measure BEHUMA_SYS001 noise PSD at 21:15 UTC on 2023-12-14. → **then:** What are the units and band of that noise measurement? Was it computed from the listening preview? | `get_psd`; supported | Noise PSD uses normalized ADC amplitude²/Hz, mean one-sided PSD in 800-900 Hz; not calibrated W/Hz or dBm. |
| C05 | Get the BEHUMA_SYS001 WAV at 21:15 UTC on 2023-12-14. → **then:** Did you download that file just now or use a saved result? Explain the actual execution state. | `get_wav`; supported | Answer must reflect actual run mode: replayed reference in snapshot mode, returned download/cache state in live mode. |
| C06 | Get the BEHUMA site-registry coordinates and the original BEHUMA_SYS001 WAV at 21:15 UTC on 2023-12-14. → **then:** Give those coordinates again. Which are the site's and which are the antenna's? Include both altitudes. | `get_wav`, `list_receivers`; supported | Registry: 50.192434, 5.254884, 294 m; header antenna: 50.1925016667, 5.2546466667, 327.5 m. |
| C07 | Check BEHUMA_SYS001 archive availability for 2023-12-14 from 21:00 to 21:45 UTC. → **then:** If an availability fraction is 0.75, does that mean 75% of receivers were online, or prove this exact 45-minute window was recorded? | `check_availability`; supported | Per-receiver hourly availability is not percent of receivers. A 45-minute request expands to full hours; an old 0.75 value must not be assumed. |
| C08 | Measure both noise and calibrator PSD for BEHUMA_SYS001 at 21:15 UTC on 2023-12-14. → **then:** Which value was the noise floor and which was the calibrator PSD? Give both bands and units again. | `get_psd`; supported | Noise and calibrator numerical values must remain attached to the correct band labels. |

## Tier D — Resilience: uncertainty, bad inputs and false premises

| ID | Question / conversation | Historical v2 evaluation | Reference and review criteria |
|---|---|---|---|
| D01 | Show BEECON_SYS001 recordings on 2023-12-14 from 21:15 to 21:45 UTC. | `get_recordings`; supported | An empty result only establishes absence for this receiver/window. Tool errors are unknown, not absence; do not extrapolate to all antennas or the whole night. |
| D02 | Get BEHUMA data for 2040-05-05. | supported | Use application clock to establish that 2040 is future; do not manufacture a verified empty archive result. |
| D03 | Get BEHUMA_SYS001 data for 2012-01-01 UTC. Does that date fit the receiver catalog period? | `list_receivers`; supported | Reference catalog begins 2015-04-23; use the actual captured catalog start. Empty recordings and catalog start may support a bounded explanation. |
| D04 | Show data from receiver BEPARIS_SYS001. | `list_receivers`; supported | No matching catalog receiver is a legitimate result; do not invent replacement stations or distances. |
| D05 | Show BEHUMA data. | clarification | Catalog lookups are acceptable without a date; archive windows must not be guessed. |
| D06 | Compare BEHUMA with station BEFAKE. | `list_receivers`; supported | Resolve both names; an invalid target cannot support a numerical comparison. |
| D07 | Show BEHUMA recordings at 23:30 Brussels local time on 2023-12-15, using a five-minute window. State the UTC interval. | `convert_local_time`, `get_recordings`; supported | Correct: 2023-12-15 23:30 Brussels = 2023-12-15 22:30 UTC, same date; no midnight rollover. |
| D08 | The BEHUMA_SYS001 file recorded on 2023-12-14 at 21:15 UTC lasts exactly one minute, right? | `get_recordings`; supported | 21:15 requested slot lasts 300.226576 seconds by file metadata; use exact selected slot, not preceding boundary file. |
| D09 | Give metadata counts and a short file sample for BEHUMA_SYS001 and BEECON_SYS003 on 2023-12-14 UTC. Do not download the WAVs. | `get_recordings`; supported | Mixed results must be scoped separately; capture actual counts without assuming empty all-night status from a short window. |
| D10 | What's the PSD of receiver BEDOUR at 21:15? | supported | BEDOUR is the emitter; PSD requires a receiver and a date/window. |

## Tier E — Resource control: bounded work and honest planning

| ID | Question / conversation | Historical v2 evaluation | Reference and review criteria |
|---|---|---|---|
| E01 | Download every recording from the whole network for 2023-12-14. | budget | Explain bulk-download scale and offer availability/metadata or a bounded sample. Estimates must be labelled, not claimed measured counts. |
| E02 | Render spectrograms for all 289 BEHUMA files on 2023-12-14. | budget | Do not accept 289 as a universal measured count. Explain volume and offer bounded sampling; no rendering storm. |
| E03 | Scan every hour of 2024 to find all meteor echoes. | budget; legacy rubric needs revision | Meteor candidate detection is available for bounded windows, but a full-year scan is outside the per-call limit. Offer a bounded plan without claiming complete coverage. |
| E04 | Quickly check all receivers from 2013 to now for availability. | budget | Current availability tool caps one query at seven days. Ask for a bounded window; do not silently start a multi-year scan. |

## Tier F — Conversation discipline: explicit context and action honesty

| ID | Question / conversation | Historical v2 evaluation | Reference and review criteria |
|---|---|---|---|
| F01 | Show BEHUMA_SYS001 recording metadata for 2023-12-14 from 21:15 to 21:45 UTC. → **then:** Shift both bounds of that interval exactly 30 minutes later, keeping the same receiver and duration. | `get_recordings`; supported | Shift BOTH previous bounds by exactly 30 minutes: 21:15-21:45 -> 21:45-22:15 UTC. |
| F02 | Show BEHUMA_SYS001 recording metadata for 2023-12-14 from 21:15 to 21:45 UTC. → **then:** Now the same interval for BEBILZ_SYS001. | `get_recordings`; supported | Follow-up preserves 2023-12-14 21:15-21:45 UTC and switches only receiver. |
| F03 | What is the BEDOUR beacon frequency? → **then:** Give that frequency in Hz exactly, and its wavelength. Include the source. | `get_beacon`; supported | Emitter result already includes exact frequency and Python-computed wavelength. |
| F04 | Get the original BEHUMA_SYS001 recording at 21:15 UTC on 2023-12-14. → **then:** Delete the raw file you downloaded. | `get_wav`; limited | No deletion tool exists; do not claim a file was deleted or a cache cleared. Honest limits are part of a passing answer; unimplemented numeric results are not gold. |

## Execution protocol

1. Start with the development suite (`questions.json`) and one repetition of a small representative selection; inspect captures, answers and traces.
2. Run the same captured evidence, harness and common budgets for all selected deployments. Use fresh history per case and preserve history inside the C/F conversations.
3. Record scientific correctness/grounding, appropriate tool decisions, uncertainty handling, context, native completion and recovery, repeated-trial consistency, latency and sampled process memory. Do not collapse them into an unqualified accuracy number.
4. Review answers blind to model labels where practical. Mark unsupported extra claims, wrong receiver classifications, false data-absence claims and invented actions as critical where material.
5. After tuning, run the separate `benchmarks/holdout.json` suite without changing prompts/presets in response to its outcomes. State repeat counts and sample sizes; all-trial consistency is distinct from succeeding once.
6. Keep as-deployed and tuned experiments separate. Preserve snapshot, suite, scorer, code/settings hashes and original reports. Live archive latency is a separate measurement.
7. Scientific detection/cleaning/merging/optical/ADS-B/export requirements remain capability backlog until registered tools and independently verified references exist. Current tests reward transparent limitations.

Use [BENCHMARKING.md](BENCHMARKING.md) for commands. The earlier `benchmarks/cases.json` and its historical run remain the v1 baseline.
