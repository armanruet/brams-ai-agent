# Conversation and geographic-query evaluation plan

> **Historical protocol:** This plan and its pilot/full JSON suites were written
> before `query_geographic_receivers` and the bundled Belgian boundary were
> integrated. The older capability-limit rubrics and controlled fixtures should
> not be used to grade today's geographic tool without a reviewed, versioned
> evaluation update. Retain the original records for comparison with past runs.

Prepared 7 November; updated 8 November 2026. Historical status: **24 pilot turns
completed in the user terminal; the old 204-turn suite was prepared**.
The full attempt from this environment failed all four connection preflights
(local gitignored attempt record: `.benchmarks/20261108-075804-732380/ATTEMPT.md`).
An evaluation updated for the geographic tool still needs to be prepared. This plan supplements
`questions.json`; it does not replace or regrade the existing benchmark.

## Purpose

Compare Ornith, Liquid, Spark and the configured public baseline on four concerns:

| Concern | What we evaluate |
| --- | --- |
| Trajectory | Observable sequence of model requests, tool arguments, results, retries, reuse, recovery and final answer |
| Query understanding | Correct intent, place constraint, calendar date, timezone, interval and requested artifact; appropriate clarification |
| Repeated questions | Consistent facts, recognition of previous evidence, sensible reuse, no loops; correct handling of an explicit refresh |
| Geographic queries | Evidence-based country/region filtering and receiver selection; honest handling of ambiguity, missing coordinates and archive failures |

Trajectory means the recorded actions and evidence, not private chain-of-thought.
An answer with a PNG is insufficient if it uses the wrong country, date or interval.

## Main user question and interpretation contract

> Can you get me a spectrogram from any station in the north of Belgium between 10pm to 11pm yeterday?

Treat `yeterday` as `yesterday`. The requested output is one archived spectrogram,
not a station list, WAV, PSD measurement or a claim of meteor detection.

* **Clock:** resolve relative dates through `get_current_utc_time`. For reproducible
  replay, use a declared benchmark clock fixture of `2026-11-07T12:00:00Z` for every
  model. This is an injected test clock, not a claim that live data was acquired
  then. Preserve real catalog/archive acquisition times separately.
* **Timezone:** retain the app's current default: unspecified BRAMS times mean UTC,
  stated to the user. The original question therefore means
  `2026-11-06T22:00:00Z`–`2026-11-06T23:00:00Z`. Belgium alone does not silently
  change this contract. If Europe/Brussels is explicitly requested, use
  `convert_local_time` for both bounds: `2026-11-06T21:00:00Z`–`22:00:00Z`.
  Clarification before retrieval is acceptable when the assumption is made explicit;
  scoring must not treat an unanswered clarification as successful retrieval.
* **Geography:** “north of Belgium” has no defined boundary in the current app.
  Ask which region is intended, or disclose a previously agreed definition.
  Do not silently equate northern Belgium with all Belgium or a guessed place name.
* **Controlled region:** for the explicit tests below, define the benchmark's
  northern region as points **inside a pinned Belgian country polygon with
  latitude >= 51.0 degrees**. Define its southern contrast as points inside that
  polygon with latitude < 50.5 degrees. These are artificial test predicates,
  not claims about administrative regions. Python performs point-in-polygon and
  filtering using registry coordinates, with geometry provenance/version recorded.
* **Selection:** any eligible receiver with a successful recording is valid. Use
  a stable Python ordering of candidates for reproducible discovery. Grade the
  returned station against the predicate and recording evidence, not one preferred
  name. A coordinate alone does not establish recording availability.
* **Time coverage:** one PNG from within the hour fulfills “a spectrogram”. Report
  its actual recording bounds, which may cover only a short part of the hour.
  Do not describe one PNG as a continuous full-hour visualization.

## Scenario matrix

Each row starts a fresh conversation. Turns within a row share real session
history. Each repetition starts fresh. The literal original question above is
used in P01 and as turn 1 of P02.

`Q` below means this complete prompt, fixed verbatim before execution:

> Get one archived spectrogram from any receiver inside Belgium with registry
> latitude at least 51.0 degrees, between 22:00 and 23:00 UTC yesterday. Use the
> benchmark country polygon and registry coordinates to choose the receiver.

| ID | Turns | Prompts in order | Expected behavior |
| --- | ---: | --- | --- |
| P01 | 1 | Original question with typo | Understand spectrogram intent; read clock when resolving yesterday; state UTC assumption; clarify undefined north rather than inventing a station/region. |
| P02 | 2 | Original question; then “Use UTC. By north I mean inside Belgium with registry latitude at least 51.0 degrees, using the benchmark country polygon.” | Use the clarification and previous question to retrieve one qualifying PNG, with source and actual recording bounds. If already completed using a disclosed definition, reconcile the clarification. |
| P03 | 3 | Q; Q again exactly; “Show that same spectrogram again with its receiver ID and exact recording time.” | Keep station, interval and artifact consistent; reuse adequate prior evidence without duplicate archive work; make existing media available again. |
| P04 | 2 | Q; “Could you show me that northern Belgian station's plot for yesterday at 10–11pm UTC once more?” | Recognize semantic repetition despite different wording; preserve the resolved date, receiver and recording. |
| P05 | 2 | Q; “Please check the archive again now for that same receiver and UTC interval, even if we already checked.” | Distinguish explicit refresh from repetition; make a new relevant archive call and label fresh vs replayed evidence honestly. |
| P06 | 2 | Q; “Correction: use 10–11pm on 6 November 2026 in Europe/Brussels instead; keep the same receiver.” | Convert both bounds to 21:00–22:00 UTC and retrieve the corrected interval. Old 22:00–23:00 UTC evidence cannot fulfill it. |
| P07 | 2 | Q; “Now use the southern benchmark region: inside Belgium with latitude below 50.5 degrees. Keep the same UTC date and hour.” | Change only geography; choose a newly eligible receiver and retrieve its plot. Do not reuse the northern receiver as the final selection. |
| P08 | 1 | Q, against a labeled fixture with no qualifying coordinates | State no qualifying candidate was established in the supplied catalog; do not choose a foreign/southern station or infer missing coordinates. |
| P09 | 1 | Q, against a labeled fixture where eligible receivers have no PNG for the interval | Explain the actual no-data scope, preserve the requested region/date and offer a bounded alternative. Do not silently change the request. |
| P10 | 1 | Q, against a labeled archive-service failure fixture | Report the returned failure as unknown availability/retrieval failure. Do not convert an outage into “no recordings”. |

Total: **10 cases / 17 user turns per model per repetition**. P08–P10 use separate,
declared controlled fixtures; they are robustness tests, not real archive facts or
live-network latency measurements. Missing candidates, missing files and service
errors must remain distinct. Do not synthesize a successful PNG.

## Expected trajectories and acceptable alternatives

Successful explicit query, conceptually:

1. Read clock; resolve yesterday under the declared timezone contract.
2. Select candidates through a Python geographic filter over registry evidence.
3. Request one actual archived PNG for an eligible receiver and resolved interval.
4. Explain the result with receiver ID, region evidence, actual recording bounds,
   UTC/local-time interpretation and the returned source URL; attach the PNG.

For an explicit local timezone, both time conversions precede the archive query.
Clarification is a valid intermediate outcome. A network-wide availability query
followed by a verified geographic selection is another acceptable route if useful.
Do not mandate an extra availability call before `get_spectrogram`: successful
retrieval already proves this recording exists. Paginating the catalog or trying
many receivers blindly is not the target workflow.

For a repeated question: identify earlier adequate evidence -> reuse its resolved
receiver/date/window -> answer and expose the same artifact. A second archive call
can still produce a factually correct answer; grade reuse/efficiency separately.
For an explicit refresh: retrieve again -> report the actual new result and source.
Snapshot replay can test the decision to refresh, but cannot prove a fresh network
observation. Do not penalize a changed answer when genuinely fresh evidence changed.

## Existing support and preparation required

Code inspected: `agent_app/benchmark.py`, `benchmark_data.py`, `harness.py`,
`service.py`, `progress.py`, `brams_tools.py` and `tool_guard.py`.

Already available:

* Four configured model slots, sequential model loading, multi-turn sessions,
  fixed-evidence replay, real media, raw tool call/results and model/tool timings.
* Named-timezone conversion, a clock tool, registry coordinates and PNG retrieval.
* Separate model answer/displayed assistant answer and Python fallback reporting.

Required before calling this an executable evaluation:

1. **Geographic tool:** add a bounded Python region selector, with country polygon
   version/source and registry coordinate provenance. Return candidate IDs,
   coordinates, predicate, excluded/missing-coordinate counts and truncation status.
   This is not currently a registered capability. Python must filter the complete
   catalog before limiting returned rows; do not give the LLM responsibility for
   geographic arithmetic or guessing country membership.
2. **Clock/replay:** support a documented per-experiment clock override, validated
   UTC gold bounds and both local-time conversions. Supply the same clock to every
   model regardless of actual execution date. Materialize all allowed candidate
   queries in the replay or provide constrained evidence-based replay derivation;
   an uncaptured but otherwise valid receiver choice is a reference-coverage issue,
   not automatically a model reasoning failure.
3. **Reuse and display:** verify that a prior artifact can be displayed again
   without a fresh download. Current media attachment discovery reads current-turn
   tool results. Add explicit verified artifact reuse if needed; do not make an
   invented path/link count as media delivery.
4. **Scoring:** add cross-turn consistency/reuse checks, geographic predicate
   checks and explicit result classes (clarified / retrieved / reused / no data /
   failed / unsupported). Current required-call scoring alone cannot reward a
   zero-call reuse turn or validate arbitrary eligible station choices.
5. **Error trajectory:** preserve `error.tool_trace` on failed benchmark rows as
   well as timings. Current failure rows can lose that trace. Link model-request
   steps to calls/results when possible; do not infer causal order from summed
   timings alone. Mask keys as in existing reports.
6. **Fixtures:** keep P08–P10 in a clearly marked fixture protocol; the real
   reference capture rejects error/setup failures and must continue doing so.
   Do not label fabricated errors or missing-data fixtures as real-tool captures.

Run an **as-is baseline** first with the current capabilities: honest clarification
or an explicit tool limitation can pass safety/grounding while remaining incomplete
for geographic retrieval. Report model errors separately from missing system
capabilities. Then run the prepared geographic workflow as a distinct experiment
with its own code/settings hashes. Do not silently fix the app between models.

## Per-model trajectory review

For every model, repetition, case and turn, review `turns.jsonl` alongside the
offline `report.html` (Answers, evidence and review -> Tool trace / Timings).
Use `conversation_geo_review.csv` for this experiment's four concern scores.

Record the observable sequence, for example:

`clock -> clarification -> user supplies boundary -> Python region selection ->
PNG retrieval -> sourced answer -> repeat -> prior-artifact reuse`

Include actual tool names/arguments, result status, call IDs, source/artifact IDs,
model-request count, tool count, retries, blocked duplicate calls, fallback,
elapsed/model/tool seconds and cache/replay/fresh acquisition status. Use the
earliest evidenced wrong step as the failure diagnosis: date resolution, timezone,
region selection, receiver selection, archive retrieval, reuse or final explanation.
Do not make up intermediate reasoning that the trace did not expose.

Use **0 = wrong, 1 = partial, 2 = complete**, plus **N/A** where the concern is
not tested. Keep task completion and infrastructure/capability failures separate.
Any invented receiver/coordinate/measurement, false successful download, wrong
country, materially wrong time window or outage misreported as no data is a
critical failure. Grade native model output and displayed assistant output
separately; Python recovery does not establish native model success.

Report four concern scores with review coverage, critical failures, completed
retrievals, repeated-answer consistency, redundant archive calls, explicit refresh
compliance, native vs fallback completion, and median/p95 latency. Report p95 as
descriptive only at these small sample sizes. Geographic accuracy uses the count
of verifiable geographic selections as its denominator, with missing/unknown
coordinates reported separately. Do not combine everything into an unexplained
single “best model” score or rank only by speed.

## Execution stages

1. Freeze the scenarios, timezone/region policies, fixture labels, protocol and
   code version. Independently calculate date bounds and geographic gold in Python;
   verify real registry sources and PNGs. Keep archives/results immutable.
2. Run the as-is capability baseline and inspect all trajectories. Then implement
   and verify the preparation items above for the enhanced experiment. Each
   experiment must expose the same tools and prompts to all four models.
3. Enhanced pilot: P01–P03, one repetition = **6 turns/model**, 18 local turns or
   24 turns including public, plus one excluded warmup conversation/model. Review
   every pilot trajectory before scaling up.
4. Enhanced full run: all ten cases, three repetitions, four models = **120 case
   executions / 204 evaluated user turns**, plus four excluded warmup conversations.
   Keep real-snapshot and controlled-fixture results split in reports. Actual
   model requests exceed user-turn counts when tools/retries are involved.
5. Load models sequentially with `--models-max 1`. Preserve configured sampling,
   quantization, common budgets, hardware and server build. Start with the current
   six-request/six-tool limits; distinguish a budget-constrained legitimate route
   from a model loop. Test higher budgets only as a separate experiment.
6. Include the hosted baseline explicitly with `--include-public` when executing.
   Review all trajectories; record results in the worksheet and existing manual
   review outputs. Preserve per-model examples of success and first failure.
7. Finally run a small separate live check with a freshly read clock to confirm
   real “yesterday” handling and Gradio media redisplay. Its archive latency and
   changing data must not be mixed with fixed-evidence comparisons.

The executable as-is pilot is `conversation_geo_pilot.json` (P01–P03, six turns).
`conversation_geo_full.json` defines the historical ten-case/17-turn baseline
experiment, with labeled non-I/O negative fixtures and separate protocol summaries.
Its rubric assumed no geographic selector or country polygon; the current app now
has both. Its controlled fixtures intercept tools rather than proving that a live
polygon lookup occurred. Four concern dimensions in `manual_review.csv` need
explicit review; structural checks do not establish scientific/geographic accuracy.
See `BENCHMARKING.md` for the full command and the pilot trajectory review. The
enhanced workflow described in the preparation section is a separate future run,
not silently mixed into the historical baseline comparison.
