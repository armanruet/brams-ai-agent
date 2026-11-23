# BRAMS grounded-assistant benchmark v2

> **Current capability mismatch:** The frozen `questions.json`,
> `benchmarks/conversation_geo_pilot.json`, and
> `benchmarks/conversation_geo_full.json` contain legacy rubrics and scoring
> assumptions from before the meteor detector and Belgian geographic selector
> were added. Some cases classify those operations as unsupported, and the
> scorer counts calls in `unsupported` turns as unexpected. These suites are
> retained to reproduce earlier evaluations, not as a reliable grading protocol
> for the current app. Use and independently review the current-capability suites
> described below and capture fresh reference evidence before model comparisons; do
> not silently regrade old runs under changed expectations.

## Current-capability evaluation

`benchmarks/current_capabilities_v2.json` is a provisional suite for general
meteor knowledge, single- and multi-slot detection and a detailed follow-up,
geographic counts, active sites and spectrogram retrieval, availability, and
spectrogram delivery. `benchmarks/current_capabilities_v3.json` extends `v2`
(`CUR01–CUR08`) with original/preview WAV delivery (`CUR09`) and Europe/Brussels
autumn DST conversion with negated archive querying (`CUR10`). Version 1 remains
available. It keeps manual scientific review mandatory. Capture fresh real-tool
evidence before snapshot replay; the previous snapshots predate meteor and
geographic tools. In particular, review availability row-presence semantics and
the detection/coverage details rather than treating a captured result as
independently established ground truth. The detector image must still be present
at its captured local artifact path for snapshot media checks.

```bash
python -m agent_app.benchmark plan --config benchmarks/models.json \
  --cases benchmarks/current_capabilities_v3.json --models ornith --repetitions 1
python -m agent_app.benchmark capture --cases benchmarks/current_capabilities_v3.json \
  --snapshot .benchmarks/current-capabilities-v3-reference.json \
  --reuse-snapshot .benchmarks/current-capabilities-v2-reference.json
python -m agent_app.benchmark run --config benchmarks/models.json \
  --cases benchmarks/current_capabilities_v3.json --mode snapshot \
  --snapshot .benchmarks/current-capabilities-v3-reference.json --models ornith --repetitions 3
```

`benchmarks/current_capabilities_failures_v2.json` is a separately labeled set
of synthetic archive failures, missing/zero/mixed availability, and partial
detection interventions; it intercepts non-time tools and must
be run in `--mode live`, never captured as live archive evidence. Run it separately
and review whether the failure was actually exercised and described as unknown.
The historical failure v1 case CURF01 mismatches its BEHUMA prompt with a BEHALL
catalog seed; keep that file only for reproducing old runs and use v2 for review.
An expected tool error is tracked as a tool error but is not an unexpected error.
The scorer can check an explicitly expected detection attachment; presence does
not prove that its image or scientific claims are correct. Separate native model,
direct-answer and recovery outcomes in reviews. Never silently replace old runs.
Model-only scientific flags and user-visible-answer flags are reported separately;
they are triage aids and do not replace manual review. The v2 live capture can
require many geographic/archive lookups and a twelve-slot detection; archive
changes or missing data can require selecting another documented reference window.
The current scorer is version 7; equivalent ISO local times with or without
explicit seconds are matched. Archived results retain their original scorer
version. Rescoring writes a separate report and does not change the old run.
The executed tool plan and the model's original plan are scored separately
when the application expands a verified detection window; the adjustment remains
visible in the trace and is not credited to the model as its own choice. Model
and displayed-answer grounding flags are separate heuristics for review, not
substitutes for independent scientific assessment.

The [conversation, repetition, geography and trajectory
evaluation](benchmarks/CONVERSATION_GEO_TEST_PLAN.md) now has an executable
historical suite: ten cases / 17 user turns per model per repetition. The pilot
completed 24 turns using the earlier tool set. The application now includes a
Belgian geographic selector and bundled country boundary; the old suite's
capability assumptions do not describe the current application.

## Historical full conversation/geography evaluation

The following command runs the archived case file with the current code. Its old
rubrics are not suitable for scoring the current geographic tool without review.
Reproducing an earlier experiment also requires its original code and tool set:

```bash
python -m agent_app.benchmark run --config benchmarks/models.json \
  --cases benchmarks/conversation_geo_full.json --mode live \
  --clock 2026-11-07T12:00:00Z \
  --models ornith liquid spark public --include-public --repetitions 3
```

This is **204 question turns / 120 case executions**, plus four excluded warmups:
168 live-archive turns and 36 explicitly controlled negative-fixture turns. Tools,
models, profiles and budgets remain the same across models. Models load sequentially.
The fixed clock resolves yesterday to 6 November even when running on a later date;
it does not alter archive acquisition timestamps or the app clock.

P01–P07 cover understanding, clarification, exact/paraphrased repetition, explicit
refresh, timezone correction and geographic correction. P08–P10 supply isolated
empty-catalog, missing-PNG and archive-error conditions. Their model calls are real,
but all non-time tools are intercepted: no live archive I/O or successful fake
artifacts are permitted. The retained singleton catalog seed uses an actual pilot
registry row with source provenance; the catalog subset and failure states are
synthetic interventions. The fixtures do not supply a country polygon.

Use `protocol_summary.csv` to compare live and controlled runs separately. Model
cards pool both protocols. Check `controlled_fixture_exercised`: an honest early
limitation may be safe, but does not establish handling of a failure result the
model never reached. Complete the four new dimensions in `manual_review.csv`
(`trajectory`, `query_understanding`, `repetition`, `geography`, each `_0_to_2`),
record reviewer/verdicts and task outcome, then summarize as usual. Use N/A where
no preceding PNG exists to support successful-artifact reuse testing. A claim
of polygon verification had no supporting tool in the historical baseline. The
current application has since gained geographic selection; the old rubrics and
controlled fixtures need review before evaluating that implementation.

The full execution attempted locally on 8 November stopped at all four connection
preflights (recorded in local gitignored run directory
`.benchmarks/20261108-075804-732380/ATTEMPT.md`).
No full-suite question turns or quality scores were produced. The offline pilot
trajectory review (`.benchmarks/pilot-trajectory-review-20261108/REVIEW.md`,
kept in local gitignored `.benchmarks/` outputs) documents the actual 24 pilot
turns and their observed issues.

## Conversation/geography capability pilot

The historical **as-is** pilot has three cases / six turns per model. It tested
understanding, clarification, geographic limitations and repeated-reference
handling before the geographic selector and country polygon were integrated.

```bash
python -m agent_app.benchmark run --config benchmarks/models.json \
  --cases benchmarks/conversation_geo_pilot.json --mode live \
  --clock 2026-11-07T12:00:00Z \
  --models ornith liquid spark public --include-public --repetitions 1
```

`--clock` overrides only the benchmark clock tool with a clearly labeled fixture;
archive requests remain live and acquisition timestamps are not rewritten. Keep
this clock for reproducing the 6 November date windows. It does not represent
today's live clock. The ordinary app clock and model/tool budgets are unchanged.
The three local models run sequentially, compatible with `--models-max 1`.

The first 7 November attempt failed every model's connection preflight. The
subsequent user-terminal run `.benchmarks/20261107-225011-318656` completed all
24 turns, including Python fallbacks; native completion was 4/6 Ornith, 6/6 Liquid,
1/6 Spark and 6/6 public. These are completion counts, not accuracy scores. A meaningful
successful-artifact reuse test still requires an actual earlier PNG; otherwise
score honest missing-reference handling and mark artifact redisplay N/A.

The evaluation now uses the corrected **54 tasks / 66 conversation turns** in
[`QUESTIONS.md`](QUESTIONS.md) and its executable twin [`questions.json`](questions.json).
Eight separate task variants live in `benchmarks/holdout.json`. The earlier
20-case `benchmarks/cases.json` and its historical runs remain available as v1.

Compare six factors separately: scientific correctness and grounding; appropriate
tool decisions; uncertainty/judgment; conversation handling; repeatable completion;
and latency/resource use. There is no automatic overall scientific accuracy score.

## Start with the prepared pilot

From the project directory, with llama.cpp running:

```bash
source .venv-brams/bin/activate
python -m agent_app.benchmark plan --config benchmarks/models.json \
  --case A01 A02 A10 A11 A15 C06 D05 E01 F01 --repetitions 1
```

The nine-case pilot has 11 turns per model: counts, coordinates, spectrogram,
noise, GPS uncertainty, coordinate provenance, missing dates, bulk-request control
and a time follow-up. A real reference for these cases has already been prepared
at `.benchmarks/reference-v2-pilot.json`, reusing the original actual capture.
The source retrieval timestamps are retained; these are saved observations, not
fresh live observations. Only the local clock entry is acquired afresh.

Run the three local deployments first:

```bash
python -m agent_app.benchmark run --config benchmarks/models.json \
  --snapshot .benchmarks/reference-v2-pilot.json \
  --case A01 A02 A10 A11 A15 C06 D05 E01 F01 \
  --models ornith liquid spark --repetitions 1
```

Include the public baseline explicitly when wanted (hosted calls may be billed):

```bash
python -m agent_app.benchmark run --config benchmarks/models.json \
  --snapshot .benchmarks/reference-v2-pilot.json \
  --case A01 A02 A10 A11 A15 C06 D05 E01 F01 \
  --models ornith liquid spark public --include-public --repetitions 1
```

The default configuration without `--config` is still
`benchmarks/models.example.json`. Your personal file is `benchmarks/models.json`.
`api_key_env` contains a variable name such as `BENCHMARK_PUBLIC_API_KEY`; the
actual credential belongs in `.env` or the terminal environment, never JSON/chat.
The public provider must support the app's existing Chat Completions tool API.

## Full development evaluation

Capture the expanded reference before the full suite. The default capture/output
is now `.benchmarks/reference-v2.json`, separate from the original v1 reference.

```bash
python -m agent_app.benchmark capture \
  --snapshot .benchmarks/reference-v2.json \
  --reuse-snapshot .benchmarks/reference.json
```

`--reuse-snapshot` is optional. It reuses exact successful queries from a previous
real-tool capture, preserves their source acquisition times and records the seed
hash. It never reuses a saved clock. Complete captured catalog rows can also be filtered in Python for equivalent names or subsets, with explicit derivation provenance; missing coordinates are never inferred. Omitting reuse retrieves every query anew.
The full suite needs 36 deduplicated tool queries, including additional metadata,
availability and a second receiver's PSD. Failed queries cannot become gold data.
An incomplete capture stays in a `.partial` file and cannot be benchmarked.
Wait for **`Saved ... actual tool results`**. Existing completed captures are
never overwritten; choose a new snapshot filename for a different acquisition.

Then inspect the snapshot and run:

```bash
python -m agent_app.benchmark plan --config benchmarks/models.json
python -m agent_app.benchmark run --config benchmarks/models.json \
  --snapshot .benchmarks/reference-v2.json \
  --models ornith liquid spark public --include-public --repetitions 3
```

For all four models this is **648 case executions / 792 timed conversation turns**,
plus four warmup conversations. Each turn can contain multiple model requests.
Start with the pilot and resolve reference/scorer problems before paying for a
large run. Tier subsets are available, e.g. `--tier C D F`. `--case` uses v2 IDs
such as `A01`, `D07`, `F01`; the earlier descriptive case IDs require the v1 file.

The historical v2 suite evaluated seven requested analysis capabilities as
**honest limitations**, including echo detection. That assessment is outdated:
candidate echo detection is now available, while cleaning, merging, optical
matching, confirmed meteor-event attribution, aircraft/ADS-B interpretation and
dataset export still have no dedicated implementation. Update and validate the
suite before scoring the current detector; do not invent measurements, matches,
exported artifacts or deletions.

## Inspect the benchmark panel and grade answers

Every new run creates an offline `report.html` in its output directory. Open it
in a browser. It needs no service, external scripts or network access. The panel
shows all selected models, including failed preflights, category filters,
questions, model versus displayed answers, actual tool traces, timing, resources,
review criteria and repeated-trial exports. Text is rendered as text, not HTML
from model answers.

| Output | Purpose |
| --- | --- |
| `report.html` | Offline model cards and model/tier/category/outcome/question filters |
| `summary.json`, `summary.csv` | Completion, query checks, source presence, timeouts, latency, known tokens, memory and reviewed quality |
| `category_summary.csv` | Factor/category breakdown with tier labels |
| `consistency.json`, `consistency.csv` | Per-case all-trial workflow and separately reviewed model/assistant consistency |
| `manual_review.csv` | Actual answers, rubrics, reviewer scores and separate verdicts |
| `turns.jsonl` | Incremental full evidence; source for later review |
| `manifest.json`, `cases.json`, `reference.json` | Exact versions, budgets, hashes, warmup and captured evidence |
| `operator_notes.md` | Hardware, server build, effective settings and experimental conditions |

Record a reviewer name and `model_verdict` / `assistant_verdict` as PASS, PARTIAL
or FAIL in `manual_review.csv`. Optional dimensions use 0 wrong/missing, 1 partial,
2 correct/complete. Mark material fabrications with `critical_error=yes`. A
critical error or a partial/wrong scored quality dimension overrides PASS. Model
PASS additionally requires an actual completed model answer. Assistant PASS can
credit verified Python recovery if the displayed answer fulfills the task.

Regenerate the panel after annotation:

```bash
python -m agent_app.benchmark summarize .benchmarks/YOUR-RUN-DIRECTORY
```

Annotations are preserved. Blank verdicts or verdicts without a reviewer remain
**ungraded**, not zero or perfect. All-trial quality remains ungraded until the
complete planned repetition/turn matrix has been reviewed. Success once and
success on every repetition are different results. Do not rank models from a
partially reviewed quality percentage without stating its coverage.

The automatic layer checks completed tool routes, media delivery, citations and
representation-aware fact presence. It accepts equivalent UTC formats, grouped
numbers, scientific notation with superscripts and explicit supported unit
conversions. Main recording facts select the requested slot instead of the prior
boundary row. Extra uncaptured auxiliary queries are reported separately from a
successful target route. Narrow claim checks flag explicit sample-count, nominal
offset and receiver-classification mistakes for review. These checks do not prove
that every sentence, unit association or interpretation is scientifically sound.

## Measure local resource use

Find the PID of your actual llama.cpp server, for example:

```bash
lsof -nP -iTCP:8080 -sTCP:LISTEN
```

Add `--server-pid YOUR_NUMERIC_PID` to the `run` command. The sampler observes
process resident memory in MiB every 0.5 seconds plus the start/end of each turn.
It records both the Python app and, for local profiles, the specified server.
Server RSS is not attributed to the hosted model. Missing/unavailable measurements
remain null; they are never reported as zero memory. RSS is **not** GPU/Metal
allocation or an exact transient peak, and should not be added to GPU memory.
Record unified-memory/GPU observations separately in operator notes if needed.

Keep hardware, power mode, background workload and local server build fixed.
Models run sequentially, so `--models-max 1` remains suitable. Warmup time,
including autoload/network/generation overhead, is recorded separately from
case latency. The app uses non-streaming model requests; TTFT and true decoding
throughput are not measured.

These are as-deployed configurations: Ornith Q4_K_M, Liquid/Spark Q8_0. Declared
sampling/context/reasoning metadata is not runtime verification. Server/provider
defaults and tokenizers differ. Keep a common budget for the baseline; record
per-model tuning as a separate protocol instead of mixing it with baseline data.

The defaults remain six model requests/six tools per turn, 180 seconds, 4096
response tokens and pre-tool recovery up to 8192. Benchmark budgets come from the
JSON `protocol`, independently of Gradio's `.env`. A final response truncated
*after* tool evidence exists uses Python fallback; the pre-tool recovery cap does
not automatically apply. Known tokens can be incomplete for cancelled requests,
even when Python returns a completed fallback. Use provider billing for actual cost.

## Holdout and live performance

After development tuning, freeze settings and capture/evaluate the eight separate
variants without further tuning on their outcomes:

```bash
python -m agent_app.benchmark capture --cases benchmarks/holdout.json \
  --snapshot .benchmarks/holdout-reference-v2.json
python -m agent_app.benchmark run --config benchmarks/models.json \
  --cases benchmarks/holdout.json --snapshot .benchmarks/holdout-reference-v2.json \
  --models ornith liquid spark --repetitions 3
```

The holdout covers other receiver coordinates, a DST-boundary conversion,
ambiguous local time, another metadata slot, an unknown receiver and bulk-request
judgment. Reference capture must verify any new query; no invented values are
embedded as gold.

Measure actual archive/network latency separately, for example:

```bash
python -m agent_app.benchmark run --config benchmarks/models.json \
  --mode live --case A07 A10 A11 --models ornith liquid spark --repetitions 3
```

Snapshot tools return fixed real evidence and explicit replay provenance. They
share semantic duration/page/PSD-band checks with live tools. Uncaptured queries
return `benchmark_unmatched_call` and never secretly contact BRAMS. Replay tool
time is replay overhead, not download/scientific compute time. Keep `.artifacts/`
with the reference to preserve original WAV/preview/PNG attachment delivery.

## Existing baseline and offline rescoring

The 276-turn v1 run is preserved. A new directory
`.benchmarks/20261101-141637-913221-evaluation-v2-preview/` contains an offline rescore and panel.
It keeps the original questions, measurements and scores in `original_score`;
it is **not a fresh v2-suite run**. No model or archive requests are needed:

```bash
python -m agent_app.benchmark rescore .benchmarks/YOUR-ORIGINAL-RUN \
  --output .benchmarks/YOUR-NEW-RESCORE-DIRECTORY
```

To deliberately reproduce v1 questions, specify both original paths:

```bash
python -m agent_app.benchmark run --config benchmarks/models.json \
  --cases benchmarks/cases.json --snapshot .benchmarks/reference.json \
  --models ornith liquid spark --repetitions 3
```

The original precision-reference row-order problem remains part of that archived
v1 case file; new v2 questions use explicit main-slot selectors. Never present a
regraded legacy run as evidence of new questions or new model behavior.

## Current preparation status

The corrected bank, runner, grader, reports and pilot reference are prepared.
The full v2 capture attempted here stopped on a BEECON catalog retrieval error;
`.benchmarks/reference-v2.json.partial` preserves the diagnostic, and no completed
full v2 reference was published. Retry the full capture command in your terminal,
where you ran the working original capture. No new model benchmark batch has been
started as part of this preparation. See `VALIDATION.md` for test coverage.
