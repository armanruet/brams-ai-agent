# Historical validation records

These dated entries preserve what was tested and implemented at the time. They
are not a current capability or test report. The app has since added local
meteor detection, Belgian geographic queries and other changes; historical notes
about `File.stream` and duration footers describe earlier implementations. See
[README.md](README.md) for current behavior, and run the current tests for a new
validation result.

# Hardening, model selector, context safety, domain scoping, and task cancellation — 14–23 November 2026

Following the meteor-detection and Belgian geographic integration, a multi-stage
hardening pass addressed archive edge cases, KV-cache latency, grounding scope,
local model selection, context-window safety, domain scoping, and real task
cancellation:

- **Archive correctness & prompt/KV-cache alignment (14 November):** Preserved raw
  hourly availability row presence (distinguishing missing rows from explicit
  zeroes), isolated per-slot failures during multi-slot detection, added a bounded
  `120 s` in-memory TTL cache for static receiver/location catalogs
  (`CATALOG_CACHE_TTL_SECONDS`, never applied to live availability or file
  listings), trimmed instructions/schemas (`3,883 -> 3,618` fixed cached prefix
  tokens), and added `benchmarks/current_capabilities_v3.json` (`10` cases /
  `11` turns). Exploratory latency and snapshot measurements are recorded in
  [docs/latency-baseline-2026-11-13.md](docs/latency-baseline-2026-11-13.md).
- **Grounding & shortcut scope audit (15 November):** Fixed false-positive
  grounding overrides and shortcut scope leaks; added focused regression tests
  (`test_grounding_scope_audit.py`, `test_shortcut_scope_audit.py`,
  `test_followup_scope.py`, `test_archive_correctness_updates.py`) and verified
  against a local `focused-scope-retest-20261115` run (retained locally under
  gitignored `.benchmarks/`).
- **Simplified local/public model settings (16 November):** Streamlined the
  **Model settings** drawer to **Local model** (with **Refresh local models**
  catalog discovery from `/v1/models`) and **Public API**, retaining
  session-scoped memory-only credentials and server-side tool-calling checks.
- **Context-window safety & retry budgeting (22 November, `test_context_budget.py`):**
  Lowered default `LLAMA_RECOVERY_MAX_TOKENS` from `8192` to `6144` (matching
  `.env.example` and `agent_app/config.py`), added `/props` `n_ctx` probing
  (`LLAMA_CTX_SIZE` override; conservative `16384` fallback; never trusting
  catalog training context), validated both `max_tokens` and
  `recovery_max_tokens` pre-flight, generalized intra-turn tool-output compaction,
  and added single-retry / verified-summary fallback on HTTP 400 context-length
  overflows with labeled token estimates.
- **Domain-scoped assistant (22 November, `test_domain_scope.py`):** Added
  `ASSISTANT_SCOPE=domain|general` (default `domain`) with a BRAMS radio-meteor
  science scope charter in `harness.py`, availability-context hardening for bare
  `"available"`/`"active"` terms, and high-precision off-domain refusal guarded by
  negative-control science tests (ensuring legitimate BRAMS/signal-processing
  queries and short follow-ups are never blocked by cue absence alone).
- **Cooperative task cancellation (22 November, `test_cancellation.py`):** Added
  `agent_app/cancellation.py` (`OperationCancelled`, `cancellation_scope`,
  chunked cancellable `httpx` archive downloads running outside `SDK_LOCK`),
  cancellation checkpoints across multi-slot YOLO detection loops, and an
  unqueued Gradio **Stop** button (`elem_id="stop-message"`) with clean composer
  rollback and `Stopped` status in the Request Monitor.

## Complete local verification — 23 November 2026

Verified locally in Python 3.13.12 (`.venv-brams`, PyBRAMS `0.1.7`, Gradio
`6.31.0`, Pydantic AI Slim `2.63.0`, ONNX Runtime) and Node.js:

- **Python unit & integration tests (`python -m unittest discover -s tests -v`):**
  **307 / 307 passed** — `306` offline tests (including `test_context_budget.py`,
  `test_domain_scope.py`, `test_cancellation.py`, `test_meteor_detection.py`,
  `test_geography.py`, `test_grounding_scope_audit.py`, and all UI/service
  suites) plus `1` live socket-bound Gradio integration test
  (`tests/test_gradio_integration.py`).
- **Fresh-clone ("weightless") vision check:** Re-ran `tests/test_meteor_detection.py`
  with `VISION_MODEL_PATH` pointed at a nonexistent file (`8 / 8 passed`;
  real-ONNX smoke test skips cleanly when local weights are absent).
- **Node.js browser-script checks:** `node tests/test_splitter.cjs` and
  `node tests/test_chat_controls.cjs` both passed (`2 / 2`).

Tests use deterministic OpenAI-compatible fixtures and mocked/synthetic archive
payloads; they validate application logic, protocol handling, and UI state, not
live LLM scientific accuracy. GitHub Actions CI (`.github/workflows/ci.yml`)
executes this same Python + Node.js suite on push and pull requests.

# Chat controls and header cleanup — 2 November 2026

Removed the duplicate connection caption below the header; the current model
remains above Request monitor and updates after applying a connection. Model
settings now toggles the drawer, with state synchronized after manual closing,
Back to workspace and successful Apply.

Enabled Gradio's individual Copy action and added individual Delete controls
using stable message IDs. Older-message deletion preserves later turns, removes
the selected text from typed model history, and retains matched tool calls and
results. Attachment deletion removes its visible bubble without deleting the
original file. Stale IDs and mismatched rendered-message counts are rejected;
controls are disabled during generation. New chat remains the whole-chat reset.

All 138 selected offline Python tests passed, including four new multi-turn/
message-history checks and two new connection/control checks. Node interaction
checks passed for both the message controls and the existing splitter. Tests use
fixture model responses; no paid model calls were made. Existing upstream
event-loop resource warnings remain nonfatal. Live Gradio socket integration
could not run because this environment denies local socket binding, so the new
UI has not been verified in a restarted live server here.

# Grounded-assistant evaluation v2 — 1 November 2026

Corrected all 54 question cards in `QUESTIONS.md` and created the executable
`questions.json` (54 cases / 66 actual conversation turns). The independent
`benchmarks/holdout.json` contains eight final-evaluation variants. Unsupported
analysis is scored as honest limitation handling; no unimplemented numeric
results are treated as gold. Repaired UTC conversion, nominal frequency offset,
site counts, availability interpretation and no-data scope. New main-recording
facts select the requested slot instead of the preceding boundary row.

Added representation-aware evidence checks (UTC, scientific notation, grouped
numbers, unit scales), completed/alternative tool-route checks, conservative
claim flags, explicit snapshot provenance and shared live/replay semantic bounds.
The scorer does not assign scientific accuracy from fact presence alone. Human
review records separate model and displayed-assistant verdicts; unreviewed fields
remain null. Reports include category factors, complete-trial consistency,
preflight failures, timeout-aware incomplete token usage, optional sampled app/
llama.cpp process RSS and an offline interactive `report.html` with safe text
rendering. No GPU-memory or TTFT measurement is claimed.

132 selected Python tests passed across the original regression modules and the
benchmark tests; targeted benchmark tests were repeated after final adjustments.
Coverage includes type/semantic validation parity, notation/date/unit matching,
slot selection, cached-catalog aliases, failed capture, real-reference reuse,
explicit replay provenance, incomplete-trial handling, human annotation
preservation, missing resource data, cancelled-request usage and safe report HTML.
Report JavaScript passes Node syntax checks. Browser verification in Brave
confirmed four model cards, 276 rows, question filtering (12 PSD rows), model
filtering (3 public PSD rows), and expandable answers/trace/checks/timing.
Non-fatal upstream Gradio/audio/event-loop resource warnings remain. Socket-bound
Gradio integration tests were excluded; the existing chat UI was not redesigned.

Prepared `.benchmarks/reference-v2-pilot.json`: ten actual reference results for
nine pilot cases, reusing the prior real capture with source timestamps preserved
and acquiring the local clock afresh. Its suite hash matches `questions.json`.
The full 36-query capture attempted here stopped at a catalog retrieval error;
`.benchmarks/reference-v2.json.partial` is diagnostic only, not usable gold.
No completed full v2 snapshot or new v2 model-performance results are claimed.

Offline-rescored the prior 276-turn run into
`.benchmarks/20261101-141637-913221-evaluation-v2-preview/`, preserving original
scores, questions, execution metadata and source hashes. The new scorer recognizes
all six PSD-value mentions per model in that legacy run and flags several explicit
sample-count/offset/receiver-classification errors for human review. It does not
fix the legacy dataset's precision-reference row choice or constitute a new-suite
execution. No model or paid API calls were made for this preparation.

# Model benchmark infrastructure — 1 November 2026

Added the CLI protocol in `BENCHMARKING.md`, three exact local model IDs from the
user's preset file, a disabled hosted slot and a 20-case/23-turn dataset. Capture
uses actual Python tools and rejects failed/missing gold data. Snapshot runs
preserve production schemas/validation but prohibit unmatched archive queries
from silently going live. Model text is retained separately from Python evidence
blocks for scoring. Budgets now load from `LLAMA_REQUEST_LIMIT` and
`LLAMA_TOOL_CALL_LIMIT`; production defaults remain 6/6. Benchmark protocol limits
come from its JSON, independently of the running Gradio connection.

114 selected offline Python tests passed, including 12 new benchmark tests and
the existing connection/tool/media/history/progress regression tests. The new
tests cover replay, timezone canonicalization, schema/argument validation, native
answer scoring, API-key handling, configurable budgets, synthetic end-to-end
runner output, failed reference capture and preserving review annotations.
Upstream event-loop/audio resource warnings remain non-fatal. Socket-bound
Gradio integration tests were excluded because local sockets are unavailable.

Actual execution attempt: `.benchmarks/connectivity-check/manifest.json` records
failed preflight connections for all three local model IDs, with zero measured
case turns. This environment cannot connect to localhost. No model ranking,
latency score, real full archive reference capture or public API measurement has
been completed here. The hosted provider/model and credential are still needed.

# Live request monitor and workspace — 31 October 2026

Resizable split: a client-side accessible separator now adjusts the chat/monitor
ratio, with pointer dragging, keyboard arrows/Home/End, double-click/Enter reset,
minimum readable panel widths and a browser-local numeric preference. It works
with document or Gradio web-component roots. Narrow layouts stack and restore
the preferred ratio when expanded. Model settings, chat content, keys and run
limits are not modified by the splitter. `node tests/test_splitter.cjs` executes
the actual initializer with DOM fixtures to check drag bounds, keyboard operation,
preference restoration, responsive stacking, unavailable storage and shadow roots.
Live-browser drag verification remains pending because browser control and local
socket binding are unavailable in this execution environment.

Limit inspection (unchanged): `service.py` uses six model requests and six tool
calls per user message. Current loaded settings use a 180-second run timeout,
4096 output tokens per response and one pre-tool recovery up to 8192 tokens.
The app sets no separate input-token cap and does not trim conversation history;
the selected server/model's context limit still applies. The 65536 context in
`config/models.example.ini` is only the supplied reference preset, not proof of
the running server's context size. This review did not raise or remove any limit.

Screenshot-driven sizing correction: the app root now has explicit full-width
constraints, and the workspace uses fixed-ratio grid tracks with zero minimum
content width. The chat viewport has the same explicit height for empty and
populated states; long messages cannot resize the columns. Below 900 CSS pixels
the layout becomes one column. Header actions and the 80×38 Send control have
non-wrapping labels. The composer uses a shared focus ring instead of a separate
textarea outline, and answer timing footers no longer add an oversized divider.
These changes address the supplied empty-chat / greeting screenshots; a fresh
live browser comparison is still needed after Gradio restarts.

Layout refinement: the left pane now uses chat bubbles, a bottom message composer
and a compact header with New chat / Model settings. Connection controls moved
to a closed-by-default sidebar drawer. Applying a valid connection closes it;
invalid fields preserve the drawer and current session for correction. The right
request monitor retains live elapsed time, model/tool timings, activity, history
and raw trace. The same 102 tests pass after the layout change, including media
serialization and connection-state assertions. A native Chrome visual review
was attempted but the computer-use native pipe failed to start; the new running
page has not been visually inspected here.

At the time, the Gradio interface had a responsive conversation/monitor layout, a
prominent elapsed-time card, live model/tool execution entries, per-step timings,
a last-20 turn history, and total-duration footers on answers. The current UI
keeps timings in the request monitor without adding footers to answers. The model wrapper
times actual non-streaming API requests; the existing tool guard times validated
Python execution. Progress is observational and does not change tool schemas,
model protocols, repeat guards, recovery or completed tool traces. No model
reasoning is displayed. Callback payloads redact credentials and UI HTML escapes
argument/question content.

**102 selected tests pass**, comprising the earlier 91 and 11 new progress/monitor
checks. The tests verify events before a delayed tool finishes, timing bounds,
request failures, blocked repeats, validation failures, concurrent session
isolation, observer failures, HTML escaping/redaction, live UI yields, background
task cancellation on disconnect, failed-turn history and the 20-turn limit.
Existing media serialization, hosted routing, scientific tools and history tests
continue to pass. API responses in these tests are fixtures, not a fresh model
benchmark. Non-fatal upstream event-loop/audio resource warnings remain.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-brams/bin/python -m unittest tests.test_connections tests.test_prototype tests.test_receivers tests.test_availability tests.test_brams_sdk tests.test_spectrogram tests.test_spectrogram_sdk tests.test_wav tests.test_wav_sdk tests.test_recovery tests.test_tool_checker tests.test_progress -q
```

Socket binding is still restricted in this environment (`Operation not permitted`),
so the server-bound integration test and live browser review of the new layout
require a user-side Gradio restart. No live render/model performance is claimed.
Timings exclude UI queue wait and browser rendering, and model duration includes
network time. Historical sections below document earlier versions.

# Local/hosted provider switching — 29 October 2026

Gradio now supports Local Ornith, official OpenAI and other OpenAI-compatible
Chat Completions endpoints. Connection profiles and credentials are scoped to
the browser session's server-side State; applying a connection clears the chat
and tool trace. Hosted keys are not persisted into `.env` or returned in the
Gradio configuration. Empty-key reuse is allowed only for the same saved
provider and base URL. Switching back restores the original local configuration.

The connection panel includes a real clock-tool test and displays elapsed time,
request count, token usage and actual tool trace. Official OpenAI mode uses the
installed OpenAI model profile and `max_completion_tokens`; local/custom mode
retains compatibility flags and `max_tokens`, with a UI override for either form.
Hosted authentication, quota and compatibility failures have specific messages.

**91 selected tests pass**, including 17 new connection tests. These exercise
authenticated HTTP routing, model/token-parameter selection, strict-schema
compatibility, profile isolation, fresh history, key redaction, Gradio password
handling, original local restoration and protection against forwarding a saved
key to another endpoint. HTTP responses are fixtures: no public API key or paid
model request was used. Existing 74 tool/science/UI callback tests also pass.
Non-fatal upstream event-loop/audio resource warnings were emitted.

```bash
.venv-brams/bin/python -m unittest tests.test_connections tests.test_prototype tests.test_receivers tests.test_availability tests.test_brams_sdk tests.test_spectrogram tests.test_spectrogram_sdk tests.test_wav tests.test_wav_sdk tests.test_recovery tests.test_tool_checker -q
```

The socket-bound Gradio integration test remains unavailable in this environment
because socket creation is restricted. Live hosted generation requires restarting
Gradio and entering a provider key/model in Connection. No claim of live hosted
performance or account/model access is made from the fixture tests.

See [README.md](README.md#switch-between-local-and-hosted-models-in-gradio) for use.

# Tool-checker improvements

See [TOOL_CHECKER_RESULTS.md](TOOL_CHECKER_RESULTS.md) for the 29 October changes:
74 passing selected tests, real local-WAV PSD verification, corrected reference
errors, new live-check commands and the remaining live-server verification limits.
Earlier validation notes below describe the versions and environments at those times.

# WAV and response-limit fixes: 28 October 2026

The current browser showed a token-limit error for a network-wide question
asking which stations had recordings. The previous instructions and wrapper
expected a receiver name, and failed turns discarded their tool traces. No WAV
tool existed at that point.

Implemented:

- At that time, `get_wav` used PyBRAMS `File.stream` and `File.save_raw` to fetch
  one original WAV. The current implementation selects from a direct archive
  listing and then uses `File.save_raw`. The unchanged original is retained for
  download. A separate peak-normalized
  PCM16 preview is created for playback, with the original sample rate and duration.
  Metadata/source references go to the model; audio bytes and local paths do not.
- Network-wide availability uses one `check_availability(search="", ...)` query
  and compact per-receiver summaries instead of every hourly bucket for every station.
- A per-turn guard stops repeated identical calls and recognized name/ID aliases
  before repeating external work.
- Completed tool results and download attachments survive repeated calls, request
  limits, and truncated final responses. A deterministic evidence summary is shown
  with a recovery notice, and history is trimmed to complete call/result pairs.
- Default generation budget is 4096 tokens. At most one retry up to 8192 tokens
  is allowed when truncation happens before any tool result. Existing overall
  timeout/request limits remain in place. Failed traces are preserved in the UI.

Validation: **57 selected tests passed**, comprising the 56-test combined run and
one additional network-wide compact-summary test. They include the actual installed
PyBRAMS raw-download path with fixture HTTP data, original-byte preservation, preview
creation, Gradio audio and original-download message serialization across follow-up
turns, duplicate download prevention, request-limit and truncation recovery, paired
history, and retained failure traces. Ordinary chat and prior receiver/spectrogram
checks continue to pass. The HTTP socket integration case remains unavailable in
the restricted command environment as recorded in earlier sections. An upstream
Gradio/Pydub ResourceWarning appeared during audio serialization; tests passed.

**A live download/playback test requires restarting Gradio with this update.**
No new packages or llama.cpp restart are required. The command environment still
cannot open localhost/external sockets, so the restart was requested from the user.
After restart, inspect Connection for a 4096-token budget and try:

```text
Play one WAV recording for Kirchberg from 22/10/2026 00:00:00 to 22/10/2026 01:00:00 UTC.
Which stations have recordings from 22/10/2026 00:00:00 to 22/10/2026 01:00:00 UTC?
```

Optional independent checks:

```bash
python -m agent_app.diagnostics --wav Kirchberg 2026-10-22T00:00:00Z 2026-10-22T01:00:00Z
python -m agent_app.diagnostics --wav-chat
```

---

# Spectrogram follow-up: 28 October 2026

The restarted browser app now reports PyBRAMS ready. A fresh live browser
availability check completed in 32.6 seconds with two model requests. Its actual
trace showed `check_availability(search="Kirchberg", start_utc="2026-10-22T00:00:00Z",
end_utc="2026-10-22T01:00:00Z")` returning `status: ok`,
`recording_availability_checked: true`, receiver `LUKIRC_SYS001`, `coverage: full`
and `availability_percent: 100`. The final answer matched those results and cited
`https://brams.aeronomie.be/v4/file.php`. Tool retrieval time was
`2026-10-28T18:01:33.831440Z`.

The user's earlier follow-up response correctly refused to display a PNG
because no spectrogram tool existed at that point.

At that time, `get_spectrogram` used receiver metadata and PyBRAMS `File.stream`
plus `File.spectrogram`. It now selects from a direct archive listing and still
uses `File.spectrogram`. One actual archived PNG is validated and saved locally;
Gradio displays it as an image message with download/fullscreen controls. The
model only receives metadata, a source URL and an opaque artifact ID. The tool
requires a unique receiver and at most a 24-hour interval, selects an overlapping
recording, and does not download WAV data or analyze image pixels.

All **44 selected tests passed** in the project's installed Python 3.13 environment:

```bash
.venv-brams/bin/python -m unittest tests.test_prototype tests.test_receivers tests.test_availability tests.test_brams_sdk tests.test_spectrogram tests.test_spectrogram_sdk -v
```

Eight new tests cover the SDK PNG fetch path, no WAV loading, ambiguous receivers,
interval overlap, argument validation, failed downloads, PNG validation, opaque
artifact resolution, a two-turn conversation, and the actual Gradio callback's
image-message serialization and subsequent transcript reuse. Model/API responses
are fixtures; the tests do not establish a successful external archive download.
The socket-based HTTP integration test remains unavailable in the restricted
command environment, as recorded below.

**The new spectrogram path still needs a Gradio restart and live browser test.**
No additional packages are needed. Leave llama.cpp running, restart `python app.py`
in `.venv-brams`, reload the browser and start a fresh conversation. For an
independent direct or model-follow-up check:

```bash
python -m agent_app.diagnostics --spectrogram Kirchberg 2026-10-22T00:00:00Z 2026-10-22T01:00:00Z
python -m agent_app.diagnostics --spectrogram-chat
```

---

# Follow-up validation: 28 October 2026

The running browser app was inspected and tested against the user's real Ornith
server. The clock request succeeded in 18.2 seconds with two model requests.
The visible trace contained `get_current_utc_time` and the actual Python result;
the final answer matched its timestamp.

The running Gradio process uses the adjacent project's Python 3.12 environment.
Its PyBRAMS catalog query returns `setup_required: python_version`, and the
original code had no recording availability tool. Repeating the exact Kirchberg
question in a fresh browser session completed in 65.3 seconds with two model
requests: Ornith called `list_receivers(search="Kirchberg")`, received the setup
error, and explained that it could not establish archive availability.

This update adds a real PyBRAMS `File.availability` tool, receiver resolution,
hour-boundary alignment, per-receiver coverage summaries, UTC defaults,
source references, failure handling and direct/live-chat diagnostics.

Checks completed for this update:

- 26 protocol, wrapper and conversation tests passed under the installed Python
  3.12 chat environment. These use fixture model/API responses, including the
  exact Kirchberg question; they do not measure Ornith's selection accuracy.
- 8 tests passed in a Python 3.13 virtual environment with the actual installed
  PyBRAMS 0.1.7 SDK and fixture API responses. They cover SDK parsing, full/partial/
  zero coverage, numeric strings, invalid metrics, multiple receivers/hours,
  sub-hour alignment, empty results, unknown receivers and configuration restoration.
- Full test discovery under Python 3.12 ran 37 cases: 26 passed, 10 SDK-dependent
  cases skipped, and the HTTP integration case was blocked at socket binding by
  `PermissionError: Operation not permitted`. The 8 availability SDK tests among
  those skipped cases were run separately and passed as described above.

**The updated end-to-end Kirchberg query is not yet verified live.** The command
environment blocks localhost/external network access and package installation;
no live recording-availability result has been obtained. A Python 3.13
`.venv-brams` was created, but dependency installation and a Gradio restart are
still required. Leave llama.cpp running, stop Gradio, and run:

```bash
source .venv-brams/bin/activate
python -m ensurepip --upgrade
python -m pip install -r requirements-brams.txt
python -m unittest discover -v
python -m agent_app.diagnostics --availability Kirchberg 2026-10-22T00:00:00Z 2026-10-22T01:00:00Z
python -m agent_app.diagnostics --availability-chat
python app.py
```

Then open/reload Gradio in a fresh chat and repeat the exact question. Unspecified
BRAMS times now mean UTC, explicitly stated in the UI. Hourly archive metrics
cannot establish live station uptime or exact mixed coverage within part of an hour.

---

# Original prototype validation (before the availability update)

Validated on 28 October 2026 with Python 3.13.16, PyBRAMS 0.1.7,
Gradio 6.31.0, Pydantic AI Slim 2.63.0, and HTTPX 0.28.1.

`python -m unittest discover -v`: **24 tests passed**.
Seven receiver-wrapper compatibility tests also passed under Python 3.12.14;
actual SDK integration requires Python 3.13+.

The tests exercised the installed libraries, not replacement framework stubs:

- Local Chat Completions URL and exact preset model ID.
- Plain chat with tools disabled and server sampling defaults retained.
- Real clock and interval function execution with tool results returned to the model.
- ISO 8601 offset normalization and rejection of timezone-naive tool arguments.
- Retry feedback for invalid arguments and unregistered tool names.
- Complete tool-call/result history across turns and separate session histories.
- Bounded model/tool loops, failed requests, and token-limit truncation.
- Model-catalog diagnostics and API-key redaction in browser error messages.
- A launched Gradio HTTP server, its queue, and Gradio Client submitting chat
  messages through the actual HTTP provider to a local fixture model server.
- The exact question "Which receivers are available at Ecotron?" generating a
  `list_receivers` call with `search: "Ecotron"` and no time arguments.
- Real SDK catalog parsing against a mocked API, fresh-query behavior, and
  SDK configuration restoration after both success and failure.
- Missing Python/dependency setup, no search matches versus API errors,
  and explicit separation of catalog listing from recording availability.

Synthetic OpenAI-compatible responses drove those tests. No Ornith weights,
BRAMS recordings, paid services, or remote inference endpoints were used.
The HTTP fixture exists only under `tests/`; the application defaults to your
real llama.cpp endpoint and never falls back to fabricated model replies.

The server's final generated answer was displayed after each completed turn;
token streaming is outside this first prototype. No model reasoning text was
included in the tool trace.

Some non-fatal upstream Gradio/Pandas deprecation and test-process resource
warnings were emitted during the integration check. The checks completed.

Your particular llama.cpp build, GGUF chat template, and Ornith tool-selection
accuracy have not been tested here. After installation, run:

```bash
python -m agent_app.diagnostics --text --tools
```

Then use the example prompts in Gradio. Those are the live checks against your
model. Passing the synthetic integration tests does not establish model accuracy.

## Read-only live SDK check

The actual `list_receivers("Ecotron")` wrapper was also run against BRAMS through
PyBRAMS 0.1.7. It retrieved a catalog of 74 receivers and returned these five
matching entries on 28 October 2026:

| Receiver ID | Name |
| --- | --- |
| BEECON_SYS001 | Ecotron 1 |
| BEECON_SYS002 | Ecotron 2 |
| BEECON_SYS003 | Ecotron 3 |
| BEECON_SYS004 | Ecotron 4 |
| BEECON_SYS005 | Ecotron 5 |

Source: `https://brams.aeronomie.be/v4/receiver.php`.
The result explicitly contained `recording_availability_checked: false`.
No WAV or PNG archive files were downloaded. Catalog results may change.

The test environment requires its configured HTTP proxy for external access;
an initial attempt without it failed and was correctly reported as an API error,
not an empty receiver list. The successful live check used normal SDK networking.

