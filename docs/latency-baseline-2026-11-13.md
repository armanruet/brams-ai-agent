# Ornith latency baseline — 13 November 2026

This is an exploratory baseline, not a statistically representative benchmark or
an accuracy score. The test used one temporary conversation and the production
harness, with the running server already loaded. Its first request had a cold
**prompt cache**; this was not a measurement of model load or server startup.

- Hardware: Apple M2 Pro, 16 GB unified memory, AC power.
- llama.cpp: b12110-cd959bae9; one slot, 16,384-token context, Q4_K_M;
  `cache-reuse=256`, `cache-ram=1024`, `fit=on`, `reasoning-budget=128`.
- Prefix reuse was observed: 6,215 cached tokens of 6,311 on one request.
- One earlier snapshot showed about 7 GB swap in use. This alone did not establish
  active thrashing, and system memory conditions subsequently changed.

| Case | Total seconds | Model requests | Notes |
| --- | ---: | ---: | --- |
| General meteor explanation | 57.85 | 2 | Grounding incorrectly demanded detector evidence. |
| Ecotron catalog | 22.38 | 2 | Catalog tool: 2.90 s. |
| Single spectrogram | 14.61 | 1 | Direct answer; archive tool: 2.66 s. |
| Detection on prior image | 18.56 | 1 | Direct answer; detector tool: 0.30 s. |
| Detailed echo follow-up | 80.16 | 2 | Grounding rejected previous evidence. |
| WAV playback | 35.48 | 2 | WAV tool: 1.39 s; second model request: 24.38 s. |

The first request's 6,219-token prompt took 26.6 s to process and generated 295
output tokens in 13.9 s. Later requests reused cached prefix tokens. Token counts
in the app monitor are summed across requests; they are not necessarily one
request's context length.

For repeatable comparisons, record exact source hashes, model/server build,
request and context sizes, output tokens, cache hits, phase timings, memory and
active swap, power mode, model preload state, archive timing and tool outcomes.
Run cold and warm trials, repeat enough times to characterize variation, and grade
scientific correctness separately. Change one setting at a time. In particular,
this baseline did not compare `cache-ram=0` with `1024`; do not infer that either
is faster from this single run. Never combine results from the historical
capability suites with the new evaluation as if their rubrics were identical.

## Controlled cache probe — 14 November 2026

The originally loaded Ornith process used `cache-ram=1024`. A temporary model
server used the same settings and local model file with `cache-ram=0`; the
original model was then restored in the existing router. The preset was not
changed. This small synthetic probe used one slot and an A, A, B, A prompt
sequence; it is not an end-to-end task-quality or memory-pressure benchmark.

| Cache RAM | First A | Repeat A | Different B | Return to A |
| --- | ---: | ---: | ---: | ---: |
| 1024 MiB | 5.54 s | 0.35 s | 5.89 s | 0.42 s |
| 0 MiB | 5.29 s | 0.35 s | 5.83 s | 5.24 s |

With 1024 MiB the return to A reused 1,281 cached prompt tokens; with 0 MiB it
reused none. Both configurations reused 1,281 tokens on the immediate A repeat.
Thus disabling cache RAM kept immediate same-prefix reuse but lost the cached
state after switching to B in this probe. This does not establish peak memory
or the best setting for every workload; repeat representative workloads under
controlled memory conditions before changing the preset.

## Post-change live smoke check — 14 November 2026

One isolated conversation used the current application code, the live archive,
and the original restored Ornith server. The first request had a cold prompt
cache after the cache comparison. These are single observations, not comparable
averages or guarantees:

| Case | Total seconds | Model requests | Outcome |
| --- | ---: | ---: | --- |
| Detect one recording and show its image | 32.72 | 1 | Live archive result and annotated image; 4,793 prompt tokens, including 18.66 s prompt processing. |
| List detected times and confidence | <0.01 | 0 | Used the full verified result from the prior turn. |
| Play its WAV | 18.33 | 1 | Live original and normalized preview; 4,789 input tokens reused from cache. |

The earlier 18.56 s detection baseline was a warm-cache follow-up that reused a
previously downloaded image, so it is not directly comparable to this cold,
fresh-download detection. An independently reviewed, repeated evaluation is
still needed to assess quality and latency across query types and memory loads.

## Prompt alignment, catalog cache, and `v3` snapshot checks — 14 November 2026

Controlled `current_capabilities_v2` / `current_capabilities_failures_v2` snapshot
and controlled-fixture runs (`ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M`, `llama.cpp`
`b12110-cd959bae9`, `--server-pid 5434`) measured the prompt/schema trim (`12,379`
instruction+schema+description chars; fixed cached prefix reduced from `3,883` to
`3,618` tokens):

| Run | Turns | Failed | Model tool plan pass | Adj. | P50 (s) | P95 (s) | Input tokens | Output tokens | Model total (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pre-trim-failures-v2-ornith` (1 rep) | 5 | 0 | 1.00 | 0 | 22.21 | 28.93 | 41,287 | 2,136 | 113.22 |
| `post-trim-failures-v2-ornith` (1 rep) | 5 | 0 | 1.00 | 0 | 20.12 | 29.09 | 38,560 | 2,068 | 108.74 |
| `pre-trim-v2-ornith` (1 rep) | 9 | 0 | 1.00 | 0 | 11.56 | 19.09 | 36,102 | 2,061 | 105.34 |
| `post-trim-v2-ornith-r2` (3 reps) | 27 | 0 | 1.00 | 0 | 11.98 | 21.45 | 110,160 | 6,806 | 364.33 |
| `post-trim-v3-ornith-r2` (1 rep) | 11 | 0 | 1.00 | 0 | 11.98 | 22.52 | 45,130 | 2,653 | 133.18 |

Notes and limitations of these runs:

- All five runs above had `0` scientific flags, `0` model grounding flags, and
  `0` visible grounding flags (`manual_review.csv` remains ungraded until human
  scientific review). In the `v2` / `v3` snapshot runs, tool replay took `~0.00 s`
  total so archive/network variance is excluded; only `failures-v2` exercised the
  5 synthetic failure/unknown-availability fixtures (`5/5`).
- With a warm KV cache (`~0.24–0.39 s` prompt prefill vs. `~4–14 s` token
  generation per step), end-to-end turn latency is dominated by generated output
  length rather than the `-265` cached-prefix tokens.
- `benchmarks/current_capabilities_v3.json` (`10` cases / `11` turns) extends
  `v2` (`CUR01–CUR08`) with `CUR09` (`get_wav` audio playback/download) and
  `CUR10` (`convert_local_time` autumn Brussels DST conversion with `"Do not
  query recordings."`), after fixing negated-directive parsing in `grounding.py`
  and explicit receiver-ID matching during detection-window expansion in
  `service.py`.
- A direct live PyBRAMS probe verified the in-memory `120.0 s` bounded TTL
  catalog/location cache (`CATALOG_CACHE_TTL_SECONDS`): `list_receivers`
  dropped from `2.0102 s` (`74` receivers, `cached=False`) to `0.0011 s`
  (`cached=True`), and `query_geographic_receivers` dropped from `0.1073 s`
  (`catalog_cached=True, locations_cached=False`) to `0.0077 s` (both `True`).
  Live recording availability and file listings are never served from cache.
