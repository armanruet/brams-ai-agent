# BRAMS project diagrams

## Current architecture

The presentation image and interactive diagram describe the current application,
including the local vision detector and geographic station queries:

- [Presentation image — light PNG](readme-architecture.png) · [dark PNG](readme-architecture-dark.png)
- [Editable light SVG](readme-architecture.svg) · [editable dark SVG](readme-architecture-dark.svg)
- [Interactive architecture](readme-architecture.html) · [interactive specification](readme-architecture.json)

The presentation image is a separately designed, editable vector overview. The
interactive diagram uses the same underlying architecture with a different
layout; it supports light/dark themes, zoom, search and image export. Both HTML
and SVG files work offline. The presentation SVGs and PNGs can be regenerated
with `python3 docs/diagrams/render_overview.py` from the project root; this uses
the `rsvg-convert` command to produce PNGs.

## How a request is handled

1. **Ask:** A user asks a question in the Gradio workspace. The interface also
   shows the request monitor, annotated images and audio playback when available.
2. **Plan:** The Pydantic AI harness and turn service work with the selected
   local or hosted language model. The model can call registered Python tools.
   Argument validation, bounded requests, grounding checks and recovery help
   manage the turn; the service currently coordinates these policies.
3. **Retrieve and analyze:** BRAMS tools query the external archive for receiver
   metadata, hourly availability, archived recordings and spectrograms. Other
   tools handle Belgian geographic queries, time conversion, beacon metadata,
   waveform metadata and PSD analysis.
4. **Detect candidate echoes:** When requested, the detector runs the local
   YOLOv11-n model with ONNX Runtime on an archived PNG or a WAV-derived
   spectrogram. It returns candidate boxes, counts, confidence scores and
   estimated times. A multi-slot scan reports what was inspected and whether
   requested coverage is incomplete.
5. **Present:** Gradio displays the answer and actual local media, including
   annotated detection images. A straightforward detection can use a concise
   deterministic answer without an additional language-model request.

The application registers twelve tools: `query_geographic_receivers`,
`detect_meteors`, `get_wav`, `get_spectrogram`, `check_availability`,
`list_receivers`, `get_beacon`, `get_recordings`, `get_psd`,
`convert_local_time`, `get_current_utc_time`, and `inspect_time_window`.
See [harness.py](../../agent_app/harness.py) for the current registration.

## Boundaries and limitations

- BRAMS is an external archive; results depend on archive availability and
  ingestion, and do not represent a continuous live stream. Hourly archive
  coverage is not the same as live receiver uptime.
- The language model may run locally or through a configured hosted endpoint.
  The meteor detector runs locally. Media bytes remain local; the selected
  language-model endpoint receives text and structured tool results.
- The meteor model reports **candidate** echoes. Its times are estimates based
  on bounding boxes, and false positives and missed echoes are possible. Missing
  or uninspected recordings limit coverage. See the
  [model preparation overview](../../models/README.md).
- `.artifacts/` stores local media and provenance. Conversation history is held
  in per-browser process memory. The media component represents files, not a SQL
  database; fresh BRAMS queries bypass PyBRAMS metadata caching.
- The default Belgian directional regions use geographic half-extents, not
  administrative regions. Geographic results disclose this definition.

## Validation and source evidence

The interactive architecture passed all **9/9 showcase artifact checks** with
zero composition errors or warnings. Its automated browser check passed at
1440×900, 1600×1000, 1920×1080 and 2048×1320, with light and dark screenshots
at the smallest and largest sizes. The final interactive viewer and both
presentation images were also visually inspected.

Local build and validation receipts are excluded from Git via `.gitignore`:

- `readme-architecture.sources.json` identifies the code files used to ground
  the architecture.
- `readme-architecture.handoff.json` records file hashes, validation and review
  status.
- `readme-architecture.visual-check.json` and `readme-architecture.visual-check.html`
  record automated browser checks and screenshots for both themes at the tested
  viewports.

## Earlier snapshots

[Earlier architecture](brams-architecture.html) and [earlier data flow](brams-dataflow.html)
are retained as historical snapshots; they predate the current hosted-model,
scientific-tool and vision integration and should not be used as the current
system description. Their historical specifications and validation receipts
remain alongside them.
