"""Small, escaped UI views for observed timings; never model-generated HTML."""

from html import escape
import json
from pathlib import Path


LAYOUT_JS = "() => { (" + Path(__file__).with_name("splitter.js").read_text() + ")(); (" + Path(__file__).with_name("chat_controls.js").read_text() + ")(); (" + Path(__file__).with_name("image_viewer.js").read_text() + ")(); }"


def current_model_view(model):
    name = escape(model, quote=True)
    return ('<div class="monitor-model"><span>Current model</span>'
            f'<strong title="{name}">{name}</strong></div>')


def totals(steps, elapsed):
    result = {"model": 0.0, "tool": 0.0, "requests": 0, "calls": 0, "input": 0, "output": 0}
    for step in steps:
        kind = step.get("kind")
        if kind not in {"model", "tool"}:
            continue
        duration = step.get("duration_seconds")
        if duration is None:
            duration = max(0, elapsed-step.get("started_at_seconds", elapsed))
        result[kind] += duration
        result["requests" if kind == "model" else "calls"] += 1
        result["input"] += step.get("input_tokens", 0)
        result["output"] += step.get("output_tokens", 0)
    return result


def timing_view(steps=(), elapsed=0, state="Ready", requests=None, tokens=None):
    values = totals(steps, elapsed)
    count = values["requests"] if requests is None else requests
    incoming, outgoing = tokens if tokens is not None else (values["input"], values["output"])
    return (f'<section class="timing-card" aria-label="Request timing"><div class="timing-heading">'
            f'<span>REQUEST TIMING</span><span class="run-state">{escape(state)}</span></div>'
            f'<div class="elapsed-value">{elapsed:.2f}<span>s</span></div><div class="elapsed-caption">Total elapsed</div>'
            f'<div class="timing-grid"><div><span>Model requests</span><strong>{values["model"]:.2f}s</strong><small>{count} request(s)</small></div>'
            f'<div><span>Tool execution</span><strong>{values["tool"]:.2f}s</strong><small>{values["calls"]} call(s)</small></div></div>'
            f'<div class="token-line">{incoming:,} input tokens <span>·</span> {outgoing:,} output tokens</div></section>')


def activity_view(steps=(), elapsed=0, state="Ready"):
    if not steps:
        message = "Waiting for the first model request…" if state == "Running" else "Model requests and tool calls will appear here as they happen."
        return f'<div class="activity-empty"><span class="activity-orbit">◎</span><strong>{escape(state)}</strong><p>{message}</p></div>'
    rows = []
    for step in steps:
        status = step.get("status", "running")
        duration = step.get("duration_seconds")
        if duration is None:
            duration = max(0, elapsed-step.get("started_at_seconds", elapsed))
        badge = {"completed": "Done", "running": "Running", "error": "Failed", "blocked": "Blocked", "cancelled": "Stopped"}.get(status, status)
        kind = "MODEL" if step["kind"] == "model" else "PYTHON TOOL"
        details = ""
        if "arguments" in step:
            arguments = escape(json.dumps(step["arguments"], indent=2, default=str))
            details = f'<details><summary>Validated arguments</summary><pre>{arguments}</pre></details>'
        if kind == "MODEL" and step.get("cached_input_tokens") is not None:
            cached = int(step["cached_input_tokens"])
            uncached = int(step.get("uncached_input_tokens", 0))
            details += (f'<p class="finding-note">Input tokens: {cached:,} cached · '
                        f'{uncached:,} newly processed</p>')
        if kind == "MODEL" and step.get("prompt_seconds") is not None and step.get("generation_seconds") is not None:
            details += (f'<p class="finding-note">Prompt processing: {step["prompt_seconds"]:.2f}s · '
                        f'Generation: {step["generation_seconds"]:.2f}s</p>')
        finding = step.get("result_status")
        if finding and finding != "ok":
            details += f'<p class="finding-note">Result: {escape(str(finding).replace("_", " "))}</p>'
        rows.append(f'<article class="activity-step {escape(status)}"><div class="step-topline"><span>{kind}</span><time>{duration:.2f}s</time></div>'
                    f'<strong>{escape(step["name"])}</strong><span class="step-status">{escape(badge)}</span>{details}</article>')
    return '<div class="activity-list">'+"".join(rows)+'</div>'


def history_view(history=()):
    if not history:
        return '<div class="activity-empty"><strong>No requests yet</strong><p>Completed and failed turns, including their total times, appear here.</p></div>'
    rows = []
    for item in reversed(history):
        rows.append(f'<tr><td>#{item["id"]}</td><td>{item["elapsed"]:.2f}s</td><td>{item["model"]:.2f}s</td><td>{item["tool"]:.2f}s</td></tr>'
                    f'<tr class="history-caption"><td colspan="4">{escape(item["question"][:90])} · {escape(item["status"])}</td></tr>')
    return '<div class="history-table"><table><thead><tr><th>Turn</th><th>Total</th><th>Model</th><th>Tools</th></tr></thead><tbody>'+"".join(rows)+'</tbody></table></div>'


CSS = """
html { scrollbar-gutter: stable; }
.gradio-container { width: 100% !important; max-width: 1560px !important; min-width: 0 !important; margin: 0 auto !important; box-sizing: border-box; }
.gradio-container > .wrap, .gradio-container main.contain { width: 100%; min-width: 0; box-sizing: border-box; }
#workspace-topbar { display: grid !important; grid-template-columns: minmax(0, 1fr) auto auto; width: 100% !important; box-sizing: border-box; padding: 4px 0 14px; align-items: center; border-bottom: 1px solid var(--border-color-primary); gap: 10px; }
#new-chat, #model-settings { white-space: nowrap !important; min-height: 38px; padding: 8px 14px !important; line-height: 1.2; }
#workspace-topbar > .gap { gap: 3px; }
.workspace-brand { display: flex; gap: 10px; align-items: center; color: var(--body-text-color); font-size: 15px; font-weight: 700; letter-spacing: -.02em; }
.brand-mark { display: grid; place-items: center; width: 33px; height: 33px; border-radius: 10px; background: #0f766e; color: white; font-size: 19px; }
.brand-detail { display: block; font-size: 11px; font-weight: 400; color: var(--body-text-color-subdued); margin-top: 1px; letter-spacing: normal; }
.chat-controls-hidden { display: none !important; }
#conversation button[aria-label="Clear"], #conversation button[aria-label="Clear conversation"] { display: none !important; }
.brams-message-actions { display: inline-flex; align-items: center; }
.brams-message-actions.standalone { align-self: flex-end; margin-top: 4px; }
.brams-delete-message { display: inline-flex; justify-content: center; align-items: center; width: 26px; height: 26px; padding: 4px; border: 0; border-radius: 5px; background: transparent; color: var(--body-text-color-subdued); cursor: pointer; }
.brams-delete-message svg { width: 16px; height: 16px; }
.brams-delete-message:hover:not(:disabled) { color: #dc2626; background: rgb(220 38 38 / .08); }
.brams-delete-message:focus-visible { outline: 2px solid #0d9488; outline-offset: 2px; }
.brams-delete-message:disabled { opacity: .35; cursor: default; }
#workspace-layout { display: grid !important; grid-template-columns: minmax(0, var(--chat-fr, 3fr)) 18px minmax(0, var(--monitor-fr, 2fr)); column-gap: 0 !important; align-items: start !important; width: 100% !important; min-width: 0 !important; box-sizing: border-box; padding-top: 6px; }
#workspace-layout > .workspace-main, #workspace-layout > .workspace-sidebar { width: 100% !important; min-width: 0 !important; max-width: 100% !important; box-sizing: border-box; }
.workspace-main { gap: 8px !important; padding-right: 12px !important; }
.workspace-sidebar { gap: 12px !important; padding-left: 14px !important; }
#workspace-divider { width: 18px !important; min-width: 0 !important; height: 100%; align-self: stretch; padding: 0 !important; background: transparent !important; border: 0 !important; }
#workspace-divider .html-container { height: 100%; }
#workspace-split-handle { position: relative; width: 18px; height: 100%; min-height: 440px; cursor: col-resize; touch-action: none; outline: none; border-radius: 6px; }
#workspace-split-handle::before { content: ""; position: absolute; top: 0; bottom: 0; left: 8px; width: 1px; background: var(--border-color-primary); }
#workspace-split-handle > span { position: absolute; left: 5px; top: 45%; width: 7px; height: 38px; border-radius: 5px; background: var(--border-color-primary); }
#workspace-split-handle:hover, #workspace-split-handle:focus-visible, .is-resizing #workspace-split-handle { background: rgb(13 148 136 / .08); }
#workspace-split-handle:hover > span, #workspace-split-handle:focus-visible > span, .is-resizing #workspace-split-handle > span { background: #0d9488; }
#workspace-split-handle:focus-visible { outline: 2px solid #0d9488; outline-offset: 1px; }
#workspace-layout.workspace-stacked { grid-template-columns: minmax(0, 1fr); gap: 18px !important; }
#workspace-layout.workspace-stacked #workspace-divider { display: none !important; }
#workspace-layout.workspace-stacked .workspace-main { padding-right: 0 !important; }
#workspace-layout.workspace-stacked .workspace-sidebar { border-top: 1px solid var(--border-color-primary); padding-left: 0 !important; padding-top: 18px !important; }
#conversation { width: 100% !important; min-width: 0 !important; height: max(360px, calc(100dvh - 315px)) !important; min-height: 360px !important; flex: 0 0 auto !important; border: 0 !important; background: transparent !important; border-radius: 0; box-shadow: none !important; }
#conversation .brams-viewable-image { cursor: zoom-in; }
#conversation .brams-viewable-image:focus-visible { outline: 3px solid #0d9488; outline-offset: 3px; }
.brams-image-viewer { box-sizing: border-box; width: min(96vw, 1500px); max-width: 96vw; max-height: 94dvh; padding: 0; border: 1px solid #64748b; border-radius: 12px; background: #111827; color: white; box-shadow: 0 20px 80px rgb(0 0 0 / .5); overflow: hidden; }
.brams-image-viewer::backdrop { background: rgb(0 0 0 / .78); }
.brams-image-viewer-toolbar { display: flex; align-items: center; gap: 10px; padding: 10px 14px; }
.brams-image-viewer-caption { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; }
.brams-image-viewer-toolbar button { padding: 7px 12px; border: 1px solid #94a3b8; border-radius: 7px; background: #1f2937; color: white; white-space: nowrap; cursor: pointer; }
.brams-image-viewer-toolbar button:focus-visible { outline: 2px solid #5eead4; outline-offset: 2px; }
.brams-image-viewer-viewport { display: flex; align-items: safe center; justify-content: safe center; width: 100%; height: min(82dvh, 1000px); overflow: auto; }
.brams-image-viewer-viewport img { display: block; width: auto; height: auto; max-width: 100%; max-height: 100%; object-fit: contain; }
.brams-image-viewer-viewport.is-actual-size img { max-width: none; max-height: none; }
#conversation .message { line-height: 1.75; font-size: 15px; }
.chat-empty { padding: 45px 20px; text-align: center; color: var(--body-text-color-subdued); }
.chat-empty strong { display: block; font-size: clamp(22px, 2.2vw, 29px); letter-spacing: -.035em; font-weight: 600; margin: 17px 0 12px; color: var(--body-text-color); }
.chat-empty p { max-width: 360px; margin: auto; line-height: 1.7; font-size: 14px; }
.chat-empty .empty-symbol { font-size: 32px; color: #0d9488; }
.starter-hints { display: flex; justify-content: center; flex-wrap: wrap; gap: 8px; margin-top: 22px; }
.starter-hints span { font-size: 11px; border: 1px solid var(--border-color-primary); border-radius: 20px; padding: 8px 10px; }
#message-composer { --block-background-fill: transparent; --input-background-fill: transparent; --input-border-width: 0px; --input-shadow: none; --input-shadow-focus: none; width: 100% !important; min-width: 0 !important; box-sizing: border-box; padding: 10px 14px !important; border: 1px solid var(--border-color-primary) !important; border-radius: 20px !important; background: var(--background-fill-primary); box-shadow: 0 4px 18px rgb(0 0 0 / .035); }
#message-composer:focus-within { border-color: #0d9488 !important; box-shadow: 0 0 0 2px rgb(13 148 136 / .14); }
#message-composer textarea { width: 100%; border: 0 !important; outline: none !important; box-shadow: none !important; padding: 7px 2px; font-size: 15px; background: transparent !important; resize: none; }
#composer-actions { display: flex !important; flex-direction: row !important; flex-wrap: nowrap !important; gap: 12px !important; align-items: center; padding-top: 4px; background: transparent !important; }
#tools-toggle { flex: 1 1 auto !important; min-width: 0 !important; border: 0 !important; background: transparent !important; box-shadow: none !important; }
#tools-toggle label { border: 0 !important; padding: 4px 0 !important; background: transparent !important; box-shadow: none !important; }
#send-message { display: inline-flex !important; align-items: center; justify-content: center; flex: 0 0 80px !important; width: 80px !important; min-width: 80px !important; max-width: 80px !important; height: 38px !important; min-height: 38px !important; padding: 0 14px !important; box-sizing: border-box; white-space: nowrap !important; word-break: normal !important; line-height: 1 !important; border: 1px solid #0f766e !important; border-radius: 12px !important; background: #0f766e !important; color: white !important; font-size: 13px; font-weight: 600; box-shadow: none !important; }
#send-message:hover:not(:disabled) { background: #115e59 !important; border-color: #115e59 !important; }
#send-message:focus-visible { outline: 2px solid #0d9488; outline-offset: 3px; }
#send-message:disabled { opacity: .5; cursor: not-allowed; }
.chat-footnote { text-align: center; color: var(--body-text-color-subdued); font-size: 10px !important; }
#connection-drawer { font-size: 13px; }
#connection-drawer h2 { font-size: 20px; letter-spacing: -.03em; }
#monitor-current-model { min-width: 0 !important; padding: 0 !important; margin: 0 !important; }
.monitor-model { display: flex; flex-direction: column; align-items: flex-end; gap: 5px; min-width: 0; }
.monitor-model > span { font-size: 10px; color: var(--body-text-color-subdued); }
.monitor-model > strong { max-width: 100%; padding: 7px 10px; border: 1px solid rgb(124 58 237 / .18); border-radius: 8px; background: rgb(124 58 237 / .08); color: var(--body-text-color); font-size: 13px; font-weight: 600; line-height: 1.5; overflow-wrap: anywhere; text-align: right; }
.monitor-heading { display: flex; gap: 8px; align-items: center; padding: 4px 0; color: var(--body-text-color); }
.monitor-heading strong { font-size: 14px; font-weight: 600; }
.monitor-heading > span:last-child { margin-left: auto; font-size: 10px; color: var(--body-text-color-subdued); }
.monitor-live-dot { width: 7px; height: 7px; border-radius: 100%; background: #0d9488; }
#monitor-tabs { border: 0; }
.context-note { font-size: 12px !important; color: var(--body-text-color-subdued) !important; }
#request-status { font-size: 12px; overflow-wrap: anywhere; }
.timing-card { padding: 20px; border: 1px solid var(--border-color-primary); border-radius: 14px; background: var(--background-fill-primary); }
.timing-heading { display: flex; align-items: center; justify-content: space-between; gap: 8px; color: var(--body-text-color-subdued); font-size: 11px; font-weight: 700; letter-spacing: .09em; }
.run-state { color: #0f766e; background: #ccfbf1; border-radius: 100px; padding: 5px 9px; letter-spacing: normal; }
.elapsed-value { font-size: 52px; line-height: 1.1; font-weight: 650; letter-spacing: -.045em; margin-top: 18px; color: var(--body-text-color); font-variant-numeric: tabular-nums; }
.elapsed-value > span { font-size: 24px; margin-left: 5px; color: var(--body-text-color-subdued); font-weight: 400; }
.elapsed-caption { margin-top: 4px; font-size: 13px; color: var(--body-text-color-subdued); }
.timing-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 22px 0 18px; }
.timing-grid > div { border-top: 1px solid var(--border-color-primary); padding-top: 14px; }
.timing-grid span, .timing-grid small { display: block; font-size: 12px; color: var(--body-text-color-subdued); }
.timing-grid strong { display: block; font-size: 23px; margin: 5px 0 2px; color: var(--body-text-color); font-variant-numeric: tabular-nums; }
.token-line { font-size: 11px; color: var(--body-text-color-subdued); }
.token-line span { margin: 0 8px; }
.activity-list { max-height: max(230px, calc(100dvh - 465px)); overflow-y: auto; padding: 6px 3px; }
.activity-step { position: relative; padding: 14px 14px 14px 18px; margin: 0 0 10px; border: 1px solid var(--border-color-primary); border-left: 3px solid #0d9488; border-radius: 9px; background: var(--background-fill-primary); }
.step-topline { display: flex; justify-content: space-between; color: var(--body-text-color-subdued); font-size: 10px; letter-spacing: .05em; margin-bottom: 7px; }
.step-topline time { font-size: 12px; letter-spacing: normal; font-variant-numeric: tabular-nums; }
.activity-step > strong { display: block; overflow-wrap: anywhere; font-size: 13px; color: var(--body-text-color); }
.step-status { display: inline-block; font-size: 11px; color: var(--body-text-color-subdued); margin-top: 6px; }
.activity-step.running { border-left-color: #3b82f6; }
.activity-step.running .step-status { color: #2563eb; }
.activity-step.error, .activity-step.blocked, .activity-step.cancelled { border-left-color: #d97706; }
.activity-step details { margin-top: 10px; font-size: 11px; color: var(--body-text-color-subdued); }
.activity-step summary { cursor: pointer; }
.activity-step pre { white-space: pre-wrap; overflow-wrap: anywhere; padding: 10px 0; max-height: 190px; overflow-y: auto; }
.finding-note { margin-top: 8px; font-size: 11px; color: var(--body-text-color-subdued); }
.activity-empty { padding: 32px 18px; text-align: center; color: var(--body-text-color-subdued); }
.activity-empty strong { display: block; font-size: 13px; margin-bottom: 10px; }
.activity-empty p { font-size: 12px; line-height: 1.7; }
.activity-orbit { display: block; font-size: 27px; color: #0d9488; margin-bottom: 14px; }
.history-table { max-height: max(230px, calc(100dvh - 465px)); overflow-y: auto; }
.history-table table { width: 100%; border-collapse: collapse; font-size: 12px; color: var(--body-text-color); }
.history-table th, .history-table td { text-align: left; padding: 9px 5px; font-variant-numeric: tabular-nums; }
.history-table th { color: var(--body-text-color-subdued); font-weight: 500; }
.history-caption td { font-size: 11px; color: var(--body-text-color-subdued); border-bottom: 1px solid var(--border-color-primary); padding-top: 0; overflow-wrap: anywhere; }
@media (max-width: 900px) { #workspace-layout { grid-template-columns: minmax(0, 1fr); gap: 18px !important; } #workspace-divider { display: none !important; } .workspace-main { padding-right: 0 !important; } .workspace-sidebar { border-top: 1px solid var(--border-color-primary); padding-left: 0 !important; padding-top: 18px !important; } .elapsed-value { font-size: 42px; } }
@media (max-width: 600px) { #workspace-topbar { grid-template-columns: minmax(0, 1fr) auto; } #workspace-topbar > .column { grid-column: 1 / -1; } #new-chat, #model-settings { width: 100%; } .starter-hints { gap: 5px; } }
"""
