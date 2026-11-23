"""Run with: python app.py"""

import logging
import inspect
import os
import asyncio
from contextlib import suppress
from dataclasses import replace
import threading
from time import perf_counter

# This prototype has no need to send Gradio usage analytics.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from agent_app.cancellation import OperationCancelled, cancellation_scope
from agent_app.config import Settings, load_settings
from agent_app.connections import ConnectionProfiles, OPENAI_BASE_URL, connection_label, apply_profile, redact
from agent_app.brams_tools import pybrams_setup_status
from agent_app.diagnostics import check_server
from agent_app.service import Session, ask, friendly_error
from agent_app.chat_history import action_manifest, bind_text_refs, existing_refs, message_ref, remove_message
from agent_app.presentation import CSS, LAYOUT_JS, timing_view, activity_view, history_view, totals, current_model_view


LOGGER = logging.getLogger(__name__)


def build_ui(settings: Settings | None = None) -> gr.Blocks:
    settings = settings or load_settings()

    async def respond(message, transcript, session, enable_tools, profiles=None, history=None, cancel_holder=None):
        active = profiles.active if profiles else settings
        original_transcript = list(transcript or [])
        transcript = list(original_transcript)
        original_session = session or Session()
        session = original_session
        history = list(history or [])
        if not message or not message.strip():
            yield (
                transcript, session, gr.update(), "Enter a message first.", [],
                gr.update(), gr.update(), gr.update(),
                gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                gr.update(visible=False, interactive=False),
            )
            return
        refs = existing_refs(transcript, session)
        user_ref = message_ref("user")
        refs.append(user_ref)
        transcript.append({"role": "user", "content": message.strip()})
        pending_session = replace(session, ui_messages=[dict(ref) for ref in refs])
        cancel_event = threading.Event()
        async_cancel = asyncio.Event()
        loop = asyncio.get_running_loop()
        started = perf_counter()
        observed = {}
        if isinstance(cancel_holder, dict):
            cancel_holder.update(
                event=cancel_event,
                async_event=async_cancel,
                loop=loop,
                observed=observed,
                started=started,
                pending_len=len(transcript),
                question=redact(message.strip(), active.api_key),
            )
        yield (
            transcript, pending_session, gr.update(value="", interactive=False),
            f"Waiting for {connection_label(active)}…", [], gr.update(interactive=False),
            gr.update(interactive=False), gr.update(interactive=False),
            timing_view(state="Running"), activity_view(state="Running"), history_view(history), history,
            gr.update(visible=True, interactive=True),
        )
        events = asyncio.Queue()
        ask_kwargs = {"on_event": events.put_nowait}
        try:
            params = inspect.signature(ask).parameters
            if "cancel_event" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
                ask_kwargs["cancel_event"] = cancel_event
        except (TypeError, ValueError):
            ask_kwargs["cancel_event"] = cancel_event
        with cancellation_scope(cancel_event):
            run_task = asyncio.create_task(ask(message, session, active, bool(enable_tools), **ask_kwargs))
        cancel_waiter = asyncio.create_task(async_cancel.wait())

        def drain():
            while not events.empty():
                event = redact(events.get_nowait(), active.api_key)
                observed[event["id"]] = event
            return list(observed.values())

        def finish_views(steps, elapsed, state, requests=None, tokens=None):
            values = totals(steps, elapsed)
            entry = {"id": history[-1]["id"]+1 if history else 1,
                     "question": redact(message.strip(), active.api_key), "elapsed": elapsed,
                     "model": values["model"], "tool": values["tool"], "status": state}
            updated = [*history, entry][-20:]
            return (timing_view(steps, elapsed, state, requests, tokens), activity_view(steps, elapsed, state),
                    history_view(updated), updated)

        try:
            while not run_task.done():
                if cancel_event.is_set() or async_cancel.is_set():
                    run_task.cancel()
                    break
                await asyncio.wait({run_task, cancel_waiter}, timeout=0.25, return_when=asyncio.FIRST_COMPLETED)
                if cancel_event.is_set() or async_cancel.is_set():
                    run_task.cancel()
                    break
                steps = drain()
                if not run_task.done():
                    elapsed = perf_counter()-started
                    running = next((step for step in reversed(steps) if step["status"] == "running"), None)
                    phase = running["name"] if running else "Preparing request"
                    yield (transcript, pending_session, gr.skip(), f"{phase} · {elapsed:.2f}s elapsed", gr.skip(),
                           gr.skip(), gr.skip(), gr.skip(), timing_view(steps, elapsed, "Running"),
                           activity_view(steps, elapsed, "Running"), gr.skip(), gr.skip(), gr.skip())
            reply = await run_task
            steps = redact(reply.timings or drain(), active.api_key)
            # calls = sum(event.get("event") == "tool_call" for event in reply.trace)
            # footer = f"\n\n*{reply.elapsed_seconds:.2f} s · {reply.requests} model request(s) · {calls} tool call(s)*"
            # transcript.append({"role": "assistant", "content": redact(reply.text+footer, active.api_key)})
            transcript.append({"role": "assistant", "content": redact(reply.text, active.api_key)})
            answer_ref = message_ref("assistant")
            refs.append(answer_ref)
            for path in reply.images:
                refs.append(message_ref("media"))
                transcript.append({"role": "assistant", "content": gr.Image(
                    value=path, format="png", interactive=False,
                    label="Archived BRAMS spectrogram", buttons=["download", "fullscreen"],
                )})
            for detection in reply.detections:
                refs.append(message_ref("media"))
                transcript.append({"role": "assistant", "content": gr.Image(
                    value=detection["path"], format="png", interactive=False,
                    label=detection["label"], buttons=["download", "fullscreen"],
                )})
            for audio in reply.audio:
                refs.extend([message_ref("media"), message_ref("media")])
                transcript.append({"role": "assistant", "content": gr.Audio(
                    value=audio["preview"], interactive=False, autoplay=False,
                    label=audio.get("label") or "Listening preview (normalized PCM16)", buttons=["download"],
                )})
                transcript.append({"role": "assistant", "content": {
                    "file": {"path": audio["original"], "orig_name": audio["filename"],
                             "mime_type": "application/octet-stream"},
                    "alt_text": f"Download unchanged original WAV: {audio['filename']}",
                }})
            # status_text = (
            #     f"{connection_label(active)} · {reply.elapsed_seconds:.1f}s · "
            #     f"{reply.requests} model request(s) · "
            #     f"{sum(event.get('event') == 'tool_call' for event in reply.trace)} tool call(s) · "
            #     f"{reply.input_tokens} input / {reply.output_tokens} output tokens"
            # )
            status_text = f"{connection_label(active)} · Ready."
            if reply.warning:
                status_text += f" · {reply.warning}"
            yield (
                transcript, bind_text_refs(refs, user_ref, answer_ref, len(session.messages), reply.session, message.strip()),
                gr.update(interactive=True), status_text,
                redact(reply.trace, active.api_key), gr.update(interactive=True), gr.update(interactive=True),
                gr.update(interactive=True),
                *finish_views(steps, reply.elapsed_seconds, "Recovered" if reply.warning else "Completed",
                              reply.requests, (reply.input_tokens, reply.output_tokens)),
                gr.update(visible=False, interactive=False),
            )
        except (asyncio.CancelledError, OperationCancelled):
            elapsed = perf_counter() - started
            steps = drain()
            for step in steps:
                if step.get("status") == "running":
                    step["status"] = "cancelled"
                    step["duration_seconds"] = max(0, elapsed - step.get("started_at_seconds", elapsed))
            yield (
                original_transcript, original_session, gr.update(interactive=True),
                "Request cancelled; earlier conversation preserved.",
                [],
                gr.update(interactive=True), gr.update(interactive=True),
                gr.update(interactive=True),
                *finish_views(steps, elapsed, "Stopped"),
                gr.update(visible=False, interactive=False),
            )
        except Exception as error:
            LOGGER.warning("Chat request failed: %s", type(error).__name__)
            refs = refs[:len(transcript)]
            transcript.append({"role": "assistant", "content": redact(friendly_error(error, active), active.api_key)})
            elapsed = getattr(error, "elapsed_seconds", perf_counter()-started)
            steps = redact(getattr(error, "timings", None) or drain(), active.api_key)
            # transcript[-1]["content"] += f"\n\n*Request stopped after {elapsed:.2f} s.*"
            refs.append(message_ref("assistant"))
            yield (
                transcript, replace(session, ui_messages=refs), gr.update(interactive=True),
                "Request failed; earlier conversation preserved. See Tool activity.",
                redact(getattr(error, "tool_trace", []), active.api_key),
                gr.update(interactive=True), gr.update(interactive=True),
                gr.update(interactive=True),
                *finish_views(steps, elapsed, "Failed"),
                gr.update(visible=False, interactive=False),
            )
        finally:
            if not cancel_waiter.done():
                cancel_waiter.cancel()
                with suppress(asyncio.CancelledError):
                    await cancel_waiter
            if not run_task.done():
                cancel_event.set()
                run_task.cancel()
                with suppress(asyncio.CancelledError):
                    await run_task
            if isinstance(cancel_holder, dict) and cancel_holder.get("event") is cancel_event:
                cancel_holder["event"] = None
                cancel_holder["async_event"] = None
                cancel_holder["loop"] = None
                cancel_holder["pending_len"] = None

    def cancel_chat(cancel_holder, transcript, session, history):
        transcript = list(transcript or [])
        session = session or Session()
        history = list(history or [])
        elapsed = 0.0
        steps = []
        question = ""
        if isinstance(cancel_holder, dict):
            event = cancel_holder.get("event")
            if event is not None:
                event.set()
            async_event = cancel_holder.get("async_event")
            target_loop = cancel_holder.get("loop")
            if async_event is not None and target_loop is not None and not target_loop.is_closed():
                target_loop.call_soon_threadsafe(async_event.set)
            pending_len = cancel_holder.get("pending_len")
            if pending_len is not None and len(transcript) == pending_len and transcript:
                transcript = transcript[:-1]
                if len(session.ui_messages) == pending_len:
                    session = replace(session, ui_messages=list(session.ui_messages[:-1]))
            started_at = cancel_holder.get("started")
            if started_at is not None:
                elapsed = max(0.0, perf_counter() - started_at)
            observed = cancel_holder.get("observed") or {}
            for step in observed.values():
                item = dict(step)
                if item.get("status") == "running":
                    item["status"] = "cancelled"
                    item["duration_seconds"] = max(0.0, elapsed - item.get("started_at_seconds", elapsed))
                steps.append(item)
            question = cancel_holder.get("question", "")
            cancel_holder["event"] = None
            cancel_holder["async_event"] = None
            cancel_holder["loop"] = None
            cancel_holder["pending_len"] = None
        values = totals(steps, elapsed)
        updated = history
        if question:
            entry = {"id": history[-1]["id"] + 1 if history else 1,
                     "question": question, "elapsed": elapsed,
                     "model": values["model"], "tool": values["tool"], "status": "Stopped"}
            updated = [*history, entry][-20:]
        return (
            transcript, session, gr.update(interactive=True),
            "Request cancelled; earlier conversation preserved.",
            gr.update(interactive=True), gr.update(interactive=True),
            gr.update(interactive=True), gr.update(visible=False, interactive=False),
            timing_view(steps, elapsed, "Stopped"), activity_view(steps, elapsed, "Stopped"),
            history_view(updated), updated,
        )

    def reset_chat():
        return [], Session(), "", "Ready.", [], timing_view(), activity_view(), history_view(), []

    def toggle_connection(is_open):
        return gr.update(open=not bool(is_open)), not bool(is_open)

    def close_connection():
        return gr.update(open=False), False

    def message_actions(transcript, session):
        return action_manifest(transcript, session or Session())

    def delete_message(payload, transcript, session):
        removed = remove_message(payload, transcript, session or Session())
        if removed is None:
            return gr.skip(), gr.skip(), gr.skip(), ""
        updated_transcript, updated_session = removed
        return updated_transcript, updated_session, "Message deleted.", ""

    def change_profile(provider, profiles):
        profiles = profiles or ConnectionProfiles.initial(settings)
        draft = profiles.local if provider == "local" else (profiles.saved_hosted.get(provider) or profiles.saved_hosted.get("openai"))
        url = draft.base_url if draft else OPENAI_BASE_URL
        model = draft.model if draft else ""
        return (gr.update(value=url, interactive=provider != "local"),
                gr.update(value=model or None, choices=[model] if model else [], interactive=True),
                gr.update(value="", visible=provider != "local",
                          placeholder="Leave blank to reuse the key for this saved endpoint" if draft and provider != "local" else "Paste API key"),
                gr.update(value=draft.token_parameter if draft else "auto", interactive=provider != "local"),
                gr.update(visible=provider == "local"))

    def apply_connection(provider, url, model, key, token_parameter, profiles):
        profiles = profiles or ConnectionProfiles.initial(settings)
        try:
            if provider == "local":
                model = (model or "").strip()
                if not model:
                    raise ValueError("Choose a local model ID.")
                profiles = replace(profiles, local=replace(profiles.local, model=model))
            elif provider == "compatible" and (url or "").strip().rstrip("/") == OPENAI_BASE_URL:
                provider = "openai"
            updated = apply_profile(profiles, provider, url, model, key, token_parameter)
        except ValueError as error:
            return (gr.skip(),)*8 + (str(error), {}) + (gr.skip(),)*6
        return (updated, [], Session(), "", "Connection applied; a fresh chat is ready.", [], "",
                False, "Connection applied. Use Check server or Test tool calling.", {},
                timing_view(), activity_view(), history_view(), [],
                current_model_view(redact(updated.active.model, updated.active.api_key)), gr.update(open=False))

    async def refresh_local_models(profiles, selected):
        local = (profiles or ConnectionProfiles.initial(settings)).local
        try:
            report = await check_server(local)
            models = [item for item in report["available_model_ids"] if isinstance(item, str) and item]
            if not models:
                return gr.skip(), "The local server returned no models. Check the server or enter a model ID."
            chosen = selected if selected in models else local.model if local.model in models else None
            return gr.update(choices=models, value=chosen), f"Found {len(models)} local model(s). Select one and apply the connection."
        except Exception as error:
            return gr.skip(), redact(friendly_error(error, local), local.api_key)

    async def server_status(profiles=None):
        active = profiles.active if profiles else settings
        try:
            report = await check_server(active)
            if report["selected_model_found"]:
                return "Server reachable; configured model found in /v1/models.", redact(report, active.api_key)
            return "Server reachable; model not listed. Confirm its ID or try Test tool calling.", redact(report, active.api_key)
        except Exception as error:
            return redact(friendly_error(error, active), active.api_key), {}

    async def test_tool_connection(profiles):
        active = profiles.active if profiles else settings
        try:
            reply = await ask("Call get_current_utc_time now. Then report the UTC time from its result.",
                              Session(), active, True)
            executed = any(event.get("event") == "tool_result" and event.get("name") == "get_current_utc_time"
                           for event in reply.trace)
            report = {"connection": connection_label(active), "base_url": active.base_url,
                      "clock_tool_executed": executed, "elapsed_seconds": reply.elapsed_seconds,
                      "model_requests": reply.requests, "input_tokens": reply.input_tokens,
                      "output_tokens": reply.output_tokens, "answer": reply.text,
                      "warning": reply.warning, "trace": reply.trace, "timings": reply.timings}
            status = "Clock tool executed" if executed else "Tool test failed: model did not execute the clock tool"
            return f"{status} · {reply.elapsed_seconds:.1f}s · {reply.requests} model request(s).", redact(report, active.api_key)
        except Exception as error:
            return redact(friendly_error(error, active), active.api_key), redact(
                {"trace": getattr(error, "tool_trace", [])}, active.api_key)

    with gr.Blocks(title="BRAMS AI Assistant", analytics_enabled=False, fill_width=True) as demo:
        profiles = gr.State(value=ConnectionProfiles.initial(settings))
        session = gr.State(value=Session())
        run_history = gr.State(value=[])
        cancel_holder = gr.State(value=lambda: {"event": None, "observed": {}, "started": None, "pending_len": None, "question": ""})
        drawer_open = gr.State(value=False)
        action_state = gr.HTML(action_manifest([], Session()), elem_id="message-action-manifest",
                               elem_classes="chat-controls-hidden")
        delete_request = gr.Textbox(value="", show_label=False, container=False, interactive=True,
                                   elem_id="delete-message-request", elem_classes="chat-controls-hidden")

        with gr.Sidebar(label="Model settings", position="left", open=False, width=360,
                        elem_id="connection-drawer") as connection_drawer:
            gr.Markdown("## Model settings\nChoose the model that powers this conversation.")
            provider = gr.Dropdown(choices=[("Local model", "local"), ("Public API", "compatible")],
                                   value="local", label="Provider")
            api_url = gr.Textbox(value=settings.base_url, label="API base URL", interactive=False,
                                 placeholder="https://your-provider.example/v1")
            model_id = gr.Dropdown(choices=[settings.model], value=settings.model, label="Model ID",
                                   interactive=True, allow_custom_value=True,
                                   info="Refresh local models, or enter an exact model ID.")
            refresh_models = gr.Button("Refresh local models", size="sm")
            api_key = gr.Textbox(value="", label="API key", type="password", visible=False,
                                 placeholder="Paste API key")
            with gr.Accordion("Advanced options & privacy", open=False):
                token_parameter = gr.Dropdown(choices=[("Automatic", "auto"), ("max_tokens", "max_tokens"),
                                                        ("max_completion_tokens", "max_completion_tokens")],
                                              value="auto", label="Token-limit parameter", interactive=False)
                gr.Textbox(value=f"{settings.max_tokens} tokens; recovery up to {settings.recovery_max_tokens}",
                           label="Response budget", interactive=False)
                gr.Markdown("Keys stay in this session's server memory and are not saved to disk. Leave the key blank to reuse it only for the same saved endpoint. Hosted models receive chat text and tool metadata; WAV/PNG files stay local.", elem_classes="context-note")
            gr.Markdown("Applying a connection starts a fresh chat.", elem_classes="context-note")
            apply_button = gr.Button("Apply connection", variant="primary")
            with gr.Row():
                check_button = gr.Button("Check server")
                test_button = gr.Button("Test tool calling")
            connection_status = gr.Markdown("Changes take effect after Apply. Tool testing makes a model request; provider charges may apply.", elem_classes="context-note")
            with gr.Accordion("Connection diagnostics", open=False):
                connection_details = gr.JSON(label="Connection / tool test results")
            close_settings = gr.Button("Back to workspace")

        with gr.Row(elem_id="workspace-topbar"):
            with gr.Column(scale=1, min_width=220):
                gr.HTML('<div class="workspace-brand"><span class="brand-mark">B</span>'
                        '<span>BRAMS<span class="brand-detail">Research assistant</span></span></div>')
            clear = gr.Button("New chat", scale=0, min_width=100, size="sm", elem_id="new-chat")
            settings_button = gr.Button("Model settings", scale=0, min_width=130, size="sm", elem_id="model-settings")

        with gr.Row(elem_id="workspace-layout", elem_classes="workspace"):
            with gr.Column(scale=6, min_width=320, elem_classes="workspace-main"):
                chatbot = gr.Chatbot(label="Conversation", show_label=False, height="calc(100dvh - 315px)",
                                     min_height=360, layout="bubble", elem_id="conversation", buttons=["copy"],
                                     group_consecutive_messages=False,
                                     placeholder='<div class="chat-empty"><span class="empty-symbol">◎</span>'
                                     '<strong>What would you like to explore?</strong><p>Ask about a receiver, a recording or a scientific measurement.</p>'
                                     '<div class="starter-hints"><span>Receiver coordinates</span><span>Recording availability</span><span>Spectrograms & PSD</span></div></div>')
                with gr.Group(elem_id="message-composer"):
                    prompt = gr.Textbox(label="Your question", show_label=False, container=False,
                                        placeholder="Message BRAMS…", lines=2, max_lines=6)
                    with gr.Row(elem_id="composer-actions"):
                        enable_tools = gr.Checkbox(value=True, label="BRAMS tools", scale=1, elem_id="tools-toggle")
                        stop = gr.Button("Stop", variant="stop", scale=0, min_width=80,
                                         size="sm", elem_id="stop-message", visible=False)
                        send = gr.Button("Send", variant="primary", scale=0, min_width=80,
                                         size="sm", elem_id="send-message")
                status = gr.Markdown("Ready.", elem_id="request-status")
                gr.Markdown("UTC by default · Source-grounded answers · Original media stays local", elem_classes="chat-footnote")
            gr.HTML('<div id="workspace-split-handle" role="separator" tabindex="0" '
                    'aria-label="Resize chat and request monitor" aria-orientation="vertical" '
                    'aria-valuemin="20" aria-valuemax="65" aria-valuenow="40" '
                    'aria-controls="conversation monitor-tabs" '
                    'title="Drag to resize. Arrow keys adjust; double-click or Enter resets."><span aria-hidden="true"></span></div>',
                    elem_id="workspace-divider", min_width=0)
            with gr.Column(scale=4, min_width=320, elem_classes="workspace-sidebar"):
                monitor_current_model = gr.HTML(current_model_view(redact(settings.model, settings.api_key)),
                                                elem_id="monitor-current-model")
                gr.HTML('<div class="monitor-heading"><span class="monitor-live-dot"></span><strong>Request monitor</strong><span>Live execution</span></div>')
                metrics = gr.HTML(timing_view())
                with gr.Tabs(elem_id="monitor-tabs"):
                    with gr.Tab("Activity"):
                        activity = gr.HTML(activity_view())
                    with gr.Tab("History"):
                        history_panel = gr.HTML(history_view())
                    with gr.Tab("Raw trace"):
                        trace = gr.JSON(value=[], label="Tool calls, results and validation feedback")
                with gr.Accordion("Timing details", open=False):
                    gr.Markdown("The elapsed timer updates during the run. Model timings include API/network and generation; tool timings measure Python execution. Total time also includes orchestration. Queue wait and browser rendering are excluded. History keeps the last 20 turns in this session.", elem_classes="context-note")
                    gr.Markdown(pybrams_setup_status()["message"], elem_classes="context-note")

        chat_event = gr.on(
            triggers=[prompt.submit, send.click],
            fn=respond,
            inputs=[prompt, chatbot, session, enable_tools, profiles, run_history, cancel_holder],
            outputs=[chatbot, session, prompt, status, trace, send, clear, enable_tools,
                     metrics, activity, history_panel, run_history, stop],
            concurrency_limit=1,
            concurrency_id="llama_server",
            api_name="chat",
        )
        stop.click(
            cancel_chat,
            inputs=[cancel_holder, chatbot, session, run_history],
            outputs=[chatbot, session, prompt, status, send, clear, enable_tools, stop,
                     metrics, activity, history_panel, run_history],
            queue=False,
            api_visibility="private",
        )
        for trigger in [clear.click, enable_tools.change, chatbot.clear]:
            trigger(
                reset_chat,
                inputs=[],
                outputs=[chatbot, session, prompt, status, trace, metrics, activity, history_panel, run_history],
                concurrency_limit=1,
                concurrency_id="llama_server",
                api_name="reset_chat",
                api_visibility="private",
            )
        check_button.click(
            server_status,
            inputs=[profiles],
            outputs=[connection_status, connection_details],
            concurrency_limit=1,
            concurrency_id="llama_server",
            api_name="check_server",
        )
        provider.change(change_profile, inputs=[provider, profiles],
                        outputs=[api_url, model_id, api_key, token_parameter, refresh_models],
                        concurrency_limit=1, concurrency_id="llama_server", api_visibility="private")
        refresh_models.click(refresh_local_models, inputs=[profiles, model_id], outputs=[model_id, connection_status],
                             concurrency_limit=1, concurrency_id="llama_server", api_visibility="private")
        apply_button.click(apply_connection,
                           inputs=[provider, api_url, model_id, api_key, token_parameter, profiles],
                           outputs=[profiles, chatbot, session, prompt, status, trace, api_key,
                                    drawer_open, connection_status, connection_details,
                                    metrics, activity, history_panel, run_history, monitor_current_model, connection_drawer],
                           concurrency_limit=1, concurrency_id="llama_server", api_visibility="private")
        settings_button.click(toggle_connection, inputs=[drawer_open], outputs=[connection_drawer, drawer_open],
                              queue=False, api_visibility="private")
        close_settings.click(close_connection, outputs=[connection_drawer, drawer_open], queue=False, api_visibility="private")
        connection_drawer.expand(lambda: True, outputs=[drawer_open], queue=False, api_visibility="private")
        connection_drawer.collapse(lambda: False, outputs=[drawer_open], queue=False, api_visibility="private")
        chatbot.change(message_actions, inputs=[chatbot, session], outputs=[action_state],
                       concurrency_limit=1, concurrency_id="llama_server", api_visibility="private")
        delete_request.input(delete_message, inputs=[delete_request, chatbot, session],
                             outputs=[chatbot, session, status, delete_request],
                             concurrency_limit=1, concurrency_id="llama_server", api_visibility="private")
        test_button.click(test_tool_connection, inputs=[profiles],
                          outputs=[connection_status, connection_details],
                          concurrency_limit=1, concurrency_id="llama_server", api_visibility="private")
    return demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_settings()
    build_ui(config).launch(
        server_name=config.gradio_host,
        server_port=config.gradio_port,
        share=False,
        inbrowser=False,
        show_error=False,
        theme=gr.themes.Soft(primary_hue="slate", secondary_hue="teal", neutral_hue="slate",
                             font=["ui-sans-serif", "system-ui", "sans-serif"],
                             font_mono=["ui-monospace", "SFMono-Regular", "monospace"]),
        css=CSS,
        js=LAYOUT_JS,
    )
