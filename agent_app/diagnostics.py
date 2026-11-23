"""Live checks: python -m agent_app.diagnostics --text --tools"""

import argparse
import asyncio
import json

import httpx
from pydantic import AwareDatetime, TypeAdapter

from .config import Settings, load_settings
from .brams_tools import check_availability, get_spectrogram, get_wav, list_receivers
from .service import Session, ask, friendly_error


async def probe_local_context_size(
    settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None,
) -> int | None:
    """Query llama.cpp /props for the served n_ctx; never trust /v1/models training context."""
    if settings.provider != "local":
        return None
    root = settings.base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        async with httpx.AsyncClient(
            transport=transport, trust_env=False,
            timeout=httpx.Timeout(3.0, connect=1.5),
        ) as client:
            response = await client.get(
                f"{root}/props",
                params={"model": settings.model},
                headers={"Authorization": f"Bearer {settings.api_key}"},
            )
            if response.status_code != 200:
                return None
            body = response.json()
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    for candidate in (
        (body.get("default_generation_settings") or {}).get("n_ctx")
        if isinstance(body.get("default_generation_settings"), dict) else None,
        (body.get("meta") or {}).get("n_ctx")
        if isinstance(body.get("meta"), dict) else None,
        body.get("n_ctx"),
    ):
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 1024:
            return candidate
    return None


async def check_server(
    settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    async with httpx.AsyncClient(
        transport=transport, trust_env=False,
        timeout=httpx.Timeout(15.0, connect=5.0),
    ) as client:
        response = await client.get(
            f"{settings.base_url}/models",
            headers={"Authorization": f"Bearer {settings.api_key}"},
        )
        response.raise_for_status()
        body = response.json()
    models = body.get("data", [])
    model_ids = [item["id"] for item in models if isinstance(item, dict) and "id" in item]
    return {
        "base_url": settings.base_url,
        "configured_model": settings.model,
        "provider": settings.provider,
        "selected_model_found": settings.model in model_ids,
        "available_model_ids": model_ids,
        "max_tokens": settings.max_tokens,
        "recovery_max_tokens": settings.recovery_max_tokens,
        "note": "A catalog entry does not prove generation or tool parsing works.",
    }


async def run_checks(args: argparse.Namespace) -> int:
    settings = load_settings()
    try:
        if args.wav:
            search, start, end = args.wav
            parse_time = TypeAdapter(AwareDatetime)
            result = await get_wav(search, parse_time.validate_python(start), parse_time.validate_python(end))
            print(json.dumps(result, indent=2), flush=True)
            if result["status"] != "ok":
                return 1
        if args.spectrogram:
            search, start, end = args.spectrogram
            parse_time = TypeAdapter(AwareDatetime)
            result = await get_spectrogram(search, parse_time.validate_python(start),
                                           parse_time.validate_python(end))
            print(json.dumps(result, indent=2), flush=True)
            if result["status"] != "ok":
                return 1
        if args.availability:
            search, start, end = args.availability
            parse_time = TypeAdapter(AwareDatetime)
            result = await check_availability(search, parse_time.validate_python(start),
                                              parse_time.validate_python(end))
            print(json.dumps(result, indent=2), flush=True)
            if result["status"] != "ok":
                return 1
        if args.receivers is not None:
            result = await list_receivers(args.receivers)
            print(json.dumps(result, indent=2), flush=True)
            if result["status"] != "ok":
                return 1
        if ((args.wav or args.spectrogram or args.availability or args.receivers is not None)
                and not (args.text or args.tools or args.availability_chat or args.spectrogram_chat or args.wav_chat)):
            return 0
        report = await check_server(settings)
        print(json.dumps(report, indent=2), flush=True)
        if not report["selected_model_found"]:
            print("FAIL: copy the exact desired model ID into LLAMA_MODEL in .env.")
            return 1
        if args.text:
            reply = await ask("Reply with one short greeting.", Session(), settings, False)
            print(f"TEXT PASS: {reply.text}", flush=True)
        if args.tools:
            reply = await ask(
                "Call get_current_utc_time now. Then report the UTC time from its result.",
                Session(), settings, True,
            )
            print(json.dumps(reply.trace, indent=2), flush=True)
            executed = any(
                event["event"] == "tool_result" and event["name"] == "get_current_utc_time"
                for event in reply.trace
            )
            if not executed:
                print("TOOL FAIL: no structured clock call was executed.")
                print("Check --jinja, the model's chat template, and the llama.cpp version.")
                print(f"Model text: {reply.text}")
                return 1
            print(f"TOOL PASS: {reply.text}", flush=True)
        if args.availability_chat or args.spectrogram_chat or args.wav_chat:
            reply = await ask(
                "Can you check if Kirchberg receiver is available from "
                "22/10/2026 00:00:00 to 22/10/2026 01:00:00 or not?",
                Session(), settings, True,
            )
            print(json.dumps(reply.trace, indent=2), flush=True)
            print(f"Model answer: {reply.text}", flush=True)
            calls = [event for event in reply.trace if event["event"] == "tool_call"
                     and event["name"] == "check_availability"]
            results = [event["result"] for event in reply.trace
                       if event["event"] == "tool_result" and event["name"] == "check_availability"]
            passed = any(
                item.get("status") == "ok" and item.get("recording_availability_checked")
                and item.get("query", "").casefold() == "kirchberg"
                and item.get("start_utc") == "2026-10-22T00:00:00Z"
                and item.get("end_utc") == "2026-10-22T01:00:00Z"
                for item in results
            )
            if not calls or not passed:
                print("AVAILABILITY FAIL: no successful structured query for the requested receiver/interval.")
                return 1
            print("AVAILABILITY PASS: real query executed for the correct receiver and interval. Review answer against trace.")
            if args.spectrogram_chat:
                image_reply = await ask("Can you please show me 1 spectrogram in this hour?",
                                        reply.session, settings, True)
                print(json.dumps(image_reply.trace, indent=2), flush=True)
                print(f"Model answer: {image_reply.text}", flush=True)
                passed = any(
                    event.get("event") == "tool_result" and event.get("name") == "get_spectrogram"
                    and event.get("result", {}).get("status") == "ok"
                    and event["result"].get("downloaded") is True
                    and event["result"].get("start_utc") == "2026-10-22T00:00:00Z"
                    and event["result"].get("end_utc") == "2026-10-22T01:00:00Z"
                    and "kir" in event["result"].get("receiver_name", "").casefold()
                    for event in image_reply.trace
                )
                if not passed or len(image_reply.images) != 1:
                    print("SPECTROGRAM FAIL: follow-up did not download one PNG for the requested receiver/interval.")
                    return 1
                print("SPECTROGRAM PASS: follow-up executed and one actual PNG is ready for display.")
            if args.wav_chat:
                audio_reply = await ask("Can you play one WAV recording from this hour?",
                                        reply.session, settings, True)
                print(json.dumps(audio_reply.trace, indent=2), flush=True)
                print(f"Model answer: {audio_reply.text}", flush=True)
                passed = any(
                    event.get("event") == "tool_result" and event.get("name") == "get_wav"
                    and event.get("result", {}).get("status") == "ok"
                    and event["result"].get("downloaded") is True
                    and event["result"].get("start_utc") == "2026-10-22T00:00:00Z"
                    and event["result"].get("end_utc") == "2026-10-22T01:00:00Z"
                    and "kir" in event["result"].get("receiver_name", "").casefold()
                    for event in audio_reply.trace
                )
                if not passed or len(audio_reply.audio) != 1:
                    print("WAV FAIL: no actual WAV and preview for the requested receiver/interval.")
                    return 1
                print("WAV PASS: original recording and listening preview are ready.")
        return 0
    except Exception as error:
        if getattr(error, "tool_trace", None):
            print(json.dumps(error.tool_trace, indent=2), flush=True)
        print(f"FAIL: {friendly_error(error, settings)}", flush=True)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", action="store_true", help="Also test a Pydantic AI text turn")
    parser.add_argument("--tools", action="store_true", help="Also test a real local tool loop")
    parser.add_argument("--receivers", nargs="?", const="", default=None, metavar="SEARCH",
                        help="Query PyBRAMS directly, e.g. --receivers Ecotron; no LLM required")
    parser.add_argument("--availability", nargs=3, metavar=("RECEIVER", "START", "END"),
                        help="Query PyBRAMS hourly availability directly; use ISO 8601 times with offsets")
    parser.add_argument("--availability-chat", action="store_true",
                        help="Test the exact Kirchberg question through Ornith and verify the actual tool result")
    parser.add_argument("--spectrogram", nargs=3, metavar=("RECEIVER", "START", "END"),
                        help="Fetch one real archived spectrogram PNG directly through PyBRAMS")
    parser.add_argument("--spectrogram-chat", action="store_true",
                        help="Test Kirchberg availability then the spectrogram follow-up with the real model")
    parser.add_argument("--wav", nargs=3, metavar=("RECEIVER", "START", "END"),
                        help="Download one actual original WAV and create a listening preview through PyBRAMS")
    parser.add_argument("--wav-chat", action="store_true",
                        help="Test Kirchberg availability then WAV playback/download with the real model")
    raise SystemExit(asyncio.run(run_checks(parser.parse_args())))


if __name__ == "__main__":
    main()
