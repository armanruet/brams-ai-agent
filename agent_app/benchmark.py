"""Reproducible BRAMS model comparison. See BENCHMARKING.md; no keys in reports."""

import argparse
import asyncio
from datetime import datetime, timezone
import csv
from functools import wraps
import hashlib
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
import os
from pathlib import Path
import platform
import random
import re
import shlex
import statistics
import sys
from time import perf_counter
from urllib.parse import urlsplit

from dotenv import load_dotenv

from .benchmark_data import SCORER_VERSION, TOOLS, SnapshotTools, bound_arguments, canonical, fact_value, reference_calls, score_turn
from .benchmark_resources import ResourceSampler
from .benchmark_scenarios import controlled_tools, fixture_exercised, validate_fixture
from .config import PROJECT_ROOT, Settings
from .connections import is_loopback
from .diagnostics import check_server
from .artifacts import detection_path, spectrogram_path, wav_path
from .service import Reply, Session, ask, friendly_error


LEGACY_CASES = PROJECT_ROOT / "benchmarks/cases.json"
DEFAULT_CASES = PROJECT_ROOT / "questions.json"
DEFAULT_SNAPSHOT = PROJECT_ROOT / ".benchmarks/reference-v2.json"
DEFAULT_CONFIG = PROJECT_ROOT / "benchmarks/models.example.json"
SNAPSHOT_NOTICE = "\nBenchmark mode: tool results replay a fixed, previously captured BRAMS reference snapshot. They are not live observations. Captured download/analysis metadata describes the earlier capture, not a new operation in this turn. Explain snapshot replay honestly and cite the returned evidence."


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def fixed_clock(value):
    """An explicit evaluation fixture; never change production clock behavior."""
    if value is None:
        return None
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("--clock requires an ISO timestamp with Z or an explicit offset.")
    return instant.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def tools_with_clock(functions, clock):
    if clock is None:
        return functions
    @wraps(TOOLS["get_current_utc_time"])
    def controlled_clock(timezone_hint: str = "UTC"):
        return {
            "utc_time": clock,
            "utc_date": clock[:10],
            "source": "benchmark_fixed_clock",
            "live_clock_read": False,
        }
    return [controlled_clock if fn.__name__ == "get_current_utc_time" else fn
            for fn in (TOOLS.values() if functions is None else functions)]


def environment():
    packages = {}
    for name in ("pydantic-ai", "pydantic-ai-slim", "pydantic", "openai", "httpx", "pybrams", "gradio"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    sources = {str(path.relative_to(PROJECT_ROOT)): digest(path)
               for path in sorted((PROJECT_ROOT / "agent_app").glob("*.py"))}
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "packages": packages, "source_sha256": sources,
            "hardware_note": "Record RAM/GPU/server build and effective settings in operator_notes.md. Platform is not a hardware inventory."}


def configuration(path):
    config = read_json(path)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark configuration version.")
    ids = [model["id"] for model in config["models"]]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-z0-9_-]+", item) for item in ids):
        raise ValueError("Model IDs must be unique, simple labels (letters, numbers, - or _).")
    def reject_secrets(value):
        if isinstance(value, dict):
            if any(key.lower() in {"api_key", "authorization", "password", "secret"} for key in value):
                raise ValueError("Store API keys in environment variables, never in benchmark JSON.")
            for child in value.values():
                reject_secrets(child)
        elif isinstance(value, list):
            for child in value:
                reject_secrets(child)
    reject_secrets(config)
    for model in config["models"]:
        variable = model.get("api_key_env")
        if not isinstance(variable, str) or not re.fullmatch(r"[A-Z_][A-Z0-9_]*", variable):
            raise ValueError(
                "api_key_env must be an environment-variable NAME, for example "
                "BENCHMARK_PUBLIC_API_KEY, not the API key itself. Store the actual "
                "key as BENCHMARK_PUBLIC_API_KEY in .env and set the JSON field to "
                '\"api_key_env\": \"BENCHMARK_PUBLIC_API_KEY\".'
            )
    allowed = {"max_tokens", "recovery_max_tokens", "timeout_seconds", "request_limit", "tool_calls_limit", "assistant_scope"}
    if set(config["protocol"]) - allowed:
        raise ValueError("Unknown benchmark protocol setting.")
    return config


def settings_for(config, model, *, require_key=True):
    key = os.getenv(model["api_key_env"], "")
    if model["provider"] != "local" and not key and require_key:
        raise ValueError(f"Set {model['api_key_env']} in your terminal or .env; do not put the key in JSON.")
    parsed = urlsplit(model["base_url"])
    if model["provider"] != "local" and parsed.scheme != "https" and not is_loopback(parsed.hostname):
        raise ValueError("Hosted API keys require an HTTPS endpoint.")
    return Settings(**config["protocol"], provider=model["provider"],
                    model=model["model"], base_url=model["base_url"],
                    token_parameter=model.get("token_parameter", "auto"),
                    api_key=key or "local-not-needed")


def selected_models(config, names=None):
    models = [model for model in config["models"] if model.get("enabled", False)]
    if names:
        known = {model["id"] for model in models}
        if set(names) - known:
            raise ValueError("Unknown or disabled model(s): " + ", ".join(sorted(set(names) - known)))
        models = [model for model in models if model["id"] in names]
    if not models:
        raise ValueError("No enabled models selected.")
    return models


def selected_cases(path, names=None, limit=None, tiers=None):
    document = read_json(path)
    cases = document["cases"]
    if document.get("schema_version") != 1 or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Invalid benchmark case document/version or duplicate case IDs.")
    # Validate the gold queries even in plan mode; malformed test data must fail early.
    reference_calls(cases)
    for case in cases:
        validate_fixture(case)
    if names:
        unknown = set(names) - {case["id"] for case in cases}
        if unknown:
            raise ValueError("Unknown cases: " + ", ".join(sorted(unknown)))
        cases = [case for case in cases if case["id"] in names]
    if tiers:
        unknown = set(tiers) - {case.get("tier") for case in cases}
        if unknown:
            raise ValueError("Unknown tiers: " + ", ".join(sorted(unknown)))
        cases = [case for case in cases if case.get("tier") in tiers]
    if limit:
        cases = cases[:limit]
    if not cases:
        raise ValueError("No benchmark cases selected.")
    return cases


def sanitize(value, keys):
    text = json.dumps(value, default=str)
    for key in keys:
        if key and key != "local-not-needed":
            # Replace JSON-escaped content, including backslashes/quotes in credentials.
            text = text.replace(json.dumps(key)[1:-1], "[redacted]")
    return json.loads(text)


async def capture(cases, path, cases_path, reuse_snapshot=None):
    if any(case.get("controlled_fixture") for case in cases):
        raise ValueError("Controlled negative fixtures are not real-tool captures. Run them as explicitly labeled live-model scenarios instead.")
    if Path(path).exists():
        raise ValueError("Snapshot already exists. Choose a new filename to preserve the reference.")
    snapshot = {"schema_version": 1, "kind": "real_tool_capture", "captured_at_utc": now(),
                "cases_sha256": digest(cases_path), "environment": environment(), "entries": []}
    seeds = {}
    seed_tools = None
    if reuse_snapshot:
        seed = read_json(reuse_snapshot)
        if seed.get("kind") != "real_tool_capture" or seed.get("schema_version") != 1:
            raise ValueError("Reuse requires a real-tool capture, not synthetic fixture data.")
        snapshot["seed_snapshot_sha256"] = digest(reuse_snapshot)
        seeds = {canonical(row["tool"], row["arguments"]): row for row in seed["entries"]
                 if row["tool"] != "get_current_utc_time" and row["result"].get("status", "ok") == "ok"}
        seed_tools = SnapshotTools(seed)
    calls = reference_calls(cases)
    partial = Path(str(path) + ".partial")
    for index, call in enumerate(calls, 1):
        name, arguments = call["tool"], call["arguments"]
        print(f"Capture {index}/{len(calls)}: {name}", flush=True)
        started = perf_counter()
        saved = seeds.get(canonical(name, arguments))
        derived = False
        if saved is None and seed_tools is not None and name == "list_receivers":
            candidate = seed_tools.result(name, arguments)
            if candidate.get("status") == "ok":
                saved = {"result": candidate}
                derived = True
        if saved:
            result = saved["result"]
        else:
            result = TOOLS[name](**bound_arguments(name, arguments))
            if inspect.isawaitable(result):
                result = await result
        snapshot["entries"].append({**call, "result": result, "elapsed_seconds": perf_counter() - started,
                                    "capture_source": "derived_from_real_snapshot" if derived else "reused_real_snapshot" if saved else "live_python_tool",
                                    "source_captured_at_utc": result.get("retrieved_at_utc", seed.get("captured_at_utc") if saved else now())})
        write_json(partial, snapshot)
        if result.get("status", "ok") not in call.get("expected_statuses", ["ok"]) or result.get("status") in {"error", "setup_required"}:
            raise ValueError(f"Reference capture failed for {name}: {result.get('status')}. Inspect {partial} to distinguish a catalog/network/setup failure from a genuinely empty recording interval. No benchmark scores were produced.")
    lookup = SnapshotTools(snapshot)
    for case in cases:
        history = []
        for turn in case["turns"]:
            history += [(call["tool"], lookup.result(call["tool"], call["arguments"])) for call in turn["required_calls"]]
            for fact in turn["facts"]:
                results = [result for name, result in history if name == fact["tool"]]
                try:
                    result = lookup.result(fact["tool"], fact["arguments"]) if "arguments" in fact else results[-1]
                    value = fact_value(result, fact)
                    if value is None:
                        raise ValueError("null reference value")
                except (KeyError, IndexError, TypeError, ValueError) as error:
                    raise ValueError(f"Missing reference for {case['id']}/{fact['path']}; inspect {partial}.") from error
    write_json(path, snapshot)
    partial.unlink(missing_ok=True)
    print(f"Saved {len(calls)} actual tool results: {path}", flush=True)


async def warmup(settings, *, transport=None, server_pid=None):
    # Keep all production schemas, but prohibit arbitrary archive downloads in preflight.
    clock = TOOLS["get_current_utc_time"]()
    replay = SnapshotTools({"entries": [{"tool": "get_current_utc_time", "arguments": {}, "result": clock}]})
    async with ResourceSampler(server_pid) as resources:
        reply = await ask("Call get_current_utc_time exactly once, then state its returned UTC time and source.",
                          Session(), settings, transport=transport, tool_functions=replay.functions())
    valid = any(row["event"] == "tool_result" and row["name"] == "get_current_utc_time"
                and row["result"] == clock for row in reply.trace)
    if not valid or not reply.model_text or replay.misses:
        raise ValueError("Structured tool-loop preflight failed. Check model ID, --jinja, chat template/parser, token budget and llama.cpp version.")
    return {"elapsed_seconds": reply.elapsed_seconds, "requests": reply.requests,
            "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
            "excluded_from_case_latency": True, "resources": resources.result()}


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)


def report(directory):
    from .benchmark_reporting import write_report
    return write_report(directory)


def _preserve_recorded_media_score(score, original, turn):
    """Use recorded execution facts; local artifacts may have moved since the run."""
    direct = original.get("direct_answer_completion")
    score["direct_answer_completion"] = direct if isinstance(direct, bool) else None
    fields = {"audio": "audio_attachments", "image": "image_attachments",
              "detection": "detection_attachments"}
    counts = {}
    for kind, field in fields.items():
        value = original.get(field)
        counts[kind] = value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
        score[field] = counts[kind]
    expected = turn.get("expected_media")
    if expected is None:
        expected = ["audio" if call["tool"] == "get_wav" else "image"
                    for call in turn["required_calls"] if call["tool"] in {"get_wav", "get_spectrogram"}]
    score["media_delivery_pass"] = (
        all(counts[kind] > 0 for kind in expected)
        if expected and all(counts[kind] is not None for kind in expected) else None
    )


def _validate_snapshot_media(cases, lookup):
    """Reject missing replay artifacts before any model is graded against them."""
    for case in cases:
        for index, turn in enumerate(case["turns"], start=1):
            required = turn.get("required_calls", [])
            expected = turn.get("expected_media")
            if expected is None:
                expected = ["audio" if call["tool"] == "get_wav" else "image"
                            for call in required if call["tool"] in {"get_wav", "get_spectrogram"}]

            def route_available(route):
                for kind in expected:
                    candidates = [call for call in route if (
                        (kind == "audio" and call["tool"] == "get_wav")
                        or (kind == "detection" and call["tool"] == "detect_meteors")
                        or (kind == "image" and (call["tool"] == "get_spectrogram"
                            or (call["tool"] == "query_geographic_receivers"
                                and call.get("arguments", {}).get("action") == "spectrogram")))
                    )]
                    found = False
                    for call in candidates:
                        result = lookup.result(call["tool"], call["arguments"])
                        available = isinstance(result, dict) and result.get("status") == "ok"
                        if available and kind == "image":
                            available = bool(result.get("downloaded") and spectrogram_path(result.get("artifact_id")))
                        elif available and kind == "audio":
                            artifact = result.get("artifact_id")
                            available = bool(result.get("downloaded") and wav_path(artifact) and wav_path(artifact, preview=True))
                        elif available and kind == "detection":
                            entries = result.get("annotated_artifacts")
                            available = bool(result.get("analysis_performed") and isinstance(entries, list) and entries
                                             and all(isinstance(entry, dict) and detection_path(entry.get("artifact_id"))
                                                     for entry in entries))
                        found = found or available
                    if not found:
                        return False
                return True

            routes = [required, *turn.get("acceptable_call_sets", [])]
            if expected and not any(route_available(route) for route in routes):
                raise ValueError(f"Snapshot fixture unavailable: {case['id']} turn {index} expects "
                                 f"{', '.join(expected)} but no complete route has usable recorded media. "
                                 "Recapture the reference and its media before running. No model requests were sent.")


def rescore(directory, output):
    """Recheck saved evidence in a new directory. No model/archive requests."""
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise ValueError("Rescore output already exists; choose a new directory to preserve results.")
    manifest = read_json(directory / "manifest.json")
    cases_document = read_json(directory / "cases.json")
    case_map = {case["id"]: case for case in cases_document["cases"]}
    reference = SnapshotTools(read_json(directory / "reference.json")) if (directory / "reference.json").exists() else None
    keys = [os.getenv(model.get("api_key_env", ""), "") for model in manifest.get("config", {}).get("models", [])]
    rows = [json.loads(line) for line in (directory / "turns.jsonl").read_text().splitlines() if line.strip()]
    histories = {}
    for row in rows:
        row["original_score"] = row["score"]
        if row["status"] != "completed":
            continue
        case = case_map[row["case_id"]]
        turn = case["turns"][row["turn"] - 1]
        key = row["model_id"], row["case_id"], row["repetition"]
        history = histories.setdefault(key, [])
        reply = Reply(row.get("displayed_answer", ""), Session(), row.get("trace", []), row["elapsed_seconds"],
                      row.get("requests", 0), row.get("input_tokens", 0), row.get("output_tokens", 0),
                      model_text=row.get("model_answer"))
        row["score"] = score_turn(turn, reply, history, reference=reference)
        _preserve_recorded_media_score(row["score"], row["original_score"], turn)
        row["score"]["controlled_fixture_exercised"] = fixture_exercised(case, row.get("trace", []))
        history.extend(reply.trace)
    output.mkdir(parents=True)
    manifest["rescoring"] = {"scorer_version": SCORER_VERSION, "rescored_at_utc": now(),
                             "original_turns_sha256": digest(directory / "turns.jsonl"),
                             "original_manifest_sha256": digest(directory / "manifest.json"),
                             "scoring_environment": environment(), "new_api_calls": 0,
                             "note": "Original cases and execution timings retained. This is not a v2-suite execution."}
    write_json(output / "manifest.json", sanitize(manifest, keys))
    write_json(output / "cases.json", cases_document)
    if reference:
        write_json(output / "reference.json", sanitize(read_json(directory / "reference.json"), keys))
    if (directory / "manual_review.csv").exists():
        text = (directory / "manual_review.csv").read_text()
        for key in keys:
            if key and key != "local-not-needed":
                text = text.replace(key, "[redacted]")
        (output / "manual_review.csv").write_text(text)
    with (output / "turns.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(sanitize(row, keys)) + "\n")
    report(output)
    print(f"Offline rescoring complete: {output / 'report.html'}; original run preserved; no API calls.")


async def run(args, config, models, cases):
    if any(model["provider"] != "local" for model in models) and not args.include_public:
        raise ValueError("Hosted calls may incur charges. Add --include-public to explicitly include them.")
    settings = {model["id"]: settings_for(config, model) for model in models}
    clock = fixed_clock(getattr(args, "clock", None))
    if args.mode == "snapshot" and any(case.get("controlled_fixture") for case in cases):
        raise ValueError("Controlled negative fixtures require --mode live (models are live; fixture archive tools never contact the archive). Do not label fixtures as real-tool captures.")
    snapshot = None
    if args.mode == "snapshot":
        if not args.snapshot.is_file():
            command = ["python", "-m", "agent_app.benchmark", "capture"]
            if args.cases != DEFAULT_CASES:
                command += ["--cases", str(args.cases)]
            command += ["--snapshot", str(args.snapshot)]
            partial = Path(str(args.snapshot) + ".partial")
            detail = (f" An incomplete capture exists at {partial}; it cannot be used as a reference."
                      if partial.exists() else "")
            raise ValueError(
                f"BRAMS reference snapshot not found: {args.snapshot}.{detail} "
                f"Create it first with: {shlex.join(command)}. "
                "Wait for 'Saved ... actual tool results' before running the snapshot benchmark. "
                "No model requests were sent."
            )
        snapshot = read_json(args.snapshot)
        if snapshot.get("kind") != "real_tool_capture" or snapshot.get("schema_version") != 1:
            raise ValueError("Expected a real-tool reference produced by the capture command.")
        if snapshot["cases_sha256"] != digest(args.cases):
            raise ValueError("Cases changed after capture. Capture a new reference for this dataset.")
        lookup = SnapshotTools(snapshot)
        for call in reference_calls(cases):
            lookup.result(call["tool"], call["arguments"])
        if lookup.misses:
            raise ValueError("Snapshot does not cover all selected cases. Capture the full case set first.")
        _validate_snapshot_media(cases, lookup)
    directory = Path(args.output or (PROJECT_ROOT / ".benchmarks" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")))
    directory.mkdir(parents=True, exist_ok=False)
    keys = [value.api_key for value in settings.values()]
    manifest = {"started_at_utc": now(), "mode": args.mode, "seed": args.seed,
                "clock_override_utc": clock,
                "clock_policy": "Explicit benchmark fixture; archive acquisition timestamps are unchanged." if clock else "Live system clock",
                "case_evidence_modes": {case["id"]: "controlled_fixture" if case.get("controlled_fixture") else args.mode for case in cases},
                "repetitions": args.repetitions, "case_ids": [case["id"] for case in cases],
                "cases_sha256": digest(args.cases), "config": config,
                "selected_models": [model["id"] for model in models], "environment": environment(),
                "assistant_scope": {model_id: value.assistant_scope for model_id, value in settings.items()},
                "suite_version": read_json(args.cases).get("suite_version", 1), "scorer_version": SCORER_VERSION,
                "case_turn_counts": {case["id"]: len(case["turns"]) for case in cases},
                "resource_sampling": {"interval_seconds": .5, "local_server_pid": args.server_pid,
                                      "note": "Sampled process RSS; GPU/Metal allocation is not measured."},
                "snapshot_sha256": digest(args.snapshot) if snapshot else None,
                "snapshot_captured_at_utc": snapshot.get("captured_at_utc") if snapshot else None,
                "sampling": "Server/provider defaults. Declared preset metadata is not runtime verification.",
                "model_status": {}, "status": "running"}
    write_json(directory / "manifest.json", sanitize(manifest, keys))
    write_json(directory / "cases.json", read_json(args.cases))
    if snapshot:
        write_json(directory / "reference.json", sanitize(snapshot, keys))
    (directory / "operator_notes.md").write_text("Record hardware/RAM/GPU, power mode, background workload, llama.cpp build/launch command, effective model sampling/reasoning/context settings, and hosted provider/model version/pricing here. Do not record API keys.\n")
    path = directory / "turns.jsonl"
    path.touch()
    print(f"Run output: {directory}", flush=True)
    try:
        with path.open("a") as stream:
            for model in models:
                model_id = model["id"]
                current = settings[model_id]
                print(f"Preflight / excluded warmup: {model_id}", flush=True)
                try:
                    try:
                        catalog = await check_server(current)
                    except Exception as error:
                        catalog = {"unavailable": True, "error_type": type(error).__name__}
                    preflight = await warmup(current, server_pid=args.server_pid if current.provider == "local" else None)
                    manifest["model_status"][model_id] = {"status": "ready", "catalog": catalog, "warmup": preflight}
                except Exception as error:
                    message = friendly_error(error, current)
                    manifest["model_status"][model_id] = {"status": "preflight_failed", "message": message,
                        "error_type": type(error).__name__, "trace": getattr(error, "tool_trace", []),
                        "timings": getattr(error, "timings", [])}
                    write_json(directory / "manifest.json", sanitize(manifest, keys))
                    print(f"{model_id}: preflight failed; see manifest.json", flush=True)
                    continue
                for repetition in range(1, args.repetitions + 1):
                    order = list(cases)
                    random.Random(args.seed + repetition).shuffle(order)
                    for case in order:
                        session = Session()
                        history = []
                        for index, turn in enumerate(case["turns"], 1):
                            print(f"{model_id}: repetition {repetition}, {case['id']}, turn {index}", flush=True)
                            replay = SnapshotTools(snapshot) if snapshot else None
                            functions = tools_with_clock(replay.functions() if replay else None, clock)
                            if case.get("controlled_fixture"):
                                functions = controlled_tools(list(TOOLS.values()) if functions is None else functions, case)
                            notice = SNAPSHOT_NOTICE if replay else ""
                            if case.get("controlled_fixture"):
                                notice += "\nControlled benchmark scenario: archive/catalog responses are labeled interventions, not live observations. Preserve their scope and uncertainty. Time tools still calculate time."
                            row = {"model_id": model_id, "case_id": case["id"], "category": case["category"],
                                   "tier": case.get("tier", "unassigned"), "capability_status": case.get("capability_status", "supported"),
                                   "repetition": repetition, "turn": index, "prompt": turn["prompt"], "started_at_utc": now()}
                            row["evidence_mode"] = "controlled_fixture" if case.get("controlled_fixture") else args.mode
                            row["controlled_fixture"] = case.get("controlled_fixture")
                            steps = []
                            started = perf_counter()
                            resources = ResourceSampler(args.server_pid if current.provider == "local" else None)
                            try:
                                async with resources:
                                    reply = await ask(turn["prompt"], session, current, on_event=lambda step: steps.append(step),
                                                      tool_functions=functions,
                                                      instructions_suffix=notice)
                                session = reply.session
                                row.update(status="completed", elapsed_seconds=reply.elapsed_seconds,
                                           model_answer=reply.model_text, displayed_answer=reply.text,
                                           warning=reply.warning, requests=reply.requests,
                                           input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
                                           trace=reply.trace, timings=reply.timings,
                                           score=score_turn(turn, reply, history, reference=SnapshotTools(snapshot) if snapshot else None))
                                history += reply.trace
                            except Exception as error:
                                # Successful prior turn history remains valid; preserve failed attempt timings.
                                final_steps = {step["id"]: step for step in steps}
                                row.update(status="error", elapsed_seconds=perf_counter() - started,
                                           error=friendly_error(error, current), error_type=type(error).__name__,
                                           trace=getattr(error, "tool_trace", []),
                                           model_answer=None, timings=list(final_steps.values()),
                                           score={"tool_plan_pass": False if turn["required_calls"] else None,
                                                  "native_completion": False, "fallback": False,
                                                  "source_mentioned": False if turn["require_source"] else None,
                                                  "review_rubric": turn.get("review_rubric", [])})
                            row["resources"] = resources.result()
                            row["score"]["controlled_fixture_exercised"] = fixture_exercised(case, row.get("trace", []))
                            row["model_seconds"] = sum(step.get("duration_seconds") or 0 for step in row["timings"] if step["kind"] == "model")
                            row["tool_seconds"] = sum(step.get("duration_seconds") or 0 for step in row["timings"] if step["kind"] == "tool")
                            stream.write(json.dumps(sanitize(row, keys)) + "\n")
                            stream.flush()
                    report(directory)
                write_json(directory / "manifest.json", sanitize(manifest, keys))
        manifest["status"] = "finished"
    finally:
        manifest["finished_at_utc"] = now()
        if manifest["status"] == "running":
            manifest["status"] = "interrupted"
        write_json(directory / "manifest.json", sanitize(manifest, keys))
        report(directory)
    print(f"Run manifest: {directory / 'manifest.json'}", flush=True)
    print("Reports: report.html, summary.json/CSV, category_summary.csv, consistency.csv, turns.jsonl and manual_review.csv", flush=True)
    summaries = report(directory)
    return 1 if (any(item["status"] == "preflight_failed" for item in manifest["model_status"].values())
                 or any(item["failed_turns"] for item in summaries)) else 0


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    for command in ("plan", "capture", "run"):
        item = sub.add_parser(command)
        item.add_argument("--cases", type=Path, default=DEFAULT_CASES)
        item.add_argument("--case", nargs="+", dest="case_ids")
        item.add_argument("--tier", nargs="+", dest="tiers")
        item.add_argument("--limit", type=int)
        if command != "capture":
            item.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
            item.add_argument("--models", nargs="+")
            item.add_argument("--repetitions", type=int, default=3)
            item.add_argument("--clock", help="Fixed benchmark UTC clock (ISO timestamp with offset); not a live observation")
        if command in {"capture", "run"}:
            item.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
        if command == "capture":
            item.add_argument("--reuse-snapshot", type=Path)
        if command == "run":
            item.add_argument("--server-pid", type=int, help="Optional local llama.cpp PID for process RSS sampling")
            item.add_argument("--mode", choices=("snapshot", "live"), default="snapshot")
            item.add_argument("--include-public", action="store_true")
            item.add_argument("--seed", type=int, default=20261101)
            item.add_argument("--output", type=Path)
    summary = sub.add_parser("summarize")
    summary.add_argument("directory", type=Path)
    recheck = sub.add_parser("rescore")
    recheck.add_argument("directory", type=Path)
    recheck.add_argument("--output", type=Path, required=True)
    return root


async def main(argv=None):
    args = parser().parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if args.command == "summarize":
        print(json.dumps(report(args.directory), indent=2))
        return 0
    if args.command == "rescore":
        rescore(args.directory, args.output)
        return 0
    if (args.limit is not None and args.limit < 1) or getattr(args, "repetitions", 1) < 1:
        raise ValueError("Limits/repetitions must be positive.")
    if getattr(args, "server_pid", None) is not None and args.server_pid < 1:
        raise ValueError("server-pid must be a positive local process ID.")
    cases = selected_cases(args.cases, args.case_ids, args.limit, args.tiers)
    if args.command == "capture":
        await capture(cases, args.snapshot, args.cases, args.reuse_snapshot)
        return 0
    config = configuration(args.config)
    models = selected_models(config, args.models)
    if args.command == "plan":
        total_turns = sum(len(case["turns"]) for case in cases)
        plan = {"suite_version": read_json(args.cases).get("suite_version", 1),
                "clock_override_utc": fixed_clock(args.clock),
                "cases_file": str(args.cases), "default_snapshot": str(DEFAULT_SNAPSHOT),
                "tier_counts": {tier: sum(case.get("tier") == tier for case in cases) for tier in sorted({case.get("tier", "unassigned") for case in cases})},
                "reference_call_count": len(reference_calls(cases)),
                "controlled_fixture_cases": sum(bool(case.get("controlled_fixture")) for case in cases),
                "turns_by_evidence_mode": {mode: sum(len(case["turns"]) for case in cases
                    if ("controlled_fixture" if case.get("controlled_fixture") else "archive") == mode)
                    for mode in ("archive", "controlled_fixture")},
                "cases": len(cases), "turns_per_repetition": total_turns,
                "repetitions": args.repetitions, "selected_model_count": len(models),
                "case_executions": len(cases) * args.repetitions * len(models),
                "timed_turns": total_turns * args.repetitions * len(models),
                "additional_warmup_turns": len(models), "protocol": config["protocol"],
                "models": [{"id": model["id"], "model": model["model"], "provider": model["provider"],
                            "key_required": model["provider"] != "local",
                            "key_present": bool(os.getenv(model["api_key_env"])),
                            "deployment": model.get("deployment")} for model in models],
                "disabled_models": [model["id"] for model in config["models"] if not model.get("enabled")],
                "note": "Plan only; no model or BRAMS requests sent. Model request counts and cost depend on actual tool loops. Configure the public baseline before including it."}
        for model in models:
            settings_for(config, model, require_key=False)
        print(json.dumps(plan, indent=2))
        return 0
    return await run(args, config, models, cases)


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("Interrupted. Completed turns remain in the run directory.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        # Friendly messages only; never emit request bodies or a key-bearing traceback.
        print(f"Benchmark stopped: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
