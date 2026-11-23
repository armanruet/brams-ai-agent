"""Factor-based reports, explicit human verdicts and an offline review panel."""

from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = int(index)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def csv_file(path, rows, fields=None):
    if not rows and fields is None:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def identity(row):
    return tuple(str(row[field]) for field in ("model_id", "case_id", "repetition", "turn"))


def has_incomplete_usage(row):
    return row["status"] != "completed" or any(
        step.get("kind") == "model" and step.get("status") in {"error", "cancelled", "running"}
        for step in row.get("timings", []))


def timeout_turn(row, timeout):
    return (any(word in row.get("error_type", "").lower() for word in ("timeout", "timedout"))
            or (row["elapsed_seconds"] >= timeout - .1 and any(
                step.get("kind") == "model" and step.get("status") == "cancelled"
                for step in row.get("timings", []))))


def reviewed_verdict(review, row, kind):
    value = review.get(kind + "_verdict", "").strip().upper()
    if not review.get("reviewer", "").strip() or value not in {"PASS", "PARTIAL", "FAIL"}:
        return None
    if review.get("critical_error", "").strip().casefold() in {"yes", "true", "1"}:
        return False
    if kind == "model" and not row["score"].get("native_completion"):
        return False
    if value == "PASS" and any(review.get(field, "").strip() in {"0", "1"}
                               for field in ("factual_correctness_0_to_2", "source_quality_0_to_2",
                                             "judgment_0_to_2", "conversation_0_to_2",
                                             "trajectory_0_to_2", "query_understanding_0_to_2",
                                             "repetition_0_to_2", "geography_0_to_2")):
        return False
    return value == "PASS"


def write_report(directory):
    directory = Path(directory)
    rows = [json.loads(line) for line in (directory / "turns.jsonl").read_text().splitlines() if line.strip()]
    manifest = json.loads((directory / "manifest.json").read_text()) if (directory / "manifest.json").exists() else {}
    timeout = manifest.get("config", {}).get("protocol", {}).get("timeout_seconds", 180)
    cases = json.loads((directory / "cases.json").read_text()).get("cases", []) if (directory / "cases.json").exists() else []
    case_map = {case["id"]: case for case in cases}
    review_path = directory / "manual_review.csv"
    existing, old_fields = {}, []
    if review_path.exists():
        with review_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            old_fields = reader.fieldnames or []
            existing = {identity(row): row for row in reader}
    fields = list(dict.fromkeys(["model_id", "case_id", "repetition", "turn", "tier", "category", "prompt", "model_answer",
                                "displayed_answer", "review_rubric", "factual_correctness_0_to_2", "source_quality_0_to_2",
                                "judgment_0_to_2", "conversation_0_to_2", "clarity_0_to_2",
                                "trajectory_0_to_2", "query_understanding_0_to_2", "repetition_0_to_2",
                                "geography_0_to_2", "task_outcome", "first_failure_stage", "critical_error",
                                "model_verdict", "assistant_verdict", "reviewer", "notes", *old_fields]))
    reviews = {}
    for row in rows:
        case = case_map.get(row["case_id"], {})
        review = {field: row.get(field, "") for field in fields}
        review["tier"] = row.get("tier", case.get("tier", ""))
        review["review_rubric"] = " | ".join(row["score"].get("review_rubric", []))
        review.update(existing.get(identity(row), {}))
        reviews[identity(row)] = review
    csv_file(review_path, list(reviews.values()), fields)
    planned_cases = manifest.get("case_ids", list(case_map))
    planned_turns = (sum(manifest.get("case_turn_counts", {}).get(case, len(case_map.get(case, {}).get("turns", [])))
                         for case in planned_cases) * manifest.get("repetitions", 1)) if planned_cases else None

    def aggregate(group, model):
        def rate(field, native_only=False):
            values = [row["score"][field] for row in group if row["score"].get(field) is not None
                      and (not native_only or row["score"].get("native_completion"))]
            return statistics.mean(values) if values else None
        durations = [row["elapsed_seconds"] for row in group if row["status"] == "completed"]
        quality = {kind: [value for row in group if (value := reviewed_verdict(reviews[identity(row)], row, kind)) is not None]
                   for kind in ("model", "assistant")}
        server_rss = [row.get("resources", {}).get("server", {}).get("peak_sampled_mib") for row in group]
        server_rss = [value for value in server_rss if value is not None]
        model_config = next((item for item in manifest.get("config", {}).get("models", []) if item.get("id") == model), {})
        return {"model_id": model, "configured_model": model_config.get("model", model),
                "preflight_status": manifest.get("model_status", {}).get(model, {}).get("status", "not_recorded"),
                "planned_turns": planned_turns, "attempted_turns": len(group),
                "failed_turns": sum(row["status"] != "completed" for row in group),
                "tool_plan_pass_rate": rate("tool_plan_pass"), "strict_tool_plan_pass_rate": rate("strict_tool_plan_pass"),
                "model_tool_plan_pass_rate": rate("model_tool_plan_pass"),
                "orchestration_adjustment_count": sum(row["score"].get("orchestration_adjustments", 0) or 0 for row in group),
                "native_completion_rate": rate("native_completion"), "fallback_rate": rate("fallback"),
                "source_mentioned_rate": rate("source_mentioned"), "source_mentioned_among_native_rate": rate("source_mentioned", True),
                "completed_turn_p50_seconds": percentile(durations, .5), "completed_turn_p95_seconds": percentile(durations, .95),
                "all_attempt_p95_seconds": percentile([row["elapsed_seconds"] for row in group], .95),
                "timeout_turns": sum(timeout_turn(row, timeout) for row in group),
                "model_seconds_total": sum(row.get("model_seconds", 0) for row in group),
                "tool_seconds_total": sum(row.get("tool_seconds", 0) for row in group),
                "model_request_attempts": sum(sum(step.get("kind") == "model" for step in row.get("timings", [])) for row in group),
                "input_tokens": sum(row.get("input_tokens", 0) for row in group),
                "output_tokens": sum(row.get("output_tokens", 0) for row in group),
                "usage_may_be_incomplete": any(has_incomplete_usage(row) for row in group),
                "snapshot_misses": sum(row["score"].get("snapshot_misses", 0) for row in group),
                "forbidden_call_attempts": sum(row["score"].get("forbidden_call_attempts", 0) for row in group),
                "call_budget_exceeded_turns": sum(row["score"].get("call_budget_exceeded", False) for row in group),
                "scientific_flagged_turns": sum(bool(row["score"].get("scientific_review_flags")) for row in group),
                "model_grounding_flagged_turns": (sum(bool(row["score"].get("model_grounding_review_flags")) for row in group)
                    if any(isinstance(row["score"].get("model_grounding_review_flags"), list) for row in group) else None),
                "visible_grounding_flagged_turns": (sum(bool(row["score"].get("visible_grounding_review_flags")) for row in group)
                    if any(isinstance(row["score"].get("visible_grounding_review_flags"), list) for row in group) else None),
                "visible_scientific_flagged_turns": (
                    sum(bool(row["score"]["visible_scientific_review_flags"]) for row in group
                        if isinstance(row["score"].get("visible_scientific_review_flags"), list))
                    if any(isinstance(row["score"].get("visible_scientific_review_flags"), list) for row in group)
                    else None),
                "visible_scientific_flag_evaluated_turns": sum(
                    isinstance(row["score"].get("visible_scientific_review_flags"), list) for row in group),
                "controlled_fixture_turns": sum(row.get("evidence_mode") == "controlled_fixture" for row in group),
                "controlled_fixture_exercised_turns": sum(row["score"].get("controlled_fixture_exercised") is True for row in group),
                "sampled_server_rss_peak_mib": max(server_rss) if server_rss else None,
                "model_reviewed_turns": len(quality["model"]),
                "model_reviewed_pass_rate": statistics.mean(quality["model"]) if quality["model"] else None,
                "assistant_reviewed_turns": len(quality["assistant"]),
                "assistant_reviewed_pass_rate": statistics.mean(quality["assistant"]) if quality["assistant"] else None}

    summaries = [aggregate([row for row in rows if row["model_id"] == model], model)
                 for model in dict.fromkeys([*manifest.get("selected_models", []), *(row["model_id"] for row in rows)])]
    groups = defaultdict(list)
    for row in rows:
        case = case_map.get(row["case_id"], {})
        groups[(row["model_id"], row.get("tier", case.get("tier", "unassigned")), row.get("category", "unassigned"))].append(row)
    categories = [{"tier": tier, "category": category, **aggregate(group, model)}
                  for (model, tier, category), group in sorted(groups.items())]
    consistency = []
    for model in dict.fromkeys(row["model_id"] for row in rows):
        for case_id in dict.fromkeys(row["case_id"] for row in rows if row["model_id"] == model):
            group = [row for row in rows if row["model_id"] == model and row["case_id"] == case_id]
            trials = sorted({row["repetition"] for row in group})
            expected_trials = manifest.get("repetitions", len(trials))
            expected_turns = len(case_map.get(case_id, {}).get("turns", [])) or max(row["turn"] for row in group)
            complete = (len(trials) == expected_trials and len(group) == expected_trials * expected_turns
                        and len({(row["repetition"], row["turn"]) for row in group}) == len(group))
            verdicts = {kind: [reviewed_verdict(reviews[identity(row)], row, kind) for row in group]
                        for kind in ("model", "assistant")}
            consistency.append({"model_id": model, "case_id": case_id, "trials_observed": len(trials),
                                "trials_expected": expected_trials, "complete_trial_matrix": complete,
                                "native_workflow_all_trials": all(row["score"].get("native_completion")
                                    and row["score"].get("tool_plan_pass") is not False
                                    and not row["score"].get("forbidden_call_attempts", 0)
                                    and not row["score"].get("call_budget_exceeded", False) for row in group) if complete else None,
                                **{kind + "_reviewed_pass_all_trials": all(values) if complete and None not in values else None
                                   for kind, values in verdicts.items()}})
    note = ("Tool/evidence presence checks are not scientific accuracy. Human verdicts require a reviewer and explicit PASS/PARTIAL/FAIL. "
            "Unreviewed quality/consistency remains null. Native model completion is separate from the displayed assistant outcome. "
            "Latency excludes warmup; completed-turn percentiles include Python fallbacks. RSS is sampled process memory, not GPU allocation. No TTFT.")
    protocols = []
    for model in manifest.get("selected_models", list(dict.fromkeys(row["model_id"] for row in rows))):
        modes = set(manifest.get("case_evidence_modes", {}).values()) or {manifest.get("mode", "unrecorded")}
        for mode in sorted(modes):
            group = [row for row in rows if row["model_id"] == model
                     and row.get("evidence_mode", manifest.get("mode", "unrecorded")) == mode]
            planned = sum(manifest.get("case_turn_counts", {}).get(case_id, len(case_map.get(case_id, {}).get("turns", [])))
                          for case_id in planned_cases if manifest.get("case_evidence_modes", {}).get(case_id, manifest.get("mode", "unrecorded")) == mode)
            protocols.append({**aggregate(group, model), "evidence_mode": mode,
                              "planned_turns": planned * manifest.get("repetitions", 1) if planned_cases else None})
    if "controlled_fixture" in manifest.get("case_evidence_modes", {}).values():
        note += " This run mixes live archive cases and controlled negative interventions. Model cards pool them; use protocol_summary.csv for separate latency/completion results. An unexercised fixture is not a passed robustness test."
    json_file(directory / "summary.json", {"report_version": 2, "models": summaries, "categories": categories, "protocols": protocols, "note": note})
    csv_file(directory / "protocol_summary.csv", protocols, list(protocols[0]) if protocols else ["model_id", "evidence_mode"])
    csv_file(directory / "summary.csv", summaries)
    csv_file(directory / "category_summary.csv", categories, list(categories[0]) if categories else ["model_id", "tier", "category"])
    csv_file(directory / "consistency.csv", consistency, list(consistency[0]) if consistency else ["model_id", "case_id"])
    json_file(directory / "consistency.json", consistency)
    scope = {"suite_version": manifest.get("suite_version", 1), "mode": manifest.get("mode", "not recorded"),
             "offline_rescore": bool(manifest.get("rescoring")), "run_status": manifest.get("status", "not recorded")}
    write_panel(directory, summaries, rows, reviews, case_map, note, scope)
    return summaries


def write_panel(directory, summaries, rows, reviews, case_map, note, scope):
    data = {"models": summaries, "note": note, "scope": scope, "rows": [dict(row,
            category=row.get("category", "unassigned"),
            tier=row.get("tier", case_map.get(row["case_id"], {}).get("tier", "")),
            reviewed_model_status=("Ungraded" if reviewed_verdict(reviews[identity(row)], row, "model") is None
                                   else "PASS" if reviewed_verdict(reviews[identity(row)], row, "model") else "FAIL / PARTIAL"),
            review=reviews[identity(row)]) for row in rows]}
    encoded = json.dumps(data, ensure_ascii=False, default=str).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BRAMS benchmark review</title><style>
:root{color-scheme:light dark;--bg:#f5f7fb;--panel:white;--ink:#203047;--muted:#627189;--line:#dce4ee;--accent:#147d78}*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}main{max-width:1600px;margin:auto;padding:36px 28px}h1{font-size:32px;margin:0}p{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px;margin:24px 0}.card{border:1px solid var(--line);border-radius:16px;background:var(--panel);padding:20px}.card h2{font-size:19px;margin-top:0}.stat{font-size:30px;font-weight:700}.row{display:flex;justify-content:space-between;gap:18px;margin:9px 0}small{color:var(--muted)}.filters{display:flex;flex-wrap:wrap;gap:12px;margin:24px 0}input,select{font:inherit;border:1px solid var(--line);border-radius:8px;padding:10px;background:var(--panel);color:var(--ink)}input{flex:1;min-width:220px}table{width:100%;border-collapse:collapse;background:var(--panel)}th,td{text-align:left;border-bottom:1px solid var(--line);padding:12px;vertical-align:top}th{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}summary{cursor:pointer;color:var(--accent)}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.55 ui-monospace,monospace;max-width:100%;max-height:600px;overflow:auto;border:1px solid var(--line);border-radius:8px;padding:12px}.scroll{overflow-x:auto}a{color:var(--accent)}.details>td{padding:0 18px}.badge{border-radius:20px;background:var(--bg);padding:3px 9px;display:inline-block}.empty{padding:20px}
@media(prefers-color-scheme:dark){:root{--bg:#111925;--panel:#1a2534;--ink:#e1e9f4;--muted:#a2b1c8;--line:#35445a;--accent:#6bcec0}}@media(max-width:650px){main{padding:22px 14px}h1{font-size:26px}}
</style><main><h1>BRAMS benchmark review</h1><p>Scientific grounding · tool decisions · judgment · conversation · repeatability · operational performance</p>
<p id="scope"></p><p id="note"></p><section class="cards" id="cards"></section><p>Review <a href="manual_review.csv">manual_review.csv</a>, record reviewer and separate model/assistant verdicts, then run <code>python -m agent_app.benchmark summarize RUN_DIRECTORY</code>. Quality remains ungraded until reviewed. <a href="category_summary.csv">Category results</a> · <a href="consistency.csv">Repeated-trial consistency</a></p>
<div class="filters"><label>Model <select id="model"><option value="">All models</option></select></label><label>Tier <select id="tier"><option value="">All tiers</option></select></label><label>Category <select id="category"><option value="">All categories</option></select></label><label>Outcome <select id="outcome"><option value="">All outcomes</option><option>native</option><option>fallback</option><option>error</option></select></label><input id="search" aria-label="Search questions" placeholder="Search case or question"></div><p id="count"></p>
<div class="scroll"><table><thead><tr><th>Model / case</th><th>Trial / turn</th><th>Question</th><th>Outcome</th><th>Tool route</th><th>Time</th><th>Reviewed model verdict</th></tr></thead><tbody id="body"></tbody></table></div></main>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),byId=id=>document.getElementById(id);
byId('note').textContent=data.note;byId('scope').textContent=(data.scope.offline_rescore?'Offline re-score of original suite v':'Recorded suite v')+data.scope.suite_version+' · '+data.scope.mode+' mode · run '+data.scope.run_status+(data.scope.offline_rescore?' · No new model calls; this is not a new-suite execution.':'');
const fmt=value=>value==null?'Not measured':value.toFixed(2),pct=value=>value==null?'Ungraded':(100*value).toFixed(1)+'%';
function cell(parent,text){const item=document.createElement('td');item.textContent=text;parent.append(item);return item;}
for(const model of data.models){const card=document.createElement('article');card.className='card';const title=document.createElement('h2');title.textContent=model.model_id;const sub=document.createElement('small');sub.style.overflowWrap='anywhere';sub.textContent=model.configured_model;card.append(title,sub);
for(const [label,value] of [['Preflight',model.preflight_status],['Observed turns',model.attempted_turns+' / '+(model.planned_turns??'?')],['Reviewed model pass',pct(model.model_reviewed_pass_rate)],['Reviewed assistant pass',pct(model.assistant_reviewed_pass_rate)],['Native completion',pct(model.native_completion_rate)],['Median / p95',fmt(model.completed_turn_p50_seconds)+' / '+fmt(model.completed_turn_p95_seconds)+' s'],['Timeout turns',model.timeout_turns],['Server RSS peak',model.sampled_server_rss_peak_mib==null?'Not measured':fmt(model.sampled_server_rss_peak_mib)+' MiB']]){const row=document.createElement('div');row.className='row';const a=document.createElement('span'),b=document.createElement('strong');a.textContent=label;b.textContent=value;row.append(a,b);card.append(row);}byId('cards').append(card);}
for(const [id,values] of [['model',data.models.map(x=>x.model_id)],['tier',[...new Set(data.rows.map(x=>x.tier))].sort()],['category',[...new Set(data.rows.map(x=>x.category))].sort()]])for(const value of values){const o=document.createElement('option');o.value=value;o.textContent=value||'Unassigned';byId(id).append(o);}
function render(){const selected=data.rows.filter(r=>(!byId('model').value||r.model_id===byId('model').value)&&(!byId('tier').value||r.tier===byId('tier').value)&&(!byId('category').value||r.category===byId('category').value)&&(!byId('outcome').value||(r.status==='error'?'error':r.score.fallback?'fallback':'native')===byId('outcome').value)&&(!byId('search').value||(r.case_id+' '+r.prompt).toLowerCase().includes(byId('search').value.toLowerCase())));byId('body').replaceChildren();byId('count').textContent=selected.length+' matching turns';
for(const row of selected){const tr=document.createElement('tr');cell(tr,row.model_id+' / '+row.case_id);cell(tr,row.repetition+' / '+row.turn);const q=cell(tr,row.prompt),d=document.createElement('details'),s=document.createElement('summary');s.textContent='Answers, evidence and review';d.append(s);let built=false;d.addEventListener('toggle',()=>{if(!d.open||built)return;built=true;for(const [label,value] of [['Model answer',row.model_answer??'No completed model answer'],['Displayed assistant answer',row.displayed_answer??row.error??''],['Review criteria',row.score.review_rubric??[]],['Tool trace',row.trace??[]],['Checks',row.score],['Timings',row.timings??[]],['Resources',row.resources??{}],['Human review',row.review]]){const h=document.createElement('h3'),p=document.createElement('pre');h.textContent=label;p.textContent=typeof value==='string'?value:JSON.stringify(value,null,2);d.append(h,p);}});q.append(d);cell(tr,row.status==='error'?'Error':row.score.fallback?'Python fallback':'Native answer');cell(tr,row.score.tool_plan_pass==null?'Not required':row.score.tool_plan_pass?'Matched':'Needs review');cell(tr,fmt(row.elapsed_seconds)+' s');cell(tr,row.reviewed_model_status);byId('body').append(tr);}}
for(const id of ['model','tier','category','outcome','search'])byId(id).addEventListener('input',render);render();
</script></html>'''
    (directory / "report.html").write_text(page.replace("__DATA__", encoded))
