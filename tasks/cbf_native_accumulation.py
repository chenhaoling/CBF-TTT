"""Causal diagnostic of native TTT fast-weight accumulation at the final checkpoint."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.build_cbf_memory_value import file_digest, REGIMES
from tasks.cbf_checkpoint_native import (
    FIELDS, LAYERS, NativeSession, load_data as load_native_data, load_model,
    native_digest, native_stats,
)
from tasks.cbf_checkpoint_readout import check_loading, parameter_digest
from tasks.cbf_selective import measure

SOURCE_SUMMARY_SHA = "00912a997f0ada7017aafa0c5ef662cd62b0aa1caca50c1b12c7d427cb0a1a0f"
ARMS = ("zero", "first_only", "last_only", "reset_each", "cumulative")
SCHEDULES = {
    "zero": (0.0, 0.0, 0.0),
    "first_only": (0.3, 0.0, 0.0),
    "last_only": (0.0, 0.0, 0.3),
    "reset_each": (0.3, 0.3, 0.3),
    "cumulative": (0.3, 0.3, 0.3),
}
CONSTRAINED = ("reset_each", "first_only", "last_only")
FORMATS = ("binding", "qa")
QUERIES = ("target", "anchor")


def _set_lr(model, value):
    for i in model.config.ttt_layers:
        model.model.layers[i].mlp.ttt_lr = value


def reset_fast_weights(session):
    """Reset only W_fast at a complete chunk boundary; retain every KV tensor."""
    for i in session.model.config.ttt_layers:
        h, target, _ = session.cache.ttt_states[i]
        if h is not None or target is not None:
            raise RuntimeError("fast-weight reset requested inside a partial chunk")
        base = session.model.model.layers[i].mlp.down_proj.weight.detach().clone()
        session.cache.TTT_update((None, None, base), i)


def rollout(model, scene, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    session = NativeSession(model); trace = []
    for index, (chunk, lr) in enumerate(zip(scene["chunks"], SCHEDULES[arm])):
        if arm == "reset_each" and index:
            kv_before = session.cache.get_seq_length()
            reset_fast_weights(session)
            if session.cache.get_seq_length() != kv_before:
                raise RuntimeError("reset changed KV length")
        _set_lr(model, lr)
        session._forward(chunk)
        trace.append(native_stats(session))
    _set_lr(model, 0.3)
    return session, trace


def score_query(session, query, choices, eos_id):
    import torch
    branch = session.clone()
    with torch.inference_mode():
        hidden = branch._forward(query["ids"], collect=False)
        logits = branch.model.lm_head(hidden[:, -1:]).float().reshape(-1)
        logp = logits.log_softmax(-1)
        choice_nll = (-logp[choices]).tolist()
        prediction = min(range(len(choices)), key=lambda i: choice_nll[i])
        answer_nll = float(-logp[query["answer_id"]])
        eos_nll = float(-logp[eos_id])
        greedy_id = int(logits.argmax())
    values = choice_nll + [answer_nll, eos_nll]
    if not all(math.isfinite(x) for x in values):
        raise RuntimeError("nonfinite query score")
    return {
        "nll": answer_nll, "answer_nll": answer_nll, "eos_nll": eos_nll,
        "answer_eos_margin": eos_nll - answer_nll, "choice_nll": choice_nll,
        "prediction": prediction, "correct": int(prediction == query["label"]),
        "greedy_id": greedy_id, "greedy_correct": int(greedy_id == query["answer_id"]),
        "eos_greedy": int(greedy_id == eos_id),
    }


def build(source, root):
    source, root = Path(source), Path(root)
    if root.exists():
        raise FileExistsError(root)
    scenes, source_design = load_native_data(source)
    if file_digest(source / "summary.json") != SOURCE_SUMMARY_SHA:
        raise ValueError("wrong source summary")
    audit = json.loads((source / "execution_audit.json").read_text())
    if not audit["audit_passed"] or audit["summary_sha256"] != SOURCE_SUMMARY_SHA:
        raise ValueError("source was not independently audited")
    root.mkdir(); data = root / "data"; data.mkdir()
    scenes_path = data / "scenes.jsonl"
    scenes_path.write_text("".join(json.dumps(r) + "\n" for r in scenes))
    design = {
        "protocol": "native_accumulation_v1", "source_root": str(source),
        "source_summary_sha256": SOURCE_SUMMARY_SHA,
        "source_data_sha256": source_design["data_sha256"],
        "data_sha256": file_digest(scenes_path), "arms": list(ARMS),
        "schedules": {k: list(v) for k, v in SCHEDULES.items()},
        "contexts": len(scenes) * len(ARMS), "queries": len(scenes) * len(ARMS) * 4,
        "groups": 8, "primary_format": "binding", "eos_id": 151645,
        "model_step": 81381, "context_tokens": 12288, "chunk_tokens": 4096,
        "confirm_scored": False,
        "context_hashes": [{k:r[k] for k in FIELDS} for r in scenes],
    }
    (data / "design.json").write_text(json.dumps(design, indent=2) + "\n")
    print(json.dumps({k:v for k,v in design.items() if k != "context_hashes"}))


def load_data(root):
    root = Path(root); design = json.loads((root / "data/design.json").read_text())
    source = Path(design["source_root"]); source_rows, source_design = load_native_data(source)
    rows = list(map(json.loads, (root / "data/scenes.jsonl").read_text().splitlines()))
    if rows != source_rows or file_digest(root / "data/scenes.jsonl") != design["data_sha256"]:
        raise ValueError("derived scenes changed")
    if file_digest(source / "summary.json") != SOURCE_SUMMARY_SHA or design["source_data_sha256"] != source_design["data_sha256"]:
        raise ValueError("source provenance changed")
    if design["arms"] != list(ARMS) or design["schedules"] != {k:list(v) for k,v in SCHEDULES.items()}:
        raise ValueError("protocol changed")
    if design["context_hashes"] != [{k:r[k] for k in FIELDS} for r in rows] or len(rows) != 64:
        raise ValueError("context matrix changed")
    return rows, design


def collect(args):
    import torch
    scenes, design = load_data(args.root)
    rows = [r for r in scenes if (r["group"] // 2 + r["group"] % 2) % 2 == args.shard]
    rows.sort(key=lambda r: (r["group"], r["id"], r["cell"]))
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    torch.manual_seed(211); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter(); model, meta = load_model(args.model, "step81381_native")
    if design["eos_id"] != model.config.eos_token_id:
        raise ValueError("unexpected EOS token")
    before = parameter_digest(model); versions = [p._version for p in model.parameters()]
    with out.open("x") as sink:
        for index, scene in enumerate(rows):
            torch.cuda.empty_cache()
            (session, trace), profile = measure(lambda: rollout(model, scene, args.arm))
            parent = native_digest(session.cache); queries = {}
            for name, formats in scene["queries"].items():
                queries[name] = {}
                for fmt, query in formats.items():
                    result, query_profile = measure(lambda q=query: score_query(session, q, scene["choice_ids"], design["eos_id"]))
                    queries[name][fmt] = {**result, **query_profile}
            if parent != native_digest(session.cache):
                raise RuntimeError("query modified parent state")
            row = {k:scene[k] for k in FIELDS}
            row.update(arm=args.arm, model=args.model, data_sha256=design["data_sha256"],
                       weights_sha256=before, parent_sha256=parent, parent_unchanged=True,
                       queries=queries, rollout=profile, native_update_trace=trace)
            sink.write(json.dumps(row) + "\n"); sink.flush()
            print(json.dumps({"arm":args.arm,"completed":index+1,"total":len(rows)}), flush=True)
            del session
    _set_lr(model, 0.3)
    if before != parameter_digest(model) or versions != [p._version for p in model.parameters()]:
        raise RuntimeError("model parameters changed")
    meta.update(arm=args.arm, shard=args.shard, rows=len(rows), rows_sha256=file_digest(out),
                weights_sha256=before, backbone_unchanged=True, seconds=time.perf_counter()-started,
                torch=torch.__version__, transformers=__import__("transformers").__version__,
                gpu=torch.cuda.get_device_name(), schedule=list(SCHEDULES[args.arm]))
    Path(str(out) + ".audit.json").write_text(json.dumps(meta, indent=2) + "\n")


def _values(rows, fmt="binding", query_names=QUERIES):
    return [r["queries"][name][fmt] for r in rows for name in query_names]


def aggregate(rows, fmt="binding", query_names=QUERIES):
    values = _values(rows, fmt, query_names)
    return {key: statistics.fmean(v[key] for v in values) for key in
            ("answer_nll", "eos_nll", "answer_eos_margin", "correct", "greedy_correct", "eos_greedy")}


def _group_metric(rows, metric, fmt="binding"):
    return {g: aggregate([r for r in rows if r["group"] == g], fmt)[metric] for g in range(8)}


def paired_bootstrap(a, b, metric, direction=1.0, samples=20000, seed=1210):
    """direction*(arm A - arm B), resampling the eight source groups."""
    av, bv = _group_metric(a, metric), _group_metric(b, metric)
    diffs = [direction * (av[g] - bv[g]) for g in range(8)]
    rng = random.Random(seed); boot = []
    for _ in range(samples):
        boot.append(statistics.fmean(diffs[rng.randrange(8)] for _ in range(8)))
    boot.sort()
    return {"mean":statistics.fmean(diffs), "ci95":[boot[int(.025*samples)], boot[int(.975*samples)-1]],
            "group_differences":diffs, "unit":"source_group", "samples":samples, "seed":seed}


def _validate_row(row, scene, arm, design):
    if any(row[k] != scene[k] for k in FIELDS) or row["arm"] != arm or row["data_sha256"] != design["data_sha256"] or not row["parent_unchanged"]:
        raise ValueError("row provenance")
    if set(row["queries"]) != set(QUERIES) or len(row["native_update_trace"]) != 3:
        raise ValueError("query/update matrix")
    for step, trace in enumerate(row["native_update_trace"]):
        if set(trace) != set(map(str, LAYERS)) or any(not math.isfinite(v) or v < 0 for v in trace.values()):
            raise ValueError("invalid update trace")
        should_be_zero = SCHEDULES[arm][step] == 0 and (arm in ("zero", "last_only") or (arm == "first_only" and step == 0))
        if should_be_zero and any(v != 0 for v in trace.values()):
            raise ValueError("zero schedule changed fast weights")
    for name in QUERIES:
        if set(row["queries"][name]) != set(FORMATS):
            raise ValueError("format matrix")
        for fmt, value in row["queries"][name].items():
            q = scene["queries"][name][fmt]; scores = value["choice_nll"]
            if len(scores) != 8 or not all(math.isfinite(x) for x in scores + [value["answer_nll"], value["eos_nll"], value["answer_eos_margin"]]):
                raise ValueError("nonfinite score")
            pred = min(range(8), key=lambda i:scores[i])
            if value["prediction"] != pred or value["correct"] != int(pred == q["label"]):
                raise ValueError("candidate score mismatch")
            if abs(value["nll"] - value["answer_nll"]) > 1e-8 or abs(value["answer_eos_margin"] - (value["eos_nll"] - value["answer_nll"])) > 1e-6:
                raise ValueError("diagnostic score mismatch")


def summarize(root):
    root = Path(root); scenes, design = load_data(root); scene_index = {(r["id"],r["cell"]):r for r in scenes}
    all_rows = {}; audits = {}; row_hashes = {}
    for arm in ARMS:
        rows=[]; audits[arm]=[]
        for shard in (0,1):
            path=root/f"{arm}_{shard}.jsonl"; part=list(map(json.loads,path.read_text().splitlines()))
            audit=json.loads(Path(str(path)+".audit.json").read_text())
            if audit["rows"] != len(part) or audit["rows_sha256"] != file_digest(path) or not audit["backbone_unchanged"] or audit["arm"] != arm or audit["schedule"] != list(SCHEDULES[arm]):
                raise ValueError("shard audit")
            check_loading(audit["loading_info"], audit["excluded_adapters"])
            for row in part:
                if (row["group"]//2 + row["group"]%2)%2 != shard or row["weights_sha256"] != audit["weights_sha256"] or row["model"] != audit["path"]:
                    raise ValueError("shard/model provenance")
                _validate_row(row, scene_index[row["id"],row["cell"]], arm, design)
            rows += part; audits[arm].append(audit); row_hashes[path.name]=file_digest(path)
        if len(rows) != 64 or len({(r["id"],r["cell"]) for r in rows}) != 64:
            raise ValueError("incomplete arm")
        if any(len({json.dumps(a[k],sort_keys=True) for a in audits[arm]}) != 1 for k in ("weights_sha256","source_files_sha256","config","path")):
            raise ValueError("mixed shards")
        all_rows[arm]=rows
    if len({audits[a][0]["weights_sha256"] for a in ARMS}) != 1 or len({json.dumps(audits[a][0]["source_files_sha256"],sort_keys=True) for a in ARMS}) != 1:
        raise ValueError("arms did not use the same checkpoint")
    tables={}
    for arm,rows in all_rows.items():
        profiles=[p for r in rows for p in [r["rollout"]]+[v for qs in r["queries"].values() for v in qs.values()]]
        tables[arm]={
            "binding":aggregate(rows,"binding"), "qa":aggregate(rows,"qa"),
            "binding_by_query":{q:aggregate(rows,"binding",(q,)) for q in QUERIES},
            "binding_by_regime":{reg:aggregate([r for r in rows if r["regime"]==reg],"binding") for reg in REGIMES},
            "binding_by_domain":{dom:aggregate([r for r in rows if r["domain"]==dom],"binding") for dom in ("fineweb","longcrawl")},
            "binding_by_group":{str(g):aggregate([r for r in rows if r["group"]==g],"binding") for g in range(8)},
            "rollout_seconds_mean":statistics.fmean(r["rollout"]["seconds"] for r in rows),
            "query_seconds_mean":statistics.fmean(v["seconds"] for r in rows for qs in r["queries"].values() for v in qs.values()),
            "peak":{k:max(p[k] for p in profiles) for k in ("peak_allocated_gib","peak_reserved_gib")},
            "relative_displacement_mean":[{str(i):statistics.fmean(r["native_update_trace"][s][str(i)] for r in rows) for i in LAYERS} for s in range(3)],
        }
    comparisons={}
    for arm in ARMS[:-1]:
        comparisons[arm]={
            "answer_nll_improvement":paired_bootstrap(all_rows["cumulative"],all_rows[arm],"answer_nll",1.0),
            "margin_improvement":paired_bootstrap(all_rows[arm],all_rows["cumulative"],"answer_eos_margin",1.0),
            "greedy_accuracy_improvement":paired_bootstrap(all_rows[arm],all_rows["cumulative"],"greedy_correct",1.0),
            "choice_accuracy_improvement":paired_bootstrap(all_rows[arm],all_rows["cumulative"],"correct",1.0),
        }
    selected=min(CONSTRAINED,key=lambda a:(tables[a]["binding"]["answer_nll"],CONSTRAINED.index(a)))
    c=comparisons[selected]
    accumulation_gate=(c["answer_nll_improvement"]["mean"]>=.25 and c["answer_nll_improvement"]["ci95"][0]>0 and
                       c["margin_improvement"]["mean"]>=.5 and c["margin_improvement"]["ci95"][0]>0 and
                       c["greedy_accuracy_improvement"]["mean"]>=.10)
    content=paired_bootstrap(all_rows["zero"],all_rows[selected],"answer_nll",1.0)
    decision={"selected_constrained_arm":selected,"accumulation_harm_supported":accumulation_gate,
              "restricted_write_vs_zero_nll_improvement":content,
              "restricted_write_content_benefit_supported":content["ci95"][0]>0,
              "rule":"lowest binding answer NLL among reset_each, first_only, last_only; frozen tie order"}
    return {"protocol":"native_accumulation_v1","contexts":320,"queries":1280,"primary_format":"binding",
            "confirm_scored":False,"data_sha256":design["data_sha256"],"source_summary_sha256":SOURCE_SUMMARY_SHA,
            "models":tables,"paired_vs_cumulative":comparisons,"decision":decision,
            "loading_audits":audits,"rows_sha256":row_hashes,"terminal_status":"completed_native_accumulation_diagnostic",
            "scope":"Observed dev mechanism diagnostic; no training, labels, controller, or held-out evaluation."}


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    b=sub.add_parser("build"); b.add_argument("--source",required=True); b.add_argument("--root",required=True)
    c=sub.add_parser("collect"); c.add_argument("--root",required=True); c.add_argument("--model",required=True); c.add_argument("--arm",choices=ARMS,required=True); c.add_argument("--shard",type=int,choices=(0,1),required=True); c.add_argument("--output",required=True)
    s=sub.add_parser("summarize"); s.add_argument("--root",required=True); s.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="build": build(args.source,args.root)
    elif args.command=="collect": collect(args)
    else: Path(args.output).write_text(json.dumps(summarize(args.root),indent=2)+"\n")


if __name__ == "__main__": main()
