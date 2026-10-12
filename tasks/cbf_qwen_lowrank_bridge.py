"""Causal content gate for a Qwen-grounded rank-one key-value memory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

from tasks.build_cbf_memory_value import file_digest
from tasks.cbf_fastweight_content_v2 import load_data as load_source
from tasks.cbf_selective import measure
from tasks.train_final_writer_binding import expand_scenes, identity, bootstrap, write_json


DEV_GROUPS = tuple(range(8))
Q_GROUPS = tuple(range(8, 16))
V_GROUPS = tuple(range(16, 24))
SCALES = (0., 1., 2., 4., 8., 16.)
EOS_ID = 151645


def build(source, root):
    root = Path(root)
    if root.exists(): raise FileExistsError(root)
    rows, source_design = load_source(source)
    rows = expand_scenes(rows)
    root.mkdir(parents=True); data = root / "data"; data.mkdir()
    path = data / "episodes.jsonl"
    path.write_text("".join(json.dumps({**identity(row), "key": row["queries"]["target"]["options"]["key"]}) + "\n" for row in rows))
    design = {"protocol": "qwen_lowrank_bridge_v1", "source_root": str(Path(source).resolve()),
              "source_data_sha256": source_design["data_sha256"], "episodes_sha256": file_digest(path),
              "dev_groups": list(DEV_GROUPS), "q_groups": list(Q_GROUPS), "v_groups": list(V_GROUPS),
              "scales": list(SCALES), "selection": "candidate accuracy, greedy accuracy, min control NLL, lower scale",
              "gate": {"candidate_accuracy": .75, "domain_candidate_accuracy": .75,
                       "greedy_accuracy": .75, "accuracy_gain": .5, "nll_gain": .2,
                       "eos_rate_max": .05}, "v_locked_until_q": True}
    write_json(data / "design.json", design); return design


def load(root):
    root = Path(root); design = json.loads((root / "data/design.json").read_text())
    source_rows, source_design = load_source(design["source_root"]); rows = expand_scenes(source_rows)
    expected = "".join(json.dumps({**identity(row), "key": row["queries"]["target"]["options"]["key"]}) + "\n" for row in rows)
    path = root / "data/episodes.jsonl"
    if path.read_text() != expected or file_digest(path) != design["episodes_sha256"]: raise ValueError("episodes changed")
    if source_design["data_sha256"] != design["source_data_sha256"]: raise ValueError("source changed")
    return rows, design


def load_model(path):
    from tasks.cbf_checkpoint_native import load_model as native
    model, meta = native(path, "step81381_native")
    return model, meta


def unit_key(model, tokenizer, text):
    import torch
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids: raise ValueError("empty entity key")
    tensor = torch.tensor(ids, device=model.device)
    value = model.get_input_embeddings()(tensor).float().mean(0)
    return value / value.norm().clamp_min(1e-12)


def unit_value(model, token_id):
    value = model.lm_head.weight[token_id].float()
    return value / value.norm().clamp_min(1e-12)


def random_value(value, seed):
    import torch
    generator = torch.Generator(device=value.device); generator.manual_seed(seed)
    sign = torch.empty_like(value).bernoulli_(.5, generator=generator).mul_(2).sub_(1)
    result = value * sign
    return result, float(result.norm() / value.norm().clamp_min(1e-12))


def base_hidden(model, query_ids):
    from cbf_ttt.event_writer import cache_for, forward
    return forward(model, query_ids, cache_for(model))[:, -1].float().reshape(-1)


def score(model, hidden, residual, scale, row):
    import torch
    state = hidden + scale * residual
    logits = model.lm_head(state.to(model.lm_head.weight.dtype)).float().reshape(-1)
    logp = logits.log_softmax(-1); choices = torch.tensor(row["choice_ids"], device=logits.device)
    candidate = logits[choices]; candidate_logp = candidate.log_softmax(-1)
    return {"nll": float(-logp[row["answer_id"]]), "candidate_nll": float(-candidate_logp[row["label"]]),
            "candidate_correct": int(candidate.argmax() == row["label"]),
            "greedy_correct": int(logits.argmax() == row["answer_id"]),
            "greedy_id": int(logits.argmax()), "eos_first": int(logits.argmax() == EOS_ID)}


def collect(args):
    import torch
    from transformers import AutoTokenizer
    rows, design = load(args.root); groups = {"dev": DEV_GROUPS, "q": Q_GROUPS, "v": V_GROUPS}[args.stage]
    groups = tuple(group for group in groups if args.shard == -1 or group % 2 == args.shard)
    rows = [row for row in rows if row["group"] in groups]
    scales = SCALES if args.stage == "dev" else (json.loads(Path(args.selection).read_text())["selected_scale"],)
    torch.manual_seed(2011); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model, meta = load_model(args.model); tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    output = Path(args.output)
    if output.exists(): raise FileExistsError(output)
    start = time.perf_counter(); records = []
    with output.open("x") as sink, torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for group in groups:
            part = sorted((row for row in rows if row["group"] == group), key=lambda row: row["label"])
            key = unit_key(model, tokenizer, part[0]["queries"]["target"]["options"]["key"])
            hidden, query_profile = measure(lambda: base_hidden(model, part[0]["query_ids"]))
            values = [unit_value(model, row["answer_id"]) for row in part]
            for row in part:
                random_residual, ratio = random_value(values[row["label"]], 2012 + group * 8 + row["label"])
                # k dot k is exactly the rank-one memory read for this entity.
                similarity = float(torch.dot(key, key))
                residuals = {"correct": values[row["label"]] * similarity,
                             "wrong": values[(row["label"] + 4) % 8] * similarity,
                             "zero": torch.zeros_like(values[0]), "random": random_residual * similarity}
                scores = {str(scale): {name: score(model, hidden, residual, scale, row)
                                       for name, residual in residuals.items()} for scale in scales}
                record = {**identity(row), "stage": args.stage, "key": row["queries"]["target"]["options"]["key"],
                          "key_self_similarity": similarity, "random_norm_ratio": ratio,
                          "scores": scores, "query_profile": query_profile,
                          "source_data_sha256": design["source_data_sha256"]}
                sink.write(json.dumps(record) + "\n"); sink.flush(); records.append(record)
        audit = {"stage": args.stage, "groups": list(groups), "rows": len(records), "rows_sha256": file_digest(output),
                 "scales": list(scales), "seconds": time.perf_counter() - start,
                 "source_data_sha256": design["source_data_sha256"], "loading_info": meta["loading_info"],
                 "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                 "transformers": __import__("transformers").__version__}
    write_json(str(output) + ".audit.json", audit); return audit


def read_rows(root, paths, stage):
    _, design = load(root); rows = []
    for path in map(Path, paths):
        part = list(map(json.loads, path.read_text().splitlines())); audit = json.loads(Path(str(path) + ".audit.json").read_text())
        if audit["rows"] != len(part) or audit["rows_sha256"] != file_digest(path) or audit["stage"] != stage: raise ValueError("artifact mismatch")
        rows.extend(part)
    groups = {"dev": DEV_GROUPS, "q": Q_GROUPS, "v": V_GROUPS}[stage]
    if len(rows) != 8 * len(groups) or {row["group"] for row in rows} != set(groups): raise ValueError("stage matrix")
    if any(abs(row["key_self_similarity"] - 1) > 1e-5 or abs(row["random_norm_ratio"] - 1) > 1e-6 for row in rows): raise ValueError("norm/key audit")
    return rows, design


def aggregate(rows, scale):
    key = str(float(scale)); policies = {}
    for policy in ("correct", "wrong", "zero", "random"):
        policies[policy] = {metric: statistics.fmean(row["scores"][key][policy][metric] for row in rows)
                            for metric in ("nll", "candidate_nll", "candidate_correct", "greedy_correct", "eos_first")}
    comparisons = {}
    groups = sorted({row["group"] for row in rows})
    for control in ("wrong", "zero", "random"):
        for metric, sign in (("nll", 1), ("candidate_correct", -1)):
            values = {group: statistics.fmean(sign * (row["scores"][key][control][metric] - row["scores"][key]["correct"][metric])
                                                   for row in rows if row["group"] == group) for group in groups}
            comparisons[f"correct_vs_{control}_{metric}_gain"] = bootstrap(values, seed=2020)
    domains = {domain: statistics.fmean(row["scores"][key]["correct"]["candidate_correct"] for row in rows if row["domain"] == domain)
               for domain in ("fineweb", "longcrawl")}
    return {"scale": scale, "policies": policies, "comparisons": comparisons, "domains": domains}


def gate(table, design):
    g = design["gate"]; p = table["policies"]; c = table["comparisons"]
    passed = (p["correct"]["candidate_correct"] >= g["candidate_accuracy"] and min(table["domains"].values()) >= g["domain_candidate_accuracy"] and
              p["correct"]["greedy_correct"] >= g["greedy_accuracy"] and p["correct"]["eos_first"] <= g["eos_rate_max"])
    for control in ("wrong", "zero"):
        passed = passed and c[f"correct_vs_{control}_candidate_correct_gain"]["mean"] >= g["accuracy_gain"]
    for control in ("wrong", "zero", "random"):
        item = c[f"correct_vs_{control}_nll_gain"]
        passed = passed and item["mean"] >= g["nll_gain"] and item["ci95"][0] > 0
    return passed


def summarize_dev(root, paths, output):
    rows, design = read_rows(root, paths, "dev"); tables = {str(scale): aggregate(rows, scale) for scale in SCALES}
    selected = max(SCALES, key=lambda scale: (tables[str(scale)]["policies"]["correct"]["candidate_correct"],
        tables[str(scale)]["policies"]["correct"]["greedy_correct"],
        min(tables[str(scale)]["comparisons"][f"correct_vs_{control}_nll_gain"]["mean"] for control in ("wrong", "zero", "random")), -scale))
    result = {"protocol": "qwen_lowrank_bridge_v1", "stage": "dev", "tables": tables,
              "selected_scale": selected, "selected_passed": gate(tables[str(selected)], design),
              "selection_rule": design["selection"], "q_scored": False, "v_scored": False}
    write_json(output, result); return result


def summarize_fixed(root, paths, stage, selection, output):
    rows, design = read_rows(root, paths, stage); selected = json.loads(Path(selection).read_text())["selected_scale"]
    table = aggregate(rows, selected); result = {"protocol": "qwen_lowrank_bridge_v1", "stage": stage, **table,
        "passed": gate(table, design), "rows": len(rows), "terminal_status": f"{stage}_passed" if gate(table, design) else f"stopped_by_{stage}",
        "v_scored": stage == "v", "rows_sha256": {Path(path).name: file_digest(Path(path)) for path in paths}}
    write_json(output, result); return result


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build"); b.add_argument("--source", required=True); b.add_argument("--root", required=True)
    c = sub.add_parser("collect"); c.add_argument("--root", required=True); c.add_argument("--model", required=True); c.add_argument("--tokenizer", required=True)
    c.add_argument("--stage", choices=("dev", "q", "v"), required=True); c.add_argument("--shard", type=int, choices=(-1, 0, 1), default=-1)
    c.add_argument("--selection"); c.add_argument("--output", required=True)
    d = sub.add_parser("summarize-dev"); d.add_argument("--root", required=True); d.add_argument("--inputs", nargs="+", required=True); d.add_argument("--output", required=True)
    s = sub.add_parser("summarize"); s.add_argument("--root", required=True); s.add_argument("--stage", choices=("q", "v"), required=True)
    s.add_argument("--inputs", nargs="+", required=True); s.add_argument("--selection", required=True); s.add_argument("--output", required=True)
    a = p.parse_args()
    if a.command == "build": print(json.dumps(build(a.source, a.root), indent=2))
    elif a.command == "collect": print(json.dumps(collect(a), indent=2))
    elif a.command == "summarize-dev": print(json.dumps(summarize_dev(a.root, a.inputs, a.output), indent=2))
    else: print(json.dumps(summarize_fixed(a.root, a.inputs, a.stage, a.selection, a.output), indent=2))


if __name__ == "__main__": main()
