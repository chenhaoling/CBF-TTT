"""Recover content binding in the pretrained final-checkpoint TTT writer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.build_cbf_memory_value import digest, file_digest
from tasks.cbf_fastweight_content_v2 import load_data as load_source
from tasks.cbf_selective import measure


TUNE_TRAIN_GROUPS = (0, 1, 2, 3, 4, 7)
TUNE_DEV_GROUPS = (5, 6)
FINAL_TRAIN_GROUPS = tuple(range(8))
Q_GROUPS = tuple(range(8, 16))
V_GROUPS = tuple(range(16, 24))
PAIRINGS = ((0, 4), (1, 5), (2, 6), (3, 7))
TUNE_ARMS = {"lr3e7": 3e-7, "lr1e6": 1e-6}
FORMAT = "options"
EOS_ID = 151645


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def tensor_digest(values):
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        h.update(name.encode())
        h.update(str((tuple(value.shape), str(value.dtype))).encode())
        h.update(value.detach().cpu().contiguous().view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


def expand_scenes(rows):
    """Counterfactually rotate the one differing color token through all choices."""
    by_group = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row)
    expanded = []
    for group in sorted(by_group):
        pair = sorted(by_group[group], key=lambda row: row["variant"])
        if len(pair) != 2:
            raise ValueError("each source group must contain one twin pair")
        left, right = pair
        differences = [(chunk, index) for chunk, (a, b) in enumerate(zip(left["chunks"], right["chunks"]))
                       for index, (x, y) in enumerate(zip(a, b)) if x != y]
        if len(differences) != 1 or differences[0][0] != 1:
            raise ValueError("expected exactly one color token in the writer block")
        chunk_index, token_index = differences[0]
        choices = left["choice_ids"]
        if len(choices) != 8 or len(set(choices)) != 8:
            raise ValueError("expected eight distinct single-token values")
        for label, answer_id in enumerate(choices):
            row = copy.deepcopy(left)
            row["chunks"][chunk_index][token_index] = answer_id
            query = copy.deepcopy(left["queries"]["target"][FORMAT])
            query.update(label=label, answer_id=answer_id)
            row.update(id=f"g{group:02d}_label{label}", label=label,
                       answer_id=answer_id, query_ids=query["ids"],
                       token_location=[chunk_index, token_index],
                       context_sha256=digest(json.dumps(row["chunks"]).encode()))
            row.pop("twin_id", None)
            row["queries"] = {"target": {FORMAT: query}}
            expanded.append(row)
    validate_expanded(expanded)
    return expanded


def validate_expanded(rows):
    if len(rows) != 24 * 8 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("expanded scene count")
    for group in range(24):
        part = [row for row in rows if row["group"] == group]
        if {row["label"] for row in part} != set(range(8)):
            raise ValueError("unbalanced group labels")
        if len({tuple(row["query_ids"]) for row in part}) != 1:
            raise ValueError("query changed across counterfactual values")
        if len({tuple(row["token_location"]) for row in part}) != 1:
            raise ValueError("counterfactual token location changed")
        base = part[0]
        for row in part:
            if len(row["chunks"]) != 2 or any(len(chunk) != 4096 for chunk in row["chunks"]):
                raise ValueError("writer layout")
            if sum(x != y for a, b in zip(base["chunks"], row["chunks"])
                   for x, y in zip(a, b)) > 1:
                raise ValueError("counterfactual changed more than the value token")
            if row["answer_id"] != row["choice_ids"][row["label"]]:
                raise ValueError("answer mapping")


def identity(row):
    return {key: row[key] for key in ("id", "group", "domain", "label", "answer_id",
                                      "sources", "token_location", "context_sha256")}


def build(source, root):
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    rows, source_design = load_source(source)
    expanded = expand_scenes(rows)
    root.mkdir(parents=True)
    data = root / "data"
    data.mkdir()
    episode_path = data / "episodes.jsonl"
    episode_path.write_text("".join(json.dumps(identity(row)) + "\n" for row in expanded))
    manifest = {
        "protocol": "final_writer_binding_v1", "source_root": str(Path(source).resolve()),
        "source_data_sha256": source_design["data_sha256"],
        "episodes_sha256": file_digest(episode_path), "format": FORMAT,
        "tune_train_groups": list(TUNE_TRAIN_GROUPS), "tune_dev_groups": list(TUNE_DEV_GROUPS),
        "final_train_groups": list(FINAL_TRAIN_GROUPS), "q_groups": list(Q_GROUPS),
        "v_groups": list(V_GROUPS), "values_per_group": 8,
        "tune_arms": TUNE_ARMS, "tune_rounds": 8, "final_rounds": 8,
        "memory_cap": .3, "candidate_weight": 1., "full_vocab_weight": .25,
        "swap_weight": 1., "zero_weight": .5, "margin": .2,
        "q_v_gate": {"candidate_accuracy": .75, "domain_candidate_accuracy": .75,
                     "greedy_accuracy": .5, "accuracy_gain": .25,
                     "nll_gain": .02, "eos_rate_max": .05},
        "v_locked_until_q": True,
    }
    write_json(data / "design.json", manifest)
    return manifest


def load(root):
    root = Path(root)
    design = json.loads((root / "data/design.json").read_text())
    rows, source_design = load_source(design["source_root"])
    expanded = expand_scenes(rows)
    expected = "".join(json.dumps(identity(row)) + "\n" for row in expanded)
    path = root / "data/episodes.jsonl"
    if path.read_text() != expected or file_digest(path) != design["episodes_sha256"]:
        raise ValueError("expanded episodes changed")
    if source_design["data_sha256"] != design["source_data_sha256"]:
        raise ValueError("source scenes changed")
    return expanded, design


def schedule(groups, rounds, seed=1729):
    rng = random.Random(seed)
    result = []
    for round_index in range(rounds):
        items = [(group, left, right) for group in groups for left, right in PAIRINGS]
        rng.shuffle(items)
        if round_index % 2:
            items = [(group, right, left) for group, left, right in items]
        result.extend(items)
    return result


def load_final_model(path):
    from tasks.cbf_checkpoint_native import load_model
    from scripts.probe_ttt_writer_training import writer_parameters
    model, meta = load_model(path, "step81381_native")
    params = writer_parameters(model)
    if len(params) != 14:
        raise ValueError("expected seven conv/proj writer pairs")
    return model, params, meta


def query_logits(model, memory, query_ids):
    from cbf_ttt.event_writer import cache_for, forward
    hidden = forward(model, query_ids, cache_for(model, memory))
    return model.lm_head(hidden[:, -1]).float().reshape(-1)


def paired_objective(left_logits, right_logits, zero_logits, left_label, right_label,
                     choice_ids, margin=.2, full_vocab_weight=.25,
                     swap_weight=1., zero_weight=.5):
    import torch
    import torch.nn.functional as F
    choice = torch.tensor(choice_ids, device=left_logits.device)
    candidate = torch.stack((left_logits[choice], right_logits[choice]))
    targets = torch.tensor([left_label, right_label], device=left_logits.device)
    candidate_ce = F.cross_entropy(candidate, targets)
    answer_ids = choice[targets]
    full_ce = (F.cross_entropy(left_logits.unsqueeze(0), answer_ids[:1]) +
               F.cross_entropy(right_logits.unsqueeze(0), answer_ids[1:])) / 2
    lp_left = candidate[0].log_softmax(-1)
    lp_right = candidate[1].log_softmax(-1)
    lp_zero = zero_logits[choice].log_softmax(-1)
    swap = (F.softplus(margin - (lp_left[left_label] - lp_right[left_label])) +
            F.softplus(margin - (lp_right[right_label] - lp_left[right_label]))) / 2
    utility = (F.softplus(margin - (lp_left[left_label] - lp_zero[left_label])) +
               F.softplus(margin - (lp_right[right_label] - lp_zero[right_label]))) / 2
    loss = candidate_ce + full_vocab_weight * full_ce + swap_weight * swap + zero_weight * utility
    accuracy = ((candidate.argmax(-1) == targets).float().mean())
    return loss, {"candidate_ce": candidate_ce, "full_ce": full_ce,
                  "swap": swap, "utility": utility, "accuracy": accuracy}


def train(args):
    import torch
    from cbf_ttt.event_writer import backbone_digest, write_memory_after_prefix
    rows, design = load(args.root)
    groups = tuple(map(int, args.groups.split(",")))
    selected = {(row["group"], row["label"]): row for row in rows if row["group"] in groups}
    if len(selected) != len(groups) * 8:
        raise ValueError("incomplete training groups")
    lr = args.lr
    if args.selection:
        selection = json.loads(Path(args.selection).read_text())
        lr = selection["selected_lr"]
    torch.manual_seed(1730)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, params, model_meta = load_final_model(args.model)
    initial_writer_sha = tensor_digest(params)
    backbone_sha = backbone_digest(model)
    frozen_versions = {name: parameter._version for name, parameter in model.named_parameters()
                       if not parameter.requires_grad}
    optimizer = torch.optim.AdamW(list(params.values()), lr=lr, weight_decay=0.)
    batches = schedule(groups, args.rounds)
    output = Path(args.output)
    output.mkdir(parents=True)
    manifest = {
        "protocol": "final_writer_binding_v1", "stage": args.stage, "arm": args.arm,
        "model": args.model, "groups": list(groups), "rounds": args.rounds,
        "updates": len(batches), "memory_writes": 2 * len(batches), "lr": lr,
        "seed": 1730, "order_seed": 1729, "memory_cap": design["memory_cap"],
        "loss_weights": {key: design[key] for key in
                         ("candidate_weight", "full_vocab_weight", "swap_weight", "zero_weight", "margin")},
        "source_data_sha256": design["source_data_sha256"],
        "episodes_sha256": design["episodes_sha256"], "initial_writer_sha256": initial_writer_sha,
        "backbone_sha256": backbone_sha, "loading_info": model_meta["loading_info"],
        "trainable_parameters": sum(parameter.numel() for parameter in params.values()),
        "writer_dtype": "float32", "compute_dtype": "bfloat16", "test_scored": False,
    }
    write_json(output / "training_manifest.json", manifest)
    records = []
    with (output / "steps.jsonl").open("x") as sink:
        for update, (group, left_label, right_label) in enumerate(batches, 1):
            left, right = selected[group, left_label], selected[group, right_label]

            def one_step():
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    left_memory = write_memory_after_prefix(model, left["chunks"][0], left["chunks"][1],
                                                            design["memory_cap"])
                    right_memory = write_memory_after_prefix(model, right["chunks"][0], right["chunks"][1],
                                                             design["memory_cap"])
                    left_logits = query_logits(model, left_memory, left["query_ids"])
                    right_logits = query_logits(model, right_memory, right["query_ids"])
                    with torch.no_grad():
                        zero_logits = query_logits(model, {}, left["query_ids"])
                    loss, components = paired_objective(
                        left_logits, right_logits, zero_logits, left_label, right_label,
                        left["choice_ids"], margin=design["margin"],
                        full_vocab_weight=design["full_vocab_weight"],
                        swap_weight=design["swap_weight"], zero_weight=design["zero_weight"])
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite writer loss")
                relative = max(float(delta.detach().float().norm() /
                                     model.model.layers[layer].mlp.down_proj.weight.float().norm())
                               for memory in (left_memory, right_memory)
                               for layer, delta in memory.items())
                if relative > .305:
                    raise RuntimeError("memory cap exceeded")
                loss.backward()
                gradients = {name: float(parameter.grad.float().norm()) if parameter.grad is not None else None
                             for name, parameter in params.items()}
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                       for parameter in params.values()):
                    raise RuntimeError("disconnected or nonfinite writer gradient")
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(list(params.values()), 1.,
                                                                      error_if_nonfinite=True))
                optimizer.step()
                return {"update": update, "group": group, "domain": left["domain"],
                        "labels": [left_label, right_label], "loss": float(loss.detach()),
                        **{key: float(value.detach()) for key, value in components.items()},
                        "unclipped_gradient_norm": gradient_norm,
                        "gradient_norms": gradients, "max_relative_delta": relative}

            record, profile = measure(one_step)
            record.update(profile)
            records.append(record)
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            if update == 1 or update % 16 == 0:
                print(json.dumps({key: record[key] for key in
                                  ("update", "loss", "candidate_ce", "swap", "utility",
                                   "accuracy", "max_relative_delta", "seconds")}), flush=True)
    if frozen_versions != {name: parameter._version for name, parameter in model.named_parameters()
                           if not parameter.requires_grad} or backbone_digest(model) != backbone_sha:
        raise RuntimeError("frozen backbone changed")
    final_writer_sha = tensor_digest(params)
    if final_writer_sha == initial_writer_sha:
        raise RuntimeError("writer did not update")
    checkpoint = output / "writer.pt"
    torch.save({"writer": {name: parameter.detach().cpu() for name, parameter in params.items()},
                "manifest": manifest}, checkpoint)
    complete = {
        "updates": len(records), "first32_loss": statistics.fmean(row["loss"] for row in records[:32]),
        "last32_loss": statistics.fmean(row["loss"] for row in records[-32:]),
        "first32_accuracy": statistics.fmean(row["accuracy"] for row in records[:32]),
        "last32_accuracy": statistics.fmean(row["accuracy"] for row in records[-32:]),
        "mean_update_seconds": statistics.fmean(row["seconds"] for row in records),
        "peak_allocated_gib": max(row["peak_allocated_gib"] for row in records),
        "peak_reserved_gib": max(row["peak_reserved_gib"] for row in records),
        "max_relative_delta": max(row["max_relative_delta"] for row in records),
        "initial_writer_sha256": initial_writer_sha, "final_writer_sha256": final_writer_sha,
        "checkpoint_sha256": file_digest(checkpoint), "steps_sha256": file_digest(output / "steps.jsonl"),
        "backbone_unchanged": True, "optimizer_state_fp32": all(
            value.dtype == torch.float32 for state in optimizer.state.values()
            for value in state.values() if isinstance(value, torch.Tensor)), "test_scored": False,
    }
    write_json(output / "training_complete.json", complete)
    return complete


def score_logits(logits, row):
    import torch
    logp = logits.log_softmax(-1)
    choices = torch.tensor(row["choice_ids"], device=logits.device)
    candidate = logits[choices]
    candidate_logp = candidate.log_softmax(-1)
    return {"nll": float(-logp[row["answer_id"]]),
            "candidate_nll": float(-candidate_logp[row["label"]]),
            "candidate_prediction": int(candidate.argmax()),
            "candidate_correct": int(candidate.argmax() == row["label"]),
            "greedy_id": int(logits.argmax()),
            "greedy_correct": int(logits.argmax() == row["answer_id"]),
            "eos_first": int(logits.argmax() == EOS_ID)}


def random_memory(memory, seed):
    import torch
    output = {}
    ratios = {}
    for layer, delta in memory.items():
        generator = torch.Generator(device=delta.device)
        generator.manual_seed(seed * 101 + layer)
        sign = torch.empty_like(delta).bernoulli_(.5, generator=generator).mul_(2).sub_(1)
        value = delta * sign
        output[layer] = value
        ratios[str(layer)] = float(value.float().norm() / delta.float().norm().clamp_min(1e-30))
    return output, ratios


def load_writer_checkpoint(model, params, checkpoint, design):
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    manifest = state["manifest"]
    if manifest["source_data_sha256"] != design["source_data_sha256"] or set(state["writer"]) != set(params):
        raise ValueError("writer checkpoint provenance")
    with torch.no_grad():
        for name, parameter in params.items():
            parameter.copy_(state["writer"][name].to(parameter.device))
    return manifest


def evaluate(args):
    import torch
    from cbf_ttt.event_writer import backbone_digest, write_memory_after_prefix
    rows, design = load(args.root)
    groups = tuple(map(int, args.groups.split(",")))
    if args.shard in (0, 1):
        groups = tuple(group for group in groups if group % 2 == args.shard)
    selected = {(row["group"], row["label"]): row for row in rows if row["group"] in groups}
    torch.manual_seed(1731)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, params, meta = load_final_model(args.model)
    manifest = load_writer_checkpoint(model, params, args.checkpoint, design)
    model.requires_grad_(False)
    backbone_sha = backbone_digest(model)
    writer_sha = tensor_digest(params)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    records = []
    with output.open("x") as sink, torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for group in groups:
            group_rows = [selected[group, label] for label in range(8)]
            memories = []
            write_profiles = []
            for row in group_rows:
                memory, profile = measure(lambda row=row: write_memory_after_prefix(
                    model, row["chunks"][0], row["chunks"][1], design["memory_cap"]))
                memories.append(memory)
                write_profiles.append(profile)
            zero_logits = query_logits(model, {}, group_rows[0]["query_ids"])
            for row in group_rows:
                correct = memories[row["label"]]
                wrong = memories[(row["label"] + 4) % 8]
                random_value, ratios = random_memory(correct, 1900 + group * 8 + row["label"])
                policies = {}
                for name, memory in (("correct", correct), ("wrong", wrong),
                                     ("zero", {}), ("random", random_value)):
                    if name == "zero":
                        logits, profile = zero_logits, {"seconds": 0., "peak_allocated_gib": 0.,
                                                       "peak_reserved_gib": 0.}
                    else:
                        logits, profile = measure(lambda memory=memory: query_logits(
                            model, memory, row["query_ids"]))
                    policies[name] = {**score_logits(logits, row), **profile}
                record = {**identity(row), "stage": args.stage, "writer_sha256": writer_sha,
                          "checkpoint_sha256": file_digest(Path(args.checkpoint)),
                          "source_data_sha256": design["source_data_sha256"],
                          "policies": policies, "random_norm_ratios": ratios,
                          "write_profile": write_profiles[row["label"]]}
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                records.append(record)
                print(json.dumps({"stage": args.stage, "completed": len(records),
                                  "total": len(groups) * 8}), flush=True)
    audit = {"stage": args.stage, "groups": list(groups), "rows": len(records),
             "rows_sha256": file_digest(output), "writer_sha256": writer_sha,
             "checkpoint_sha256": file_digest(Path(args.checkpoint)),
             "backbone_sha256": backbone_sha, "backbone_unchanged": True,
             "source_data_sha256": design["source_data_sha256"], "training_manifest": manifest,
             "seconds": time.perf_counter() - started, "gpu": torch.cuda.get_device_name(),
             "torch": torch.__version__, "transformers": __import__("transformers").__version__,
             "loading_info": meta["loading_info"]}
    write_json(str(output) + ".audit.json", audit)
    return audit


def bootstrap(values, seed=1732, samples=20000):
    groups = sorted(values)
    mean = statistics.fmean(values.values())
    rng = random.Random(seed)
    draws = sorted(statistics.fmean(values[rng.choice(groups)] for _ in groups) for _ in range(samples))
    return {"mean": mean, "ci95": [draws[int(.025 * samples)], draws[int(.975 * samples)]]}


def summarize_files(root, paths, stage, save=None):
    rows, design = load(root)
    raw = []
    audits = []
    for path in map(Path, paths):
        part = list(map(json.loads, path.read_text().splitlines()))
        audit = json.loads(Path(str(path) + ".audit.json").read_text())
        if audit["rows"] != len(part) or audit["rows_sha256"] != file_digest(path):
            raise ValueError("evaluation artifact mismatch")
        raw.extend(part)
        audits.append(audit)
    expected_groups = TUNE_DEV_GROUPS if stage == "dev" else Q_GROUPS if stage == "q" else V_GROUPS
    if {row["group"] for row in raw} != set(expected_groups) or len(raw) != 8 * len(expected_groups):
        raise ValueError("evaluation matrix")
    if len({row["id"] for row in raw}) != len(raw) or len({row["writer_sha256"] for row in raw}) != 1:
        raise ValueError("mixed or duplicate evaluation")
    policies = {}
    for policy in ("correct", "wrong", "zero", "random"):
        policies[policy] = {metric: statistics.fmean(row["policies"][policy][metric] for row in raw)
                            for metric in ("nll", "candidate_nll", "candidate_correct",
                                           "greedy_correct", "eos_first")}
    comparisons = {}
    for control in ("wrong", "zero", "random"):
        for metric, sign in (("nll", 1), ("candidate_correct", -1)):
            values = {group: statistics.fmean(
                sign * (row["policies"][control][metric] - row["policies"]["correct"][metric])
                for row in raw if row["group"] == group) for group in expected_groups}
            comparisons[f"correct_vs_{control}_{metric}_gain"] = bootstrap(values)
    domains = {domain: statistics.fmean(row["policies"]["correct"]["candidate_correct"]
                                        for row in raw if row["domain"] == domain)
               for domain in ("fineweb", "longcrawl")}
    gate = design["q_v_gate"]
    passed = (policies["correct"]["candidate_correct"] >= gate["candidate_accuracy"] and
              min(domains.values()) >= gate["domain_candidate_accuracy"] and
              policies["correct"]["greedy_correct"] >= gate["greedy_accuracy"] and
              policies["correct"]["eos_first"] <= gate["eos_rate_max"])
    for control in ("wrong", "zero"):
        passed = passed and comparisons[f"correct_vs_{control}_candidate_correct_gain"]["mean"] >= gate["accuracy_gain"]
    for control in ("wrong", "zero", "random"):
        item = comparisons[f"correct_vs_{control}_nll_gain"]
        passed = passed and item["mean"] >= gate["nll_gain"] and item["ci95"][0] > 0
    summary = {"protocol": "final_writer_binding_v1", "stage": stage,
               "groups": list(expected_groups), "rows": len(raw), "policies": policies,
               "domains": domains, "comparisons": comparisons, "passed": passed,
               "writer_sha256": raw[0]["writer_sha256"],
               "checkpoint_sha256": raw[0]["checkpoint_sha256"],
               "rows_sha256": {Path(path).name: file_digest(Path(path)) for path in paths},
               "source_data_sha256": design["source_data_sha256"],
               "terminal_status": f"{stage}_passed" if passed else f"stopped_by_{stage}",
               "v_scored": stage == "v", "audit_count": len(audits)}
    if save:
        write_json(save, summary)
    return summary


def select_tune(root, arms, output):
    candidates = []
    for arm in arms:
        path = Path(root) / arm / "dev_summary.json"
        summary = json.loads(path.read_text())
        train_complete = json.loads((Path(root) / arm / "training_complete.json").read_text())
        score = min(summary["comparisons"][f"correct_vs_{control}_nll_gain"]["mean"]
                    for control in ("wrong", "zero", "random"))
        candidates.append({"arm": arm, "lr": TUNE_ARMS[arm], "score": score,
                           "candidate_accuracy": summary["policies"]["correct"]["candidate_correct"],
                           "greedy_accuracy": summary["policies"]["correct"]["greedy_correct"],
                           "passed_full_gate": summary["passed"], "summary_sha256": file_digest(path),
                           "training_complete_sha256": file_digest(Path(root) / arm / "training_complete.json"),
                           "last32_accuracy": train_complete["last32_accuracy"]})
    selected = max(candidates, key=lambda item: (item["passed_full_gate"], item["score"],
                                                  item["candidate_accuracy"], item["greedy_accuracy"],
                                                  -item["lr"]))
    result = {"protocol": "final_writer_binding_v1", "stage": "tune_selection",
              "candidates": candidates, "selected_arm": selected["arm"],
              "selected_lr": selected["lr"], "selection_rule":
              "gate, minimum Correct-vs-control NLL gain, candidate accuracy, greedy accuracy, lower LR",
              "q_scored": False, "v_scored": False}
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build"); b.add_argument("--source", required=True); b.add_argument("--root", required=True)
    t = sub.add_parser("train"); t.add_argument("--root", required=True); t.add_argument("--model", required=True)
    t.add_argument("--output", required=True); t.add_argument("--stage", choices=("tune", "final"), required=True)
    t.add_argument("--arm", required=True); t.add_argument("--groups", required=True); t.add_argument("--rounds", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-7); t.add_argument("--selection")
    e = sub.add_parser("evaluate"); e.add_argument("--root", required=True); e.add_argument("--model", required=True)
    e.add_argument("--checkpoint", required=True); e.add_argument("--groups", required=True); e.add_argument("--stage", choices=("dev", "q", "v"), required=True)
    e.add_argument("--shard", type=int, choices=(-1, 0, 1), default=-1); e.add_argument("--output", required=True)
    s = sub.add_parser("summarize"); s.add_argument("--root", required=True); s.add_argument("--stage", choices=("dev", "q", "v"), required=True)
    s.add_argument("--inputs", nargs="+", required=True); s.add_argument("--output", required=True)
    q = sub.add_parser("select-tune"); q.add_argument("--root", required=True); q.add_argument("--arms", nargs="+", required=True); q.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "build": print(json.dumps(build(args.source, args.root), indent=2))
    elif args.command == "train": print(json.dumps(train(args), indent=2))
    elif args.command == "evaluate": print(json.dumps(evaluate(args), indent=2))
    elif args.command == "summarize": print(json.dumps(summarize_files(args.root, args.inputs, args.stage, args.output), indent=2))
    else: print(json.dumps(select_tune(args.root, args.arms, args.output), indent=2))


if __name__ == "__main__":
    main()
