"""Train continuous history/last-2/last-1 forgetting on Qwen memory."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import time

from tasks.diagnose_cbf_qwen_contextual_geometry import (
    apply_contextual_transform, extract_contextual,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import write_json
from tasks.train_cbf_lowrank_forgetting import (
    episode_candidates,
)
from tasks.train_cbf_qwen_lowrank_forgetting import QwenEmbeddings, SEEDS
from tasks.train_cbf_structured_forgetting import (
    boundary_truth, load_split, parse_block,
)


PROTOCOL = "qwen_continuous_cohort_forgetting_v1"
DATA_SEED = 202610151
GROUP_COUNTS = {"train": 128, "dev": 32, "test": 64}
GEOMETRY_HASHES = {
    "summary.json": "4fa6e0aba9fa5ab335ee9650b31547379a7525cd4c52db3b02d49aa49fbd272c",
    "training.json": "01ecc4e9e5875485750b1f5a6121826eb254d9d865442a2c45ba79456ebb170c",
    "training_manifest.json": "d13cfa35397f65ed5231aace10d0338480482f08aba3f14f4e4f5b55c9347510",
    "contextual_transforms.pt": "524d213a82f158fddadab8973a28770a687138f528e269659430490affe74ae5",
}
GRID = tuple(sorted(itertools.product((0., .5, 1.), repeat=3), reverse=True))
FIXED = {
    "fixed_retain": (1., 1., 1.),
    "fixed_global_clear": (0., 0., 0.),
    "fixed_global_half": (.5, .5, .5),
    "fixed_window2": (0., 0., 1.),
    "fixed_window3": (0., 1., 1.),
}
POLICIES = tuple(FIXED) + (
    "deterministic_random", "reference_blind", "grid_classifier", "learned",
    "physics_cohort", "exact_reference", "oracle_grid",
)
CONTROLLER_SEEDS = (2611, 2617, 2621)
FEATURE_NAMES = (
    "memory_norm", "delta_norm", "memory_delta_alignment",
    "history_norm_fraction", "last2_norm_fraction", "last1_norm_fraction",
    "history_delta_alignment", "last2_delta_alignment", "last1_delta_alignment",
    "has_reference", "reference_last1", "reference_last2", "reference_history",
    "target_mass_fraction", "reset_or_revoke_prelude", "block_index",
    "cohort_count", "active_fact_count", "fact_overlap", "fact_duplicate",
)
REFERENCE_INDICES = tuple(FEATURE_NAMES.index(name) for name in (
    "has_reference", "reference_last1", "reference_last2", "reference_history",
    "target_mass_fraction", "reset_or_revoke_prelude"))


def verify_inputs(data_root, geometry_root):
    data_root, geometry_root = Path(data_root), Path(geometry_root)
    manifest = json.loads((data_root / "manifest.json").read_text())
    if (manifest.get("protocol") != "cbf_event_curriculum_v1"
            or manifest.get("seed") != DATA_SEED
            or manifest.get("group_counts") != GROUP_COUNTS
            or manifest.get("audit", {}).get("causal_answers_verified") is not True
            or manifest.get("audit", {}).get("entities_disjoint") is not True
            or manifest.get("audit", {}).get("templates_disjoint") is not True):
        raise ValueError("unexpected fresh event-curriculum release")
    observed = {name: sha(geometry_root / name) for name in GEOMETRY_HASHES}
    if observed != GEOMETRY_HASHES:
        raise ValueError("frozen contextual geometry dependency changed")
    summary = json.loads((geometry_root / "summary.json").read_text())
    if any((summary.get("stage_passed") is not True,
            summary.get("selected_variant") != "context_last",
            summary.get("test_scored") is not False,
            summary["selected_configs"]["context_last"] != {
                "internal": "context_last_r0.01", "regularizer": .01,
                "unknown_raw": .75, "train_accuracy": .9988787615740741})):
        raise ValueError("unexpected contextual geometry configuration")
    return manifest, summary["selected_configs"]["context_last"]


def clone_state(state):
    return [{"batch": row["batch"], "facts": dict(row["facts"]),
             "delta": row["delta"], "weight": float(row["weight"])}
            for row in state]


def apply_retention(state, retention, assignment, delta):
    if len(retention) != 3 or any(not 0 <= value <= 1 for value in retention):
        raise ValueError("cohort retention requires three values in [0,1]")
    output, size = clone_state(state), len(state)
    for index, row in enumerate(output):
        cohort = 2 if index == size - 1 else 1 if index == size - 2 else 0
        row["weight"] *= float(retention[cohort])
    output.append({"batch": assignment["batch"], "facts": dict(assignment["facts"]),
                   "delta": delta, "weight": 1.})
    return output


def apply_exact_reference(state, assignment, delta):
    output = clone_state(state)
    if assignment["prelude_lines"] and not assignment["old_refs"]:
        output = []
    elif assignment["old_refs"]:
        target = assignment["old_refs"][0]
        output = [row for row in output if row["batch"] != target]
    output.append({"batch": assignment["batch"], "facts": dict(assignment["facts"]),
                   "delta": delta, "weight": 1.})
    return output


def matrix(state, dim):
    import torch
    if not state:
        return torch.zeros((dim, dim), dtype=torch.float32)
    output = torch.zeros_like(state[0]["delta"])
    for row in state:
        output += row["weight"] * row["delta"]
    return output


def batch_delta(assignment, block_index, embeddings):
    import torch
    first = next(iter(assignment["facts"]))
    device = embeddings.get("entity", first).device
    delta = torch.zeros((embeddings.dim, embeddings.dim), dtype=torch.float32,
                        device=device)
    strength = 1. + .15 * block_index
    for entity, value in assignment["facts"].items():
        delta += strength * torch.outer(
            embeddings.get("value", value), embeddings.get("entity", entity))
    return delta


def query_metrics_from_memory(memory, query, candidates, embeddings,
                              unknown_raw=.75, temperature=.2):
    import torch
    output = memory @ embeddings.get("entity", query["key"])
    logits = torch.stack([
        output.new_tensor(unknown_raw) if value == "unknown" else
        torch.dot(embeddings.get("value", value), output)
        for value in candidates]) / temperature
    target = candidates.index(query["answer"])
    prediction = int(logits.argmax())
    return {"nll": float(-torch.log_softmax(logits, -1)[target]),
            "correct": int(prediction == target),
            "prediction": candidates[prediction]}


def query_metrics(state, query, candidates, embeddings, unknown_raw=.75):
    return query_metrics_from_memory(
        matrix(state, embeddings.dim), query, candidates, embeddings, unknown_raw)


def outcome(state, retention, assignment, delta, truth, candidates, embeddings,
            unknown_raw=.75):
    candidate = apply_retention(state, retention, assignment, delta)
    candidate_memory = matrix(candidate, embeddings.dim)
    metrics = [query_metrics_from_memory(
        candidate_memory, row, candidates, embeddings, unknown_raw) for row in truth]
    removed = 0.0
    size = len(state)
    for index, row in enumerate(state):
        cohort = 2 if index == size - 1 else 1 if index == size - 2 else 0
        removed += (1 - retention[cohort]) * row["weight"]
    return {"nll": statistics.fmean(row["nll"] for row in metrics),
            "accuracy": statistics.fmean(row["correct"] for row in metrics),
            "removed_mass": removed}


def counterfactual_label(state, assignment, delta, truth, candidates, embeddings):
    values = {retention: outcome(state, retention, assignment, delta, truth,
                                 candidates, embeddings) for retention in GRID}
    best = min(row["nll"] for row in values.values())
    near = [retention for retention in GRID if values[retention]["nll"] <= best + 1e-6]
    selected = min(near, key=lambda value: (
        values[value]["removed_mass"], GRID.index(value)))
    return selected, values, len(near) > 1


def cohort_matrices(state, dim):
    import torch
    zero = torch.zeros((dim, dim), dtype=torch.float32,
                       device=state[0]["delta"].device if state else None)
    values = [zero.clone(), zero.clone(), zero.clone()]
    size = len(state)
    for index, row in enumerate(state):
        cohort = 2 if index == size - 1 else 1 if index == size - 2 else 0
        values[cohort] += row["weight"] * row["delta"]
    return values


def features(state, assignment, delta, block_index, history=True):
    import torch
    cohorts = cohort_matrices(state, delta.shape[0])
    memory = sum(cohorts, torch.zeros_like(delta))
    memory_norm = memory.norm(); delta_norm = delta.norm()
    cohort_norms = [value.norm() for value in cohorts]
    reference = assignment["old_refs"][0] if assignment["old_refs"] else None
    position = next((index for index, row in enumerate(state)
                     if row["batch"] == reference), None)
    age = None if position is None else len(state) - position
    target_mass = (0. if position is None else
                   state[position]["weight"] * float(state[position]["delta"].norm()))
    old_facts = {key: value for row in state for key, value in row["facts"].items()}
    overlap = sum(key in old_facts for key in assignment["facts"]) / len(assignment["facts"])
    duplicate = sum(old_facts.get(key) == value for key, value in assignment["facts"].items()) \
        / len(assignment["facts"])
    reference_values = [float(reference is not None), float(age == 1),
                        float(age == 2), float(age is not None and age > 2),
                        target_mass / float(memory_norm.clamp_min(1e-8)),
                        min(assignment["prelude_lines"], 2) / 2.]
    if not history:
        reference_values = [0.] * len(reference_values)
    return [
        math.log1p(float(memory_norm)) / 5., math.log1p(float(delta_norm)) / 5.,
        float(torch.sum(memory * delta) / (memory_norm * delta_norm).clamp_min(1e-8)),
        *[float(value / memory_norm.clamp_min(1e-8)) for value in cohort_norms],
        *[float(torch.sum(value * delta) / (value.norm() * delta_norm).clamp_min(1e-8))
          for value in cohorts],
        *reference_values, block_index / 8., len(state) / 8., len(old_facts) / 20.,
        overlap, duplicate,
    ]


def build_examples(pairs, embeddings, history=True):
    rows, ties = [], 0
    for inp, target in pairs:
        state, candidates = [], episode_candidates(target)
        boundaries = set(inp["decision_boundaries"])
        for block in inp["blocks"]:
            assignment = parse_block(block["text"], state)
            delta = batch_delta(assignment, block["index"], embeddings)
            if block["index"] in boundaries:
                truth = boundary_truth(inp, target, block["index"])
                label, outcomes, tied = counterfactual_label(
                    state, assignment, delta, truth, candidates, embeddings)
                ties += tied
                rows.append({
                    "group_id": inp["group_id"], "episode_id": inp["id"],
                    "block_index": block["index"],
                    "features": features(state, assignment, delta, block["index"], history),
                    "retention": list(label), "label": GRID.index(label),
                    "outcomes": {str(value): outcomes[value] for value in GRID},
                })
                state = apply_retention(state, label, assignment, delta)
            else:
                state = apply_retention(state, (1., 1., 1.), assignment, delta)
    interior = sum(any(value == .5 for value in row["retention"]) for row in rows)
    return rows, {
        "examples": len(rows), "ties": ties,
        "interior_examples": interior, "interior_fraction": interior / len(rows),
        "label_counts": dict(Counter(str(tuple(row["retention"])) for row in rows)),
    }


def train_controllers(rows, blind_rows, seed, epochs=300):
    import torch
    x = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    xb = torch.tensor([row["features"] for row in blind_rows], dtype=torch.float32)
    y = torch.tensor([row["retention"] for row in rows], dtype=torch.float32)
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.long)

    def regressor(values, model_seed):
        mean, std = values.mean(0), values.std(0).clamp_min(1e-5)
        normalized = (values - mean) / std
        torch.manual_seed(model_seed)
        model = torch.nn.Sequential(torch.nn.Linear(values.shape[1], 64), torch.nn.Tanh(),
                                    torch.nn.Linear(64, 32), torch.nn.Tanh(),
                                    torch.nn.Linear(32, 3), torch.nn.Sigmoid())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=1e-4)
        curve = []
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad(set_to_none=True); prediction = model(normalized)
            loss = torch.nn.functional.mse_loss(prediction, y)
            loss.backward(); optimizer.step()
            if epoch in (1, 10, 50, 100, 250, epochs):
                curve.append({"epoch": epoch, "loss": float(loss.detach()),
                              "mae": float((prediction.detach() - y).abs().mean())})
        return {"kind": "regressor", "model": model.eval(),
                "mean": mean, "std": std}, curve

    learned, curve = regressor(x, seed)
    blind, blind_curve = regressor(xb, seed + 1)
    mean, std = x.mean(0), x.std(0).clamp_min(1e-5); normalized = (x - mean) / std
    torch.manual_seed(seed + 2)
    classifier = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.Tanh(),
                                     torch.nn.Linear(64, 32), torch.nn.Tanh(),
                                     torch.nn.Linear(32, len(GRID)))
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=.01, weight_decay=1e-4)
    classifier_curve = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True); logits = classifier(normalized)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward(); optimizer.step()
        if epoch in (1, 10, 50, 100, 250, epochs):
            classifier_curve.append({"epoch": epoch, "loss": float(loss.detach()),
                                     "accuracy": float((logits.detach().argmax(1)
                                                        == labels).float().mean())})
    return {
        "learned": learned, "reference_blind": blind,
        "grid_classifier": {"kind": "classifier", "model": classifier.eval(),
                            "mean": mean, "std": std},
    }, {"learned": curve, "reference_blind": blind_curve,
        "grid_classifier": classifier_curve}


def controller_retention(controller, values):
    import torch
    x = (torch.tensor([values], dtype=torch.float32) - controller["mean"]) \
        / controller["std"]
    with torch.no_grad():
        output = controller["model"](x)
    if controller["kind"] == "classifier":
        return GRID[int(output.argmax(1))]
    return tuple(float(value) for value in output[0])


def physics_retention(state, assignment):
    if assignment["prelude_lines"] and not assignment["old_refs"]:
        return (0., 0., 0.)
    if not assignment["old_refs"]:
        return (1., 1., 1.)
    target = assignment["old_refs"][0]
    position = next((index for index, row in enumerate(state)
                     if row["batch"] == target), None)
    age = None if position is None else len(state) - position
    return ((1., 1., 0.) if age == 1 else (1., 0., 1.) if age == 2
            else (0., 1., 1.) if age is not None else (1., 1., 1.))


def select(policy, state, assignment, delta, block_index, truth, candidates,
           embeddings, controllers, episode_id):
    if policy in FIXED:
        return FIXED[policy], None
    if policy == "deterministic_random":
        digest = hashlib.sha256(
            f"{embeddings.seed}|{episode_id}|{block_index}".encode()).hexdigest()
        return GRID[int(digest[:12], 16) % len(GRID)], None
    if policy == "physics_cohort":
        return physics_retention(state, assignment), None
    if policy == "oracle_grid":
        return counterfactual_label(
            state, assignment, delta, truth, candidates, embeddings)[0], None
    if policy == "exact_reference":
        return None, apply_exact_reference(state, assignment, delta)
    history = policy != "reference_blind"
    values = features(state, assignment, delta, block_index, history)
    return controller_retention(controllers[policy], values), None


def evaluate_policy(pairs, policy, embeddings, controllers):
    records = []
    for inp, target in pairs:
        state, queries, decisions = [], [], []
        candidates, boundaries = episode_candidates(target), set(inp["decision_boundaries"])
        for block in inp["blocks"]:
            assignment = parse_block(block["text"], state)
            delta = batch_delta(assignment, block["index"], embeddings)
            if block["index"] in boundaries:
                truth = boundary_truth(inp, target, block["index"])
                oracle, _, _ = counterfactual_label(
                    state, assignment, delta, truth, candidates, embeddings)
                retention, direct_state = select(
                    policy, state, assignment, delta, block["index"], truth,
                    candidates, embeddings, controllers, inp["id"])
                state = (direct_state if direct_state is not None else
                         apply_retention(state, retention, assignment, delta))
                decisions.append({
                    "block": block["index"],
                    "retention": None if retention is None else list(retention),
                    "oracle_retention": list(oracle),
                    "retention_mae": (None if retention is None else
                                      statistics.fmean(abs(a - b)
                                                       for a, b in zip(retention, oracle))),
                    "grid_action_match": (None if retention is None else
                        int(tuple(min((0., .5, 1.), key=lambda point: abs(point - value))
                                  for value in retention) == oracle)),
                    "retained_mass": sum(row["weight"] for row in state),
                })
                current_memory = matrix(state, embeddings.dim)
                for row in truth:
                    queries.append({**row, **query_metrics_from_memory(
                        current_memory, row, candidates, embeddings, unknown_raw=.75)})
            else:
                state = apply_retention(state, (1., 1., 1.), assignment, delta)
        records.append({"episode_id": inp["id"], "group_id": inp["group_id"],
                        "family": target["family"], "policy": policy,
                        "queries": queries, "decisions": decisions})
    return records


def aggregate(records):
    queries = [row for episode in records for row in episode["queries"]]
    decisions = [row for episode in records for row in episode["decisions"]]
    groups, families, roles = {}, {}, {}
    for group in sorted({row["group_id"] for row in records}):
        values = [q for episode in records if episode["group_id"] == group
                  for q in episode["queries"]]
        groups[group] = {"accuracy": statistics.fmean(q["correct"] for q in values),
                         "nll": statistics.fmean(q["nll"] for q in values)}
    for family in sorted({row["family"] for row in records}):
        values = [q for episode in records if episode["family"] == family
                  for q in episode["queries"]]
        families[family] = statistics.fmean(q["correct"] for q in values)
    for role in ("current", "historical", "anchor"):
        values = [row for row in queries if row["role"] == role]
        roles[role] = statistics.fmean(row["correct"] for row in values)
    retained = [row for row in decisions if row["retention_mae"] is not None]
    return {
        "accuracy": statistics.fmean(row["correct"] for row in queries),
        "nll": statistics.fmean(row["nll"] for row in queries),
        "roles": roles, "families": families, "groups": groups,
        "retention_mae": (statistics.fmean(row["retention_mae"] for row in retained)
                          if retained else None),
        "grid_action_accuracy": (statistics.fmean(row["grid_action_match"] for row in retained)
                                 if retained else None),
        "mean_retained_mass": statistics.fmean(row["retained_mass"] for row in decisions),
        "queries": len(queries), "decisions": len(decisions),
    }


def multi_seed_summary(per_seed):
    output = {}
    for policy in POLICIES:
        rows = [per_seed[str(seed)][policy] for seed in SEEDS]
        output[policy] = {
            "mean_accuracy": statistics.fmean(row["accuracy"] for row in rows),
            "mean_nll": statistics.fmean(row["nll"] for row in rows),
            "min_seed_accuracy": min(row["accuracy"] for row in rows),
            "mean_retention_mae": (None if rows[0]["retention_mae"] is None else
                                   statistics.fmean(row["retention_mae"] for row in rows)),
            "mean_grid_action_accuracy": (None if rows[0]["grid_action_accuracy"] is None else
                statistics.fmean(row["grid_action_accuracy"] for row in rows)),
            "roles": {role: statistics.fmean(row["roles"][role] for row in rows)
                      for role in rows[0]["roles"]},
            "families": {family: statistics.fmean(row["families"][family] for row in rows)
                         for family in rows[0]["families"]},
            "per_seed": {str(seed): per_seed[str(seed)][policy] for seed in SEEDS},
        }
    return output


def comparison(primary, baseline, seed):
    groups = sorted(next(iter(primary["per_seed"].values()))["groups"])
    accuracy, nll = [], []
    for group in groups:
        accuracy.append(statistics.fmean(
            primary["per_seed"][str(s)]["groups"][group]["accuracy"]
            - baseline["per_seed"][str(s)]["groups"][group]["accuracy"]
            for s in SEEDS))
        nll.append(statistics.fmean(
            baseline["per_seed"][str(s)]["groups"][group]["nll"]
            - primary["per_seed"][str(s)]["groups"][group]["nll"]
            for s in SEEDS))
    return {"accuracy_gain": statistics.fmean(accuracy),
            "accuracy_bootstrap_95ci": _bootstrap(accuracy, seed),
            "nll_gain": statistics.fmean(nll),
            "nll_bootstrap_95ci": _bootstrap(nll, seed + 1)}


def development_gate(summary, training_meta):
    best_fixed = max(FIXED, key=lambda name: summary[name]["mean_accuracy"])
    learned = summary["learned"]
    gain = comparison(learned, summary[best_fixed], 2631)
    checks = {
        "accuracy_gain_at_least_point_05": gain["accuracy_gain"] >= .05,
        "accuracy_ci_lower_above_point_02": gain["accuracy_bootstrap_95ci"][0] > .02,
        "nll_ci_lower_above_zero": gain["nll_bootstrap_95ci"][0] > 0.,
        "every_seed_beats_fixed": all(
            learned["per_seed"][str(seed)]["accuracy"]
            > summary[best_fixed]["per_seed"][str(seed)]["accuracy"]
            for seed in SEEDS),
        "all_roles_at_least_point_85": min(learned["roles"].values()) >= .85,
        "all_families_at_least_point_80": min(learned["families"].values()) >= .80,
        "retention_mae_at_most_point_15": learned["mean_retention_mae"] <= .15,
        "interior_label_fraction_at_least_point_10": min(
            value["interior_fraction"] for value in training_meta.values()) >= .10,
    }
    return {"checks": checks, "passed": all(checks.values()),
            "best_fixed": best_fixed, "comparison": gain}


def serialize_controller(controller):
    return {"kind": controller["kind"], "mean": controller["mean"],
            "std": controller["std"], "state": controller["model"].state_dict()}


def move_embeddings(embeddings, device):
    return QwenEmbeddings(embeddings.seed, embeddings.dim,
                          {key: value.to(device) for key, value in embeddings.values.items()})


def extract_embeddings(pairs, transforms, model_path, tokenizer_path, batch_size):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model
    model, model_meta = load_model(model_path, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    raw, names, extraction = extract_contextual(model, tokenizer, pairs, batch_size)
    embeddings, hashes = apply_contextual_transform(
        raw, names, transforms, "context_last", .01)
    del model, raw
    torch.cuda.empty_cache()
    return embeddings, hashes, extraction, model_meta


def run(args):
    import torch
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    data_manifest, config = verify_inputs(args.data_root, args.geometry_root)
    transform_artifact = torch.load(
        Path(args.geometry_root) / "contextual_transforms.pt",
        map_location="cpu", weights_only=False)
    transforms = transform_artifact["transforms"]
    train_pairs, dev_pairs = load_split(args.data_root, "train"), load_split(
        args.data_root, "dev")
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    embeddings, embedding_hashes, extraction, model_meta = extract_embeddings(
        train_pairs + dev_pairs, transforms, args.model, args.tokenizer, args.batch_size)
    extraction_seconds = time.perf_counter() - started

    controllers_saved, training_meta, dev_seed, raw_hashes = {}, {}, {}, {}
    for offset, seed in enumerate(SEEDS):
        memory = move_embeddings(embeddings[str(seed)], "cuda")
        train_rows, meta = build_examples(train_pairs, memory, True)
        blind_rows, blind_meta = build_examples(train_pairs, memory, False)
        controllers, curves = train_controllers(
            train_rows, blind_rows, CONTROLLER_SEEDS[offset], args.epochs)
        controllers_saved[str(seed)] = {name: serialize_controller(value)
                                        for name, value in controllers.items()}
        training_meta[str(seed)] = {**meta, "blind": blind_meta, "curves": curves}
        dev_seed[str(seed)] = {}
        for policy in POLICIES:
            records = evaluate_policy(dev_pairs, policy, memory, controllers)
            dev_seed[str(seed)][policy] = aggregate(records)
            path = root / f"dev_seed{seed}_{policy}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            raw_hashes[path.name] = sha(path)
        del memory
        torch.cuda.empty_cache()
    dev = multi_seed_summary(dev_seed)
    gate = development_gate(dev, training_meta)

    test_scored, test, test_embedding_meta = False, None, None
    if gate["passed"]:
        test_pairs = load_split(args.data_root, "test")
        test_embeddings, test_hashes, test_extraction, _ = extract_embeddings(
            test_pairs, transforms, args.model, args.tokenizer, args.batch_size)
        test_seed = {}
        for seed in SEEDS:
            memory = move_embeddings(test_embeddings[str(seed)], "cuda")
            controllers = restore_controllers(controllers_saved[str(seed)], len(FEATURE_NAMES))
            test_seed[str(seed)] = {}
            for policy in POLICIES:
                records = evaluate_policy(test_pairs, policy, memory, controllers)
                test_seed[str(seed)][policy] = aggregate(records)
                path = root / f"test_seed{seed}_{policy}.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for row in records))
                raw_hashes[path.name] = sha(path)
        test = multi_seed_summary(test_seed); test_scored = True
        test_embedding_meta = {"hashes": test_hashes, "extraction": test_extraction}

    checkpoint = root / "controllers.pt"
    torch.save({"protocol": PROTOCOL, "controllers": controllers_saved,
                "features": FEATURE_NAMES, "grid": GRID, "projection_seeds": SEEDS,
                "controller_seeds": CONTROLLER_SEEDS, "epochs": args.epochs}, checkpoint)
    training = {
        "seconds": time.perf_counter() - started,
        "train_dev_extraction_seconds": extraction_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "embedding_hashes": embedding_hashes, "extraction": extraction,
        "test_embedding_meta": test_embedding_meta, "seeds": training_meta,
        "checkpoint_sha256": sha(checkpoint), "raw_rows_sha256": raw_hashes,
    }
    manifest = {
        "protocol": PROTOCOL, "data_manifest_sha256": sha(
            Path(args.data_root) / "manifest.json"),
        "geometry_hashes": GEOMETRY_HASHES, "contextual_config": config,
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"], "grid": GRID,
        "fixed": FIXED, "policies": POLICIES, "features": FEATURE_NAMES,
        "projection_seeds": SEEDS, "controller_seeds": CONTROLLER_SEEDS,
        "epochs": args.epochs, "test_loaded": test_scored, "test_scored": test_scored,
    }
    result = {
        "protocol": PROTOCOL, "dev": dev, "dev_gate": gate,
        "test": test, "test_scored": test_scored,
        "stage_passed": gate["passed"],
        "next_stage": ("report_one_time_test" if gate["passed"]
                       else "stop_continuous_cohort_branch"),
        "terminal_status": "completed_qwen_continuous_cohort_forgetting",
    }
    write_json(root / "training.json", training)
    write_json(root / "manifest.json", manifest)
    write_json(root / "summary.json", result)
    return result


def restore_controllers(saved, feature_dim):
    import torch
    output = {}
    for name, row in saved.items():
        width = len(GRID) if row["kind"] == "classifier" else 3
        final = torch.nn.Linear(32, width)
        model = torch.nn.Sequential(torch.nn.Linear(feature_dim, 64), torch.nn.Tanh(),
                                    torch.nn.Linear(64, 32), torch.nn.Tanh(), final)
        if row["kind"] == "regressor":
            model.add_module("sigmoid", torch.nn.Sigmoid())
        model.load_state_dict(row["state"])
        output[name] = {"kind": row["kind"], "mean": row["mean"],
                        "std": row["std"], "model": model.eval()}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--geometry-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    result = run(parser.parse_args())
    print(json.dumps({"status": result["terminal_status"],
                      "gate": result["dev_gate"], "test_scored": result["test_scored"],
                      "next_stage": result["next_stage"]}, indent=2))


if __name__ == "__main__":
    main()
