"""Nested grouped OOF Barrier-Q controller for natural session forgetting."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import time

from tasks.build_qwen_natural_causal_memory_v1 import load
from tasks.build_qwen_natural_forgetting_utility_pilot_v1 import (
    BOOTSTRAP_DRAWS,
    EXPECTED,
    SCOPE_SEED,
    UNIQUE_ACTIONS,
    scope_examples,
    state_before,
)
from tasks.evaluate_qwen_natural_scope_v5_development import load_v5_gate
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_scope_transfer_v2 import build_scorer


PROTOCOL = "qwen_natural_forgetting_barrier_q_v1"
DATA_PROTOCOL = "qwen_natural_forgetting_utility_formal_v1"
PLAN = "QWEN_NATURAL_FORGETTING_BARRIER_Q_V1_PLAN.md"
ACTIONS = tuple(UNIQUE_ACTIONS)
ACTION_INDEX = {name: index for index, name in enumerate(ACTIONS)}
SEEDS = (9401, 9402, 9403)
HIDDENS = (0, 32, 64)
LEARNING_RATES = (.001, .005)
RISK_WEIGHTS = (1., 3.)
EPOCHS = 100
BARRIERS = (.05, .10, .20, .30)
MIN_GAINS = (0., .005, .01, .02)
TIE_TOLERANCES = (0., .005, .01)
CLASS_CONFIDENCES = (0., .5, .7)
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAP_SEED = 202610231


def fold_for(session_id):
    value = hashlib.sha256(f"barrier-q-v1|{session_id}".encode()).hexdigest()
    return int(value, 16) % OUTER_FOLDS


def inner_fold_for(session_id, outer_fold):
    value = hashlib.sha256(
        f"barrier-q-v1-inner|{outer_fold}|{session_id}".encode()
    ).hexdigest()
    return int(value, 16) % INNER_FOLDS


def lexical_features(left, right):
    def tokens(value):
        return set(re.findall(r"[\w]+", value.casefold()))
    a, b = tokens(left), tokens(right)
    union = a | b; intersection = a & b
    jaccard = len(intersection) / max(1, len(union))
    containment = len(intersection) / max(1, min(len(a), len(b)))
    ratio = min(len(left), len(right)) / max(1, max(len(left), len(right)))
    return (jaccard, containment, ratio)


def immediate_mass(label, action):
    before = label["memory_size_before"]
    if action == "retain_all":
        return float(before + 1)
    meta = label["actions"][action]["intervention"]
    count = len(meta["indices"])
    return float(before + 1 - count * (1 - meta["alpha"]))


def load_labels(root):
    root = Path(root); summary = json.loads((root / "summary.json").read_text())
    if summary["protocol"] != DATA_PROTOCOL or not summary["gate"]["controller_training_authorized"]:
        raise ValueError("formal utility resource is not released for controller training")
    rows = []
    for path in sorted(root.glob("labels_shard*.jsonl")):
        if summary["label_files_sha256"].get(path.name) != sha(path):
            raise ValueError("formal label shard hash mismatch")
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    rows.sort(key=lambda row: (row["session_id"], row["event_step"]))
    if len(rows) != 774 or len({(row["session_id"], row["event_step"]) for row in rows}) != 774:
        raise ValueError("formal controller labels are incomplete")
    return summary, rows


def target_tables(labels):
    import torch

    gain, nll, risk, mass = [], [], [], []
    for row in labels:
        retain = row["actions"]["retain_all"]
        gains, nlls, risks, masses = [], [], [], []
        for action in ACTIONS:
            value = row["actions"][action]
            gains.append(value["candidate_accuracy"] - retain["candidate_accuracy"])
            nlls.append(retain["mean_answer_nll"] - value["mean_answer_nll"])
            risks.append(float(value["unrelated_correct"] < retain["unrelated_correct"]))
            masses.append(immediate_mass(row, action))
        gain.append(gains); nll.append(nlls); risk.append(risks); mass.append(masses)
    return (torch.tensor(gain, dtype=torch.float32),
            torch.tensor(nll, dtype=torch.float32),
            torch.tensor(risk, dtype=torch.float32),
            torch.tensor(mass, dtype=torch.float32))


def extract_features(args, sessions, labels):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    input_paths = {
        "data_manifest": Path(args.data_root) / "manifest.json",
        "v5_checkpoint": Path(args.v5_root) / "result" / "constrained_scope_gate.pt",
        "transforms": Path(args.context_root) / "transforms.pt",
        "scope": Path(args.scope_root) / "scope_gates.pt",
    }
    observed = {key: sha(path) for key, path in input_paths.items()}
    if observed != EXPECTED:
        raise ValueError("Barrier-Q frozen input mismatch")
    torch.manual_seed(BOOTSTRAP_SEED); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter(); model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    source_summary = json.loads((Path(args.context_root) / "summary.json").read_text())
    transforms = torch.load(input_paths["transforms"], map_location="cpu", weights_only=False)
    scope_artifact = torch.load(input_paths["scope"], map_location="cpu", weights_only=False)
    scorer, feature_meta = build_scorer(
        scope_examples(sessions), model, tokenizer, transforms["transforms"],
        source_summary["selected_config"]["regularizer"], scope_artifact,
        args.batch_size,
    )
    gate, threshold, reader_selected = load_v5_gate(
        input_paths["v5_checkpoint"], scope_artifact["heads"]["7301"]
    )
    if threshold != .30:
        raise ValueError("Barrier-Q requires frozen V5 threshold 0.30")
    v5_artifact = torch.load(input_paths["v5_checkpoint"], map_location="cpu", weights_only=False)
    base_dim = v5_artifact["feature_dim"]
    sessions_by_id = {row["id"]: row for row in sessions}; features = []
    names = [f"reader_{index}" for index in range(base_dim)] + [
        "has_memory", "memory_size", "event_position", "top_recency",
        "top_cosine", "cosine_margin", "cosine_mean", "cosine_std",
        "cosine_entropy", "v5_probability", "lexical_jaccard",
        "lexical_containment", "length_ratio",
    ]
    with torch.no_grad():
        for row in labels:
            session = sessions_by_id[row["session_id"]]
            memory, _ = state_before(session, row["event_step"])
            query = row["incoming_prompt"]
            if not memory:
                base = torch.zeros(base_dim); engineered = [0.] * 13
                engineered[2] = row["event_step"] / max(1, len(session["events"]) - 1)
            else:
                result, base = scorer.encode(
                    query, [record["prompt"] for record in memory], SCOPE_SEED
                )
                probability = float(gate(base.unsqueeze(0)).sigmoid()[0])
                scores = result["mean_scores"].float(); top = result["top"]
                if len(scores) > 1:
                    top_two = torch.topk(scores, 2).values; margin = float(top_two[0] - top_two[1])
                else:
                    margin = float(scores[0] + 1.)
                distribution = torch.softmax(scores / .07, dim=0)
                entropy = (0. if len(scores) == 1 else float(
                    -(distribution * distribution.clamp_min(1e-12).log()).sum()
                    / math.log(len(scores))))
                lexical = lexical_features(query, memory[top]["prompt"])
                engineered = [
                    1., len(memory) / 12.,
                    row["event_step"] / max(1, len(session["events"]) - 1),
                    (len(memory) - 1 - top) / max(1, len(memory) - 1),
                    float(scores[top]), margin, float(scores.mean()),
                    float(scores.std(unbiased=False)), entropy, probability, *lexical,
                ]
            features.append(torch.cat((base.float().cpu(), torch.tensor(engineered))))
    resource = {
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "frozen_inputs": observed, "feature_meta": feature_meta,
        "reader_selected": reader_selected, "reader_threshold": threshold,
        "model": args.model, "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"], "gpu": torch.cuda.get_device_name(),
    }
    del scorer, gate, model; torch.cuda.empty_cache()
    return torch.stack(features), names, resource


class BarrierQModel:
    """Thin wrapper that keeps torch imports optional for score-only utilities."""

    @staticmethod
    def create(input_dim, hidden, seed):
        import torch
        torch.manual_seed(seed)
        output = 3 * len(ACTIONS)
        if hidden == 0:
            return torch.nn.Linear(input_dim, output)
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, output),
        )


class ClassifierModel:
    @staticmethod
    def create(input_dim, hidden, seed):
        import torch
        torch.manual_seed(seed + 100_000)
        if hidden == 0:
            return torch.nn.Linear(input_dim, len(ACTIONS))
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, len(ACTIONS)),
        )


def normalize(features, indices):
    mean = features[indices].mean(0); std = features[indices].std(0, unbiased=False).clamp_min(1e-6)
    return mean, std


def fit_q(features, gain, nll, risk, indices, config, seed):
    import torch

    mean, std = normalize(features, indices); x = (features[indices] - mean) / std
    model = BarrierQModel.create(features.shape[1], config["hidden"], seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.)
    positives = risk[indices].sum(0); negatives = len(indices) - positives
    pos_weight = torch.where(positives > 0, negatives / positives.clamp_min(1),
                             torch.ones_like(positives)).clamp(1, 20)
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True); raw = model(x)
        pred_gain, pred_nll, pred_risk = raw.chunk(3, dim=-1)
        loss = (torch.nn.functional.smooth_l1_loss(pred_gain, gain[indices])
                + torch.nn.functional.smooth_l1_loss(pred_nll, nll[indices])
                + config["risk_weight"] * torch.nn.functional.binary_cross_entropy_with_logits(
                    pred_risk, risk[indices], pos_weight=pos_weight))
        loss.backward(); optimizer.step()
    return model, mean, std


def predict_q(model, mean, std, features):
    import torch
    with torch.no_grad():
        gain, nll, risk = model((features - mean) / std).chunk(3, dim=-1)
    return gain, nll, risk.sigmoid()


def fit_classifier(features, labels, indices, hidden, learning_rate, seed):
    import torch

    mean, std = normalize(features, indices); x = (features[indices] - mean) / std
    model = ClassifierModel.create(features.shape[1], hidden, seed)
    counts = torch.bincount(labels[indices], minlength=len(ACTIONS)).float()
    weights = torch.where(counts > 0, len(indices) / (len(ACTIONS) * counts),
                          torch.zeros_like(counts))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.)
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True); logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, labels[indices], weight=weights)
        loss.backward(); optimizer.step()
    return model, mean, std


def predict_classifier(model, mean, std, features):
    import torch
    with torch.no_grad(): return model((features - mean) / std).softmax(-1)


def select_q_actions(gain, nll, risk, mass, barrier, min_gain, tie, use_barrier=True):
    output = []
    for row in range(len(gain)):
        eligible = [0]
        for action in range(1, len(ACTIONS)):
            if gain[row, action] < min_gain:
                continue
            if use_barrier and risk[row, action] > barrier:
                continue
            eligible.append(action)
        best_gain = max(float(gain[row, action]) for action in eligible)
        tied = [action for action in eligible
                if float(gain[row, action]) >= best_gain - tie]
        output.append(max(tied, key=lambda action: (
            float(nll[row, action]), -float(mass[row, action]), -action
        )))
    return output


def select_classifier_actions(probability, confidence):
    output = []
    for row in probability:
        value, action = row.max(0)
        output.append(int(action) if float(value) >= confidence else 0)
    return output


def policy_metrics(labels, actions, indices=None):
    if indices is None: indices = range(len(labels))
    correct = nll = queries = unrelated = 0
    distribution = Counter()
    for index, action_index in zip(indices, actions):
        action = ACTIONS[action_index]; value = labels[index]["actions"][action]
        retain = labels[index]["actions"]["retain_all"]
        correct += value["correct"]; nll += value["total_answer_nll"]
        queries += value["queries"]; distribution[action] += 1
        unrelated += max(0, retain["unrelated_correct"] - value["unrelated_correct"])
    return {"candidate_accuracy": correct / queries, "mean_answer_nll": nll / queries,
            "unrelated_errors_added": unrelated,
            "action_distribution": dict(sorted(distribution.items())),
            "queries": queries, "correct": correct}


def source_actions(labels, indices):
    return [ACTION_INDEX["scope_top1_a0"] if labels[index]["desired_source_action"] == "local"
            else 0 for index in indices]


def inner_search(features, gain, nll, risk, mass, class_labels, labels,
                 outer_train, outer_fold):
    import torch

    predictions = {}; searches = []
    configs = [
        {"hidden": hidden, "learning_rate": lr, "risk_weight": weight}
        for hidden in HIDDENS for lr in LEARNING_RATES for weight in RISK_WEIGHTS
    ]
    for config_index, config in enumerate(configs):
        by_seed = {}
        for seed in SEEDS:
            held_gain = torch.empty((len(outer_train), len(ACTIONS)))
            held_nll = torch.empty_like(held_gain); held_risk = torch.empty_like(held_gain)
            position = {index: offset for offset, index in enumerate(outer_train)}
            for inner in range(INNER_FOLDS):
                validation = [index for index in outer_train
                              if inner_fold_for(labels[index]["session_id"], outer_fold) == inner]
                training = [index for index in outer_train if index not in set(validation)]
                model, mean, std = fit_q(features, gain, nll, risk, training, config, seed)
                pg, pn, pr = predict_q(model, mean, std, features[validation])
                for local, index in enumerate(validation):
                    held_gain[position[index]] = pg[local]; held_nll[position[index]] = pn[local]
                    held_risk[position[index]] = pr[local]
            by_seed[seed] = (held_gain, held_nll, held_risk)
        predictions[config_index] = by_seed
        for barrier in BARRIERS:
            for min_gain in MIN_GAINS:
                for tie in TIE_TOLERANCES:
                    metrics = []
                    for seed in SEEDS:
                        pg, pn, pr = by_seed[seed]
                        actions = select_q_actions(
                            pg, pn, pr, mass[outer_train], barrier, min_gain, tie, True)
                        metrics.append(policy_metrics(labels, actions, outer_train))
                    searches.append({
                        "config_index": config_index, "config": config,
                        "barrier": barrier, "min_gain": min_gain, "tie": tie,
                        "mean_candidate_accuracy": statistics.fmean(
                            row["candidate_accuracy"] for row in metrics),
                        "mean_unrelated_errors": statistics.fmean(
                            row["unrelated_errors_added"] for row in metrics),
                    })
    retain = policy_metrics(labels, [0] * len(outer_train), outer_train)
    source = policy_metrics(labels, source_actions(labels, outer_train), outer_train)
    for row in searches:
        row["feasible"] = (row["mean_unrelated_errors"] <= source["unrelated_errors_added"]
                           and row["mean_candidate_accuracy"]
                           >= retain["candidate_accuracy"] + .005)
    feasible = [row for row in searches if row["feasible"]]
    if feasible:
        selected = max(feasible, key=lambda row: (
            row["mean_candidate_accuracy"], -row["mean_unrelated_errors"],
            -row["config"]["hidden"],
            -abs(row["barrier"] - .10), -abs(row["min_gain"] - .01),
            -abs(row["tie"] - .005),
        )); fold_feasible = True
    else:
        selected = min(searches, key=lambda row: (
            row["mean_unrelated_errors"], -row["mean_candidate_accuracy"],
            row["config"]["hidden"], abs(row["barrier"] - .10),
            abs(row["min_gain"] - .01), abs(row["tie"] - .005),
        )); fold_feasible = False

    config = selected["config"]; classifier_predictions = {}
    for seed in SEEDS:
        held = torch.empty((len(outer_train), len(ACTIONS)))
        position = {index: offset for offset, index in enumerate(outer_train)}
        for inner in range(INNER_FOLDS):
            validation = [index for index in outer_train
                          if inner_fold_for(labels[index]["session_id"], outer_fold) == inner]
            training = [index for index in outer_train if index not in set(validation)]
            model, mean, std = fit_classifier(
                features, class_labels, training, config["hidden"],
                config["learning_rate"], seed)
            probability = predict_classifier(model, mean, std, features[validation])
            for local, index in enumerate(validation): held[position[index]] = probability[local]
        classifier_predictions[seed] = held
    confidence_grid = []
    for confidence in CLASS_CONFIDENCES:
        metrics = [policy_metrics(
            labels, select_classifier_actions(classifier_predictions[seed], confidence), outer_train
        ) for seed in SEEDS]
        confidence_grid.append({
            "confidence": confidence,
            "mean_candidate_accuracy": statistics.fmean(row["candidate_accuracy"] for row in metrics),
            "mean_unrelated_errors": statistics.fmean(row["unrelated_errors_added"] for row in metrics),
        })
    selected_confidence = max(confidence_grid, key=lambda row: (
        row["mean_candidate_accuracy"], -row["mean_unrelated_errors"], row["confidence"]))
    return selected, fold_feasible, searches, selected_confidence, retain, source


def paired_bootstrap(labels, per_seed_left, right, seed):
    groups = sorted({row["session_id"] for row in labels}); values = {}
    for group in groups:
        indices = [index for index, row in enumerate(labels) if row["session_id"] == group]
        seed_values = []
        for actions in per_seed_left.values():
            left = policy_metrics(labels, [actions[index] for index in indices], indices)
            baseline = policy_metrics(labels, [right[index] for index in indices], indices)
            seed_values.append(left["candidate_accuracy"] - baseline["candidate_accuracy"])
        values[group] = statistics.fmean(seed_values)
    rng = random.Random(seed); draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        draws.append(statistics.fmean(values[group] for group in sample))
    draws.sort()
    def q(value): return draws[min(len(draws)-1, max(0, math.ceil(value*len(draws))-1))]
    return {"gain": statistics.fmean(values.values()), "bootstrap_95ci": [q(.025), q(.975)],
            "sessions": len(groups)}


def run(args):
    import torch

    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    data_summary, labels = load_labels(args.utility_root)
    _, sessions = load(args.data_root, "calibration")
    features, feature_names, feature_resource = extract_features(args, sessions, labels)
    gain, nll, risk, mass = target_tables(labels)
    class_labels = torch.tensor([ACTION_INDEX[row["safe_oracle_action"]] for row in labels])
    torch.save({"features": features, "gain": gain, "nll": nll, "risk": risk,
                "mass": mass, "feature_names": feature_names}, root / "features.pt")
    feature_hash = sha(root / "features.pt")

    primary = {seed: [None] * len(labels) for seed in SEEDS}
    gain_only = {seed: [None] * len(labels) for seed in SEEDS}
    classifier = {seed: [None] * len(labels) for seed in SEEDS}
    folds, all_searches = {}, {}
    started = time.perf_counter()
    for outer in range(OUTER_FOLDS):
        validation = [index for index, row in enumerate(labels)
                      if fold_for(row["session_id"]) == outer]
        training = [index for index in range(len(labels)) if index not in set(validation)]
        selected, feasible, search, selected_confidence, retain, source = inner_search(
            features, gain, nll, risk, mass, class_labels, labels, training, outer)
        config = selected["config"]
        for seed in SEEDS:
            model, mean, std = fit_q(features, gain, nll, risk, training, config, seed)
            pg, pn, pr = predict_q(model, mean, std, features[validation])
            chosen = select_q_actions(
                pg, pn, pr, mass[validation], selected["barrier"],
                selected["min_gain"], selected["tie"], True)
            no_barrier = select_q_actions(
                pg, pn, pr, mass[validation], selected["barrier"],
                selected["min_gain"], selected["tie"], False)
            cmodel, cmean, cstd = fit_classifier(
                features, class_labels, training, config["hidden"],
                config["learning_rate"], seed)
            cp = predict_classifier(cmodel, cmean, cstd, features[validation])
            direct = select_classifier_actions(cp, selected_confidence["confidence"])
            for local, index in enumerate(validation):
                primary[seed][index] = chosen[local]; gain_only[seed][index] = no_barrier[local]
                classifier[seed][index] = direct[local]
        folds[str(outer)] = {
            "train_sessions": sorted({labels[index]["session_id"] for index in training}),
            "validation_sessions": sorted({labels[index]["session_id"] for index in validation}),
            "selected": selected, "feasible": feasible,
            "classifier_confidence": selected_confidence,
            "inner_retain": retain, "inner_source_rule": source,
        }
        all_searches[str(outer)] = search

    if any(any(action is None for action in values) for table in (primary, gain_only, classifier)
           for values in table.values()):
        raise ValueError("OOF controller predictions are incomplete")
    fixed_actions = {action: [ACTION_INDEX[action]] * len(labels) for action in ACTIONS}
    fixed_metrics = {action: policy_metrics(labels, values)
                     for action, values in fixed_actions.items()}
    strongest_fixed = max(ACTIONS, key=lambda action: fixed_metrics[action]["candidate_accuracy"])
    retain_actions = fixed_actions["retain_all"]
    source = source_actions(labels, range(len(labels)))
    safe_oracle = [ACTION_INDEX[row["safe_oracle_action"]] for row in labels]
    unconstrained = [ACTION_INDEX[row["unconstrained_oracle_action"]] for row in labels]
    frozen_scope = [ACTION_INDEX["scope_top1_a0"]
                    if features[index, feature_names.index("v5_probability")] >= .30 else 0
                    for index in range(len(labels))]
    policies = {
        "barrier_q": primary, "gain_q": gain_only, "direct_classification": classifier,
    }
    metrics = {name: {str(seed): policy_metrics(labels, actions)
                      for seed, actions in table.items()} for name, table in policies.items()}
    baselines = {**fixed_metrics,
                 "frozen_scope_delete_if_retrieve": policy_metrics(labels, frozen_scope),
                 "source_action_rule": policy_metrics(labels, source),
                 "safe_oracle": policy_metrics(labels, safe_oracle),
                 "unconstrained_oracle": policy_metrics(labels, unconstrained)}
    nonretain_rate = statistics.fmean(action != 0 for actions in primary.values() for action in actions)
    random_metrics = {}
    for seed in SEEDS:
        rng = random.Random(seed + 700_000); actions = []
        for _ in labels:
            actions.append(ACTION_INDEX["scope_top1_a0"] if rng.random() < nonretain_rate else 0)
        random_metrics[str(seed)] = policy_metrics(labels, actions)
    metrics["rate_matched_random"] = random_metrics

    comparison_retain = paired_bootstrap(labels, primary, retain_actions, BOOTSTRAP_SEED)
    comparison_fixed = paired_bootstrap(
        labels, primary, fixed_actions[strongest_fixed], BOOTSTRAP_SEED + 1)
    mean_metric = lambda name, key: statistics.fmean(
        metrics[name][str(seed)][key] for seed in SEEDS)
    primary_accuracy = mean_metric("barrier_q", "candidate_accuracy")
    gainq_accuracy = mean_metric("gain_q", "candidate_accuracy")
    primary_errors = mean_metric("barrier_q", "unrelated_errors_added")
    gainq_errors = mean_metric("gain_q", "unrelated_errors_added")
    retain_accuracy = baselines["retain_all"]["candidate_accuracy"]
    oracle_accuracy = baselines["safe_oracle"]["candidate_accuracy"]
    headroom = (primary_accuracy - retain_accuracy) / (oracle_accuracy - retain_accuracy)
    checks = {
        "gain_over_retain_at_least_point_0075": primary_accuracy - retain_accuracy >= .0075,
        "retain_gain_ci_lower_above_zero": comparison_retain["bootstrap_95ci"][0] > 0,
        "gain_over_strongest_fixed_at_least_point_0025":
            primary_accuracy - baselines[strongest_fixed]["candidate_accuracy"] >= .0025,
        "fixed_gain_ci_lower_above_zero": comparison_fixed["bootstrap_95ci"][0] > 0,
        "captures_at_least_point_30_oracle_headroom": headroom >= .30,
        "unrelated_errors_no_more_than_source_rule":
            primary_errors <= baselines["source_action_rule"]["unrelated_errors_added"],
        "barrier_reduces_gain_q_errors": primary_errors < gainq_errors,
        "barrier_within_point_005_of_gain_q": primary_accuracy >= gainq_accuracy - .005,
        "at_least_four_feasible_outer_folds": sum(row["feasible"] for row in folds.values()) >= 4,
        "all_seeds_improve_retain": all(
            metrics["barrier_q"][str(seed)]["candidate_accuracy"] > retain_accuracy for seed in SEEDS),
        "all_seeds_beat_strongest_fixed": all(
            metrics["barrier_q"][str(seed)]["candidate_accuracy"]
            > baselines[strongest_fixed]["candidate_accuracy"] for seed in SEEDS),
    }
    selected_values = {key: [folds[str(outer)]["selected"][key]
                             for outer in range(OUTER_FOLDS)]
                       for key in ("barrier", "min_gain", "tie")}
    selected_configs = [folds[str(outer)]["selected"]["config"] for outer in range(OUTER_FOLDS)]
    final_config = {
        "hidden": int(statistics.median(row["hidden"] for row in selected_configs)),
        "learning_rate": statistics.median(row["learning_rate"] for row in selected_configs),
        "risk_weight": statistics.median(row["risk_weight"] for row in selected_configs),
        "barrier": statistics.median(selected_values["barrier"]),
        "min_gain": statistics.median(selected_values["min_gain"]),
        "tie": statistics.median(selected_values["tie"]),
    }
    checkpoints = {}
    for seed in SEEDS:
        model, mean, std = fit_q(features, gain, nll, risk, list(range(len(labels))),
                                 final_config, seed)
        checkpoints[str(seed)] = {"state": model.state_dict(), "mean": mean, "std": std}
    checkpoint_path = root / "barrier_q.pt"
    torch.save({"protocol": PROTOCOL, "actions": ACTIONS, "feature_names": feature_names,
                "config": final_config, "input_dim": features.shape[1],
                "seeds": SEEDS, "checkpoints": checkpoints}, checkpoint_path)
    with (root / "oof_predictions.jsonl").open("x") as sink:
        for index, row in enumerate(labels):
            sink.write(json.dumps({
                "session_id": row["session_id"], "event_step": row["event_step"],
                "outer_fold": fold_for(row["session_id"]),
                "safe_oracle_action": row["safe_oracle_action"],
                "predictions": {name: {str(seed): ACTIONS[table[seed][index]]
                                       for seed in SEEDS}
                                for name, table in policies.items()},
            }) + "\n")
    write_json(root / "search.json", all_searches)
    summary = {
        "protocol": PROTOCOL, "metrics": metrics, "baselines": baselines,
        "strongest_fixed_deployable": strongest_fixed,
        "comparisons": {"barrier_q_vs_retain": comparison_retain,
                        "barrier_q_vs_strongest_fixed": comparison_fixed},
        "mean": {"barrier_q_candidate_accuracy": primary_accuracy,
                 "gain_q_candidate_accuracy": gainq_accuracy,
                 "barrier_q_unrelated_errors": primary_errors,
                 "gain_q_unrelated_errors": gainq_errors,
                 "oracle_headroom_captured": headroom},
        "folds": folds, "final_config": final_config,
        "gate": {"checks": checks, "development_authorized": all(checks.values())},
        "resources": {"feature_extraction": feature_resource,
                      "training_seconds": time.perf_counter() - started},
        "hashes": {"features": feature_hash, "oof_predictions": sha(root / "oof_predictions.jsonl"),
                   "search": sha(root / "search.json"), "checkpoint": sha(checkpoint_path),
                   "utility_summary": sha(Path(args.utility_root) / "summary.json"),
                   "plan": sha(PLAN)},
        "model_grid": {"hiddens": HIDDENS, "learning_rates": LEARNING_RATES,
                       "risk_weights": RISK_WEIGHTS, "epochs": EPOCHS,
                       "barriers": BARRIERS, "min_gains": MIN_GAINS,
                       "tie_tolerances": TIE_TOLERANCES,
                       "classifier_confidences": CLASS_CONFIDENCES},
        "seeds": SEEDS, "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
        "calibration_scored": True, "development_loaded": False,
        "test_loaded": False, "external_benchmarks_loaded": False,
        "data_summary": data_summary,
    }
    write_json(root / "summary.json", summary); return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("utility_root", "data_root", "v5_root", "model", "tokenizer",
                 "context_root", "scope_root", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
