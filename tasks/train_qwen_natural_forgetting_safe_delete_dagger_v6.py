"""Collect one cross-fit DAgger rollout and retrain lifecycle safe-delete V6."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from tasks.build_qwen_natural_causal_memory_v1 import load
from tasks.build_qwen_natural_forgetting_utility_pilot_v1 import SCOPE_SEED, scope_examples
from tasks.evaluate_qwen_natural_causal_memory_v1 import chain_map, record_for
from tasks.evaluate_qwen_natural_forgetting_online_v3 import EXPECTED_V5, engineered_features
from tasks.evaluate_qwen_natural_scope_v5_development import load_v5_gate
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_barrier_q_v1 import OUTER_FOLDS, SEEDS, fold_for
from tasks.train_qwen_natural_forgetting_factor_gate_v2 import BinaryHead, ENGINEERED, NO_SCHEDULE
from tasks.train_qwen_natural_forgetting_safe_delete_v5 import (
    PROTOCOL as SOURCE_PROTOCOL,
    fit_head,
    inner_search,
    state_id,
)
from tasks.train_qwen_natural_scope_transfer_v2 import build_scorer


PROTOCOL = "qwen_natural_forgetting_safe_delete_dagger_v6"
PLAN = "QWEN_NATURAL_FORGETTING_SAFE_DELETE_DAGGER_V6_PLAN.md"


class FoldSafeDeleteController:
    def __init__(self, artifact, outer):
        self.outer = outer
        self.feature_names = tuple(artifact["feature_names"])
        self.fold = artifact["folds"][str(outer)]
        self.models = {}
        for seed in SEEDS:
            checkpoint = self.fold["checkpoints"][str(seed)]
            model = BinaryHead.create(
                len(self.feature_names), self.fold["config"]["hidden"], seed,
            )
            model.load_state_dict(checkpoint["state"])
            model.eval()
            self.models[seed] = {
                "model": model, "mean": checkpoint["mean"], "std": checkpoint["std"],
            }

    def probability(self, engineered):
        import torch

        vector = torch.tensor([
            engineered[ENGINEERED.index(name)] for name in self.feature_names
        ], dtype=torch.float32)
        values = []
        with torch.no_grad():
            for row in self.models.values():
                values.append(row["model"](((vector - row["mean"]) / row["std"])
                                            .unsqueeze(0)).squeeze().sigmoid())
        return float(torch.stack(values).mean())

    @property
    def threshold(self):
        return float(self.fold["threshold"])


def load_v5(v5_root):
    import torch

    root = Path(v5_root)
    summary_path = root / "result" / "summary.json"
    audit_path = root / "execution_audit.json"
    checkpoint_path = root / "result" / "safe_delete_crossfit.pt"
    summary = json.loads(summary_path.read_text())
    audit = json.loads(audit_path.read_text())
    if summary.get("protocol") != SOURCE_PROTOCOL or not audit.get("passed"):
        raise ValueError("V5 source is not an audited safe-delete result")
    if summary["gate"]["online_calibration_authorized"]:
        raise ValueError("DAgger branch requires the registered V5 offline stop")
    if (root / "terminal_status.txt").read_text().strip() != \
            "completed_qwen_natural_forgetting_safe_delete_v5_offline_stop":
        raise ValueError("unexpected V5 terminal status")
    if summary["hashes"]["checkpoint"] != sha(checkpoint_path):
        raise ValueError("V5 checkpoint hash mismatch")
    artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return summary, audit, artifact


def rollout(session, scorer, v5_gate, controller):
    memory = []
    chains = chain_map(session)
    rows = []
    for step, event in enumerate(session["events"]):
        chain = chains[event["chain_id"]]
        version = chain["versions"][event["version"]]
        features, top, reader_probability = engineered_features(
            version["prompt"], memory, step, len(session["events"]), scorer, v5_gate,
        )
        probability = controller.probability(features)
        delete = bool(memory) and probability >= controller.threshold
        records = [(row["chain_id"], int(row["version"])) for row in memory]
        safe = int(
            event["action"] == "local" and top is not None
            and memory[int(top)]["chain_id"] == event["chain_id"]
        )
        row = {
            "protocol": PROTOCOL,
            "state_id": state_id(session["id"], step, records),
            "session_id": session["id"], "outer_fold": fold_for(session["id"]),
            "event_step": step, "memory": [list(value) for value in records],
            "top": top, "incoming_chain_id": event["chain_id"],
            "incoming_version": int(event["version"]), "event_kind": event["kind"],
            "desired_source_action": event["action"],
            "features": [features[ENGINEERED.index(name)] for name in NO_SCHEDULE],
            "safe_delete": safe, "origin_arms": ["v5_crossfit_oof_policy"],
            "v5_probability": probability, "v5_threshold": controller.threshold,
            "v5_action": "scope_top1_a0" if delete else "retain_all",
            "reader_probability": reader_probability,
        }
        rows.append(row)
        if delete:
            memory.pop(int(top))
        memory.append({**record_for(chain, version), "strength": 1.0})
    return rows


def collect(args):
    import torch
    from transformers import AutoTokenizer
    from tasks.cbf_checkpoint_native import load_model

    root = Path(args.root)
    outputs = [root / f"dagger_rows_shard{args.shard}.jsonl",
               root / f"dagger_resource_shard{args.shard}.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("V6 DAgger shard exists")
    root.mkdir(parents=True, exist_ok=True)
    v5_summary, _, artifact = load_v5(args.safe_delete_root)
    controllers = {fold: FoldSafeDeleteController(artifact, fold)
                   for fold in range(OUTER_FOLDS)}
    _, sessions = load(args.data_root, "calibration")
    selected = [session for index, session in enumerate(sessions)
                if index % args.num_shards == args.shard]
    torch.manual_seed(202610271)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model, model_meta = load_model(args.model, "step81381_native")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    context, scope = Path(args.context_root), Path(args.scope_root)
    source_summary = json.loads((context / "summary.json").read_text())
    transforms = torch.load(context / "transforms.pt", map_location="cpu",
                            weights_only=False)
    scope_artifact = torch.load(scope / "scope_gates.pt", map_location="cpu",
                                weights_only=False)
    scorer, feature_meta = build_scorer(
        scope_examples(selected), model, tokenizer, transforms["transforms"],
        source_summary["selected_config"]["regularizer"], scope_artifact,
        args.batch_size,
    )
    reader_path = Path(args.v5_root) / "result" / "constrained_scope_gate.pt"
    if sha(reader_path) != EXPECTED_V5:
        raise ValueError("V5 reader changed")
    reader, reader_threshold, reader_selected = load_v5_gate(
        reader_path, scope_artifact["heads"]["7301"],
    )
    rows = []
    with outputs[0].open("x") as sink, torch.no_grad():
        for index, session in enumerate(selected):
            part = rollout(session, scorer, reader,
                           controllers[fold_for(session["id"])])
            for row in part:
                sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            sink.flush()
            rows.extend(part)
            print(json.dumps({"shard": args.shard, "sessions": index + 1,
                              "rows": len(rows)}), flush=True)
    frozen = {
        "safe_delete_summary": sha(Path(args.safe_delete_root) / "result" / "summary.json"),
        "safe_delete_audit": sha(Path(args.safe_delete_root) / "execution_audit.json"),
        "safe_delete_checkpoint": sha(Path(args.safe_delete_root) / "result" /
                                       "safe_delete_crossfit.pt"),
        "data_manifest": sha(Path(args.data_root) / "manifest.json"),
        "reader_checkpoint": sha(reader_path),
        "context_transforms": sha(context / "transforms.pt"),
        "scope_artifact": sha(scope / "scope_gates.pt"),
    }
    resource = {
        "protocol": PROTOCOL, "stage": "dagger_collect", "shard": args.shard,
        "num_shards": args.num_shards, "sessions": len(selected), "rows": len(rows),
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "rows_sha256": sha(outputs[0]), "frozen_inputs": frozen,
        "feature_meta": feature_meta, "reader_selected": reader_selected,
        "reader_threshold": reader_threshold, "loading_info": model_meta["loading_info"],
        "gpu": torch.cuda.get_device_name(), "v5_gate_passed": False,
        "v5_false_positives": v5_summary["metrics"]["false_positives"],
        "calibration_loaded": True, "development_loaded": False,
        "test_loaded": False, "external_benchmarks_loaded": False,
    }
    write_json(outputs[1], resource)
    return resource


def summarize(args):
    root = Path(args.root)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows, resources = [], []
    for shard in range(args.num_shards):
        path = root / f"dagger_rows_shard{shard}.jsonl"
        resource = json.loads((root / f"dagger_resource_shard{shard}.json").read_text())
        if resource["rows_sha256"] != sha(path):
            raise ValueError("V6 DAgger shard hash mismatch")
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
        resources.append(resource)
    rows.sort(key=lambda row: (row["session_id"], row["event_step"]))
    base = [json.loads(line) for line in
            (Path(args.safe_delete_root) / "result" / "safe_delete_corpus.jsonl")
            .read_text().splitlines() if line]
    base_ids = {row["state_id"] for row in base}
    identities = [row["state_id"] for row in rows]
    checks = {
        "all_65_sessions_collected": len({row["session_id"] for row in rows}) == 65,
        "all_774_events_collected": len(rows) == 774,
        "one_state_per_session_step": len(identities) == len(set(identities)),
        "feature_width_is_ten": all(len(row["features"]) == len(NO_SCHEDULE)
                                    for row in rows),
        "heldout_sources_sealed": all(
            resource["calibration_loaded"] and not resource["development_loaded"]
            and not resource["test_loaded"] and not resource["external_benchmarks_loaded"]
            for resource in resources),
    }
    summary = {
        "protocol": PROTOCOL, "stage": "dagger_collect", "rows": len(rows),
        "sessions": len({row["session_id"] for row in rows}),
        "overlap_with_v5_corpus": sum(value in base_ids for value in identities),
        "new_states": sum(value not in base_ids for value in identities),
        "label_distribution": dict(sorted(Counter(row["safe_delete"] for row in rows).items())),
        "action_distribution": dict(sorted(Counter(row["v5_action"] for row in rows).items())),
        "resources": resources, "checks": checks,
        "training_authorized": all(checks.values()),
        "rows_sha256": {f"dagger_rows_shard{i}.jsonl": sha(
            root / f"dagger_rows_shard{i}.jsonl") for i in range(args.num_shards)},
        "plan_sha256": sha(PLAN), "calibration_loaded": True,
        "development_loaded": False, "test_loaded": False,
        "external_benchmarks_loaded": False,
    }
    write_json(output, summary)
    return summary


def core_row(row):
    return {key: row[key] for key in (
        "state_id", "session_id", "outer_fold", "event_step", "memory", "top",
        "incoming_chain_id", "incoming_version", "event_kind",
        "desired_source_action", "features", "safe_delete",
    )}


def merged_corpus(safe_delete_root, dagger_root):
    base = [json.loads(line) for line in
            (Path(safe_delete_root) / "result" / "safe_delete_corpus.jsonl")
            .read_text().splitlines() if line]
    dagger = []
    for shard in range(2):
        dagger.extend(json.loads(line) for line in
                      (Path(dagger_root) / f"dagger_rows_shard{shard}.jsonl")
                      .read_text().splitlines() if line)
    merged = {}
    for row in [*base, *dagger]:
        identity = row["state_id"]
        if identity in merged:
            if core_row(row) != core_row(merged[identity]):
                raise ValueError("V5/DAgger overlapping state disagreement")
            merged[identity]["origin_arms"] = sorted(set(
                merged[identity]["origin_arms"] + row["origin_arms"]))
        else:
            merged[identity] = {**core_row(row), "protocol": PROTOCOL,
                                "origin_arms": list(row["origin_arms"])}
    rows = sorted(merged.values(), key=lambda row: (
        row["session_id"], row["event_step"], row["state_id"]))
    return rows


def oof_metrics(rows, oof):
    labels = [row["safe_delete"] for row in rows]
    actions = [row["action"] == "scope_top1_a0" for row in oof]
    tp = sum(action and target for action, target in zip(actions, labels))
    fp = sum(action and not target for action, target in zip(actions, labels))
    fn = sum(not action and target for action, target in zip(actions, labels))
    tn = len(labels) - tp - fp - fn
    metrics = {
        "rows": len(labels), "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
        "delete_rate": (tp + fp) / max(1, len(labels)), "by_outer_fold": {},
    }
    for outer in range(OUTER_FOLDS):
        indices = [index for index, row in enumerate(rows) if row["outer_fold"] == outer]
        values = Counter()
        for index in indices:
            action, target = actions[index], bool(labels[index])
            values["tp" if action and target else "fp" if action else
                   "fn" if target else "tn"] += 1
        metrics["by_outer_fold"][str(outer)] = {
            "rows": len(indices), "true_positives": values["tp"],
            "false_positives": values["fp"], "false_negatives": values["fn"],
            "precision": values["tp"] / max(1, values["tp"] + values["fp"]),
            "recall": values["tp"] / max(1, values["tp"] + values["fn"]),
        }
    return metrics


def train(args):
    import torch

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    dagger_summary = json.loads((Path(args.dagger_root) / "dagger_summary.json").read_text())
    dagger_audit = json.loads((Path(args.dagger_root) / "dagger_execution_audit.json").read_text())
    if not dagger_summary["training_authorized"] or not dagger_audit["passed"]:
        raise ValueError("V6 DAgger collection is not audited for training")
    rows = merged_corpus(args.safe_delete_root, args.dagger_root)
    corpus_path = output / "safe_delete_dagger_corpus.jsonl"
    with corpus_path.open("x") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    features = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    targets = torch.tensor([row["safe_delete"] for row in rows], dtype=torch.float32)
    feature_path = output / "safe_delete_dagger_features.pt"
    torch.save({"protocol": PROTOCOL, "feature_names": NO_SCHEDULE,
                "features": features, "targets": targets}, feature_path)
    artifact = {"protocol": PROTOCOL, "feature_names": NO_SCHEDULE,
                "seeds": SEEDS, "folds": {}}
    oof = [None] * len(rows)
    searches = {}
    for outer in range(OUTER_FOLDS):
        training, search, selected = inner_search(features, targets, rows, outer)
        validation = [index for index, row in enumerate(rows)
                      if row["outer_fold"] == outer]
        predictions, checkpoints = [], {}
        for seed in SEEDS:
            model, mean, std = fit_head(
                features, targets, training, selected["config"], seed)
            with torch.no_grad():
                values = model(((features[validation] - mean) / std)).squeeze(-1).sigmoid()
            predictions.append(values)
            checkpoints[str(seed)] = {"state": model.state_dict(), "mean": mean, "std": std}
        ensemble = torch.stack(predictions).mean(0).tolist()
        for local, index in enumerate(validation):
            probability = float(ensemble[local])
            oof[index] = {
                "protocol": PROTOCOL, "state_id": rows[index]["state_id"],
                "session_id": rows[index]["session_id"], "outer_fold": outer,
                "safe_delete": rows[index]["safe_delete"], "probability": probability,
                "action": "scope_top1_a0" if probability >= selected["threshold"]
                          else "retain_all",
            }
        artifact["folds"][str(outer)] = {
            "config": selected["config"], "threshold": selected["threshold"],
            "inner_feasible": selected["feasible"],
            "inner_selected_metrics": selected["metrics"],
            "train_sessions": sorted({rows[index]["session_id"] for index in training}),
            "validation_sessions": sorted({rows[index]["session_id"] for index in validation}),
            "checkpoints": checkpoints,
        }
        searches[str(outer)] = search
    checkpoint_path = output / "safe_delete_dagger_crossfit.pt"
    torch.save(artifact, checkpoint_path)
    oof_path = output / "oof_predictions.jsonl"
    with oof_path.open("x") as sink:
        for row in oof:
            sink.write(json.dumps(row, sort_keys=True) + "\n")
    search_path = output / "inner_search.json"
    write_json(search_path, searches)
    metrics = oof_metrics(rows, oof)
    active = [value for value in metrics["by_outer_fold"].values()
              if value["true_positives"] + value["false_positives"]]
    checks = {
        "precision_at_least_point_98": metrics["precision"] >= .98,
        "recall_at_least_point_50": metrics["recall"] >= .50,
        "false_positive_deletes_at_most_four": metrics["false_positives"] <= 4,
        "every_active_outer_fold_precision_at_least_point_95": min(
            (value["precision"] for value in active), default=1.0) >= .95,
        "delete_rate_between_point_05_and_point_50": .05 <= metrics["delete_rate"] <= .50,
        "all_sessions_outer_held_out": all(
            not set(value["train_sessions"]) & set(value["validation_sessions"])
            for value in artifact["folds"].values()),
        "same_registered_feature_set": tuple(NO_SCHEDULE) == tuple(
            json.loads((Path(args.safe_delete_root) / "result" / "summary.json")
                       .read_text())["feature_names"]),
        "development_test_external_sealed": True,
    }
    summary = {
        "protocol": PROTOCOL, "source_protocol": SOURCE_PROTOCOL,
        "stage": "offline_crossfit_after_one_dagger", "rows": len(rows),
        "sessions": len({row["session_id"] for row in rows}),
        "base_rows": 2213, "dagger_rows": dagger_summary["rows"],
        "dagger_new_states": dagger_summary["new_states"],
        "label_distribution": dict(sorted(Counter(row["safe_delete"] for row in rows).items())),
        "feature_names": NO_SCHEDULE, "metrics": metrics,
        "fold_selections": {fold: {
            "config": value["config"], "threshold": value["threshold"],
            "inner_feasible": value["inner_feasible"],
            "inner_selected_metrics": value["inner_selected_metrics"],
        } for fold, value in artifact["folds"].items()},
        "gate": {"checks": checks, "offline_crossfit_passed": all(checks.values()),
                 "online_calibration_authorized": all(checks.values())},
        "hashes": {
            "plan": sha(PLAN), "corpus": sha(corpus_path), "features": sha(feature_path),
            "checkpoint": sha(checkpoint_path), "oof": sha(oof_path),
            "inner_search": sha(search_path),
            "v5_summary": sha(Path(args.safe_delete_root) / "result" / "summary.json"),
            "v5_audit": sha(Path(args.safe_delete_root) / "execution_audit.json"),
            "dagger_summary": sha(Path(args.dagger_root) / "dagger_summary.json"),
            "dagger_audit": sha(Path(args.dagger_root) / "dagger_execution_audit.json"),
        },
        "seconds": time.perf_counter() - started, "calibration_loaded": True,
        "development_loaded": False, "test_loaded": False,
        "external_benchmarks_loaded": False,
    }
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    for name in ("root", "safe_delete_root", "data_root", "v5_root", "model",
                 "tokenizer", "context_root", "scope_root"):
        collect_parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    collect_parser.add_argument("--shard", type=int, required=True)
    collect_parser.add_argument("--num-shards", type=int, default=2)
    collect_parser.add_argument("--batch-size", type=int, default=128)
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--root", required=True)
    summary_parser.add_argument("--safe-delete-root", required=True)
    summary_parser.add_argument("--output", required=True)
    summary_parser.add_argument("--num-shards", type=int, default=2)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--safe-delete-root", required=True)
    train_parser.add_argument("--dagger-root", required=True)
    train_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = collect(args) if args.command == "collect" else \
        summarize(args) if args.command == "summarize" else train(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
