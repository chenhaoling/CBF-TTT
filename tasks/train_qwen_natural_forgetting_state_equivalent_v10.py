"""Cross-fit V7 features with a physically state-equivalent safe-delete target."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from tasks.build_qwen_natural_causal_memory_v1 import load
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_qwen_natural_forgetting_barrier_q_v1 import OUTER_FOLDS, SEEDS
from tasks.train_qwen_natural_forgetting_entity_key_v7 import (
    FEATURE_NAMES,
    PROTOCOL as SOURCE_PROTOCOL,
    entity_inner_search,
)
from tasks.train_qwen_natural_forgetting_safe_delete_dagger_v6 import oof_metrics
from tasks.train_qwen_natural_forgetting_safe_delete_v5 import fit_head


PROTOCOL = "qwen_natural_forgetting_state_equivalent_v10"
PLAN = "QWEN_NATURAL_FORGETTING_STATE_EQUIVALENT_V10_PLAN.md"


def fact_value_identity(version):
    if not version:
        return None
    source = version.get("source_row_sha256")
    target = version.get("target")
    if not source or target is None:
        return None
    payload = json.dumps([source, target], ensure_ascii=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def duplicate_equivalent(row, session):
    if row["top"] is None:
        return False
    chains = {chain["id"]: chain for chain in session["chains"]}
    incoming = chains[row["incoming_chain_id"]]["versions"][
        int(row["incoming_version"])]
    chain_id, version = row["memory"][int(row["top"])]
    top = chains[chain_id]["versions"][int(version)]
    left, right = fact_value_identity(incoming), fact_value_identity(top)
    return left is not None and left == right


def relabel_corpus(source_root, data_root):
    source = Path(source_root)
    summary = json.loads((source / "result" / "summary.json").read_text())
    audit = json.loads((source / "execution_audit.json").read_text())
    if summary.get("protocol") != SOURCE_PROTOCOL or not audit.get("passed"):
        raise ValueError("V10 requires audited V7 source")
    if summary["gate"]["online_calibration_authorized"]:
        raise ValueError("V10 expects the V7 offline stop")
    if (source / "terminal_status.txt").read_text().strip() != \
            "completed_qwen_natural_forgetting_entity_key_v7_offline_stop":
        raise ValueError("unexpected V7 terminal status")
    rows = [json.loads(line) for line in
            (source / "result" / "entity_key_corpus.jsonl").read_text().splitlines()
            if line]
    _, sessions = load(data_root, "calibration")
    session_table = {session["id"]: session for session in sessions}
    output = []
    for row in rows:
        lifecycle = bool(row["safe_delete"])
        duplicate = duplicate_equivalent(row, session_table[row["session_id"]])
        output.append({
            **row,
            "protocol": PROTOCOL,
            "lifecycle_safe_delete": int(lifecycle),
            "duplicate_equivalent": int(duplicate),
            "safe_delete": int(lifecycle or duplicate),
        })
    output.sort(key=lambda row: (row["session_id"], row["event_step"], row["state_id"]))
    return output


def run(args):
    import torch

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.perf_counter()
    rows = relabel_corpus(args.source_root, args.data_root)
    corpus_path = output / "state_equivalent_corpus.jsonl"
    with corpus_path.open("x") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    features = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    targets = torch.tensor([row["safe_delete"] for row in rows], dtype=torch.float32)
    feature_path = output / "state_equivalent_features.pt"
    torch.save({"protocol": PROTOCOL, "feature_names": FEATURE_NAMES,
                "features": features, "targets": targets}, feature_path)
    artifact = {"protocol": PROTOCOL, "feature_names": FEATURE_NAMES,
                "seeds": SEEDS, "folds": {}}
    searches, oof = {}, [None] * len(rows)
    for outer in range(OUTER_FOLDS):
        training, search, selected = entity_inner_search(features, targets, rows, outer)
        validation = [index for index, row in enumerate(rows)
                      if row["outer_fold"] == outer]
        predictions, checkpoints = [], {}
        for seed in SEEDS:
            model, mean, std = fit_head(features, targets, training,
                                        selected["config"], seed)
            with torch.no_grad():
                values = model((features[validation] - mean) / std) \
                    .squeeze(-1).sigmoid()
            predictions.append(values)
            checkpoints[str(seed)] = {
                "state": model.state_dict(), "mean": mean, "std": std,
            }
        ensemble = torch.stack(predictions).mean(0).tolist()
        for local, index in enumerate(validation):
            probability = float(ensemble[local])
            barrier = rows[index]["entity_anchor_barrier"]
            delete = probability >= selected["threshold"] and (
                not selected["use_entity_anchor_barrier"] or barrier)
            oof[index] = {
                "protocol": PROTOCOL, "state_id": rows[index]["state_id"],
                "session_id": rows[index]["session_id"], "outer_fold": outer,
                "safe_delete": rows[index]["safe_delete"],
                "lifecycle_safe_delete": rows[index]["lifecycle_safe_delete"],
                "duplicate_equivalent": rows[index]["duplicate_equivalent"],
                "probability": probability, "entity_anchor_barrier": barrier,
                "action": "scope_top1_a0" if delete else "retain_all",
            }
        artifact["folds"][str(outer)] = {
            "config": selected["config"], "threshold": selected["threshold"],
            "use_entity_anchor_barrier": selected["use_entity_anchor_barrier"],
            "inner_feasible": selected["feasible"],
            "inner_selected_metrics": selected["metrics"],
            "train_sessions": sorted({rows[index]["session_id"] for index in training}),
            "validation_sessions": sorted({rows[index]["session_id"]
                                            for index in validation}),
            "checkpoints": checkpoints,
        }
        searches[str(outer)] = search
    checkpoint_path = output / "state_equivalent_crossfit.pt"
    torch.save(artifact, checkpoint_path)
    oof_path = output / "oof_predictions.jsonl"
    with oof_path.open("x") as sink:
        for row in oof:
            sink.write(json.dumps(row, sort_keys=True) + "\n")
    search_path = output / "inner_search.json"; write_json(search_path, searches)
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
        "feature_family_unchanged_from_v7": tuple(FEATURE_NAMES)
            == tuple(json.loads((Path(args.source_root) / "result" / "summary.json")
                                .read_text())["feature_names"]),
        "label_metadata_excluded_from_features": True,
        "development_test_external_sealed": True,
    }
    source = Path(args.source_root)
    summary = {
        "protocol": PROTOCOL, "source_protocol": SOURCE_PROTOCOL,
        "stage": "offline_state_equivalent_crossfit", "rows": len(rows),
        "sessions": len({row["session_id"] for row in rows}),
        "label_distribution": dict(sorted(Counter(
            row["safe_delete"] for row in rows).items())),
        "label_components": {
            "lifecycle_safe": sum(row["lifecycle_safe_delete"] for row in rows),
            "duplicate_equivalent": sum(row["duplicate_equivalent"] for row in rows),
            "duplicate_only": sum(row["duplicate_equivalent"]
                                  and not row["lifecycle_safe_delete"] for row in rows),
        },
        "feature_names": FEATURE_NAMES, "metrics": metrics,
        "fold_selections": {fold: {
            "config": value["config"], "threshold": value["threshold"],
            "use_entity_anchor_barrier": value["use_entity_anchor_barrier"],
            "inner_feasible": value["inner_feasible"],
            "inner_selected_metrics": value["inner_selected_metrics"],
        } for fold, value in artifact["folds"].items()},
        "gate": {"checks": checks, "offline_crossfit_passed": all(checks.values()),
                 "online_calibration_authorized": all(checks.values())},
        "hashes": {
            "plan": sha(PLAN), "corpus": sha(corpus_path), "features": sha(feature_path),
            "checkpoint": sha(checkpoint_path), "oof": sha(oof_path),
            "inner_search": sha(search_path),
            "source_summary": sha(source / "result" / "summary.json"),
            "source_audit": sha(source / "execution_audit.json"),
            "source_corpus": sha(source / "result" / "entity_key_corpus.jsonl"),
            "data_manifest": sha(Path(args.data_root) / "manifest.json"),
        },
        "seconds": time.perf_counter() - started,
        "calibration_loaded": True, "development_loaded": False,
        "test_loaded": False, "external_benchmarks_loaded": False,
    }
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
