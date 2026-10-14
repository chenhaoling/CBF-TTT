"""Independent nested-OOF audit for the V18 causal deletion policy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tasks.evaluate_mquake_qwen_outcome_v16 import load_jsonl
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json
from tasks.train_v18_causal_deletion_policy import (
    EPOCHS, FEATURE_NAMES, GRADIENT_CLIP, HIDDEN, LEARNING_RATE, LOSS_WEIGHTS,
    PROTOCOL, SEEDS, THRESHOLDS, WEIGHT_DECAY, policy_states, predict,
    select_hyperparameters, selection_metrics, verify_examples_release,
)


def audit(examples_root, policy_root, output):
    import torch
    examples_root, policy_root = Path(examples_root), Path(policy_root)
    dependencies = verify_examples_release(examples_root)
    rows = load_jsonl(examples_root / "data/examples.jsonl")
    summary = json.loads((policy_root / "summary.json").read_text())
    training = json.loads((policy_root / "training.json").read_text())
    predictions = load_jsonl(policy_root / "oof_policy_states.jsonl")
    inner_records = load_jsonl(policy_root / "inner_selection_scores.jsonl")
    checkpoint_path = policy_root / "controllers.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    selections_ok = True
    inner_index = {(row["outer_fold"], row["epoch"]): row for row in inner_records}
    recomputed_selection = {}
    for outer in range(5):
        inner = (outer + 1) % 5
        inner_rows = [row for row in rows if row["fold"] == inner]
        grid = {}
        for epoch in EPOCHS:
            record = inner_index.get((outer, epoch), {})
            selections_ok &= (record.get("inner_fold") == inner
                              and record.get("example_ids")
                              == [row["example_id"] for row in inner_rows]
                              and len(record.get("ensemble_scores", [])) == len(inner_rows)
                              and all(math.isfinite(value)
                                      for value in record.get("ensemble_scores", [])))
            grid[str(epoch)] = {str(threshold): selection_metrics(
                inner_rows, record.get("ensemble_scores", []), threshold)
                for threshold in THRESHOLDS}
        selected = select_hyperparameters(grid)
        recomputed_selection[str(outer)] = {"inner_fold": inner, **selected, "grid": grid}
        selections_ok &= summary["selection"][str(outer)] == recomputed_selection[str(outer)]
    prediction_index = {row["episode_id"]: row for row in predictions}
    states_ok = (len(predictions) == len(prediction_index) == 320)
    for outer in range(5):
        outer_rows = [row for row in rows if row["fold"] == outer]
        threshold = recomputed_selection[str(outer)]["threshold"]
        for seed in SEEDS:
            artifact = checkpoint["checkpoints"][str(outer)][str(seed)]
            scores = predict(artifact, outer_rows, device)
            expected = {row["episode_id"]: row for row in policy_states(
                outer_rows, scores, threshold)}
            for episode_id, value in expected.items():
                observed = prediction_index.get(episode_id, {}).get("per_seed", {}).get(
                    str(seed), {})
                states_ok &= observed == value
                states_ok &= (value["current_session_id"] in value["retained_session_ids"]
                              and value["current_session_id"] not in value["deleted_session_ids"])
    hyperparameters_ok = (
        summary.get("seeds") == list(SEEDS) and summary.get("epochs") == list(EPOCHS)
        and summary.get("thresholds") == list(THRESHOLDS)
        and summary.get("feature_names") == list(FEATURE_NAMES)
        and summary.get("hidden") == HIDDEN and summary.get("loss_weights") == LOSS_WEIGHTS
        and summary.get("optimizer") == {"name": "AdamW",
                                         "learning_rate": LEARNING_RATE,
                                         "weight_decay": WEIGHT_DECAY,
                                         "gradient_clip": GRADIENT_CLIP}
        and checkpoint.get("protocol") == PROTOCOL
        and tuple(checkpoint.get("feature_names", ())) == tuple(FEATURE_NAMES)
        and tuple(checkpoint.get("seeds", ())) == SEEDS
        and checkpoint.get("selections") == summary.get("selection"))
    checks = {
        "audited_causal_example_release": summary.get("dependencies") == dependencies,
        "registered_architecture_losses_and_optimizer": hyperparameters_ok,
        "independent_inner_grid_and_hyperparameter_selection": selections_ok,
        "independent_outer_fold_checkpoint_predictions": states_ok,
        "complete_subject_template_disjoint_oof_coverage": (
            len(rows) == 9280 and len(predictions) == 320
            and {row["fold"] for row in predictions} == set(range(5))
            and all(row["selection"]["inner_fold"] == (row["fold"] + 1) % 5
                    for row in predictions)),
        "artifacts_hashes_resources_and_access_locks": (
            summary.get("protocol") == PROTOCOL and summary.get("rows") == 9280
            and summary.get("episodes") == 320
            and summary.get("checkpoint_sha256") == sha(checkpoint_path)
            and summary.get("oof_policy_states_sha256")
                == sha(policy_root / "oof_policy_states.jsonl")
            and summary.get("inner_selection_scores_sha256")
                == sha(policy_root / "inner_selection_scores.jsonl")
            and training.get("checkpoint_sha256") == summary.get("checkpoint_sha256")
            and math.isfinite(summary.get("training_seconds", math.nan))
            and summary.get("training_seconds", 0) > 0
            and math.isfinite(summary.get("gpu_peak_allocated_gib", math.nan))
            and math.isfinite(summary.get("gpu_peak_reserved_gib", math.nan))
            and summary.get("query_features_used") is False
            and summary.get("current_update_always_retained") is True
            and summary.get("external_sets_loaded") is False
            and summary.get("terminal_status") == "trained_oof_deletion_policy"),
    }
    result = {"protocol": f"{PROTOCOL}_audit", "audit_passed": all(checks.values()),
              "checks": checks, "summary_sha256": sha(policy_root / "summary.json"),
              "training_sha256": sha(policy_root / "training.json"),
              "checkpoint_sha256": sha(checkpoint_path),
              "oof_policy_states_sha256": sha(policy_root / "oof_policy_states.jsonl"),
              "inner_selection_scores_sha256": sha(
                  policy_root / "inner_selection_scores.jsonl"),
              "external_sets_loaded": False}
    write_json(output, result)
    if not result["audit_passed"]:
        raise RuntimeError("V18 causal deletion policy audit failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-root", required=True)
    parser.add_argument("--policy-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
