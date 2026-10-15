"""Independently reconstruct S4 controls and the selected oracle checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tasks.evaluate_pair_localized_relation_memory_s4_oracle import (
    PROTOCOL,
    evaluate_matrices,
    load_jsonl,
    load_s4_embedding_set,
    oracle_subset,
    random_view,
    select_view,
    selection_key,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_retrieval_evidence_forgetting_s3_projection import (
    ProjectionHead,
    load_embedding_set as load_s3_embedding_set,
    project_matrix,
)


def close(left, right, tolerance=1e-7) -> bool:
    if isinstance(left, dict):
        return set(left) <= set(right) and all(
            close(value, right[key], tolerance) for key, value in left.items()
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            close(a, b, tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, float):
        return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)
    return left == right


def audit(args) -> dict:
    import torch

    data_root, s3_data_root = Path(args.data_root), Path(args.s3_data_root)
    embedding_root, s3_embedding_root = Path(args.embedding_root), Path(args.s3_embedding_root)
    selection_path, weights_path = Path(args.selection), Path(args.weights)
    selection = json.loads(selection_path.read_text())
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    manifest = json.loads((data_root / "manifest.json").read_text())
    s3_manifest = json.loads((s3_data_root / "manifest.json").read_text())
    train_mentions = load_jsonl(data_root / "train_localized_mentions.jsonl")
    calibration_mentions = load_jsonl(data_root / "calibration_localized_mentions.jsonl")
    train_ids = [row["source_row_sha256"] for row in train_mentions]
    calibration_ids = [row["source_row_sha256"] for row in calibration_mentions]
    train_relations = [row["relation"] for row in train_mentions]
    calibration_relations = [row["relation"] for row in calibration_mentions]
    train_candidates, _ = load_s4_embedding_set(
        embedding_root, "train_candidates", manifest["counts"]["train"]["candidates"]
    )
    calibration_candidates, _ = load_s4_embedding_set(
        embedding_root, "calibration_candidates",
        manifest["counts"]["calibration"]["candidates"],
    )
    train_controls, _ = load_s4_embedding_set(
        embedding_root, "train_controls",
        3 * len(train_mentions) + manifest["counts"]["train"][
            "dialogre_trigger_supervised_mentions"
        ],
    )
    calibration_controls, _ = load_s4_embedding_set(
        embedding_root, "calibration_controls",
        3 * len(calibration_mentions) + manifest["counts"]["calibration"][
            "dialogre_trigger_supervised_mentions"
        ],
    )
    train_full, _ = load_s3_embedding_set(
        s3_embedding_root, "train", s3_manifest["counts"]["train"]["mentions"]
    )
    calibration_full, _ = load_s3_embedding_set(
        s3_embedding_root, "calibration",
        s3_manifest["counts"]["calibration"]["mentions"],
    )
    pairs = load_jsonl(s3_data_root / "calibration_relation_pairs.jsonl")
    s3_checkpoint = torch.load(
        args.s3_projection_weights, map_location="cpu", weights_only=False
    )
    fixed_head = ProjectionHead.create(train_full["matrix"].shape[1], 512).to("cuda")
    fixed_head.load_state_dict(s3_checkpoint["state_dict"])
    raw_controls = {
        "full_dialogue": (train_full["matrix"], calibration_full["matrix"]),
        "random_window": (random_view(train_candidates, train_ids),
                          random_view(calibration_candidates, calibration_ids)),
        **{f"deterministic_r{radius}": (
            select_view(train_controls, train_ids, f"deterministic_r{radius}"),
            select_view(calibration_controls, calibration_ids, f"deterministic_r{radius}"),
        ) for radius in (0, 1, 2)},
    }
    reconstructed_controls = {
        name: evaluate_matrices(
            train_ids, train_relations, project_matrix(fixed_head, train_raw),
            calibration_ids, calibration_relations,
            project_matrix(fixed_head, calibration_raw), pairs,
        )
        for name, (train_raw, calibration_raw) in raw_controls.items()
    }
    train_oracle, calibration_oracle = map(
        oracle_subset, (train_controls, calibration_controls)
    )
    oracle_head = ProjectionHead.create(train_oracle["matrix"].shape[1], 512).to("cuda")
    oracle_head.load_state_dict(weights["state_dict"])
    reconstructed_oracle = evaluate_matrices(
        train_oracle["ids"], train_oracle["relations"],
        project_matrix(oracle_head, train_oracle["matrix"]),
        calibration_oracle["ids"], calibration_oracle["relations"],
        project_matrix(oracle_head, calibration_oracle["matrix"]), pairs,
    )
    reconstructed_oracle.update({
        "epoch": weights["epoch"],
        "last_training_loss": selection["selected_oracle"]["last_training_loss"],
    })
    selected_from_summaries = max(selection["oracle_summaries"], key=selection_key)
    gates = {
        "oracle_direct_pair_roc_auc_at_least_075": (
            reconstructed_oracle["direct_pair_roc_auc"] >= .75
        ),
        "oracle_has_both_pair_classes": (
            reconstructed_oracle["direct_pair_positive"] > 0
            and reconstructed_oracle["direct_pair_negative"] > 0
        ),
    }
    checks = {
        "protocol": selection.get("protocol") == PROTOCOL
            and weights.get("protocol") == PROTOCOL,
        "weights": sha(weights_path) == selection["weights_sha256"]
            and weights["s3_initialization_sha256"] == sha(args.s3_projection_weights),
        "controls": close(reconstructed_controls, selection["controls"]),
        "oracle": close(reconstructed_oracle, selection["selected_oracle"]),
        "selection": close(selected_from_summaries, selection["selected_oracle"])
            and weights["epoch"] == selection["selected_oracle"]["epoch"],
        "counts": len(train_oracle["ids"]) == selection["oracle_train_mentions"]
            and len(calibration_oracle["ids"]) == selection["oracle_calibration_mentions"],
        "resource": math.isfinite(selection["resource"]["seconds"])
            and selection["resource"]["peak_reserved_gib"] < 28,
        "provenance": selection["input_audit_sha256"]["data"]
            == sha(data_root / "data_audit.json")
            and selection["input_audit_sha256"]["embeddings"]
            == sha(embedding_root / "embeddings_audit.json"),
        "sealed": selection.get("dialogre_dev_loaded") is False
            and selection.get("dialogre_test_loaded") is False
            and selection.get("pubmed_validation_loaded") is False,
    }
    passed = all(checks.values())
    result = {
        "protocol": "qwen_pair_localized_relation_memory_s4_oracle_audit_v1",
        "passed": passed,
        "checks": checks,
        "registered_gates": gates,
        "release_learned_localizer": all(gates.values()),
        "terminal_status": ("oracle_passed" if all(gates.values())
                            else "stopped_localized_representation_failure"),
        "selected_oracle": reconstructed_oracle,
        "controls": reconstructed_controls,
        "selection_sha256": sha(selection_path),
        "weights_sha256": sha(weights_path),
        "dialogre_dev_rows_parsed": 0,
        "dialogre_test_rows_parsed": 0,
    }
    if not passed:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--s3-data-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--s3-embedding-root", required=True)
    parser.add_argument("--s3-projection-weights", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = audit(args)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
