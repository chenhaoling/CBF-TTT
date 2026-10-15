"""Audit V8 relation-posterior feature extraction and reconstruction."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.audit_qwen_utility_record_controller_v5_residual_training import (
    load_checkpoint,
)
from scripts.inspect_lama_pararel_utility_source import sha256
from tasks.extract_qwen_utility_record_controller_v8_relation_posterior_features import (
    AUDIT_PROTOCOL,
    FEATURE_NAMES,
    PROTOCOL,
    add_arguments,
    augment_row,
    load_domains,
    predict_relation_logits,
    prepare_relation_training,
    relation_mentions,
)
from tasks.pilot_qwen_utility_semantic_record_gate_v1 import load_jsonl


def close(left, right, tolerance=2e-5):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            close(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            close(a, b, tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isclose(
            float(left), float(right), rel_tol=0, abs_tol=tolerance
        )
    return left == right


def audit(args) -> dict:
    import torch

    root = Path(args.feature_root)
    summary_path = root / "summary.json"
    embedding_path = root / "relation_embeddings.pt"
    head_path = root / "relation_heads.pt"
    summary = json.loads(summary_path.read_text())
    domains, audit_hashes = load_domains(args)
    mentions = relation_mentions(domains)
    embeddings = load_checkpoint(embedding_path)
    heads = load_checkpoint(head_path)
    texts = embeddings["texts"]
    vectors = embeddings["vectors"]
    text_to_index = {text: index for index, text in enumerate(texts)}
    training = prepare_relation_training(mentions, text_to_index, vectors)
    head_artifacts = heads["heads"]
    logits = predict_relation_logits(
        vectors, training, head_artifacts, torch.device("cpu"),
    )
    reconstruction_checks = []
    output_hashes = {}
    row_counts = {}
    for name, domain in domains.items():
        original = {
            (row["episode_id"], row["boundary_index"]): row
            for row in domain["features"]
        }
        seen = set(); rows = 0; candidates = 0
        for shard in (0, 1):
            path = root / name / f"features_shard{shard}.jsonl"
            output_hashes[f"{name}_shard{shard}"] = sha256(path)
            for observed in load_jsonl(path):
                identity = (observed["episode_id"], observed["boundary_index"])
                if identity in seen or identity not in original:
                    reconstruction_checks.append(False); continue
                seen.add(identity)
                expected = augment_row(
                    domain, original[identity], text_to_index, logits
                )
                reconstruction_checks.append(close(observed, expected))
                rows += 1; candidates += len(observed["features"])
        reconstruction_checks.append(seen == set(original))
        row_counts[name] = {"rows": rows, "candidates": candidates}
    vector_norms = vectors.float().norm(dim=1)
    checks = {
        "protocol_plan_and_summary_passed": (
            summary.get("protocol") == PROTOCOL
            and summary.get("plan")
            == "QWEN_UTILITY_RECORD_CONTROLLER_V8_RELATION_POSTERIOR_PLAN.md"
            and summary.get("passed")
        ),
        "upstream_audit_hashes_reconstructed": (
            summary.get("upstream_audit_hashes") == audit_hashes
            and embeddings.get("upstream_audit_hashes") == audit_hashes
        ),
        "embedding_hash_order_and_norms_valid": (
            sha256(embedding_path) == summary.get("embedding_sha256")
            and texts == mentions["required_texts"]
            and len(texts) == len(set(texts)) == len(vectors)
            and torch.isfinite(vectors).all().item()
            and float((vector_norms - 1).abs().max()) <= .002
        ),
        "relation_heads_hash_and_training_support_valid": (
            sha256(head_path) == summary.get("relation_heads_sha256")
            and heads.get("protocol") == PROTOCOL
            and heads.get("relations") == training["relations"]
            and heads.get("relation_counts") == training["counts"]
            and heads.get("training_examples") == len(training["y"])
            and heads.get("ambiguous_training_texts")
            == mentions["ambiguous_training_texts"]
            and len(head_artifacts) == 3
            and all(
                torch.isfinite(value).all().item()
                for artifact in head_artifacts
                for value in artifact["state_dict"].values()
            )
        ),
        "all_augmented_features_reconstructed": all(reconstruction_checks),
        "feature_hashes_reconstructed": (
            output_hashes == summary.get("feature_sha256")
        ),
        "registered_counts_and_dimension": (
            row_counts == {
                "templama_train": {"rows": 3072, "candidates": 9216},
                "lama_train": {"rows": 3072, "candidates": 9216},
                "templama_dev": {"rows": 1536, "candidates": 4608},
                "lama_dev": {"rows": 384, "candidates": 1152},
            }
            and tuple(summary.get("feature_names", ())) == FEATURE_NAMES
        ),
        "finite_resources": all(
            math.isfinite(float(summary[name])) and float(summary[name]) > 0
            for name in (
                "embedding_seconds", "relation_head_seconds",
                "peak_allocated_gib", "peak_reserved_gib",
            )
        ),
        "heldout_sources_sealed": all(
            not summary[name] for name in (
                "templama_test_loaded", "lama_confirmation_loaded",
                "longmemeval_loaded", "locomo_loaded",
            )
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "passed": passed,
        "checks": checks,
        "summary_sha256": sha256(summary_path),
        "embedding_sha256": sha256(embedding_path),
        "relation_heads_sha256": sha256(head_path),
        "feature_sha256": output_hashes,
        "row_counts": row_counts,
        "next_stage": "v8_relation_posterior_controller_training" if passed else "stop",
        "terminal_status": (
            "v8_relation_posterior_feature_audit_passed" if passed
            else "v8_relation_posterior_feature_audit_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
