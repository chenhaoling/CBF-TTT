"""Nested-OOF diagnostic of causal and subject-local Qwen views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from tasks.build_natural_semantic_relation_pairs import load_jsonl
from tasks.evaluate_natural_semantic_embedding_pairs import (
    FOLDS, PAIR_MANIFEST_SHA256, PAIR_ROWS_SHA256,
    binary_metrics, rank_metrics, select_threshold,
)
from tasks.extract_natural_semantic_multiview_embeddings import (
    VIEWS, load_multiview_embeddings,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "natural_semantic_multiview_pairs_v1"
MULTIVIEW_AUDIT_SHA256 = (
    "ea817bab098ce41446309fa79a76f5eb4cc6afa6037a78c481f775e8057f28fb"
)
MISSING_MARKER_SURFACES = {
    "7b4d11c33b3b143e64bb21ce14a3f153a3d900fc3c8970c5b923c14644e8fded",
    "e60b68e8659d5b34b729880bb728de4295a6859fa6025b22cff7eaaf04354647",
}
ARMS = (*VIEWS, "causal_local_concat", "all_views_concat")


def select_arm(candidates):
    return max(candidates, key=lambda value: (
        value["inner_constraints_feasible"],
        value["inner_metrics"]["balanced_accuracy"],
        value["inner_metrics"]["auroc"],
        value["inner_metrics"]["f1"],
        -ARMS.index(value["arm"]),
    ))


def build_arms(views):
    import torch
    values = dict(views)
    values["causal_local_concat"] = torch.nn.functional.normalize(torch.cat([
        views["last_token"], views["last8_mean"],
        views["subject_context8_mean"]], dim=1), p=2, dim=1)
    values["all_views_concat"] = torch.nn.functional.normalize(torch.cat([
        views[view] for view in VIEWS], dim=1), p=2, dim=1)
    return values


def pair_scores(values, pairs, surface_index):
    import torch
    left = torch.tensor([surface_index[row["left_surface_id"]] for row in pairs])
    right = torch.tensor([surface_index[row["right_surface_id"]] for row in pairs])
    return (values[left] * values[right]).sum(dim=1).tolist()


def metric(labels, scores, threshold):
    result = binary_metrics(labels, [int(value >= threshold) for value in scores])
    result.update(rank_metrics(labels, scores))
    return result


def verify_release(pair_root, embedding_root):
    pair_root, embedding_root = Path(pair_root), Path(embedding_root)
    audit = json.loads((embedding_root / "audit.json").read_text())
    checks = (
        sha(pair_root / "manifest.json") == PAIR_MANIFEST_SHA256,
        sha(pair_root / "pairs.jsonl") == PAIR_ROWS_SHA256,
        sha(embedding_root / "audit.json") == MULTIVIEW_AUDIT_SHA256,
        audit.get("audit_passed") is True,
        audit.get("views") == list(VIEWS),
        audit.get("subject_marker_counts", {}).get("missing_fallback_last8") == 2,
        (embedding_root / "runner_exit_code.txt").read_text().strip() == "0",
        (embedding_root / "terminal_status.txt").read_text().strip()
            == "completed_natural_semantic_multiview_embeddings",
    )
    if not all(checks):
        raise ValueError("multiview scoring requires the frozen audited release")
    return {"pair_manifest_sha256": sha(pair_root / "manifest.json"),
            "pairs_sha256": sha(pair_root / "pairs.jsonl"),
            "multiview_audit_sha256": sha(embedding_root / "audit.json")}


def run(args):
    dependencies = verify_release(args.pair_root, args.embedding_root)
    pairs = load_jsonl(Path(args.pair_root) / "pairs.jsonl")
    surface_ids, raw_views = load_multiview_embeddings(args.embedding_root)
    surface_index = {value: index for index, value in enumerate(surface_ids)}
    arms = build_arms(raw_views)
    started = time.perf_counter()
    all_scores = {arm: pair_scores(values, pairs, surface_index)
                  for arm, values in arms.items()}
    fold_results, chosen_rows = {}, []
    for outer in range(FOLDS):
        inner = (outer + 1) % FOLDS
        inner_indices = [index for index, row in enumerate(pairs)
                         if int(row["outer_fold"]) == inner]
        outer_indices = [index for index, row in enumerate(pairs)
                         if int(row["outer_fold"]) == outer]
        candidates = []
        for arm in ARMS:
            inner_labels = [int(pairs[index]["same_relation"])
                            for index in inner_indices]
            inner_scores = [all_scores[arm][index] for index in inner_indices]
            threshold, inner_metric, feasible = select_threshold(
                inner_labels, inner_scores)
            inner_metric.update(rank_metrics(inner_labels, inner_scores))
            outer_labels = [int(pairs[index]["same_relation"])
                            for index in outer_indices]
            outer_scores = [all_scores[arm][index] for index in outer_indices]
            candidates.append({
                "arm": arm, "threshold": threshold,
                "inner_constraints_feasible": feasible,
                "inner_metrics": inner_metric,
                "outer_metrics": metric(outer_labels, outer_scores, threshold),
            })
        chosen = select_arm(candidates)
        fold_results[str(outer)] = {
            "outer_fold": outer, "inner_fold": inner,
            "selected_arm": chosen["arm"], "selected_threshold": chosen["threshold"],
            "inner_constraints_feasible": chosen["inner_constraints_feasible"],
            "inner_metrics": chosen["inner_metrics"],
            "outer_metrics": chosen["outer_metrics"],
            "candidates": candidates,
        }
        for index in outer_indices:
            score = all_scores[chosen["arm"]][index]
            row = pairs[index]
            chosen_rows.append({
                "pair_id": row["pair_id"], "outer_fold": outer,
                "kind": row["kind"], "same_relation": int(row["same_relation"]),
                "selected_arm": chosen["arm"], "threshold": chosen["threshold"],
                "score": score, "prediction": int(score >= chosen["threshold"]),
                "missing_marker_in_pair": int(
                    row["left_surface_id"] in MISSING_MARKER_SURFACES
                    or row["right_surface_id"] in MISSING_MARKER_SURFACES),
            })
    chosen_rows.sort(key=lambda row: row["pair_id"])

    def aggregate(rows):
        labels = [row["same_relation"] for row in rows]
        scores = [row["score"] for row in rows]
        result = binary_metrics(labels, [row["prediction"] for row in rows])
        result.update(rank_metrics(labels, scores))
        return result

    full = aggregate(chosen_rows)
    clean_rows = [row for row in chosen_rows if not row["missing_marker_in_pair"]]
    clean = aggregate(clean_rows)
    arm_macro = {}
    for arm in ARMS:
        values = [fold_results[str(fold)]["candidates"][ARMS.index(arm)]["outer_metrics"]
                  for fold in range(FOLDS)]
        arm_macro[arm] = {key: statistics.fmean(row[key] for row in values)
                          for key in ("balanced_accuracy", "positive_recall",
                                      "negative_specificity", "auroc",
                                      "average_precision")}
    checks = {
        "complete_nested_oof_coverage": (
            len(chosen_rows) == len(pairs)
            and len({row["pair_id"] for row in chosen_rows}) == len(pairs)),
        "all_inner_view_and_threshold_constraints_feasible": all(
            row["inner_constraints_feasible"] for row in fold_results.values()),
        "all_selected_outer_fold_auroc_at_least_point_60": all(
            row["outer_metrics"]["auroc"] >= .60 for row in fold_results.values()),
        "selected_balanced_accuracy_at_least_point_65":
            full["balanced_accuracy"] >= .65,
        "missing_marker_sensitivity_under_point_1pp":
            abs(full["balanced_accuracy"] - clean["balanced_accuracy"]) < .001,
    }
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    rows_path = output / "oof_multiview.jsonl"
    with rows_path.open("x") as sink:
        for row in chosen_rows:
            sink.write(json.dumps(row) + "\n")
    result = {
        "protocol": PROTOCOL, "rows": len(chosen_rows),
        "nested_selected_oof": full,
        "without_missing_marker_pairs": clean,
        "pairs_with_missing_marker": len(chosen_rows) - len(clean_rows),
        "folds": fold_results, "arm_outer_fold_macro": arm_macro,
        "diagnostic_gate": {"checks": checks, "passed": all(checks.values())},
        "dependencies": dependencies, "oof_rows_sha256": sha(rows_path),
        "model_outcomes_loaded": False, "sealed_benchmarks_loaded": False,
        "resources": {"seconds": time.perf_counter() - started},
        "next_stage": ("train_multiview_semantic_projection"
                       if full["balanced_accuracy"] >= .65
                       else "revise_relation_text_supervision"),
        "terminal_status": "completed_natural_semantic_multiview_pair_diagnostic",
    }
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
