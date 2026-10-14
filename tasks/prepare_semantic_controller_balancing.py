"""Prepare nested-fold, relation/source-balanced weights for semantic training."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "post_v20_semantic_controller_nested_balancing_v2"
SEED = "post-v20-semantic-controller-balance-v2"
TEMPERATURE_EXPONENT = .5


def stable(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source]


def roles(rows, outer_fold, folds=5):
    inner_fold = (outer_fold + 1) % folds
    output = {}
    for row in rows:
        fold = int(row["outer_fold"])
        output[row["row_id"]] = "outer_test" if fold == outer_fold else (
            "inner_validation" if fold == inner_fold else "train")
    return output, inner_fold


def relation_source_weights(rows, exponent=TEMPERATURE_EXPONENT):
    """Temper relation-source frequency while using every row once.

    A row in a relation-source group of size ``n`` receives raw weight
    ``n**(-exponent)``.  The registered square-root exponent reduces dominant
    groups without letting a singleton receive hundreds of times the weight of
    an ordinary row.  Weights are normalized to mean one.
    """
    if not rows:
        raise ValueError("cannot balance an empty training split")
    by_relation_source = Counter((row["relation_id"], row["source"]) for row in rows)
    if not 0 <= exponent <= 1:
        raise ValueError("weight exponent must be in [0, 1]")
    raw = {row["row_id"]: by_relation_source[(row["relation_id"], row["source"])] ** (-exponent)
           for row in rows}
    scale = len(rows) / math.fsum(raw[key] for key in sorted(raw))
    weights = {}
    for row in rows:
        if row["row_id"] in weights:
            raise ValueError(f"duplicate row_id: {row['row_id']}")
        weights[row["row_id"]] = raw[row["row_id"]] * scale
    return weights


def validate_balance(rows, weights, exponent=TEMPERATURE_EXPONENT, tolerance=1e-9):
    counts = Counter((row["relation_id"], row["source"]) for row in rows)
    group_row_weights = defaultdict(list)
    for row in rows:
        weight = weights[row["row_id"]]
        group_row_weights[(row["relation_id"], row["source"])].append(weight)
    within_group_equal = all(max(values) - min(values) <= tolerance * max(1.0, max(values))
                             for values in group_row_weights.values())
    scale_values = []
    for key, values in group_row_weights.items():
        scale_values.append(values[0] * (counts[key] ** exponent))
    formula_exact = max(scale_values) - min(scale_values) <= tolerance * max(1.0, max(scale_values))
    mean_one = abs(math.fsum(weights[key] for key in sorted(weights)) / len(rows) - 1.0) <= tolerance
    ess_fraction = effective_sample_size(weights) / len(rows)
    max_weight = max(weights.values())
    if not within_group_equal or not formula_exact or not mean_one:
        raise RuntimeError("balance-weight invariant failed")
    return {"within_relation_source_group_equal": True,
            "inverse_sqrt_group_frequency_exact": exponent == .5 and formula_exact,
            "mean_row_weight_one": True,
            "effective_sample_size_fraction": ess_fraction,
            "maximum_row_weight": max_weight,
            "resource_gate_passed": ess_fraction >= .8 and max_weight <= 32.0}


def effective_sample_size(weights):
    values = [weights[key] for key in sorted(weights)]
    return math.fsum(values) ** 2 / math.fsum(value * value for value in values)


def prepare(rows, folds=5):
    fold_outputs, summaries = {}, {}
    row_ids = {row["row_id"] for row in rows}
    if len(row_ids) != len(rows):
        raise ValueError("semantic inventory has duplicate row IDs")
    for outer in range(folds):
        role_map, inner = roles(rows, outer, folds)
        train = [row for row in rows if role_map[row["row_id"]] == "train"]
        weights = relation_source_weights(train)
        checks = validate_balance(train, weights)
        if not checks["resource_gate_passed"]:
            raise RuntimeError(f"tempered weighting resource gate failed for outer fold {outer}")
        weighted_rows = [{"row_id": row["row_id"], "outer_fold": outer,
                          "role": role_map[row["row_id"]],
                          "balance_weight": weights.get(row["row_id"])}
                         for row in sorted(rows, key=lambda item: stable(
                             f"{SEED}|{outer}|{item['row_id']}"))]
        fold_outputs[outer] = weighted_rows
        by_source = Counter(row["source"] for row in train)
        summaries[str(outer)] = {
            "inner_fold": inner,
            "train_folds": sorted(set(range(folds)) - {outer, inner}),
            "train_rows": len(train),
            "train_relations": len({row["relation_id"] for row in train}),
            "train_subjects": len({row["subject_normalized"] for row in train}),
            "train_source_rows": dict(sorted(by_source.items())),
            "balance_weight_min": min(weights.values()),
            "balance_weight_max": max(weights.values()),
            "effective_sample_size": effective_sample_size(weights),
            "checks": checks,
        }
    return fold_outputs, summaries


def write_jsonl(path, rows):
    with Path(path).open("x") as sink:
        for row in rows:
            sink.write(json.dumps(row, sort_keys=True) + "\n")


def build(args):
    inventory = Path(args.inventory)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    rows = read_jsonl(inventory / "semantic_rows.jsonl")
    fold_outputs, summaries = prepare(rows, args.folds)
    output.mkdir(parents=True)
    files = {}
    for outer, values in fold_outputs.items():
        name = f"outer_{outer}.jsonl"
        write_jsonl(output / name, values); files[name] = sha(output / name)
    manifest = {
        "protocol": PROTOCOL, "seed": SEED, "folds": args.folds,
        "inventory_manifest_sha256": sha(inventory / "manifest.json"),
        "semantic_rows_sha256": sha(inventory / "semantic_rows.jsonl"),
        "weighting": "inverse square-root relation-source group frequency",
        "temperature_exponent": TEMPERATURE_EXPONENT,
        "row_usage": "once_per_epoch_without_cross_fold_copying",
        "inner_fold_rule": "(outer_fold + 1) mod 5",
        "fold_summaries": summaries, "files_sha256": files,
        "model_outcomes_loaded": False, "sealed_benchmarks_loaded": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
