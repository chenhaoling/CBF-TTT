"""Independent integrity audit for forgetting-only validation outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


POLICIES = (
    "fixed_0", "fixed_0.5", "fixed_1",
    "controller_seed42", "controller_seed43", "controller_seed44",
)
CONDITIONS = ("full_context", "correct_memory", "wrong_memory", "empty_memory")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return abs(left - right) <= tolerance


def audit(root: Path, reference: Path | None = None) -> dict:
    if (root / "status.txt").read_text().strip() != "completed":
        raise ValueError("run status is not completed")
    summary = json.loads((root / "summary.json").read_text())
    rows = {}
    hashes = {}
    for policy in POLICIES:
        path = root / "results" / f"{policy}.jsonl"
        hashes[path.name] = _sha256(path)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (policy, row["id"])
                if key in rows or row["policy_name"] != policy:
                    raise ValueError(f"duplicate or mismatched row {key}")
                if row["group_id"] == row["donor_group_id"]:
                    raise ValueError(f"same-group wrong memory for {key}")
                if row["memory_norm"] <= 0 or row["donor_memory_norm"] <= 0:
                    raise ValueError(f"empty fast memory for {key}")
                if set(row["query_losses"]) != set(CONDITIONS):
                    raise ValueError(f"condition mismatch for {key}")
                if len(row["query_kinds"]) != 2 or set(row["query_kinds"]) != {"historical", "current_or_new"}:
                    raise ValueError(f"query-kind mismatch for {key}")
                if any(not math.isfinite(value) for values in row["query_losses"].values() for value in values):
                    raise ValueError(f"nonfinite loss for {key}")
                rows[key] = row

    ids = {policy: {sample_id for row_policy, sample_id in rows if row_policy == policy}
           for policy in POLICIES}
    expected_ids = ids[POLICIES[0]]
    if len(expected_ids) != 300 or any(policy_ids != expected_ids for policy_ids in ids.values()):
        raise ValueError("formal policies do not have identical 300-scenario coverage")
    if len({rows[(POLICIES[0], sample_id)]["group_id"] for sample_id in expected_ids}) != 100:
        raise ValueError("expected 100 independent source groups")

    donor_pairs = {}
    for policy in POLICIES:
        for sample_id in expected_ids:
            row = rows[(policy, sample_id)]
            donor_pairs[(policy, sample_id)] = row["donor_id"]
            donor = rows.get((policy, row["donor_id"]))
            if donor is None or donor["donor_id"] != sample_id or donor["regime"] != row["regime"]:
                raise ValueError(f"wrong-memory pairing is not reciprocal for {(policy, sample_id)}")

    recomputed_means = {}
    for policy in POLICIES:
        policy_rows = [rows[(policy, sample_id)] for sample_id in sorted(expected_ids)]
        recomputed_means[policy] = {
            condition: statistics.mean(
                value
                for row in policy_rows
                for value in row["query_losses"][condition]
            )
            for condition in CONDITIONS
        }
        recorded = summary["policies"][policy]["mean_nll"]
        if any(not _close(recomputed_means[policy][condition], recorded[condition])
               for condition in CONDITIONS):
            raise ValueError(f"summary mean mismatch for {policy}")

    reference_check = None
    if reference is not None:
        old = defaultdict(dict)
        with reference.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    old[row["policy"]][row["id"]] = row["mean_loss"]
        mapping = {"fixed_0": "baseline", "fixed_0.5": "0.5", "fixed_1": "1",
                   "controller_seed42": "controller"}
        errors = {}
        for new_policy, old_policy in mapping.items():
            if set(old[old_policy]) != expected_ids:
                raise ValueError(f"reference coverage mismatch for {old_policy}")
            errors[new_policy] = max(
                abs(rows[(new_policy, sample_id)]["mean_nll"]["full_context"] - old[old_policy][sample_id])
                for sample_id in expected_ids
            )
            if errors[new_policy] > 1e-5:
                raise ValueError(f"old full-context rollout was not reproduced for {new_policy}")
        reference_check = {"path": str(reference), "sha256": _sha256(reference), "max_abs_errors": errors}

    result = {
        "passed": True,
        "scenarios_per_policy": len(expected_ids),
        "source_groups": 100,
        "policies": list(POLICIES),
        "raw_sha256": hashes,
        "summary_sha256": _sha256(root / "summary.json"),
        "recomputed_mean_nll": recomputed_means,
        "reciprocal_wrong_memory_pairs": True,
        "finite_losses": True,
        "nonempty_fast_memory": True,
        "reference_full_context_reproduction": reference_check,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.reference)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
