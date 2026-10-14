"""Build a leakage-audited source inventory for the post-V20 controller.

This is a data-only stage.  It never reads model outcomes or sealed natural
benchmarks.  CounterFact is treated as already-observed development data,
MQuAKE-T dev rows are excluded, and only the previously materialized ZsRE
calibration split is accepted.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from tasks.build_mquake_temporal_sessions import split_subjects as split_mquake_subjects
from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "post_v20_semantic_controller_source_inventory_v1"
FOLD_SEED = "post-v20-semantic-controller-folds-v1"
KNOWN_HASHES = {
    "counterfact": "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f",
    "mquake": "58500a1d4aaf9e036c23a1d9dd6be2d73d693e35a695614842f211ea5aff59f2",
    "zsre_calibration": "d19bb92ca6f487ce046e75a0a65e384b831a7bbc988e81436df68b3cd7beec6f",
}


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def stable(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def normalize(value: str) -> str:
    return " ".join(str(value).casefold().strip().split())


def row_hash(row) -> str:
    return stable(canonical(row))


def _target(value) -> str:
    if isinstance(value, dict):
        value = value.get("str", "")
    return " ".join(str(value or "").strip().split())


def _unique(values):
    output, seen = [], set()
    for value in values:
        value = " ".join(str(value or "").strip().split())
        key = normalize(value)
        if value and key not in seen:
            output.append(value); seen.add(key)
    return output


def mask_surface(surface: str, subject: str, values) -> str:
    """Mask exact subject/value spans without exposing them to the encoder."""
    output = surface
    replacements = [(subject, "[SUBJECT]")] + [(value, "[VALUE]") for value in values]
    for value, token in sorted(replacements, key=lambda item: -len(item[0])):
        value = str(value).strip()
        if value:
            output = re.sub(re.escape(value), token, output, flags=re.IGNORECASE)
    return " ".join(output.split())


def counterfact_records(rows):
    records, excluded = [], []
    for index, row in enumerate(rows):
        rewrite = row.get("requested_rewrite", {})
        subject = str(rewrite.get("subject", "")).strip()
        relation = str(rewrite.get("relation_id", "")).strip()
        old, new = _target(rewrite.get("target_true")), _target(rewrite.get("target_new"))
        surfaces = []
        prompt = rewrite.get("prompt", "")
        try:
            if prompt:
                surfaces.append(prompt.format(subject))
        except (IndexError, KeyError, ValueError):
            pass
        surfaces.extend(row.get("paraphrase_prompts", []))
        surfaces = _unique(surfaces)
        identifier = f"counterfact:{row.get('case_id', index)}"
        if not subject or not relation or not old or not new or not surfaces:
            excluded.append({"source": "counterfact", "row_id": identifier,
                             "source_row_sha256": row_hash(row), "reason": "invalid_identity"})
            continue
        records.append({
            "source": "counterfact", "source_split": "observed_development",
            "row_id": identifier, "source_row_sha256": row_hash(row),
            "relation_id": relation, "subject": subject,
            "subject_normalized": normalize(subject), "values": [old, new],
            "surfaces": surfaces,
            "masked_surfaces": _unique(mask_surface(item, subject, [old, new])
                                       for item in surfaces),
        })
    return records, excluded


def _mquake_rewrite(row):
    values = row.get("requested_rewrite", [])
    if not isinstance(values, list) or len(values) != 1:
        return None
    return values[0]


def mquake_records(rows):
    assignment, _ = split_mquake_subjects(rows)
    records, excluded = [], []
    for index, row in enumerate(rows):
        rewrite = _mquake_rewrite(row)
        identifier = f"mquake:{row.get('case_id', index)}"
        if rewrite is None:
            excluded.append({"source": "mquake", "row_id": identifier,
                             "source_row_sha256": row_hash(row), "reason": "invalid_rewrite"})
            continue
        subject = str(rewrite.get("subject", "")).strip()
        source_split = assignment.get(subject)
        if source_split == "dev":
            excluded.append({"source": "mquake", "row_id": identifier,
                             "source_row_sha256": row_hash(row), "reason": "mquake_dev_reserved"})
            continue
        relation = str(rewrite.get("relation_id", "")).strip()
        old, new = _target(rewrite.get("target_true")), _target(rewrite.get("target_new"))
        surfaces = []
        prompt = rewrite.get("question") or rewrite.get("prompt", "")
        try:
            if prompt:
                surfaces.append(prompt.format(subject))
        except (IndexError, KeyError, ValueError):
            pass
        edit_indices = row.get("orig", {}).get("edit_triples_idx", [])
        if len(edit_indices) == 1:
            idx = int(edit_indices[0])
            for key in ("single_hops", "new_single_hops"):
                hops = row.get(key, [])
                if 0 <= idx < len(hops):
                    surfaces.append(hops[idx].get("cloze") or hops[idx].get("question", ""))
        surfaces = _unique(surfaces)
        if (source_split not in {"train", "tune"} or not subject or not relation
                or not old or not new or not surfaces):
            excluded.append({"source": "mquake", "row_id": identifier,
                             "source_row_sha256": row_hash(row), "reason": "invalid_identity"})
            continue
        records.append({
            "source": "mquake", "source_split": source_split,
            "row_id": identifier, "source_row_sha256": row_hash(row),
            "relation_id": relation, "subject": subject,
            "subject_normalized": normalize(subject), "values": [old, new],
            "surfaces": surfaces,
            "masked_surfaces": _unique(mask_surface(item, subject, [old, new])
                                       for item in surfaces),
        })
    return records, excluded


def load_zsre_calibration(path):
    facts = []
    with Path(path).open() as source:
        for line in source:
            session = json.loads(line)
            if session.get("split") != "calibration":
                raise ValueError("ZsRE input contains a non-calibration session")
            for fact in session.get("facts", []):
                facts.append(fact)
    if not facts or len({normalize(fact["subject"]) for fact in facts}) != len(facts):
        raise ValueError("ZsRE calibration must contain unique non-empty subjects")
    return facts


def assign_relation_folds(records, folds=5):
    if folds < 2:
        raise ValueError("at least two folds are required")
    counts = Counter(row["relation_id"] for row in records)
    relation_fold, loads = {}, [0] * folds
    ordered = sorted(counts, key=lambda relation: (-counts[relation],
                     stable(f"{FOLD_SEED}|relation|{relation}")))
    for relation in ordered:
        fold = min(range(folds), key=lambda value: (loads[value], value))
        relation_fold[relation] = fold; loads[fold] += counts[relation]
    return relation_fold


def enforce_subject_disjoint(records, relation_fold):
    by_subject = defaultdict(list)
    for row in records:
        by_subject[row["subject_normalized"]].append(row)
    retained, excluded, subject_fold = [], [], {}
    for subject, rows in sorted(by_subject.items()):
        counts = Counter(relation_fold[row["relation_id"]] for row in rows)
        chosen = min(counts, key=lambda fold: (-counts[fold],
                     stable(f"{FOLD_SEED}|subject|{subject}|{fold}")))
        subject_fold[subject] = chosen
        for row in rows:
            fold = relation_fold[row["relation_id"]]
            if fold == chosen:
                retained.append({**row, "outer_fold": fold})
            else:
                excluded.append({"source": row["source"], "row_id": row["row_id"],
                                 "source_row_sha256": row["source_row_sha256"],
                                 "reason": "cross_fold_subject_conflict",
                                 "assigned_relation_fold": fold,
                                 "retained_subject_fold": chosen})
    return sorted(retained, key=lambda row: (row["outer_fold"], row["source"], row["row_id"])), excluded


def overlap_matrix(records, field):
    values = defaultdict(set)
    for row in records:
        values[row["source"]].add(row[field])
    sources = sorted(values)
    return {left: {right: len(values[left] & values[right]) for right in sources}
            for left in sources}


def validate_folds(records, folds):
    relation_sets, subject_sets = defaultdict(set), defaultdict(set)
    for row in records:
        relation_sets[row["outer_fold"]].add(row["relation_id"])
        subject_sets[row["outer_fold"]].add(row["subject_normalized"])
    relation_overlap = any(relation_sets[a] & relation_sets[b]
                           for a in range(folds) for b in range(a + 1, folds))
    subject_overlap = any(subject_sets[a] & subject_sets[b]
                          for a in range(folds) for b in range(a + 1, folds))
    nonempty = all(relation_sets[index] and subject_sets[index] for index in range(folds))
    if relation_overlap or subject_overlap or not nonempty:
        raise RuntimeError("unable to construct non-empty relation/subject-disjoint folds")
    return {"relation_disjoint": True, "subject_disjoint": True, "all_folds_nonempty": True}


def construct(counterfact_path, mquake_path, zsre_calibration_path, folds=5,
              enforce_known_hashes=True):
    paths = {"counterfact": Path(counterfact_path), "mquake": Path(mquake_path),
             "zsre_calibration": Path(zsre_calibration_path)}
    hashes = {name: sha(path) for name, path in paths.items()}
    if enforce_known_hashes and hashes != KNOWN_HASHES:
        raise ValueError(f"source hash mismatch: {hashes}")
    counterfact = json.loads(paths["counterfact"].read_text())
    mquake = json.loads(paths["mquake"].read_text())
    cf_records, cf_excluded = counterfact_records(counterfact)
    mq_records, mq_excluded = mquake_records(mquake)
    allowed = cf_records + mq_records
    relation_fold = assign_relation_folds(allowed, folds)
    retained, crossfold_excluded = enforce_subject_disjoint(allowed, relation_fold)
    checks = validate_folds(retained, folds)

    retained_subjects = {row["subject_normalized"] for row in retained}
    zsre, zsre_excluded = [], []
    for fact in load_zsre_calibration(paths["zsre_calibration"]):
        subject = normalize(fact["subject"])
        compact = {
            "source": "zsre", "source_split": "calibration",
            "row_id": f"zsre:{fact['source_row_sha256'][:20]}",
            "source_row_sha256": fact["source_row_sha256"],
            "subject": fact["subject"], "subject_normalized": subject,
            "prompt": fact["prompt"], "rephrase_prompt": fact["rephrase_prompt"],
            "original": fact["original"], "updated": fact["updated"],
        }
        if subject in retained_subjects:
            zsre_excluded.append({"source": "zsre", "row_id": compact["row_id"],
                                  "source_row_sha256": compact["source_row_sha256"],
                                  "reason": "semantic_subject_overlap"})
        else:
            zsre.append(compact)

    excluded = sorted(cf_excluded + mq_excluded + crossfold_excluded + zsre_excluded,
                      key=lambda row: (row["source"], row["row_id"], row["reason"]))
    fold_counts = {}
    for fold in range(folds):
        rows = [row for row in retained if row["outer_fold"] == fold]
        fold_counts[str(fold)] = {
            "rows": len(rows), "relations": len({row["relation_id"] for row in rows}),
            "subjects": len({row["subject_normalized"] for row in rows}),
            "counterfact_rows": sum(row["source"] == "counterfact" for row in rows),
            "mquake_rows": sum(row["source"] == "mquake" for row in rows),
        }
    manifest = {
        "protocol": PROTOCOL, "fold_seed": FOLD_SEED, "folds": folds,
        "source_sha256": hashes,
        "source_rows": {"counterfact": len(counterfact), "mquake": len(mquake),
                        "zsre_calibration": len(zsre) + len(zsre_excluded)},
        "semantic_rows_before_subject_filter": len(allowed),
        "semantic_rows_retained": len(retained), "zsre_calibration_retained": len(zsre),
        "excluded_counts": dict(sorted(Counter(row["reason"] for row in excluded).items())),
        "fold_counts": fold_counts,
        "source_subject_overlap_before_filter": overlap_matrix(allowed, "subject_normalized"),
        "source_relation_overlap_before_filter": overlap_matrix(allowed, "relation_id"),
        "source_subject_overlap_after_filter": overlap_matrix(retained, "subject_normalized"),
        "source_relation_overlap_after_filter": overlap_matrix(retained, "relation_id"),
        "checks": checks,
        "counterfact_role": "already_observed_development_only",
        "mquake_splits_used": ["train", "tune"], "mquake_dev_used": False,
        "zsre_role": "readout_calibration_only", "zsre_dev_or_test_loaded": False,
        "locomo_loaded": False, "longmemeval_loaded": False,
        "model_outcomes_loaded": False,
    }
    return retained, zsre, excluded, manifest


def write_jsonl(path, rows):
    with Path(path).open("x") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    retained, zsre, excluded, manifest = construct(
        args.counterfact, args.mquake, args.zsre_calibration, args.folds,
        enforce_known_hashes=not args.allow_unknown_hashes)
    output.mkdir(parents=True)
    write_jsonl(output / "semantic_rows.jsonl", retained)
    write_jsonl(output / "zsre_calibration.jsonl", zsre)
    write_jsonl(output / "excluded_rows.jsonl", excluded)
    relation_folds = {row["relation_id"]: row["outer_fold"] for row in retained}
    (output / "relation_folds.json").write_text(
        json.dumps(relation_folds, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    manifest["files_sha256"] = {name: sha(output / name) for name in (
        "semantic_rows.jsonl", "zsre_calibration.jsonl", "excluded_rows.jsonl",
        "relation_folds.json")}
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfact", required=True)
    parser.add_argument("--mquake", required=True)
    parser.add_argument("--zsre-calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--allow-unknown-hashes", action="store_true",
                        help="Tests only: production inputs must match registered hashes")
    print(json.dumps(build(parser.parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
