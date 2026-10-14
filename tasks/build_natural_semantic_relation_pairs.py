"""Build relation-disjoint supervision pairs for the semantic residual."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "natural_semantic_relation_pairs_v1"
SEED = 2201
RELATION_CAP = 512
HARD_CANDIDATES = 64
INVENTORY_MANIFEST_SHA256 = (
    "df3002f4913663ef829b865b3d1502305b712f5a96c401568c413a9fa2862836"
)
INVENTORY_ROWS_SHA256 = (
    "222f7c51d384a9e26da93ecfd0dda91dd0add148c9901e5765f2fe4458614a99"
)
HEADROOM_SUMMARY_SHA256 = (
    "5301773ddf293959ac7c9ac0fd015cc3dfc504f1d44ac0aef821c1db5c13a830"
)
HEADROOM_AUDIT_SHA256 = (
    "00e8ca36d7275014fb939fc71a7e5bffeff010dc3d446a99ee0410a89f31549d"
)


def load_jsonl(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def stable(*values):
    return hashlib.sha256(":.:".join(map(str, values)).encode()).hexdigest()


def normalize(value):
    return " ".join(str(value).casefold().split())


def trigrams(value):
    value = f"  {normalize(value)}  "
    return {value[index:index + 3] for index in range(max(0, len(value) - 2))}


def trigram_jaccard(left, right):
    left, right = trigrams(left), trigrams(right)
    return len(left & right) / len(left | right) if left or right else 1.0


def row_surface(row, index=0):
    values = [" ".join(value.split()) for value in row.get("masked_surfaces", [])
              if isinstance(value, str) and value.strip()]
    unique = list(dict.fromkeys(values))
    return unique[index] if len(unique) > index else None


def verify_release(inventory_root, headroom_root):
    inventory_root, headroom_root = Path(inventory_root), Path(headroom_root)
    manifest_path, rows_path = (inventory_root / "manifest.json",
                                inventory_root / "semantic_rows.jsonl")
    inventory_audit_path = inventory_root.parent / "audit.json"
    summary_path, audit_path = headroom_root / "summary.json", headroom_root / "audit.json"
    manifest, inventory_audit = (json.loads(manifest_path.read_text()),
                                 json.loads(inventory_audit_path.read_text()))
    summary, audit = json.loads(summary_path.read_text()), json.loads(audit_path.read_text())
    checks = (
        sha(manifest_path) == INVENTORY_MANIFEST_SHA256,
        sha(rows_path) == INVENTORY_ROWS_SHA256,
        inventory_audit.get("passed") is True,
        inventory_audit.get("manifest_sha256") == INVENTORY_MANIFEST_SHA256,
        sha(summary_path) == HEADROOM_SUMMARY_SHA256,
        sha(audit_path) == HEADROOM_AUDIT_SHA256,
        summary.get("formal_gate", {}).get("passed") is True,
        summary.get("next_stage") == "train_natural_semantic_residual_controller",
        summary.get("terminal_status") == "formal_semantic_headroom_supported",
        audit.get("audit_passed") is True,
        audit.get("summary_sha256") == HEADROOM_SUMMARY_SHA256,
        (headroom_root / "runner_exit_code.txt").read_text().strip() == "0",
        (headroom_root / "terminal_status.txt").read_text().strip()
            == "completed_post_v19_natural_semantic_formal",
    )
    if not all(checks):
        raise ValueError("semantic pair construction requires the audited formal release")
    return rows_path, {
        "inventory_manifest_sha256": sha(manifest_path),
        "inventory_audit_sha256": sha(inventory_audit_path),
        "semantic_rows_sha256": sha(rows_path),
        "headroom_summary_sha256": sha(summary_path),
        "headroom_audit_sha256": sha(audit_path),
    }


def retained_anchors(rows, seed=SEED, cap=RELATION_CAP):
    groups = defaultdict(list)
    for row in rows:
        if row_surface(row) and row.get("subject_normalized") and row.get("relation_id"):
            groups[(int(row["outer_fold"]), row["relation_id"])].append(row)
    output, excluded = [], []
    for key, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda row: stable(
            seed, "anchor", row["source_row_sha256"]))
        output.extend(ordered[:cap])
        excluded.extend({"source_row_sha256": row["source_row_sha256"],
                         "outer_fold": key[0], "relation_id": key[1],
                         "reason": "relation_cap"} for row in ordered[cap:])
    return output, excluded


def candidate_indices(rows):
    by_fold, by_relation, by_trigram = defaultdict(list), defaultdict(list), defaultdict(set)
    for index, row in enumerate(rows):
        fold, relation = int(row["outer_fold"]), row["relation_id"]
        by_fold[fold].append(index)
        by_relation[(fold, relation)].append(index)
        for value in trigrams(row_surface(row)):
            by_trigram[(fold, value)].add(index)
    for values in by_fold.values():
        values.sort(key=lambda index: rows[index]["source_row_sha256"])
    return by_fold, by_relation, by_trigram


def different(row, other):
    return (row["relation_id"] != other["relation_id"]
            and row["subject_normalized"] != other["subject_normalized"])


def next_positive(anchor_index, rows, by_relation, seed=SEED):
    anchor = rows[anchor_index]
    values = [index for index in by_relation[(int(anchor["outer_fold"]),
                                               anchor["relation_id"])]
              if index != anchor_index
              and rows[index]["subject_normalized"] != anchor["subject_normalized"]]
    return min(values, key=lambda index: stable(
        seed, "positive", anchor["source_row_sha256"], rows[index]["source_row_sha256"])) \
        if values else None


def negative_candidates(anchor_index, rows, by_fold, by_trigram, seed=SEED):
    anchor = rows[anchor_index]
    fold = int(anchor["outer_fold"])
    lexical = set()
    for value in trigrams(row_surface(anchor)):
        lexical.update(by_trigram[(fold, value)])
    lexical = [index for index in lexical if different(anchor, rows[index])]
    lexical = sorted(lexical, key=lambda index: stable(
        seed, "hard-pool", anchor["source_row_sha256"],
        rows[index]["source_row_sha256"]))[:HARD_CANDIDATES]
    if not lexical:
        lexical = [index for index in by_fold[fold] if different(anchor, rows[index])]
        lexical = sorted(lexical, key=lambda index: stable(
            seed, "hard-fallback", anchor["source_row_sha256"],
            rows[index]["source_row_sha256"]))[:HARD_CANDIDATES]
    hard = max(lexical, key=lambda index: (
        trigram_jaccard(row_surface(anchor), row_surface(rows[index])),
        stable(seed, "hard-tie", rows[index]["source_row_sha256"])))
    ordered = by_fold[fold]
    start = int(stable(seed, "random-start", anchor["source_row_sha256"])[:12], 16) \
        % len(ordered)
    random_index = None
    for offset in range(len(ordered)):
        index = ordered[(start + offset) % len(ordered)]
        if index != hard and different(anchor, rows[index]):
            random_index = index
            break
    if random_index is None:
        raise ValueError("unable to construct different-relation negatives")
    return hard, random_index


def pair_row(kind, anchor, other, left, right, label, seed=SEED):
    pair_id = stable(seed, kind, anchor["source_row_sha256"],
                     other["source_row_sha256"] if other else "within", left, right)[:24]
    return {
        "pair_id": pair_id,
        "kind": kind,
        "outer_fold": int(anchor["outer_fold"]),
        "anchor_row_id": anchor["row_id"],
        "other_row_id": other["row_id"] if other else anchor["row_id"],
        "anchor_source_row_sha256": anchor["source_row_sha256"],
        "other_source_row_sha256": (other["source_row_sha256"] if other
                                     else anchor["source_row_sha256"]),
        "anchor_source": anchor["source"],
        "other_source": other["source"] if other else anchor["source"],
        "anchor_relation_id": anchor["relation_id"],
        "other_relation_id": other["relation_id"] if other else anchor["relation_id"],
        "left_surface": left,
        "right_surface": right,
        "same_relation": int(label),
        "locality_risk": int(not label),
        "character_trigram_jaccard": trigram_jaccard(left, right),
    }


def construct(rows, seed=SEED, cap=RELATION_CAP):
    anchors, exclusions = retained_anchors(rows, seed, cap)
    by_fold, by_relation, by_trigram = candidate_indices(anchors)
    pairs = []
    for index, anchor in enumerate(anchors):
        surface0, surface1 = row_surface(anchor, 0), row_surface(anchor, 1)
        if surface1 and normalize(surface0) != normalize(surface1):
            pairs.append(pair_row("within_row_positive", anchor, None,
                                  surface0, surface1, True, seed))
        positive = next_positive(index, anchors, by_relation, seed)
        if positive is not None:
            other = anchors[positive]
            pairs.append(pair_row("cross_row_positive", anchor, other,
                                  surface0, row_surface(other), True, seed))
        hard, random_index = negative_candidates(
            index, anchors, by_fold, by_trigram, seed)
        for kind, other_index in (("lexical_hard_negative", hard),
                                  ("random_negative", random_index)):
            other = anchors[other_index]
            pairs.append(pair_row(kind, anchor, other, surface0,
                                  row_surface(other), False, seed))
    unique = {}
    for row in pairs:
        identity = (row["kind"], row["anchor_source_row_sha256"],
                    row["other_source_row_sha256"], row["left_surface"],
                    row["right_surface"])
        unique.setdefault(identity, row)
    pairs = sorted(unique.values(), key=lambda row: (row["outer_fold"], row["pair_id"]))
    surfaces = {}
    for row in pairs:
        for field in ("left_surface", "right_surface"):
            text = row[field]
            surface_id = hashlib.sha256(text.encode()).hexdigest()
            surfaces.setdefault(surface_id, {"surface_id": surface_id, "text": text})
            row[field.replace("surface", "surface_id")] = surface_id
    return pairs, sorted(surfaces.values(), key=lambda row: row["surface_id"]), exclusions


def counts(pairs, surfaces, exclusions):
    by_fold = {}
    for fold in range(5):
        rows = [row for row in pairs if row["outer_fold"] == fold]
        by_fold[str(fold)] = {
            "pairs": len(rows),
            "positive": sum(row["same_relation"] for row in rows),
            "negative": sum(not row["same_relation"] for row in rows),
            "locality_risk_positive": sum(row["locality_risk"] for row in rows),
            "relations": len({row["anchor_relation_id"] for row in rows}),
            "sources": sorted({row["anchor_source"] for row in rows}),
        }
    return {"pairs": len(pairs), "surfaces": len(surfaces),
            "excluded_by_relation_cap": len(exclusions), "by_fold": by_fold,
            "by_kind": dict(sorted(Counter(row["kind"] for row in pairs).items()))}


def build(output, inventory_root, headroom_root, seed=SEED, relation_cap=RELATION_CAP):
    output = Path(output)
    rows_path, dependencies = verify_release(inventory_root, headroom_root)
    pairs, surfaces, exclusions = construct(load_jsonl(rows_path), seed, relation_cap)
    summary = counts(pairs, surfaces, exclusions)
    if not all(value["positive"] >= 500 and value["negative"] >= 500
               and value["locality_risk_positive"] >= 500
               for value in summary["by_fold"].values()):
        raise RuntimeError("semantic pair fold coverage gate failed")
    output.mkdir(parents=True, exist_ok=False)
    files = {"pairs.jsonl": pairs, "surfaces.jsonl": surfaces,
             "excluded.jsonl": exclusions}
    hashes = {}
    for name, values in files.items():
        path = output / name
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in values))
        hashes[name] = sha(path)
    manifest = {
        "protocol": PROTOCOL,
        "seed": seed,
        "relation_cap": relation_cap,
        "hard_candidate_pool": HARD_CANDIDATES,
        "counts": summary,
        "files_sha256": hashes,
        "dependencies": dependencies,
        "relation_ids_are_training_labels_only": True,
        "model_outcomes_loaded": False,
        "embeddings_extracted": False,
        "sealed_benchmarks_loaded": False,
        "terminal_status": "frozen_natural_semantic_pairs_before_embeddings",
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inventory-root", required=True)
    parser.add_argument("--headroom-root", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--relation-cap", type=int, default=RELATION_CAP)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
