"""Independently audit S1 data and sealing without importing its builder."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


PROTOCOL = "qwen_structured_fact_comparator_s1"
KINDS = {
    "duplicate", "update", "same_relation_other_entity",
    "same_entity_other_relation", "unrelated",
}
EXPECTED_FIELDS = {
    "duplicate": (1, 1, 1, "duplicate"),
    "update": (1, 1, 0, "update"),
    "same_relation_other_entity": (0, 1, None, "retain"),
    "same_entity_other_relation": (1, 0, None, "retain"),
    "unrelated": (0, 0, 0, "retain"),
}
SOURCE_HASHES = {
    "semeval_rel2id.json": "c3153e7d20a127c82fbc9b7542fcdc8e48714bcf9165fafd611dadaed22c7de8",
    "semeval_train.txt": "5a53be5e2dfbf865f77c287deec7b9e8a12be333ffb6157bfb5a692250a06220",
    "semeval_val.txt": "a91170e7c19354a895a58e582bf7fad43994467a0d27a8632319a70840b88e02",
    "semeval_test.txt": "1f33efe3db4dc6ff99f52416331682f3ec6759de526786d3629e6b43f8020795",
}
SOURCE_ROWS = {"semeval_train.txt": 6507, "semeval_val.txt": 1493,
               "semeval_test.txt": 2717}
FEWREL_HASHES = {
    "train_wiki.json": "8e9e58a05d559be773cbe18a8d4a16ba17e6d397663f9f05a056f894a94b92c1",
    "val_wiki.json": "76bb44c2485b6faed358a3ce040eadafcc8a8835245cd7e6afc1bef672d4a0bf",
}


def file_sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def stable(*values) -> str:
    return hashlib.sha256(":".join(str(value) for value in values).encode()).hexdigest()


def expected_train_relations(train: dict, external: dict) -> set[str]:
    ordered = sorted(train, key=lambda value: hashlib.sha256(
        f"2311:relation:{value}".encode()).hexdigest())
    if len(ordered) != 64 or len(external) != 16:
        raise ValueError("unexpected FewRel relation inventory")
    return set(ordered[:48])


def audit(data_root: str | Path, fewrel_root: str | Path,
          semeval_root: str | Path) -> dict:
    data_root, fewrel_root, semeval_root = map(Path, (data_root, fewrel_root, semeval_root))
    manifest = json.loads((data_root / "manifest.json").read_text())
    checks = {}
    checks["protocol"] = manifest["protocol"] == PROTOCOL
    checks["fixed_rule"] = manifest["fixed_operation_rule"] is True
    checks["model_free"] = manifest["model_outputs_loaded"] is False
    checks["test_sealed"] = (manifest["semeval_test_parsed"] is False
                              and manifest["semeval_test_examples_written"] is False
                              and not (data_root / "test.jsonl").exists())
    checks["fewrel_nontrain_absent"] = manifest["fewrel_nontrain_examples_written"] is False

    source_ok, source_rows_ok = True, True
    for name, expected in SOURCE_HASHES.items():
        source_ok &= file_sha(semeval_root / name) == expected
        if name in SOURCE_ROWS:
            # Never JSON-decode the sealed test: this is byte-level cardinality only.
            count = sum(1 for line in (semeval_root / name).open("rb") if line.strip())
            source_rows_ok &= count == SOURCE_ROWS[name]
    checks["semeval_source_hashes"] = source_ok
    checks["semeval_source_rows"] = source_rows_ok
    checks["fewrel_source_hashes"] = all(
        file_sha(fewrel_root / "data" / name) == expected
        for name, expected in FEWREL_HASHES.items())

    train_source = json.loads((fewrel_root / "data/train_wiki.json").read_text())
    external_source = json.loads((fewrel_root / "data/val_wiki.json").read_text())
    expected_relations = expected_train_relations(train_source, external_source)

    split_rows, all_ids = {}, set()
    per_split_checks = {}
    source_hashes = {}
    for split in ("train", "calibration", "development"):
        path = data_root / f"{split}.jsonl"
        rows = load_jsonl(path)
        split_rows[split] = rows
        row_ids = [row["pair_id"] for row in rows]
        source_hashes[split] = {row["left_source_row_sha256"] for row in rows}
        grouped = defaultdict(list)
        valid = True
        for row in rows:
            grouped[row["left_source_row_sha256"]].append(row)
            expected_id = stable(PROTOCOL, manifest["seed"], split, row["kind"],
                                 row["left_source_row_sha256"],
                                 row["right_source_row_sha256"])[:24]
            expected = EXPECTED_FIELDS.get(row["kind"])
            actual = (row["same_entity"], row["same_relation"],
                      row["same_value"], row["operation"])
            field_ok = (expected is not None and actual[0] == expected[0]
                        and actual[1] == expected[1] and actual[3] == expected[3]
                        and (expected[2] is None or actual[2] == expected[2]))
            relation_ok = ((row["left_relation_audit"] == row["right_relation_audit"])
                           == bool(row["same_relation"]))
            tags_ok = all(text.count("[SUBJECT]") == 1
                          and text.count("[/SUBJECT]") == 1
                          and text.count("[VALUE]") == 1
                          and text.count("[/VALUE]") == 1
                          for text in (row["left_text"], row["right_text"]))
            valid &= (row["split"] == split and row["source"]
                      == ("fewrel" if split == "train" else "semeval")
                      and expected_id == row["pair_id"] and field_ok
                      and relation_ok and tags_ok)
        group_ok = all({row["kind"] for row in values} == KINDS
                       and len(values) == len(KINDS)
                       and len({row["left_text"] for row in values}) == 1
                       for values in grouped.values())
        counts = manifest["counts"][split]
        count_ok = (len(rows) == counts["pairs"]
                    and len(grouped) == counts["anchors"]
                    and Counter(row["kind"] for row in rows) == Counter(counts["by_kind"])
                    and Counter(row["operation"] for row in rows)
                    == Counter(counts["by_operation"]))
        hash_ok = file_sha(path) == manifest["files_sha256"][path.name]
        unique_ok = len(row_ids) == len(set(row_ids)) and not (all_ids & set(row_ids))
        all_ids.update(row_ids)
        per_split_checks[split] = {
            "rows_valid": bool(valid), "groups_complete": group_ok,
            "counts": count_ok, "hash": hash_ok, "unique_ids": unique_ok,
        }
    checks["split_rows"] = per_split_checks
    checks["fewrel_train_relations"] = ({row["left_relation_audit"]
                                          for row in split_rows["train"]}
                                         == expected_relations)
    checks["source_rows_disjoint"] = all(
        not (source_hashes[left] & source_hashes[right])
        for left, right in (("train", "calibration"), ("train", "development"),
                            ("calibration", "development")))
    excluded = manifest["eligibility_exclusions"]
    checks["single_registered_exclusion"] = (
        not excluded["train"] and not excluded["calibration"]
        and len(excluded["development"]) == 1
        and excluded["development"][0]["relation"] == "Entity-Destination(e2,e1)"
        and excluded["development"][0]["reason"]
        == "undercovered_relation_for_structured_pair")

    def flatten(value):
        if isinstance(value, dict):
            return all(flatten(item) for item in value.values())
        return bool(value)
    passed = flatten(checks)
    result = {"protocol": "qwen_structured_fact_comparator_s1_data_audit_v1",
              "passed": passed, "checks": checks,
              "rows_audited": sum(len(rows) for rows in split_rows.values()),
              "test_rows_parsed": 0,
              "manifest_sha256": file_sha(data_root / "manifest.json")}
    if not passed:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fewrel-root", required=True)
    parser.add_argument("--semeval-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(args.data_root, args.fewrel_root, args.semeval_root)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
