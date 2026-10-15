"""Independently reconstruct and audit the S4 pair-localized corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.inspect_pair_localization_s4_source import inspect
from tasks.build_pair_localized_relation_memory_s4 import (
    PROTOCOL,
    load_jsonl,
    localize_mention,
    summarize,
)
from tasks.build_retrieval_evidence_forgetting_s3 import PROTOCOL as S3_PROTOCOL
from tasks.pack_cbf_event_warmup import sha


ALLOWED_CANDIDATE_KEYS = {
    "candidate_id", "center_turn", "turn_indices", "model_text", "features",
}
ALLOWED_FEATURE_KEYS = {
    "arg1_present", "arg2_present", "distance_to_arg1", "distance_to_arg2",
    "normalized_position",
}


def candidate_schema_safe(candidate: dict) -> bool:
    if set(candidate) != ALLOWED_CANDIDATE_KEYS:
        return False
    if set(candidate["features"]) != ALLOWED_FEATURE_KEYS:
        return False
    forbidden = ("gold", "trigger", "relation", "entity", "split", "score", "logit")
    return (isinstance(candidate["model_text"], str)
            and not any(token in key.casefold() for key in candidate for token in forbidden)
            and not any(token in key.casefold()
                        for key in candidate["features"] for token in forbidden))


def audit(args) -> dict:
    dialogre_root, s3_root, s4_root = map(Path, (
        args.dialogre_root, args.s3_data_root, args.s4_data_root
    ))
    manifest_path = s4_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    s3_manifest_path, s3_audit_path = s3_root / "manifest.json", s3_root / "data_audit.json"
    s3_manifest = json.loads(s3_manifest_path.read_text())
    s3_audit = json.loads(s3_audit_path.read_text())
    source_inspection_path = Path(args.source_inspection)
    source_inspection = json.loads(source_inspection_path.read_text())
    reconstructed_inspection = inspect(dialogre_root)
    raw = json.loads((dialogre_root / "data_v2/en/data/train.json").read_text())

    row_reconstruction, candidate_schema, identifiers, counts, hashes = True, True, True, {}, {}
    seen_candidates = set()
    for split in ("train", "calibration"):
        s3_rows = load_jsonl(s3_root / f"{split}_mentions.jsonl")
        path = s4_root / f"{split}_localized_mentions.jsonl"
        rows = load_jsonl(path)
        hashes[path.name] = sha(path)
        if len(rows) != len(s3_rows):
            row_reconstruction = False
        for source_row, actual in zip(s3_rows, rows):
            expected = localize_mention(source_row, raw)
            row_reconstruction &= actual == expected
            identifiers &= actual["source_row_sha256"] == source_row["source_row_sha256"]
            identifiers &= actual["source_instance_sha256"] == source_row["source_instance_sha256"]
            identifiers &= actual["source_split"] == split
            candidate_schema &= all(candidate_schema_safe(item)
                                    for item in actual["candidates"])
            ids = [item["candidate_id"] for item in actual["candidates"]]
            identifiers &= len(ids) == len(set(ids))
            identifiers &= not (set(ids) & seen_candidates)
            seen_candidates.update(ids)
            if actual["source"] == "dialogre":
                candidate_schema &= [item["center_turn"] for item in actual["candidates"]] \
                    == list(range(actual["turn_count"]))
            candidate_schema &= all(
                0 <= index < actual["turn_count"]
                for item in actual["candidates"] for index in item["turn_indices"]
            )
            candidate_schema &= all(
                0 <= index < len(actual["candidates"])
                for index in actual["gold_trigger_candidate_indices_audit"]
            )
        counts[split] = summarize(rows)

    checks = {
        "protocol": manifest.get("protocol") == PROTOCOL,
        "source_inspection": source_inspection == reconstructed_inspection
            and manifest["source"]["source_inspection_sha256"] == sha(source_inspection_path),
        "s3_provenance": s3_manifest.get("protocol") == S3_PROTOCOL
            and s3_audit.get("passed") is True
            and manifest["source"]["s3_manifest_sha256"] == sha(s3_manifest_path)
            and manifest["source"]["s3_audit_sha256"] == sha(s3_audit_path),
        "row_reconstruction": row_reconstruction,
        "candidate_schema_and_gold_isolation": candidate_schema,
        "identifiers": identifiers,
        "counts": counts == manifest["counts"]
            and all(counts[split]["mentions"] == s3_manifest["counts"][split]["mentions"]
                    for split in ("train", "calibration")),
        "hashes": hashes == manifest["files_sha256"],
        "no_model_outputs": manifest.get("model_outputs_loaded") is False,
        "sealed": manifest.get("dialogre_dev_parsed") is False
            and manifest.get("dialogre_test_parsed") is False
            and manifest.get("pubmed_validation_parsed") is False
            and source_inspection.get("dialogre_dev_rows_parsed") == 0
            and source_inspection.get("dialogre_test_rows_parsed") == 0,
    }
    result = {
        "protocol": "qwen_pair_localized_relation_memory_s4_data_audit_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "counts": counts,
        "manifest_sha256": sha(manifest_path),
        "files_sha256": hashes,
        "dialogre_dev_rows_parsed": 0,
        "dialogre_test_rows_parsed": 0,
    }
    if not result["passed"]:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogre-root", required=True)
    parser.add_argument("--s3-data-root", required=True)
    parser.add_argument("--s4-data-root", required=True)
    parser.add_argument("--source-inspection", required=True)
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
