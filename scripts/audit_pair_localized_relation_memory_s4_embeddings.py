"""Audit completeness, provenance, and normalization of all S4 embeddings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tasks.extract_pair_localized_relation_memory_s4_embeddings import (
    DATASETS,
    PROTOCOL,
    load_rows,
)
from tasks.pack_cbf_event_warmup import sha


def audit(data_root: str | Path, embedding_root: str | Path) -> dict:
    import numpy as np

    data_root, embedding_root = Path(data_root), Path(embedding_root)
    checks, datasets, all_pass = {}, {}, True
    for dataset in DATASETS:
        expected, _ = load_rows(data_root, dataset)
        values, shard_audits, shard_hashes = [], [], {}
        protocol_ok = resource_ok = sealed = True
        for shard in range(2):
            path = embedding_root / f"{dataset}_shard_{shard}.npz"
            audit_path = embedding_root / f"{dataset}_shard_{shard}.audit.json"
            shard_audit = json.loads(audit_path.read_text())
            protocol_ok &= (shard_audit.get("protocol") == PROTOCOL
                            and shard_audit.get("dataset") == dataset
                            and shard_audit.get("shard") == shard
                            and shard_audit.get("num_shards") == 2
                            and shard_audit.get("truncation") is False
                            and shard_audit.get("ttt_mode") is False
                            and sha(path) == shard_audit.get("output_sha256"))
            resource_ok &= (math.isfinite(shard_audit.get("seconds", math.nan))
                            and shard_audit.get("peak_reserved_gib", math.inf) < 28
                            and shard_audit.get("length_max", math.inf)
                            <= shard_audit.get("max_length", -1))
            sealed &= (shard_audit.get("dialogre_dev_loaded") is False
                       and shard_audit.get("dialogre_test_loaded") is False
                       and shard_audit.get("pubmed_validation_loaded") is False)
            with np.load(path) as arrays:
                # NPZ members are compressed independently.  Read each member
                # once; repeated ``arrays[key][index]`` access would decompress
                # the complete member once per row.
                values.extend(zip(
                    arrays["indices"].astype("int64").tolist(),
                    arrays["view_ids"].tolist(), arrays["source_ids"].tolist(),
                    arrays["relations"].tolist(), arrays["sources"].tolist(),
                    arrays["dialogue_ids"].tolist(), arrays["views"].tolist(),
                    arrays["center_turns"].astype("int64").tolist(),
                    arrays["side_features"].astype("float32"),
                    arrays["embeddings"].astype("float32"),
                ))
            shard_audits.append(shard_audit)
            shard_hashes[str(shard)] = sha(path)
        values.sort(key=lambda row: row[0])
        indices_ok = (len(values) == len(expected)
                      and [row[0] for row in values] == list(range(len(expected))))
        metadata_ok = indices_ok and all(
            actual[1] == wanted["view_id"]
            and actual[2] == wanted["source_id"]
            and actual[3] == wanted["relation"]
            and actual[4] == wanted["source"]
            and actual[5] == wanted["dialogue_id"]
            and actual[6] == wanted["view"]
            and actual[7] == wanted["center_turn"]
            and np.allclose(actual[8], wanted["features"], atol=1e-7)
            for actual, wanted in zip(values, expected)
        )
        finite = all(np.isfinite(row[8]).all() and np.isfinite(row[9]).all()
                     for row in values)
        normalized = all(abs(float(np.linalg.norm(row[9])) - 1.0) <= .002
                         for row in values)
        dataset_checks = {
            "protocol_and_hash": protocol_ok,
            "complete_indices": indices_ok,
            "metadata": metadata_ok,
            "finite": finite,
            "normalized": normalized,
            "resource": resource_ok,
            "sealed": sealed,
        }
        checks[dataset] = dataset_checks
        all_pass &= all(dataset_checks.values())
        datasets[dataset] = {
            "rows": len(values),
            "dimension": len(values[0][9]) if values else 0,
            "shard_sha256": shard_hashes,
            "seconds": [value["seconds"] for value in shard_audits],
            "peak_reserved_gib": [value["peak_reserved_gib"] for value in shard_audits],
            "length_max": [value["length_max"] for value in shard_audits],
        }
    result = {
        "protocol": "qwen_pair_localized_relation_memory_s4_embeddings_audit_v1",
        "passed": all_pass,
        "checks": checks,
        "datasets": datasets,
        "data_manifest_sha256": sha(data_root / "manifest.json"),
        "data_audit_sha256": sha(data_root / "data_audit.json"),
        "dialogre_dev_rows_parsed": 0,
        "dialogre_test_rows_parsed": 0,
    }
    if not all_pass:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = audit(args.data_root, args.embedding_root)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
