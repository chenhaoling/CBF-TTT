"""Build exact two-record inputs for the hku-gpu2 sparse resource ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_2x5090_resource_inputs_v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "IN_PLACE_TTT_BASELINE_2X5090_RESOURCE_INPUT_ADDENDUM.md"
SPECS = {
    8192: "stage1_32k",
    16384: "stage1_32k",
    32768: "stage1_32k",
    65536: "stage2_128k",
    131072: "stage2_128k",
}
SOURCES = ("fineweb", "longcrawl")
SOURCE_PROFILES = ("pilot", "preliminary_100m", "full")
SOURCE_MANIFEST_SHA256 = "3b2e7f4821692dfe75bae48e4cb52c6ac39f4b9b3b4c75f96aeb32ab7dec7206"
ASSETS = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "added_tokens.json", "config.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def token_hash(ids: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def tokenizer_assets(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for name in ASSETS:
        asset = path / name
        if asset.is_file():
            result[name] = {"sha256": sha256_file(asset), "size_bytes": asset.stat().st_size}
    for required in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        if required not in result:
            raise FileNotFoundError(path / required)
    return result


def has_plaintext(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in {"text", "content_split"} or has_plaintext(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(has_plaintext(item) for item in value)
    return False


def stable_prefix(text: str, tokenizer: Any, target_content: int) -> dict[str, Any]:
    parent_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(parent_ids) < target_content:
        raise ValueError("parent record is shorter than requested target")
    if len(parent_ids) == target_content:
        output_text, ids, raw_count = text, parent_ids, target_content
    else:
        raw_count = target_content
        while raw_count:
            output_text = tokenizer.decode(
                parent_ids[:raw_count], skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            ids = tokenizer.encode(output_text, add_special_tokens=False)
            if len(ids) <= target_content:
                break
            raw_count -= 1
        if raw_count == 0:
            raise ValueError("no stable tokenizer prefix")
    synthetic_eos = target_content - len(ids)
    if synthetic_eos:
        output_text += tokenizer.eos_token * synthetic_eos
        ids = tokenizer.encode(output_text, add_special_tokens=False)
    if len(ids) != target_content:
        raise ValueError("EOS padding did not produce the exact target")
    training_ids = ids + [tokenizer.eos_token_id]
    return {
        "text": output_text,
        "raw_parent_tokens_consumed": raw_count,
        "synthetic_eos_tokens": synthetic_eos,
        "content_token_sha256": token_hash(ids),
        "training_token_sha256": token_hash(training_ids),
        "training_ids": training_ids,
    }


def first_pair(data_path: Path, provenance_path: Path) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    with data_path.open(encoding="utf-8") as data, \
            provenance_path.open(encoding="utf-8") as provenance:
        for expected_source in SOURCES:
            data_line, provenance_line = data.readline(), provenance.readline()
            if not data_line or not provenance_line:
                raise ValueError("parent stage has fewer than two records")
            row, record = json.loads(data_line), json.loads(provenance_line)
            if (set(row) != {"content_split"}
                    or not isinstance(row["content_split"], str)
                    or record.get("source") != expected_source
                    or record.get("stage_record_index") != len(rows)):
                raise ValueError("parent first-pair order/content is invalid")
            rows.append((row["content_split"], record))
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).expanduser().resolve()
    source_manifest_path = source_root / "manifest.json"
    source_audit_path = Path(args.source_audit).expanduser().resolve()
    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    plan_path = Path(args.plan).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    partial = output_root.with_name(output_root.name + ".partial")
    if output_root.exists() or partial.exists():
        raise FileExistsError("refusing to overwrite resource inputs")
    partial.mkdir(parents=True)
    try:
        source_manifest = json.loads(source_manifest_path.read_text())
        source_audit = json.loads(source_audit_path.read_text())
        source_manifest_sha = sha256_file(source_manifest_path)
        expected_source_sha = args.expected_source_manifest_sha256
        required_profile = args.required_source_profile
        if (source_manifest.get("protocol") != "in_place_ttt_public_corpus_v1"
                or source_manifest.get("profile") != required_profile
                or source_manifest_sha != expected_source_sha
                or source_audit.get("passed") is not True
                or source_audit.get("bindings", {}).get("manifest_sha256")
                != source_manifest_sha):
            raise ValueError("source public-corpus pilot/audit is invalid")

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=True, fix_mistral_regex=False, local_files_only=True,
        )
        if tokenizer.encode(tokenizer.eos_token, add_special_tokens=False) \
                != [tokenizer.eos_token_id]:
            raise ValueError("tokenizer EOS is not stable")
        stage_records = {stage["name"]: stage for stage in source_manifest["stages"]}
        parent_pairs = {}
        parent_artifacts = {}
        for stage_name in set(SPECS.values()):
            stage = stage_records[stage_name]
            data_path = source_root / f"{stage_name}.jsonl"
            provenance_path = source_root / f"{stage_name}.provenance.jsonl"
            for path in (data_path, provenance_path):
                saved = stage["artifacts"][path.name]
                if (sha256_file(path) != saved["sha256"]
                        or path.stat().st_size != saved["size_bytes"]):
                    raise ValueError(f"parent artifact drift: {path.name}")
            parent_pairs[stage_name] = first_pair(data_path, provenance_path)
            parent_artifacts[stage_name] = {
                data_path.name: stage["artifacts"][data_path.name],
                provenance_path.name: stage["artifacts"][provenance_path.name],
            }

        outputs = {}
        for seq_len, stage_name in SPECS.items():
            output_path = partial / f"resource_{seq_len}.jsonl"
            stream_hasher = hashlib.sha256()
            records = []
            with output_path.open("x", encoding="utf-8") as sink:
                for source, (text, parent_provenance) in zip(
                        SOURCES, parent_pairs[stage_name], strict=True):
                    derived = stable_prefix(text, tokenizer, seq_len - 1)
                    sink.write(json.dumps(
                        {"content_split": derived.pop("text")}, ensure_ascii=False
                    ) + "\n")
                    training_ids = derived.pop("training_ids")
                    stream_hasher.update(struct.pack(f"<{len(training_ids)}I", *training_ids))
                    records.append({
                        "source": source,
                        "parent_stage_record_index": parent_provenance["stage_record_index"],
                        "parent_training_token_sha256":
                            parent_provenance["training_token_sha256"],
                        **derived,
                    })
            outputs[str(seq_len)] = {
                "parent_stage": stage_name,
                "records": 2,
                "training_tokens": 2 * seq_len,
                "ordered_training_tokens_sha256": stream_hasher.hexdigest(),
                "record_metadata": records,
                "artifact": {
                    "path": output_path.name,
                    "sha256": sha256_file(output_path),
                    "size_bytes": output_path.stat().st_size,
                },
            }
        manifest = {
            "protocol": PROTOCOL,
            "bindings": {
                "plan_sha256": sha256_file(plan_path),
                "builder_sha256": sha256_file(Path(__file__).resolve()),
                "source_manifest_sha256": source_manifest_sha,
                "source_audit_sha256": sha256_file(source_audit_path),
                "required_source_profile": required_profile,
                "expected_source_manifest_sha256": expected_source_sha,
                "tokenizer_assets": tokenizer_assets(tokenizer_path),
                "parent_artifacts": parent_artifacts,
            },
            "outputs": outputs,
        }
        if has_plaintext(manifest):
            raise AssertionError("plaintext leaked into manifest")
        manifest_path = partial / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        partial.replace(output_root)
        return manifest
    except BaseException:
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-audit", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument(
        "--expected-source-manifest-sha256",
        default=SOURCE_MANIFEST_SHA256,
        help="Exact audited source manifest hash (legacy pilot hash by default).",
    )
    parser.add_argument(
        "--required-source-profile", choices=SOURCE_PROFILES, default="pilot",
        help="Require the source corpus profile used to derive resource inputs.",
    )
    args = parser.parse_args()
    result = build(args)
    print(json.dumps({"protocol": result["protocol"],
                      "outputs": sorted(result["outputs"], key=int)}, sort_keys=True))


if __name__ == "__main__":
    main()
