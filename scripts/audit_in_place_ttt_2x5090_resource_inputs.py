"""Independently audit exact inputs for the hku-gpu2 resource ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from itertools import zip_longest
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_2x5090_resource_inputs_audit_v1"
SOURCE_MANIFEST_SHA256 = "3b2e7f4821692dfe75bae48e4cb52c6ac39f4b9b3b4c75f96aeb32ab7dec7206"
SPECS = {
    8192: "stage1_32k",
    16384: "stage1_32k",
    32768: "stage1_32k",
    65536: "stage2_128k",
    131072: "stage2_128k",
}
SOURCES = ("fineweb", "longcrawl")
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
    return {
        name: {"sha256": sha256_file(path / name),
               "size_bytes": (path / name).stat().st_size}
        for name in ASSETS if (path / name).is_file()
    }


def forbidden(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in {"text", "content_split"} or forbidden(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(forbidden(item) for item in value)
    return False


def derive_expected(text: str, tokenizer: Any, target: int) -> dict[str, Any]:
    parent = tokenizer.encode(text, add_special_tokens=False)
    if len(parent) < target:
        raise ValueError("parent record is too short")
    if len(parent) == target:
        output, ids, consumed = text, parent, target
    else:
        consumed = target
        while consumed:
            output = tokenizer.decode(
                parent[:consumed], skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            ids = tokenizer.encode(output, add_special_tokens=False)
            if len(ids) <= target:
                break
            consumed -= 1
        if consumed == 0:
            raise ValueError("stable parent prefix does not exist")
    padding = target - len(ids)
    if padding:
        output += tokenizer.eos_token * padding
        ids = tokenizer.encode(output, add_special_tokens=False)
    if len(ids) != target:
        raise ValueError("independent EOS padding failed")
    training = ids + [tokenizer.eos_token_id]
    return {
        "text": output,
        "raw_parent_tokens_consumed": consumed,
        "synthetic_eos_tokens": padding,
        "content_token_sha256": token_hash(ids),
        "training_token_sha256": token_hash(training),
        "training_ids": training,
    }


def read_first_pair(data_path: Path, provenance_path: Path) -> list[tuple[str, dict[str, Any]]]:
    result = []
    with data_path.open(encoding="utf-8") as data, \
            provenance_path.open(encoding="utf-8") as provenance:
        for index, source in enumerate(SOURCES):
            data_line, provenance_line = data.readline(), provenance.readline()
            if not data_line or not provenance_line:
                raise ValueError("parent stage lacks its first source pair")
            row, record = json.loads(data_line), json.loads(provenance_line)
            if (set(row) != {"content_split"}
                    or not isinstance(row.get("content_split"), str)
                    or record.get("source") != source
                    or record.get("stage_record_index") != index):
                raise ValueError("parent source order is invalid")
            result.append((row["content_split"], record))
    return result


def audit(root: Path, source_root: Path, source_audit_path: Path,
          tokenizer_path: Path, plan_path: Path, builder_path: Path,
          expected_source_manifest_sha256: str = SOURCE_MANIFEST_SHA256,
          required_source_profile: str = "pilot") -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, bool] = {}
    manifest_path = root / "manifest.json"
    required = [manifest_path] + [root / f"resource_{length}.jsonl" for length in SPECS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {"protocol": PROTOCOL, "passed": False,
                "errors": [f"missing artifacts: {missing}"]}
    manifest = json.loads(manifest_path.read_text())
    source_manifest_path = source_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_audit = json.loads(source_audit_path.read_text())
    source_sha = sha256_file(source_manifest_path)
    checks["protocol_and_paths"] = (
        manifest.get("protocol") == "in_place_ttt_2x5090_resource_inputs_v1"
        and not root.name.endswith(".partial")
        and not root.with_name(root.name + ".partial").exists()
        and set(manifest.get("outputs", {})) == {str(value) for value in SPECS}
    )
    checks["source_corpus"] = (
        source_sha == expected_source_manifest_sha256
        and source_manifest.get("protocol") == "in_place_ttt_public_corpus_v1"
        and source_manifest.get("profile") == required_source_profile
        and source_audit.get("passed") is True
        and source_audit.get("bindings", {}).get("manifest_sha256") == source_sha
    )
    bindings = manifest.get("bindings", {})
    checks["code_plan_and_tokenizer"] = (
        bindings.get("plan_sha256") == sha256_file(plan_path)
        and bindings.get("builder_sha256") == sha256_file(builder_path)
        and bindings.get("source_manifest_sha256") == source_sha
        and bindings.get("source_audit_sha256") == sha256_file(source_audit_path)
        and bindings.get("required_source_profile") == required_source_profile
        and bindings.get("expected_source_manifest_sha256")
        == expected_source_manifest_sha256
        and bindings.get("tokenizer_assets") == tokenizer_assets(tokenizer_path)
    )
    checks["no_plaintext_metadata"] = not forbidden(manifest)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=True, fix_mistral_regex=False, local_files_only=True,
    )
    checks["tokenizer_eos"] = (
        tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)
        == [tokenizer.eos_token_id]
    )
    stage_records = {stage["name"]: stage for stage in source_manifest["stages"]}
    pairs = {}
    parent_valid = True
    for stage_name in set(SPECS.values()):
        data_path = source_root / f"{stage_name}.jsonl"
        provenance_path = source_root / f"{stage_name}.provenance.jsonl"
        saved = stage_records[stage_name]["artifacts"]
        observed = {}
        for path in (data_path, provenance_path):
            observed[path.name] = {
                "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
            }
            parent_valid &= observed[path.name] == saved[path.name]
        parent_valid &= bindings.get("parent_artifacts", {}).get(stage_name) == observed
        pairs[stage_name] = read_first_pair(data_path, provenance_path)
    checks["parent_artifacts_and_order"] = parent_valid

    output_valid = True
    observed_outputs = {}
    for length, stage_name in SPECS.items():
        output_path = root / f"resource_{length}.jsonl"
        saved = manifest["outputs"].get(str(length), {})
        stream = hashlib.sha256()
        records = []
        with output_path.open(encoding="utf-8") as handle:
            output_rows = [json.loads(line) for line in handle if line.strip()]
        if len(output_rows) != 2:
            output_valid = False
        for index, values in enumerate(zip_longest(
                output_rows, pairs[stage_name], SOURCES)):
            row, parent, source = values
            if row is None or parent is None or source is None:
                output_valid = False
                continue
            if not isinstance(row, dict):
                output_valid = False
                continue
            parent_text, parent_provenance = parent
            expected = derive_expected(parent_text, tokenizer, length - 1)
            text = row.get("content_split") if isinstance(row, dict) else None
            if set(row) != {"content_split"} or text != expected.pop("text"):
                output_valid = False
                continue
            training_ids = tokenizer.encode(text, add_special_tokens=False) \
                + [tokenizer.eos_token_id]
            expected_ids = expected.pop("training_ids")
            if training_ids != expected_ids or len(training_ids) != length:
                output_valid = False
            stream.update(struct.pack(f"<{len(training_ids)}I", *training_ids))
            record = {
                "source": source,
                "parent_stage_record_index": parent_provenance["stage_record_index"],
                "parent_training_token_sha256":
                    parent_provenance["training_token_sha256"],
                **expected,
            }
            records.append(record)
        artifact = {"path": output_path.name, "sha256": sha256_file(output_path),
                    "size_bytes": output_path.stat().st_size}
        observed = {
            "parent_stage": stage_name,
            "records": len(output_rows),
            "training_tokens": len(output_rows) * length,
            "ordered_training_tokens_sha256": stream.hexdigest(),
            "record_metadata": records,
            "artifact": artifact,
        }
        observed_outputs[str(length)] = observed
        output_valid &= observed == saved
        if length in {32768, 131072}:
            output_valid &= all(
                output_rows[index]["content_split"] == pairs[stage_name][index][0]
                for index in range(min(len(output_rows), 2))
            )
    checks["exact_outputs"] = output_valid
    for name, passed in checks.items():
        if not passed:
            errors.append(f"failed check: {name}")
    return {
        "protocol": PROTOCOL,
        "passed": not errors,
        "errors": errors,
        "checks": checks,
        "outputs": {
            length: {
                "artifact_sha256": value["artifact"]["sha256"],
                "ordered_training_tokens_sha256":
                    value["ordered_training_tokens_sha256"],
                "records": value["records"],
                "training_tokens": value["training_tokens"],
            }
            for length, value in observed_outputs.items()
        },
        "bindings": {
            "manifest_sha256": sha256_file(manifest_path),
            "source_manifest_sha256": source_sha,
            "source_audit_sha256": sha256_file(source_audit_path),
            "required_source_profile": required_source_profile,
            "expected_source_manifest_sha256": expected_source_manifest_sha256,
            "plan_sha256": sha256_file(plan_path),
            "builder_sha256": sha256_file(builder_path),
            "auditor_sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument(
        "--expected-source-manifest-sha256",
        default=SOURCE_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--required-source-profile", choices=("pilot", "full"), default="pilot",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve() \
        if args.output else root / "independent_audit.json"
    result = audit(
        root,
        source_root=args.source_root.expanduser().resolve(),
        source_audit_path=args.source_audit.expanduser().resolve(),
        tokenizer_path=args.tokenizer.expanduser().resolve(),
        plan_path=args.plan.expanduser().resolve(),
        builder_path=args.builder.expanduser().resolve(),
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        required_source_profile=args.required_source_profile,
    )
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    partial.replace(output)
    print(json.dumps({"output": str(output), "passed": result["passed"],
                      "errors": result["errors"]}, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
