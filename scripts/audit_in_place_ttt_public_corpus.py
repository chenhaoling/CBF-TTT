"""Independently audit a two-stage Level-B public corpus or frozen pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from itertools import zip_longest
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_public_corpus_v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "IN_PLACE_TTT_BASELINE_PUBLIC_CORPUS_PLAN.md"
DEFAULT_BUILDER = ROOT / "scripts/build_in_place_ttt_public_corpus.py"
PROFILES = {
    "pilot": (("stage1_32k", 32768, 64), ("stage2_128k", 131072, 8)),
    "preliminary_100m": (
        ("stage1_32k", 32768, 864),
        ("stage2_128k", 131072, 160),
    ),
    "full": (("stage1_32k", 32768, 305184), ("stage2_128k", 131072, 57216)),
}
TRAINING_PROFILES = frozenset({"preliminary_100m", "full"})
SOURCES = ("fineweb", "longcrawl")
ASSETS = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "added_tokens.json", "config.json")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def token_bytes(ids: list[int]) -> bytes:
    return struct.pack(f"<{len(ids)}I", *ids)


def token_sha(ids: list[int]) -> str:
    return hashlib.sha256(token_bytes(ids)).hexdigest()


def forbidden(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in {"text", "content_split"} or forbidden(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(forbidden(item) for item in value)
    return False


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error


def document_entry(span: dict[str, Any]) -> dict[str, Any]:
    return {key: span[key] for key in (
        "source", "document_index", "locator", "document_sha256",
        "content_token_count", "token_count_with_boundary",
    )}


class SourceContinuity:
    def __init__(self, source: str, inventory_source: dict[str, Any]):
        self.source = source
        self.inventory_source = inventory_source
        # Only unfinished/prefetched documents are needed to validate the
        # next span. Keeping every past entry makes formal-scale audit memory
        # grow with millions of FineWeb documents.
        self.entries: dict[int, dict[str, Any]] = {}
        self.cursors: dict[int, int] = {}
        self.documents_seen = 0
        self.peak_active_documents = 0
        self.hasher = hashlib.sha256()
        self.last_document_index = -1
        self.last_locator: tuple[int, int] | None = None

    def _add_entry(self, entry: dict[str, Any], start: int) -> bool:
        index = entry.get("document_index")
        locator = entry.get("locator")
        valid = (
            entry.get("source") == self.source
            and isinstance(index, int) and index == self.documents_seen
            and start == 0 and isinstance(locator, dict)
            and isinstance(entry.get("document_sha256"), str)
            and len(entry["document_sha256"]) == 64
            and isinstance(entry.get("content_token_count"), int)
            and entry.get("token_count_with_boundary") == entry["content_token_count"] + 1
        )
        if not valid:
            return False
        file_index = locator.get("file_index")
        row_index = locator.get("parquet_row_index")
        files = self.inventory_source.get("files", [])
        if (not isinstance(file_index, int) or not 0 <= file_index < len(files)
                or locator.get("repository_path") != files[file_index].get("repository_path")
                or not isinstance(row_index, int) or row_index < 0):
            return False
        location = (file_index, row_index)
        if self.last_locator is not None and location <= self.last_locator:
            return False
        identity = locator.get("source_identity_sha256")
        if identity is not None and (not isinstance(identity, str) or len(identity) != 64):
            return False
        self.last_locator = location
        self.entries[index] = entry
        self.cursors[index] = 0
        self.documents_seen += 1
        self.peak_active_documents = max(
            self.peak_active_documents, len(self.entries)
        )
        self.hasher.update((canonical(entry) + "\n").encode("utf-8"))
        return True

    def consume_span(self, span: dict[str, Any]) -> tuple[bool, int]:
        try:
            entry = document_entry(span)
            index, start, end = span["document_index"], span["token_start"], span["token_end"]
            total = span["token_count_with_boundary"]
        except (KeyError, TypeError):
            return False, 0
        if not isinstance(index, int) or not isinstance(start, int) or not isinstance(end, int):
            return False, 0
        if index not in self.entries and not self._add_entry(entry, start):
            return False, 0
        if (self.entries.get(index) != entry or start != self.cursors.get(index)
                or not 0 <= start < end <= total or index < self.last_document_index):
            return False, 0
        if index > self.last_document_index:
            if index != self.last_document_index + 1:
                return False, 0
            if (self.last_document_index >= 0
                    and self.cursors[self.last_document_index]
                    != self.entries[self.last_document_index]["token_count_with_boundary"]):
                return False, 0
            if self.last_document_index >= 0:
                del self.entries[self.last_document_index]
                del self.cursors[self.last_document_index]
            self.last_document_index = index
        self.cursors[index] = end
        return True, end - start

    def validate_snapshot(self, snapshot: dict[str, Any]) -> bool:
        segments = snapshot.get("pending_segments")
        if not isinstance(segments, list):
            return False
        for segment in segments:
            try:
                entry = document_entry(segment)
                index, cursor = segment["document_index"], segment["cursor"]
            except (KeyError, TypeError):
                return False
            if index not in self.entries and not self._add_entry(entry, 0):
                return False
            if self.entries.get(index) != entry or self.cursors.get(index) != cursor:
                return False
        pending_count = sum(
            segment["token_count_with_boundary"] - segment["cursor"] for segment in segments
        )
        pending_indices = {segment.get("document_index") for segment in segments}
        for index in set(self.entries) - pending_indices:
            if (index != self.last_document_index
                    or self.cursors[index] != self.entries[index]["token_count_with_boundary"]):
                return False
        return (
            snapshot.get("source") == self.source
            and snapshot.get("documents_loaded") == self.documents_seen
            and snapshot.get("pending_token_count") == pending_count
            and isinstance(snapshot.get("pending_tokens_sha256"), str)
            and len(snapshot["pending_tokens_sha256"]) == 64
            and snapshot.get("document_stream_sha256") == self.hasher.copy().hexdigest()
        )


def assets(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for name in ASSETS:
        asset = path / name
        if asset.is_file():
            result[name] = {"sha256": sha(asset), "size_bytes": asset.stat().st_size}
    return result


def finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def audit(root: Path, inventory_path: Path, inventory_audit_path: Path,
          tokenizer_path: Path, plan_path: Path, builder_path: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory_audit = json.loads(inventory_audit_path.read_text(encoding="utf-8"))
    profile = manifest.get("profile")
    checks: dict[str, bool] = {
        "protocol_and_profile": manifest.get("protocol") == PROTOCOL and profile in PROFILES,
        "no_partial_paths": not root.with_name(root.name + ".partial").exists(),
    }
    bindings = manifest.get("bindings", {})
    checks["plan_and_builder"] = (
        bindings.get("plan", {}).get("sha256") == sha(plan_path)
        and bindings.get("builder", {}).get("sha256") == sha(builder_path)
    )
    checks["inventory_dependencies"] = (
        bindings.get("inventory", {}).get("sha256") == sha(inventory_path)
        and bindings.get("inventory_audit", {}).get("sha256") == sha(inventory_audit_path)
        and inventory_audit.get("passed")
        and inventory_audit.get("bindings", {}).get("inventory_sha256") == sha(inventory_path)
    )
    observed_assets = assets(tokenizer_path)
    checks["tokenizer_assets"] = (
        all(name in observed_assets for name in ("tokenizer.json", "tokenizer_config.json", "config.json"))
        and observed_assets == bindings.get("tokenizer_assets")
    )
    if profile not in PROFILES:
        return {"protocol": PROTOCOL + "_independent_audit", "passed": False,
                "checks": checks, "error": "unknown profile"}

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, use_fast=True, fix_mistral_regex=False, local_files_only=True,
    )
    continuity = {source: SourceContinuity(source, inventory["sources"][source])
                  for source in SOURCES}
    previous_after = None
    combined_hasher = hashlib.sha256()
    total_records = total_tokens = total_synthetic = 0
    observed_stages = []
    stages_valid = len(manifest.get("stages", [])) == len(PROFILES[profile])
    provenance_clean = True
    artifacts_valid = True
    resources_valid = True

    for stage_index, expected in enumerate(PROFILES[profile]):
        name, seq_len, per_source = expected
        saved = manifest["stages"][stage_index]
        data_path = root / f"{name}.jsonl"
        provenance_path = root / f"{name}.provenance.jsonl"
        stage_valid = (
            saved.get("name") == name and saved.get("seq_len") == seq_len
            and saved.get("records_per_source") == per_source
            and saved.get("combined_records") == 2 * per_source
            and saved.get("combined_training_tokens") == 2 * per_source * seq_len
            and saved.get("global_batch_size")
            == (64 if profile in TRAINING_PROFILES else None)
            and saved.get("training_steps")
            == (2 * per_source // 64 if profile in TRAINING_PROFILES else None)
            and saved.get("source_records") == {source: per_source for source in SOURCES}
            and saved.get("source_training_tokens")
            == {source: per_source * seq_len for source in SOURCES}
        )
        before = saved.get("source_state_before", {})
        if stage_index == 0:
            stage_valid &= all(
                before.get(source, {}).get("documents_loaded") == 0
                and before.get(source, {}).get("pending_token_count") == 0
                and before.get(source, {}).get("pending_segments") == []
                for source in SOURCES
            )
        else:
            stage_valid &= before == previous_after

        stage_hasher = hashlib.sha256()
        combined_hasher.update((name + "\n").encode("utf-8"))
        stage_records = stage_tokens = stage_synthetic = 0
        per_source_synthetic = {source: 0 for source in SOURCES}
        for record_index, pair in enumerate(zip_longest(
                read_jsonl(data_path), read_jsonl(provenance_path))):
            row, provenance = pair
            if row is None or provenance is None:
                stage_valid = False
                break
            expected_source = SOURCES[record_index % 2]
            expected_source_index = record_index // 2
            provenance_clean &= not forbidden(provenance)
            if (set(row) != {"content_split"}
                    or not isinstance(row.get("content_split"), str)):
                stage_valid = False
                continue
            ids = tokenizer.encode(row["content_split"], add_special_tokens=False)
            training_ids = ids + [tokenizer.eos_token_id]
            payload = token_bytes(training_ids)
            stage_hasher.update(payload); combined_hasher.update(payload)
            spans = provenance.get("spans")
            raw_count = 0
            if not isinstance(spans, list) or not spans:
                stage_valid = False; spans = []
            for span in spans:
                valid, consumed = continuity[expected_source].consume_span(span)
                stage_valid &= valid
                raw_count += consumed
            synthetic = provenance.get("synthetic_eos_tokens")
            if not isinstance(synthetic, int) or synthetic < 0:
                stage_valid = False; synthetic = 0
            stage_valid &= (
                provenance.get("source") == expected_source
                and provenance.get("stage") == name
                and provenance.get("stage_record_index") == record_index
                and provenance.get("source_record_index") == expected_source_index
                and provenance.get("content_token_count") == seq_len - 1
                and provenance.get("training_token_count") == seq_len
                and provenance.get("raw_source_tokens_consumed") == raw_count
                and provenance.get("training_token_sha256") == token_sha(training_ids)
                and len(ids) == seq_len - 1 and len(training_ids) == seq_len
            )
            per_source_synthetic[expected_source] += synthetic
            stage_synthetic += synthetic
            stage_records += 1; stage_tokens += len(training_ids)

        after = saved.get("source_state_after", {})
        for source in SOURCES:
            stage_valid &= continuity[source].validate_snapshot(after.get(source, {}))
        previous_after = after
        stage_valid &= (
            stage_records == 2 * per_source
            and stage_tokens == 2 * per_source * seq_len
            and saved.get("synthetic_eos_tokens") == per_source_synthetic
            and saved.get("ordered_training_tokens_sha256") == stage_hasher.hexdigest()
        )
        artifact_records = saved.get("artifacts", {})
        artifacts_valid &= (
            artifact_records.get(data_path.name, {}).get("sha256") == sha(data_path)
            and artifact_records.get(data_path.name, {}).get("size_bytes") == data_path.stat().st_size
            and artifact_records.get(provenance_path.name, {}).get("sha256") == sha(provenance_path)
            and artifact_records.get(provenance_path.name, {}).get("size_bytes")
            == provenance_path.stat().st_size
        )
        stage_resources = saved.get("resources", {})
        resources_valid &= (
            finite_positive(stage_resources.get("wall_seconds"))
            and finite_positive(stage_resources.get("peak_host_rss_mib"))
            and all(finite_positive(stage_resources.get("source_pack_seconds", {}).get(source))
                    and finite_positive(stage_resources.get("source_tokens_per_second", {}).get(source))
                    for source in SOURCES)
        )
        observed_stages.append({"name": name, "passed": bool(stage_valid),
                                "records": stage_records, "training_tokens": stage_tokens,
                                "synthetic_eos_tokens": stage_synthetic,
                                "ordered_training_tokens_sha256": stage_hasher.hexdigest()})
        stages_valid &= stage_valid
        total_records += stage_records; total_tokens += stage_tokens
        total_synthetic += stage_synthetic

    totals = manifest.get("totals", {})
    checks["stage_records_tokens_order_and_continuity"] = bool(stages_valid)
    checks["provenance_has_no_plaintext"] = provenance_clean
    checks["artifact_hashes"] = bool(artifacts_valid)
    checks["combined_curriculum"] = (
        totals.get("records") == total_records
        and totals.get("training_tokens") == total_tokens
        and totals.get("synthetic_eos_tokens") == total_synthetic
        and totals.get("combined_curriculum_sha256") == combined_hasher.hexdigest()
    )
    top_resources = manifest.get("resources", {})
    checks["resources"] = bool(
        resources_valid and finite_positive(top_resources.get("wall_seconds"))
        and finite_positive(top_resources.get("peak_host_rss_mib"))
        and (profile != "pilot" or top_resources["peak_host_rss_mib"] <= 16384)
    )
    return {
        "protocol": PROTOCOL + "_independent_audit",
        "profile": profile,
        "passed": all(checks.values()),
        "checks": checks,
        "observed_stages": observed_stages,
        "observed_totals": {
            "records": total_records, "training_tokens": total_tokens,
            "synthetic_eos_tokens": total_synthetic,
            "combined_curriculum_sha256": combined_hasher.hexdigest(),
            "continuity_peak_active_documents": {
                source: continuity[source].peak_active_documents for source in SOURCES
            },
            "continuity_documents_seen": {
                source: continuity[source].documents_seen for source in SOURCES
            },
        },
        "bindings": {
            "manifest_sha256": sha(manifest_path),
            "inventory_sha256": sha(inventory_path),
            "inventory_audit_sha256": sha(inventory_audit_path),
            "plan_sha256": sha(plan_path),
            "builder_sha256": sha(builder_path),
            "auditor_sha256": sha(Path(__file__).resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--inventory-audit", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--builder", default=str(DEFAULT_BUILDER))
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else root / "independent_audit.json"
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {output}")
    result = audit(
        root,
        Path(args.inventory).expanduser().resolve(),
        Path(args.inventory_audit).expanduser().resolve(),
        Path(args.tokenizer).expanduser().resolve(),
        Path(args.plan).expanduser().resolve(),
        Path(args.builder).expanduser().resolve(),
    )
    partial.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    os.replace(partial, output)
    print(json.dumps({"output": str(output), "passed": result["passed"],
                      "checks": result["checks"]}, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
