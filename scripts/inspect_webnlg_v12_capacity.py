"""Inspect registered WebNLG train/1triples capacity without model execution."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from tasks.build_qwen_utility_semantic_record_gate_v1 import write_json


PROTOCOL = "qwen_utility_record_controller_v12_webnlg_capacity"
SOURCE_COMMIT = "711a8ca37eff2d694d664edcaeb07b0ccf8f56c2"
SOURCE_MANIFEST_SHA256 = "d72d28824c8912c34fc652ff45050323568a94bf584395d466bcfa0aa3ee21a4"
MIN_ROWS = 28
MIN_SUBJECTS = 20
MIN_VALUES = 20
MIN_TEMPLATES = 7
MIN_ELIGIBLE_RELATIONS = 8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value: str) -> str:
    value = str(value).replace("_", " ").strip().strip('"')
    return " ".join(value.casefold().split())


def display(value: str) -> str:
    return " ".join(str(value).replace("_", " ").strip().strip('"').split())


def split_triple(text: str):
    values = [value.strip() for value in str(text).split("|")]
    if len(values) != 3 or not all(values):
        raise ValueError("invalid modified triple")
    return values


def entity_roles(entry) -> dict[str, str]:
    roles = {}
    for node in entry.findall(".//entity"):
        values = [value.strip() for value in (node.text or "").split("|")]
        if len(values) == 2 and values[0] and values[1]:
            roles[values[0]] = values[1]
    return roles


def slot_template(entry, lexicalization, subject: str, value: str) -> str | None:
    template = lexicalization.findtext("template")
    if template:
        output = template
        roles = entity_roles(entry)
        subject_roles = [role for role, entity in roles.items()
                         if normalize(entity) == normalize(subject)]
        value_roles = [role for role, entity in roles.items()
                       if normalize(entity) == normalize(value)]
        for role in sorted(subject_roles, key=len, reverse=True):
            output = re.sub(rf"\b{re.escape(role)}\b", "[X]", output)
        for role in sorted(value_roles, key=len, reverse=True):
            output = re.sub(rf"\b{re.escape(role)}\b", "[Y]", output)
        if output.count("[X]") == 1 and output.count("[Y]") == 1:
            return " ".join(output.split())
    text = lexicalization.findtext("text") or ""
    output = text
    subject_display, value_display = display(subject), display(value)
    subject_pattern = re.compile(re.escape(subject_display), re.IGNORECASE)
    value_pattern = re.compile(re.escape(value_display), re.IGNORECASE)
    if len(subject_pattern.findall(output)) == 1 and len(value_pattern.findall(output)) == 1:
        output = subject_pattern.sub("[X]", output, count=1)
        output = value_pattern.sub("[Y]", output, count=1)
        if output.count("[X]") == 1 and output.count("[Y]") == 1:
            return " ".join(output.split())
    return None


def inspect(source_root: Path) -> dict:
    manifest_path = source_root / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha256(manifest_path) != SOURCE_MANIFEST_SHA256 \
            or manifest.get("repository_commit") != SOURCE_COMMIT \
            or manifest.get("xml_or_json_parsed") is not False:
        raise ValueError("WebNLG source identity differs from V12 registration")
    files = sorted((source_root / "repository/data/v1.6/en/train/1triples").glob("*.xml"))
    if not files:
        raise ValueError("missing registered WebNLG train/1triples XML")
    rows, exclusion = [], Counter(); raw_entries = raw_lexicalizations = 0
    for path in files:
        tree = ET.parse(path)
        for entry in tree.findall(".//entry"):
            raw_entries += 1
            triples = [node.text or "" for node in entry.findall(".//modifiedtripleset/mtriple")]
            if len(triples) != 1:
                exclusion["not_exactly_one_modified_triple"] += 1; continue
            try:
                subject, relation, value = split_triple(triples[0])
            except ValueError:
                exclusion["invalid_modified_triple"] += 1; continue
            if not normalize(subject) or not normalize(value) \
                    or normalize(subject) == normalize(value):
                exclusion["invalid_subject_or_value"] += 1; continue
            lexicalizations = entry.findall(".//lex")
            raw_lexicalizations += len(lexicalizations)
            valid = 0
            for lex_index, lexicalization in enumerate(lexicalizations):
                pattern = slot_template(entry, lexicalization, subject, value)
                if not pattern:
                    exclusion["unaligned_lexicalization"] += 1; continue
                payload = ET.tostring(entry, encoding="utf-8") + f":{lex_index}".encode()
                rows.append({
                    "relation": relation, "subject": display(subject),
                    "value": display(value), "subject_norm": normalize(subject),
                    "value_norm": normalize(value), "pattern": pattern,
                    "pattern_norm": normalize(pattern),
                    "source_file": str(path.relative_to(source_root / "repository")),
                    "entry_id": entry.attrib.get("eid"),
                    "lexicalization_index": lex_index,
                    "source_row_sha256": hashlib.sha256(payload).hexdigest(),
                }); valid += 1
            if not valid:
                exclusion["entry_without_valid_lexicalization"] += 1
    by_relation = defaultdict(list)
    for row in rows: by_relation[row["relation"]].append(row)
    capacity = {}
    for relation, values in sorted(by_relation.items()):
        pairs = {(row["subject_norm"], row["value_norm"]) for row in values}
        subjects = {row["subject_norm"] for row in values}
        objects = {row["value_norm"] for row in values}
        templates = {row["pattern_norm"] for row in values}
        capacity[relation] = {
            "lexicalizations": len(values), "rows": len(pairs),
            "subjects": len(subjects), "values": len(objects),
            "templates": len(templates),
            "eligible": len(pairs) >= MIN_ROWS and len(subjects) >= MIN_SUBJECTS
                and len(objects) >= MIN_VALUES and len(templates) >= MIN_TEMPLATES,
        }
    eligible = sorted(key for key, value in capacity.items() if value["eligible"])
    checks = {
        "source_commit_and_manifest_locked": True,
        "train_one_triple_only": all("/train/1triples/" in row["source_file"] for row in rows),
        "no_model_or_benchmark_access": True,
        "minimum_eight_eligible_relations": len(eligible) >= MIN_ELIGIBLE_RELATIONS,
    }
    return {
        "protocol": PROTOCOL, "passed": all(checks.values()), "checks": checks,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_commit": SOURCE_COMMIT,
        "counts": {"files": len(files), "raw_entries": raw_entries,
                   "raw_lexicalizations": raw_lexicalizations,
                   "valid_lexicalizations": len(rows),
                   "relations": len(capacity), "eligible_relations": len(eligible)},
        "exclusions": dict(sorted(exclusion.items())),
        "thresholds": {"rows": MIN_ROWS, "subjects": MIN_SUBJECTS,
                       "values": MIN_VALUES, "templates": MIN_TEMPLATES},
        "eligible_relations": eligible, "capacity": capacity,
        "model_loaded": False, "sealed_benchmarks_loaded": False,
        "next_stage": "v12_webnlg_corpus_construction" if all(checks.values())
                      else "stop_v12_webnlg_capacity_failure",
        "terminal_status": "v12_webnlg_capacity_passed" if all(checks.values())
                           else "v12_webnlg_capacity_failed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); result = inspect(args.source_root)
    write_json(args.output, result); print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
