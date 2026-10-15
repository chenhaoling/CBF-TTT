"""Inspect aggregate TempLAMA trajectory eligibility without writing examples."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

from scripts.inspect_lama_pararel_utility_source import normalize, sha256


PROTOCOL = "qwen_utility_record_controller_v2_templama_source_inspection"
SOURCE_PROTOCOL = "qwen_utility_record_controller_v2_templama_sources"
IDENTITY = re.compile(r"^(Q[0-9]+)_(P[0-9]+)_([0-9]{4})$")
SPLITS = ("train", "val", "test")
OWNER_NAMESPACE = "261016:templama-subject-owner"


def load_templates(path: Path) -> dict[str, dict[str, str]]:
    output = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            relation = row["Wikidata ID"].strip()
            template = row["Template"].strip()
            if not relation or template.count("<subject>") != 1 \
                    or template.count("<object>") != 1:
                raise ValueError("invalid TempLAMA template")
            if relation in output:
                raise ValueError("duplicate TempLAMA template")
            output[relation] = {
                "name": row["Relation"].strip(),
                "template": template,
            }
    return output


def template_regex(template: str) -> re.Pattern:
    pieces = re.split(r"(<subject>|<object>)", template)
    expression = []
    for piece in pieces:
        if piece == "<subject>":
            expression.append(r"(?P<subject>.+?)")
        elif piece == "<object>":
            expression.append(re.escape("_X_"))
        else:
            expression.append(re.escape(piece))
    return re.compile("^" + "".join(expression) + "$")


def extract_subject(query: str, template: str) -> str | None:
    match = template_regex(template).fullmatch(query)
    if match is None:
        return None
    subject = re.sub(r"\s+", " ", match.group("subject").strip())
    return subject or None


def singleton_answer(row: dict) -> tuple[str, str] | None:
    answers = row.get("answer")
    if not isinstance(answers, list) or not answers:
        return None
    values = {
        (answer.get("wikidata_id"), normalize(answer.get("name", "")))
        for answer in answers if isinstance(answer, dict)
    }
    if len(values) != 1:
        return None
    identity, name = next(iter(values))
    if not isinstance(identity, str) or not re.fullmatch(r"Q[0-9]+", identity) or not name:
        return None
    return identity, name


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            rows.append(row)
    return rows


def choose_owner(subject: str, owners: set[str]) -> str:
    ordered = sorted(owners)
    if not ordered:
        raise ValueError("subject has no source partition")
    digest = hashlib.sha256(f"{OWNER_NAMESPACE}:{subject}".encode()).hexdigest()
    return ordered[int(digest[:16], 16) % len(ordered)]


def inspect(source_root: Path) -> dict:
    manifest_path = source_root / "metadata" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") != SOURCE_PROTOCOL or manifest.get("content_parsed"):
        raise ValueError("unexpected or already parsed TempLAMA manifest")
    template_path = (
        source_root / "sources" / "google-research-language"
        / "language" / "templama" / "templates.csv"
    )
    templates = load_templates(template_path)
    template_patterns = {
        relation: template_regex(value["template"])
        for relation, value in templates.items()
    }

    split_results = {}
    subjects_by_split = {}
    surfaces_by_split = {}
    surfaces_by_subject_by_split = {}
    eligible_by_split = {}
    eligible_keys_by_split = {}
    rejection_totals = Counter()
    for split in SPLITS:
        path = source_root / "downloads" / f"{split}.json"
        locked = manifest["sources"]["downloads"][split]
        if path.stat().st_size != locked["bytes"] or sha256(path) != locked["sha256"]:
            raise ValueError(f"TempLAMA {split} file changed after acquisition")
        rows = load_rows(path)
        trajectories = defaultdict(dict)
        subjects = set(); surfaces = set(); rejected = Counter()
        surfaces_by_subject = defaultdict(set)
        row_relations = Counter()
        structurally_valid_rows = 0
        for row in rows:
            match = IDENTITY.fullmatch(str(row.get("id", "")))
            if match is None:
                rejected["invalid_identity"] += 1; continue
            subject_id, relation, identity_year = match.groups()
            if relation != row.get("relation") or identity_year != str(row.get("date")):
                rejected["identity_field_mismatch"] += 1; continue
            if relation not in templates:
                rejected["missing_official_template"] += 1; continue
            query = row.get("query")
            if not isinstance(query, str):
                rejected["missing_query"] += 1; continue
            template_match = template_patterns[relation].fullmatch(query)
            if template_match is None:
                rejected["query_template_mismatch"] += 1; continue
            subject_text = re.sub(
                r"\s+", " ", template_match.group("subject").strip()
            )
            if not subject_text:
                rejected["missing_subject_surface"] += 1; continue
            structurally_valid_rows += 1
            subjects.add(subject_id); row_relations[relation] += 1
            answer = singleton_answer(row)
            if answer is None:
                rejected["ambiguous_or_missing_answer"] += 1; continue
            year = int(identity_year); key = (subject_id, relation)
            if year in trajectories[key]:
                rejected["duplicate_trajectory_year"] += 1; continue
            answer_id, answer_name = answer
            trajectories[key][year] = {
                "subject_text": subject_text,
                "answer_id": answer_id,
                "answer_name": answer_name,
                "query": query,
            }
            surface = normalize(query.replace("_X_", answer_name))
            surfaces.add(surface); surfaces_by_subject[subject_id].add(surface)

        eligible = {}
        relation_eligible = Counter(); transitions = 0
        for key, by_year in trajectories.items():
            ordered = sorted(by_year.items())
            subject_names = {value["subject_text"] for _, value in ordered}
            answer_ids = [value["answer_id"] for _, value in ordered]
            changes = sum(left != right for left, right in zip(answer_ids, answer_ids[1:]))
            if len(subject_names) != 1:
                rejected["unstable_subject_surface"] += len(ordered); continue
            if len(set(answer_ids)) < 2 or changes < 1:
                rejected["no_value_transition"] += len(ordered); continue
            eligible[key] = ordered
            relation_eligible[key[1]] += 1
            transitions += changes
        rejection_totals.update(rejected)
        eligible_subjects = {subject for subject, _ in eligible}
        subjects_by_split[split] = subjects
        surfaces_by_split[split] = surfaces
        surfaces_by_subject_by_split[split] = dict(surfaces_by_subject)
        eligible_by_split[split] = eligible_subjects
        eligible_keys_by_split[split] = set(eligible)
        split_results[split] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "rows": len(rows),
            "structurally_valid_rows": structurally_valid_rows,
            "usable_single_answer_rows": sum(
                len(years) for years in trajectories.values()
            ),
            "subjects": len(subjects),
            "relations": len(row_relations),
            "row_relations": dict(sorted(row_relations.items())),
            "trajectories": len(trajectories),
            "eligible_trajectories": len(eligible),
            "eligible_subjects": len(eligible_subjects),
            "eligible_relations": len(relation_eligible),
            "eligible_by_relation": dict(sorted(relation_eligible.items())),
            "value_transitions": transitions,
            "rejected": dict(rejected),
        }

    subject_overlaps = {}
    eligible_subject_overlaps = {}
    surface_overlaps = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            name = f"{left}_{right}"
            subject_overlaps[name] = len(subjects_by_split[left] & subjects_by_split[right])
            eligible_subject_overlaps[name] = len(
                eligible_by_split[left] & eligible_by_split[right]
            )
            surface_overlaps[name] = len(surfaces_by_split[left] & surfaces_by_split[right])

    source_owners = defaultdict(set)
    for split in SPLITS:
        for subject in subjects_by_split[split]:
            source_owners[subject].add(split)
    assigned_owner = {
        subject: choose_owner(subject, owners)
        for subject, owners in source_owners.items()
    }
    owned_subjects = {
        split: {subject for subject in subjects_by_split[split]
                if assigned_owner[subject] == split}
        for split in SPLITS
    }
    owned_eligible_keys = {
        split: {key for key in eligible_keys_by_split[split]
                if assigned_owner[key[0]] == split}
        for split in SPLITS
    }
    owned_surfaces = {
        split: {
            surface
            for subject, values in surfaces_by_subject_by_split[split].items()
            if assigned_owner[subject] == split
            for surface in values
        }
        for split in SPLITS
    }
    colliding_surfaces = set()
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            colliding_surfaces.update(owned_surfaces[left] & owned_surfaces[right])
    collision_subjects = {
        split: {
            subject
            for subject, values in surfaces_by_subject_by_split[split].items()
            if assigned_owner[subject] == split and set(values) & colliding_surfaces
        }
        for split in SPLITS
    }
    filtered_subjects = {
        split: owned_subjects[split] - collision_subjects[split] for split in SPLITS
    }
    filtered_eligible_keys = {
        split: {key for key in owned_eligible_keys[split]
                if key[0] in filtered_subjects[split]}
        for split in SPLITS
    }
    filtered_surfaces = {
        split: owned_surfaces[split] - colliding_surfaces for split in SPLITS
    }
    post_subject_overlaps = {}
    post_eligible_subject_overlaps = {}
    post_surface_overlaps = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            name = f"{left}_{right}"
            post_subject_overlaps[name] = len(
                filtered_subjects[left] & filtered_subjects[right]
            )
            post_eligible_subject_overlaps[name] = len(
                {key[0] for key in filtered_eligible_keys[left]}
                & {key[0] for key in filtered_eligible_keys[right]}
            )
            post_surface_overlaps[name] = len(
                filtered_surfaces[left] & filtered_surfaces[right]
            )
    post_decontamination = {}
    for split in SPLITS:
        relation_counts = Counter(key[1] for key in filtered_eligible_keys[split])
        post_decontamination[split] = {
            "subjects": len(filtered_subjects[split]),
            "removed_cross_split_subjects": (
                len(subjects_by_split[split] - owned_subjects[split])
            ),
            "removed_surface_collision_subjects": len(collision_subjects[split]),
            "eligible_trajectories": len(filtered_eligible_keys[split]),
            "eligible_relations": len(relation_counts),
            "eligible_by_relation": dict(sorted(relation_counts.items())),
            "rendered_surfaces": len(filtered_surfaces[split]),
        }

    checks = {
        "official_nine_templates": len(templates) == 9,
        "all_download_hashes_locked": all(
            split_results[split]["sha256"]
            == manifest["sources"]["downloads"][split]["sha256"]
            for split in SPLITS
        ),
        "all_rows_parse_against_templates": all(
            split_results[split]["structurally_valid_rows"]
            == split_results[split]["rows"]
            for split in SPLITS
        ),
        "deterministic_subject_ownership_disjoint": all(
            count == 0 for count in post_subject_overlaps.values()
        ),
        "eligible_subject_ownership_disjoint": all(
            count == 0 for count in post_eligible_subject_overlaps.values()
        ),
        "rendered_surface_decontamination": all(
            count == 0 for count in post_surface_overlaps.values()
        ),
        "formal_partition_capacity": all(
            post_decontamination[split]["eligible_trajectories"] >= 128
            and post_decontamination[split]["eligible_relations"] >= 6
            for split in ("val", "test")
        ),
        "resource_pilot_capacity": (
            post_decontamination["train"]["eligible_trajectories"] >= 72
            and post_decontamination["train"]["eligible_relations"] == 9
        ),
        "v1_confirmation_and_forbidden_sources_sealed": (
            manifest.get("v1_confirmation_loaded") is False
            and manifest.get("forbidden_sources_loaded") is False
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": PROTOCOL,
        "source_manifest_sha256": sha256(manifest_path),
        "repository_commit": manifest["sources"]["repository"]["commit"],
        "template_sha256": sha256(template_path),
        "templates": templates,
        "splits": split_results,
        "cross_split_subject_overlaps": subject_overlaps,
        "cross_split_eligible_subject_overlaps": eligible_subject_overlaps,
        "cross_split_rendered_surface_overlaps": surface_overlaps,
        "subject_owner_namespace": OWNER_NAMESPACE,
        "cross_split_subject_overlaps_after_decontamination": post_subject_overlaps,
        "cross_split_eligible_subject_overlaps_after_decontamination": (
            post_eligible_subject_overlaps
        ),
        "cross_split_rendered_surface_overlaps_after_decontamination": (
            post_surface_overlaps
        ),
        "colliding_rendered_surfaces_removed_from_all_splits": len(colliding_surfaces),
        "post_decontamination": post_decontamination,
        "rejected_total": dict(rejection_totals),
        "checks": checks,
        "passed": passed,
        "content_written": False,
        "v1_confirmation_loaded": False,
        "forbidden_sources_loaded": False,
        "next_stage": "templama_resource_pilot_construction" if passed else "stop",
        "terminal_status": (
            "templama_source_inspection_passed" if passed
            else "templama_source_inspection_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "splits": result["splits"],
        "checks": result["checks"],
        "passed": result["passed"],
        "next_stage": result["next_stage"],
    }, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
