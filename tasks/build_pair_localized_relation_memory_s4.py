"""Build pair-localized candidate windows for the registered S4 experiment."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from scripts.inspect_pair_localization_s4_source import (
    nearest_entity_window,
    normalized,
    phrase_turns,
)
from tasks.build_retrieval_evidence_forgetting_s3 import (
    PROTOCOL as S3_PROTOCOL,
    source_hash,
    verify_dialogre,
)
from tasks.build_structured_fact_comparator_s1 import canonical_sha
from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_pair_localized_relation_memory_s4_data_v1"
CANDIDATE_RADIUS = 1


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def mask_dialogue_turns(dialogue: list[str], left: str, right: str) -> list[str]:
    if normalized(left) == normalized(right):
        raise ValueError("identical DialogRE arguments")
    replacements = sorted(((left, "[ARG1]"), (right, "[ARG2]")),
                          key=lambda item: len(item[0]), reverse=True)
    output = []
    for turn in dialogue:
        masked = str(turn)
        for value, marker in replacements:
            pattern = r"(?<!\w)" + re.escape(str(value)) + r"(?!\w)"
            masked = re.sub(pattern, marker, masked, flags=re.IGNORECASE)
        output.append(" ".join(masked.split()))
    return output


def format_turns(masked_turns: list[str], indices: list[int], center: int | None) -> str:
    labels = []
    for index in indices:
        if center is None:
            label = "[TURN]"
        elif index < center:
            label = "[PREVIOUS]"
        elif index == center:
            label = "[CURRENT]"
        else:
            label = "[NEXT]"
        labels.append(f"{label} {masked_turns[index]}")
    body = " ".join(labels)
    return "Arguments: [ARG1] ; [ARG2] . [WINDOW] " + body


def centered_indices(center: int, count: int, radius: int = CANDIDATE_RADIUS) -> list[int]:
    return list(range(max(0, center - radius), min(count, center + radius + 1)))


def nearest_distance(center: int, turns: list[int], count: int) -> float:
    if not turns:
        return 1.0
    return min(abs(center - value) for value in turns) / max(1, count - 1)


def dialogue_candidates(masked_turns: list[str], left_turns: list[int],
                        right_turns: list[int], mention_id: str) -> list[dict]:
    candidates = []
    for center in range(len(masked_turns)):
        indices = centered_indices(center, len(masked_turns))
        body = " ".join(masked_turns[index] for index in indices)
        candidates.append({
            "candidate_id": source_hash("s4-candidate", mention_id, center),
            "center_turn": center,
            "turn_indices": indices,
            "model_text": format_turns(masked_turns, indices, center),
            "features": {
                "arg1_present": "[ARG1]" in body,
                "arg2_present": "[ARG2]" in body,
                "distance_to_arg1": nearest_distance(center, left_turns, len(masked_turns)),
                "distance_to_arg2": nearest_distance(center, right_turns, len(masked_turns)),
                "normalized_position": center / max(1, len(masked_turns) - 1),
            },
        })
    return candidates


def deterministic_windows(masked_turns: list[str], left_turns: list[int],
                          right_turns: list[int]) -> dict[str, dict]:
    windows = {}
    for radius in (0, 1, 2):
        indices = sorted(nearest_entity_window(
            left_turns, right_turns, len(masked_turns), radius
        ))
        fallback = not indices
        if fallback:
            indices = list(range(len(masked_turns)))
        windows[str(radius)] = {
            "turn_indices": indices,
            "model_text": format_turns(masked_turns, indices, None),
            "full_dialogue_fallback": fallback,
        }
    return windows


def localize_dialogre(row: dict, raw_value: list) -> dict:
    dialogue, annotations = raw_value
    annotation = annotations[row["annotation_index_audit"]]
    dialogue_sha256 = canonical_sha(raw_value)
    if dialogue_sha256 != row["dialogue_sha256"]:
        raise ValueError("S3 dialogue hash mismatch")
    left, right = str(annotation["x"]), str(annotation["y"])
    relation = row["relation"].removeprefix("dialogre:")
    if relation not in annotation["r"]:
        raise ValueError("S3 relation not present in DialogRE annotation")
    instance_id = source_hash(
        "dialogre-instance", dialogue_sha256, row["annotation_index_audit"]
    )
    mention_id = source_hash("dialogre-mention", instance_id, relation)
    if instance_id != row["source_instance_sha256"] or mention_id != row["source_row_sha256"]:
        raise ValueError("S3 source identifier mismatch")

    masked_turns = mask_dialogue_turns(dialogue, left, right)
    left_turns, right_turns = phrase_turns(dialogue, left), phrase_turns(dialogue, right)
    triggers = sorted({normalized(value) for value in annotation.get("t", [])
                       if normalized(value)})
    trigger_turns = sorted({index for trigger in triggers
                            for index in phrase_turns(dialogue, trigger)})
    candidates = dialogue_candidates(masked_turns, left_turns, right_turns, mention_id)
    gold_candidates = [
        index for index, candidate in enumerate(candidates)
        if set(candidate["turn_indices"]) & set(trigger_turns)
    ]
    oracle_indices = sorted({index for trigger_turn in trigger_turns
                             for index in centered_indices(trigger_turn, len(dialogue))})
    return {
        "source": row["source"],
        "source_split": row["source_split"],
        "source_row_sha256": mention_id,
        "source_instance_sha256": instance_id,
        "dialogue_sha256": dialogue_sha256,
        "relation": row["relation"],
        "full_model_text": row["masked_text"],
        "turn_count": len(dialogue),
        "candidates": candidates,
        "deterministic_windows": deterministic_windows(
            masked_turns, left_turns, right_turns
        ),
        "trigger_supervision_available_audit": bool(trigger_turns),
        "gold_trigger_turns_audit": trigger_turns,
        "gold_trigger_candidate_indices_audit": gold_candidates,
        "oracle_window_audit": {
            "available": bool(oracle_indices),
            "turn_indices": oracle_indices,
            "model_text": format_turns(masked_turns, oracle_indices, None)
            if oracle_indices else None,
        },
        "dialogue_index_audit": row["dialogue_index_audit"],
        "annotation_index_audit": row["annotation_index_audit"],
    }


def localize_fewrel(row: dict) -> dict:
    text = row["masked_text"]
    candidate = {
        "candidate_id": source_hash("s4-candidate", row["source_row_sha256"], 0),
        "center_turn": 0,
        "turn_indices": [0],
        "model_text": text,
        "features": {
            "arg1_present": "[ARG1]" in text,
            "arg2_present": "[ARG2]" in text,
            "distance_to_arg1": 0.0,
            "distance_to_arg2": 0.0,
            "normalized_position": 0.0,
        },
    }
    window = {"turn_indices": [0], "model_text": text, "full_dialogue_fallback": False}
    return {
        "source": row["source"],
        "source_split": row["source_split"],
        "source_row_sha256": row["source_row_sha256"],
        "source_instance_sha256": row["source_instance_sha256"],
        "dialogue_sha256": row["dialogue_sha256"],
        "relation": row["relation"],
        "full_model_text": text,
        "turn_count": 1,
        "candidates": [candidate],
        "deterministic_windows": {str(radius): dict(window) for radius in (0, 1, 2)},
        "trigger_supervision_available_audit": False,
        "gold_trigger_turns_audit": [],
        "gold_trigger_candidate_indices_audit": [],
        "oracle_window_audit": {"available": False, "turn_indices": [], "model_text": None},
    }


def localize_mention(row: dict, dialogre_raw: list) -> dict:
    if row["source"] == "fewrel":
        return localize_fewrel(row)
    if row["source"] != "dialogre":
        raise ValueError(f"unsupported source: {row['source']}")
    return localize_dialogre(row, dialogre_raw[row["dialogue_index_audit"]])


def write_jsonl(path: Path, rows: list[dict]) -> str:
    with path.open("x") as sink:
        for row in rows:
            sink.write(json.dumps(row, sort_keys=True) + "\n")
    return sha(path)


def summarize(rows: list[dict]) -> dict:
    sources = Counter(row["source"] for row in rows)
    return {
        "mentions": len(rows),
        "sources": dict(sorted(sources.items())),
        "candidates": sum(len(row["candidates"]) for row in rows),
        "dialogre_trigger_supervised_mentions": sum(
            row["source"] == "dialogre" and row["trigger_supervision_available_audit"]
            for row in rows
        ),
        "deterministic_full_dialogue_fallback_mentions": sum(
            row["source"] == "dialogre"
            and row["deterministic_windows"]["0"]["full_dialogue_fallback"]
            for row in rows
        ),
    }


def build(args) -> dict:
    dialogre_root, s3_root, output = map(Path, (
        args.dialogre_root, args.s3_data_root, args.output_root
    ))
    if output.exists():
        raise FileExistsError(output)
    metadata = verify_dialogre(dialogre_root)
    s3_manifest_path, s3_audit_path = s3_root / "manifest.json", s3_root / "data_audit.json"
    s3_manifest, s3_audit = map(lambda path: json.loads(path.read_text()),
                                (s3_manifest_path, s3_audit_path))
    if s3_manifest.get("protocol") != S3_PROTOCOL or not s3_audit.get("passed"):
        raise ValueError("S4 requires the audited S3 corpus")
    source_inspection_path = Path(args.source_inspection)
    source_inspection = json.loads(source_inspection_path.read_text())
    if source_inspection.get("dialogre_dev_rows_parsed") != 0:
        raise ValueError("invalid S4 source inspection")
    raw = json.loads((dialogre_root / "data_v2/en/data/train.json").read_text())
    outputs, summaries = {}, {}
    output.mkdir(parents=True)
    for split in ("train", "calibration"):
        source_rows = load_jsonl(s3_root / f"{split}_mentions.jsonl")
        localized = [localize_mention(row, raw) for row in source_rows]
        name = f"{split}_localized_mentions.jsonl"
        outputs[name] = write_jsonl(output / name, localized)
        summaries[split] = summarize(localized)
        if len(localized) != s3_manifest["counts"][split]["mentions"]:
            raise RuntimeError("S3/S4 mention count mismatch")
    manifest = {
        "protocol": PROTOCOL,
        "candidate_radius": CANDIDATE_RADIUS,
        "counts": summaries,
        "files_sha256": outputs,
        "source": {
            "dialogre": metadata,
            "s3_manifest_sha256": sha(s3_manifest_path),
            "s3_audit_sha256": sha(s3_audit_path),
            "source_inspection_sha256": sha(source_inspection_path),
        },
        "model_outputs_loaded": False,
        "dialogre_dev_parsed": False,
        "dialogre_test_parsed": False,
        "pubmed_validation_parsed": False,
        "terminal_status": "data_built_pending_independent_audit",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogre-root", required=True)
    parser.add_argument("--s3-data-root", required=True)
    parser.add_argument("--source-inspection", required=True)
    parser.add_argument("--output-root", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
