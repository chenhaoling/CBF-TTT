"""Build entity-disjoint natural long-session episodes from official MQuAKE-T."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from tasks.build_long_session_sparse_corpus import LENGTHS, UNITS_PER_SESSION, fact_text, text_id
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "mquake_temporal_sessions_v1"
SOURCE_URL = "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-T.json"
LICENSE_URL = "https://github.com/princeton-nlp/MQuAKE/blob/main/LICENSE"
SPLIT_WEIGHTS = {"train": .6, "tune": .2, "dev": .2}


def stable(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def rewrite(row):
    values = row.get("requested_rewrite", [])
    if len(values) != 1:
        raise ValueError("MQuAKE-T row must contain exactly one requested rewrite")
    return values[0]


def subject(row):
    return rewrite(row)["subject"]


def split_subjects(rows):
    """Greedily balance case counts while keeping every edited subject in one split."""
    groups = Counter(subject(row) for row in rows)
    total = len(rows); loads = {name: 0 for name in SPLIT_WEIGHTS}; assignment = {}
    ordered = sorted(groups.items(), key=lambda item: (-item[1], stable(f"{PROTOCOL}|{item[0]}")))
    for name, count in ordered:
        split = min(SPLIT_WEIGHTS, key=lambda key: (
            (loads[key] + count) / (total * SPLIT_WEIGHTS[key]), loads[key], key))
        assignment[name] = split; loads[split] += count
    return assignment, loads


def unit(question, answer, table):
    question = " ".join(question.strip().split())
    answer = " ".join(str(answer).strip().split())
    if not question or not answer:
        raise ValueError("empty MQuAKE question or answer")
    return text_id(fact_text(question, answer), table)


def edit_index(row):
    values = row.get("orig", {}).get("edit_triples_idx", [])
    if len(values) != 1:
        raise ValueError("MQuAKE-T row must identify exactly one edited hop")
    value = int(values[0])
    if not 0 <= value < len(row["single_hops"]):
        raise ValueError("invalid edited hop index")
    return value


def case_units(row, use_new, table):
    idx = edit_index(row)
    hops = row["new_single_hops"] if use_new else row["single_hops"]
    values = []
    for position, hop in enumerate(hops):
        if position == idx:
            continue
        values.append(unit(hop.get("cloze") or hop["question"], hop["answer"], table))
    return values


def anchor(row, use_new, table):
    item = rewrite(row); idx = edit_index(row)
    if use_new:
        hop = row["new_single_hops"][idx]
        prompt = hop.get("cloze") or hop["question"]
        answer = item["target_new"]["str"]
    else:
        prompt = item.get("question") or item["prompt"].format(item["subject"])
        answer = item["target_true"]["str"]
    return unit(prompt, answer, table)


def ranked_pool(rows, primary, key, same_relation=False):
    relation = rewrite(primary)["relation_id"]
    candidates = [row for row in rows if row["case_id"] != primary["case_id"]
                  and subject(row) != subject(primary)]
    if same_relation:
        candidates = [row for row in candidates if rewrite(row)["relation_id"] == relation]
    return sorted(candidates, key=lambda row: stable(f"{PROTOCOL}|{key}|{row['case_id']}"))


def fill_units(anchor_id, context, fallback, table):
    values = [anchor_id]
    for identifier in context:
        if identifier not in values:
            values.append(identifier)
        if len(values) == UNITS_PER_SESSION:
            return values
    cursor = 0
    while len(values) < UNITS_PER_SESSION:
        row = fallback[cursor % len(fallback)]; cursor += 1
        identifier = anchor(row, False, table)
        if identifier not in values:
            values.append(identifier)
    return values


def build_episode(primary, rows, length, split, table):
    key = f"{split}|{length}|{primary['case_id']}"
    general = ranked_pool(rows, primary, key)
    related = ranked_pool(rows, primary, key, same_relation=True)
    if len(general) < UNITS_PER_SESSION:
        raise ValueError("insufficient entity-distinct MQuAKE distractors")
    related_ids = {row["case_id"] for row in related}
    hard_pool = related + [row for row in general if row["case_id"] not in related_ids]
    old_position, update_position = length // 4, 3 * length // 4
    hard_positions = {old_position + 1: 0, update_position - 1: 1}
    sessions = []
    for position in range(length):
        desired, target, kind = "retain", None, "distractor"
        if position == old_position:
            row, use_new, kind = primary, False, "primary_old"
        elif position == update_position:
            row, use_new, kind, desired = primary, True, "primary_update", "local"
            target = f"{stable(key)[:12]}-s{old_position:02d}"
        elif position in hard_positions:
            row, use_new, kind = hard_pool[hard_positions[position]], False, "hard_retain"
        else:
            row, use_new = general[position % len(general)], False
        anchor_id = anchor(row, use_new, table)
        context = case_units(row, use_new, table)
        units = fill_units(anchor_id, context, general, table)
        sessions.append({"session_id": f"{stable(key)[:12]}-s{position:02d}",
                         "units": units, "desired": desired,
                         "target_session_id": target, "event_kind": kind,
                         "hard_relation_matched": (None if kind != "hard_retain" else
                                                   rewrite(row)["relation_id"]
                                                   == rewrite(primary)["relation_id"])})
    return {"episode_id": stable(key)[:20], "split": split, "length": length,
            "primary_chain_id": f"mquake-t-{primary['case_id']}",
            "primary_subject_hash": stable(subject(primary)), "sessions": sessions}


def write_jsonl(path, rows):
    with Path(path).open("x") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(args):
    source = Path(args.source); output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = json.loads(source.read_text())
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate MQuAKE case_id")
    assignment, loads = split_subjects(rows)
    partitions = defaultdict(list)
    for row in rows:
        partitions[assignment[subject(row)]].append(row)
    table, files, counts = {}, {}, {}
    for split in SPLIT_WEIGHTS:
        episodes = [build_episode(row, partitions[split], length, split, table)
                    for row in partitions[split] for length in LENGTHS]
        path = output / f"{split}.jsonl"; write_jsonl(path, episodes)
        files[path.name] = sha(path)
        hard = [event for episode in episodes for event in episode["sessions"]
                if event["event_kind"] == "hard_retain"]
        counts[split] = {"source_cases": len(partitions[split]),
                         "source_subjects": len({subject(row) for row in partitions[split]}),
                         "episodes": len(episodes),
                         "events": sum(item["length"] for item in episodes),
                         "hard_retain_events": len(hard),
                         "same_relation_hard_retain_events": sum(
                             event["hard_relation_matched"] for event in hard)}
    text_path = output / "texts.json"; write_json(text_path, table); files[text_path.name] = sha(text_path)
    split_sets = {name: {subject(row) for row in partitions[name]} for name in SPLIT_WEIGHTS}
    disjoint = not any(split_sets[left] & split_sets[right]
                       for left in SPLIT_WEIGHTS for right in SPLIT_WEIGHTS if left < right)
    manifest = {"protocol": PROTOCOL, "source": "MQuAKE-T", "source_url": SOURCE_URL,
                "license_url": LICENSE_URL, "source_sha256": sha(source),
                "source_cases": len(rows), "unique_edited_subjects": len(assignment),
                "split_method": "edited_subject_disjoint_greedy_case_balance",
                "split_weights": SPLIT_WEIGHTS, "split_case_loads": loads,
                "lengths": list(LENGTHS), "units_per_session": UNITS_PER_SESSION,
                "counts": counts, "unique_texts": len(table), "files_sha256": files,
                "subject_disjoint": disjoint, "real_temporal_updates": True,
                "mquake_cf_used": False, "longmemeval_used": False, "locomo_used": False,
                "chain_identity_model_feature": False}
    if not disjoint:
        raise RuntimeError("MQuAKE split leakage")
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True); parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
