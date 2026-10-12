"""Build audited scope-pair supervision from sequential forgetting sessions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

from tasks.build_counterfact_zero_shot_forgetting import PROTOCOL as COUNTERFACT_PROTOCOL
from tasks.build_zsre_counterfactual_forgetting import PROTOCOL as ZSRE_PROTOCOL
from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_event_writer import write_json


PROTOCOL = "scoped_delta_corpus_v1"
MAX_KEYS = 8


def mask_subject(text, subject):
    """Remove an explicitly supplied subject span without changing other text."""
    pattern = re.compile(re.escape(subject), flags=re.IGNORECASE)
    masked, count = pattern.subn("<ENTITY>", text)
    return " ".join(masked.split()), count > 0


def tag_sessions(sessions, source):
    output = copy.deepcopy(sessions)
    for session in output:
        old_session = session["id"]
        session["id"] = f"{source}:{old_session}"
        mapping = {chain["id"]: f"{source}:{chain['id']}" for chain in session["chains"]}
        for chain in session["chains"]:
            chain["id"] = mapping[chain["id"]]
            if "parent_chain_id" in chain:
                chain["parent_chain_id"] = mapping[chain["parent_chain_id"]]
        for event in session["events"]:
            event["chain_id"] = mapping[event["chain_id"]]
    return output


def decision_groups(sessions, source, split):
    groups = []
    for session in sessions:
        chains = {chain["id"]: chain for chain in session["chains"]}
        memory = []
        for step, event in enumerate(session["events"]):
            chain = chains[event["chain_id"]]
            version = chain["versions"][event["version"]]
            matches = [index for index, row in enumerate(memory)
                       if row["chain_id"] == event["chain_id"]]
            if memory:
                if event["action"] == "local" and len(matches) != 1:
                    raise ValueError("local event must have exactly one active predecessor")
                if event["action"] == "retain" and matches:
                    raise ValueError("retain event unexpectedly matches an active chain")
                if len(memory) > MAX_KEYS:
                    raise ValueError("candidate memory exceeds registered width")
                query_relation, query_found = mask_subject(version["prompt"], chain["subject"])
                target = matches[0] if matches else MAX_KEYS
                candidates = []
                for index, row in enumerate(memory):
                    if index == target:
                        kind = "same_chain_positive"
                    elif row["subject"].casefold() == chain["subject"].casefold():
                        kind = "same_subject_different_chain"
                    else:
                        kind = "different_subject"
                    candidates.append({
                        "index": index,
                        "prompt": row["prompt"],
                        "subject": row["subject"],
                        "relation_prompt": row["relation_prompt"],
                        "subject_span_found": row["subject_span_found"],
                        "negative_type": kind,
                        "label": int(index == target),
                        "age": step - row["written_step"],
                        "chain_id": row["chain_id"],
                        "source_row_sha256": row["source_row_sha256"],
                    })
                groups.append({
                    "source": source,
                    "split": split,
                    "session_id": session["id"],
                    "group_id": chain.get("parent_chain_id", chain["id"]),
                    "step": step,
                    "event_kind": event["kind"],
                    "desired_action": event["action"],
                    "query": version["prompt"],
                    "query_subject": chain["subject"],
                    "query_relation": query_relation,
                    "query_subject_span_found": query_found,
                    "query_source_row_sha256": version.get("source_row_sha256"),
                    "target": target,
                    "candidates": candidates,
                })
            if event["action"] == "local":
                memory = [row for row in memory if row["chain_id"] != event["chain_id"]]
            relation_prompt, found = mask_subject(version["prompt"], chain["subject"])
            memory.append({"chain_id": event["chain_id"], "prompt": version["prompt"],
                           "subject": chain["subject"], "relation_prompt": relation_prompt,
                           "subject_span_found": found, "written_step": step,
                           "source_row_sha256": version.get("source_row_sha256")})
    return groups


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def corpus_stats(groups):
    candidates = [candidate for group in groups for candidate in group["candidates"]]
    local = [group for group in groups if group["target"] < MAX_KEYS]
    return {
        "groups": len(groups),
        "local_groups": len(local),
        "retain_groups": len(groups) - len(local),
        "candidates": len(candidates),
        "candidate_types": {kind: sum(candidate["negative_type"] == kind
                                      for candidate in candidates)
                            for kind in ("same_chain_positive",
                                         "same_subject_different_chain",
                                         "different_subject")},
        "query_subject_span_rate": sum(group["query_subject_span_found"]
                                       for group in groups) / max(1, len(groups)),
        "candidate_subject_span_rate": sum(candidate["subject_span_found"]
                                           for candidate in candidates) / max(1, len(candidates)),
    }


def build(zsre_root, counterfact_root, output):
    zsre_root, counterfact_root, output = map(Path, (zsre_root, counterfact_root, output))
    if output.exists():
        raise FileExistsError(output)
    zsre_manifest = json.loads((zsre_root / "manifest.json").read_text())
    counterfact_manifest = json.loads((counterfact_root / "manifest.json").read_text())
    if zsre_manifest["protocol"] != ZSRE_PROTOCOL:
        raise ValueError("ZsRE protocol mismatch")
    if counterfact_manifest["protocol"] != COUNTERFACT_PROTOCOL:
        raise ValueError("CounterFact protocol mismatch")
    groups = []
    for split in ("calibration", "dev", "test"):
        path = zsre_root / f"{split}.jsonl"
        if sha(path) != zsre_manifest["files_sha256"][path.name]:
            raise ValueError("ZsRE data hash mismatch")
        sessions = tag_sessions(load_jsonl(path), f"zsre_{split}")
        groups.extend(decision_groups(sessions, "zsre", split))
    path = counterfact_root / "transfer_test.jsonl"
    if sha(path) != counterfact_manifest["files_sha256"][path.name]:
        raise ValueError("CounterFact data hash mismatch")
    # Preserve CounterFact identifiers so V4 predictions can be compared
    # event-for-event with the frozen V2 OOF trace.
    sessions = load_jsonl(path)
    groups.extend(decision_groups(sessions, "counterfact", "observed_development"))
    output.mkdir(parents=True)
    data_path = output / "groups.jsonl"
    data_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in groups))
    manifest = {
        "protocol": PROTOCOL,
        "max_keys": MAX_KEYS,
        "sources": {
            "zsre_manifest_sha256": sha(zsre_root / "manifest.json"),
            "counterfact_manifest_sha256": sha(counterfact_root / "manifest.json"),
        },
        "statistics": {
            source: corpus_stats([group for group in groups if group["source"] == source])
            for source in ("zsre", "counterfact")
        },
        "chain_identity_in_model_features": False,
        "subject_metadata_role": "oracle_segmentation_diagnostic",
        "relation_labels_used": False,
        "longmemeval_scored": False,
        "locomo_scored": False,
        "files_sha256": {data_path.name: sha(data_path)},
        "content_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zsre-root", required=True)
    parser.add_argument("--counterfact-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
