"""Train a physics-informed forgetting controller on structured session memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import time
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import write_json


ACTIONS = ("retain", "last1", "last2", "global")
FIXED = tuple(f"fixed_{action}" for action in ACTIONS)
BATCH_RE = re.compile(r"batch_[0-9a-f]+")
ENTITY_RE = re.compile(r"(?:asset|unit|sensor)_[A-Za-z0-9_]+")
VALUE_RE = re.compile(r"code_[0-9]{5}")


def load_split(root: str | Path, split: str) -> list[tuple[dict, dict]]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["protocol"] != "cbf_event_curriculum_v1":
        raise ValueError("structured forgetting corpus protocol mismatch")
    paths = [root / f"{split}.{kind}.jsonl" for kind in ("inputs", "targets")]
    for path in paths:
        if sha(path) != manifest["files_sha256"][path.name]:
            raise ValueError("structured forgetting corpus hash mismatch")
    inputs = [json.loads(line) for line in paths[0].read_text().splitlines() if line]
    targets = [json.loads(line) for line in paths[1].read_text().splitlines() if line]
    if len(inputs) != len(targets) or any(a["id"] != b["id"] for a, b in zip(inputs, targets)):
        raise ValueError("structured forgetting corpus alignment mismatch")
    return list(zip(inputs, targets))


def parse_block(text: str, state: list[dict]) -> dict:
    known = {item["batch"] for item in state}
    refs = BATCH_RE.findall(text)
    new = [value for value in dict.fromkeys(refs) if value not in known]
    old = [value for value in dict.fromkeys(refs) if value in known]
    if len(new) != 1 or len(old) > 1:
        raise ValueError("visible block does not identify one new and at most one old batch")
    facts = {}
    for line in text.splitlines():
        entities, values = ENTITY_RE.findall(line), VALUE_RE.findall(line)
        if entities or values:
            if len(entities) != 1 or len(values) != 1:
                raise ValueError("visible assignment line is ambiguous")
            facts[entities[0]] = values[0]
    if not facts:
        raise ValueError("visible block has no assignments")
    lines = text.splitlines()
    new_line = next(index for index, line in enumerate(lines) if new[0] in line)
    return {"batch": new[0], "facts": facts, "old_refs": old,
            "prelude_lines": new_line, "text": text}


def apply_action(state: list[dict], action: str, assignment: dict) -> list[dict]:
    output = [{"batch": item["batch"], "facts": dict(item["facts"])} for item in state]
    if action == "global":
        output = []
    elif action in ("last1", "last2"):
        offset = 1 if action == "last1" else 2
        if len(output) >= offset:
            del output[-offset]
    elif action != "retain":
        raise ValueError("unknown forgetting action")
    output.append({"batch": assignment["batch"], "facts": dict(assignment["facts"])})
    return output


def lookup(state: list[dict], key: str) -> str:
    value = "unknown"
    for item in state:
        if key in item["facts"]:
            value = item["facts"][key]
    return value


def visible_features(state: list[dict], assignment: dict, block_index: int,
                     history: bool = True) -> list[float]:
    old = assignment["old_refs"][0] if assignment["old_refs"] else None
    position = next((index for index, item in enumerate(state) if item["batch"] == old), None)
    age = None if position is None else len(state) - position
    recency = [float(age == 1), float(age == 2), float(age is not None and age > 2)]
    if not history:
        recency = [0., 0., 0.]
    return [
        len(state) / 8.,
        float(old is not None),
        *recency,
        min(assignment["prelude_lines"], 2) / 2.,
        block_index / 8.,
        len(assignment["facts"]) / 6.,
        len({key for item in state for key in item["facts"]}) / 20.,
    ]


def boundary_truth(inp: dict, target: dict, block_index: int) -> list[dict]:
    answers = {row["id"]: row for row in target["answers"]}
    output = []
    for query in inp["queries"]:
        if query["after_block"] != block_index:
            continue
        entities = ENTITY_RE.findall(query["text"])
        if len(entities) != 1:
            raise ValueError("query does not identify exactly one entity")
        answer = answers[query["id"]]
        output.append({"id": query["id"], "key": entities[0], "answer": answer["answer"],
                       "role": answer["role"]})
    if len(output) != 3:
        raise ValueError("decision boundary must have three queries")
    return output


def counterfactual_label(state: list[dict], assignment: dict, truth: list[dict]) -> tuple[str, dict]:
    scores = {}
    for action in ACTIONS:
        candidate = apply_action(state, action, assignment)
        scores[action] = sum(lookup(candidate, row["key"]) == row["answer"] for row in truth)
    best = max(scores.values())
    winners = [action for action in ACTIONS if scores[action] == best]
    return winners[0], {"scores": scores, "winners": winners, "best": best}


def build_examples(pairs: list[tuple[dict, dict]], history: bool = True) -> tuple[list[dict], dict]:
    examples, ties = [], 0
    for inp, target in pairs:
        state = []
        boundaries = set(inp["decision_boundaries"])
        for block in inp["blocks"]:
            parsed = parse_block(block["text"], state)
            if block["index"] in boundaries:
                truth = boundary_truth(inp, target, block["index"])
                label, counterfactual = counterfactual_label(state, parsed, truth)
                ties += len(counterfactual["winners"]) > 1
                examples.append({"episode_id": inp["id"], "group_id": inp["group_id"],
                                 "family": target["family"], "block_index": block["index"],
                                 "features": visible_features(state, parsed, block["index"], history),
                                 "label": ACTIONS.index(label), "action": label,
                                 "counterfactual": counterfactual})
                state = apply_action(state, label, parsed)
            else:
                state = apply_action(state, "retain", parsed)
    counts = {action: sum(row["action"] == action for row in examples) for action in ACTIONS}
    return examples, {"examples": len(examples), "ties": ties, "label_counts": counts}


def train_controller(examples: list[dict], seed: int, epochs: int = 500):
    import torch

    torch.manual_seed(seed)
    model = torch.nn.Linear(len(examples[0]["features"]), len(ACTIONS))
    optimizer = torch.optim.Adam(model.parameters(), lr=.05, weight_decay=1e-4)
    x = torch.tensor([row["features"] for row in examples], dtype=torch.float32)
    y = torch.tensor([row["label"] for row in examples], dtype=torch.long)
    curve = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x); loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward(); optimizer.step()
        if epoch in (1, 10, 50, 100, 250, epochs):
            curve.append({"epoch": epoch, "loss": float(loss.detach()),
                          "accuracy": float((logits.detach().argmax(-1) == y).float().mean())})
    return model.eval(), curve


def choose_action(policy: str, state: list[dict], assignment: dict, block_index: int,
                  truth: list[dict], models: dict, episode_id: str) -> str:
    if policy.startswith("fixed_"):
        return policy.removeprefix("fixed_")
    if policy == "random":
        digest = hashlib.sha256(f"{episode_id}|{block_index}".encode()).hexdigest()
        return ACTIONS[int(digest[:8], 16) % len(ACTIONS)]
    if policy == "oracle":
        return counterfactual_label(state, assignment, truth)[0]
    if policy == "physics":
        old = assignment["old_refs"][0] if assignment["old_refs"] else None
        if old is not None:
            age = len(state) - next(index for index, item in enumerate(state) if item["batch"] == old)
            return "last1" if age == 1 else "last2" if age == 2 else "retain"
        return "global" if assignment["prelude_lines"] > 0 else "retain"
    if policy in ("learned", "history_blind"):
        import torch

        features = visible_features(state, assignment, block_index, policy == "learned")
        with torch.no_grad():
            return ACTIONS[int(models[policy](torch.tensor([features], dtype=torch.float32)).argmax(-1))]
    raise ValueError("unknown structured forgetting policy")


def evaluate_policy(pairs: list[tuple[dict, dict]], policy: str, models: dict) -> list[dict]:
    records = []
    for inp, target in pairs:
        state, actions, queries = [], [], []
        boundaries = set(inp["decision_boundaries"])
        for block in inp["blocks"]:
            parsed = parse_block(block["text"], state)
            if block["index"] in boundaries:
                truth = boundary_truth(inp, target, block["index"])
                action = choose_action(policy, state, parsed, block["index"], truth, models, inp["id"])
                oracle = counterfactual_label(state, parsed, truth)[0]
                state = apply_action(state, action, parsed)
                actions.append({"block": block["index"], "selected": action, "oracle": oracle})
                queries.extend({**row, "correct": lookup(state, row["key"]) == row["answer"]}
                               for row in truth)
            else:
                state = apply_action(state, "retain", parsed)
        records.append({"episode_id": inp["id"], "group_id": inp["group_id"],
                        "family": target["family"], "decisions": target["decisions"],
                        "twin": target["twin"], "policy": policy, "actions": actions,
                        "queries": queries, "retained_batches": len(state)})
    return records


def aggregate(records: list[dict]) -> dict:
    query_rows = [(row, query) for row in records for query in row["queries"]]
    actions = [action for row in records for action in row["actions"]]
    result = {
        "episodes": len(records), "queries": len(query_rows), "decisions": len(actions),
        "query_accuracy": statistics.fmean(query["correct"] for _, query in query_rows),
        "action_accuracy": statistics.fmean(action["selected"] == action["oracle"] for action in actions),
        "mean_retained_batches": statistics.fmean(row["retained_batches"] for row in records),
        "by_family": {}, "by_role": {}, "groups": {},
    }
    for family in sorted({row["family"] for row in records}):
        subset = [query["correct"] for row, query in query_rows if row["family"] == family]
        result["by_family"][family] = {"queries": len(subset), "accuracy": statistics.fmean(subset)}
    for role in sorted({query["role"] for _, query in query_rows}):
        subset = [query["correct"] for _, query in query_rows if query["role"] == role]
        result["by_role"][role] = {"queries": len(subset), "accuracy": statistics.fmean(subset)}
    for group in sorted({row["group_id"] for row in records}):
        subset = [query["correct"] for row, query in query_rows if row["group_id"] == group]
        result["groups"][group] = statistics.fmean(subset)
    return result


def comparison(primary: dict, baseline: dict, seed: int) -> dict:
    groups = sorted(primary["groups"])
    values = [primary["groups"][group] - baseline["groups"][group] for group in groups]
    return {"mean_accuracy_gain": statistics.fmean(values),
            "source_group_bootstrap_95ci": _bootstrap(values, seed)}


def gate(summaries: dict) -> dict:
    best_name = max(FIXED, key=lambda name: summaries[name]["query_accuracy"])
    gain = comparison(summaries["learned"], summaries[best_name], 1511)
    checks = {
        "learned_action_accuracy": summaries["learned"]["action_accuracy"] >= .95,
        "learned_query_accuracy": summaries["learned"]["query_accuracy"] >= .98,
        "best_fixed_margin": gain["mean_accuracy_gain"] >= .15,
        "best_fixed_confidence": gain["source_group_bootstrap_95ci"][0] > .10,
    }
    return {"checks": checks, "best_fixed": best_name, "comparison": gain,
            "passed": all(checks.values())}


def run(args) -> dict:
    import torch

    root, data_root = Path(args.output), Path(args.data_root)
    root.mkdir(parents=True)
    manifest = json.loads((data_root / "manifest.json").read_text())
    if manifest["group_counts"] != {"train": 64, "dev": 16, "test": 16}:
        raise ValueError("structured forgetting requires registered 64/16/16 corpus")
    train_pairs, dev_pairs = load_split(data_root, "train"), load_split(data_root, "dev")
    full_examples, full_meta = build_examples(train_pairs, True)
    blind_examples, blind_meta = build_examples(train_pairs, False)
    if [row["label"] for row in full_examples] != [row["label"] for row in blind_examples]:
        raise ValueError("ablation changed labels")
    start = time.perf_counter()
    learned, learned_curve = train_controller(full_examples, args.seed, args.epochs)
    blind, blind_curve = train_controller(blind_examples, args.seed, args.epochs)
    train_seconds = time.perf_counter() - start
    models = {"learned": learned, "history_blind": blind}
    policies = FIXED + ("random", "physics", "history_blind", "learned", "oracle")
    dev_records, dev_summaries = {}, {}
    for policy in policies:
        records = evaluate_policy(dev_pairs, policy, models)
        dev_records[policy] = records; dev_summaries[policy] = aggregate(records)
    dev_gate = gate(dev_summaries)
    test_scored, test_summaries = False, None
    if dev_gate["passed"]:
        test_pairs = load_split(data_root, "test")
        test_summaries = {policy: aggregate(evaluate_policy(test_pairs, policy, models))
                          for policy in policies}
        test_scored = True
    checkpoint = root / "controllers.pt"
    torch.save({"learned": learned.state_dict(), "history_blind": blind.state_dict(),
                "feature_dim": len(full_examples[0]["features"]), "actions": ACTIONS,
                "seed": args.seed, "epochs": args.epochs}, checkpoint)
    training = {"seed": args.seed, "epochs": args.epochs, "seconds": train_seconds,
                "full_examples": full_meta, "blind_examples": blind_meta,
                "learned_curve": learned_curve, "history_blind_curve": blind_curve,
                "checkpoint_sha256": sha(checkpoint)}
    write_json(root / "training.json", training)
    for policy, records in dev_records.items():
        path = root / f"dev_{policy}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
    run_manifest = {"protocol": "structured_forgetting_v1", "data_protocol": manifest["protocol"],
                    "data_manifest_sha256": sha(data_root / "manifest.json"),
                    "group_counts": manifest["group_counts"], "actions": ACTIONS,
                    "features": ["memory_size", "known_reference", "reference_last1",
                                 "reference_last2", "reference_deeper", "prelude_lines",
                                 "block_position", "fact_count", "active_key_count"],
                    "controller": "linear_softmax", "update_gate": 1,
                    "test_release_rule": "score once only if development gate passes",
                    "test_scored": test_scored}
    write_json(root / "training_manifest.json", run_manifest)
    result = {"protocol": "structured_forgetting_v1", "training": training,
              "dev": dev_summaries, "dev_gate": dev_gate, "test": test_summaries,
              "test_scored": test_scored, "stage_passed": dev_gate["passed"],
              "terminal_status": "completed_structured_forgetting"}
    write_json(root / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1517); parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
