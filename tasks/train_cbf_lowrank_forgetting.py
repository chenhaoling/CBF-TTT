"""Counterfactual forgetting on an additive low-rank fast-weight memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

from tasks.pack_cbf_event_warmup import sha
from tasks.train_cbf_associative_binding import _bootstrap
from tasks.train_cbf_event_writer import write_json
from tasks.train_cbf_structured_forgetting import (
    BATCH_RE, ENTITY_RE, VALUE_RE, boundary_truth, load_split, parse_block,
)


ACTIONS = ("retain", "half", "local", "global")
GLOBAL_ACTIONS = ("retain", "half", "global")
SEEDS = (1701, 1702, 1703)
POLICIES = ("fixed_retain", "fixed_half", "fixed_local", "fixed_global", "random",
            "reference_blind", "global_learned", "learned", "physics", "oracle")


class Embeddings:
    def __init__(self, seed: int, dim: int):
        self.seed, self.dim, self.cache = seed, dim, {}

    def get(self, kind: str, text: str):
        import torch

        key = (kind, text)
        if key not in self.cache:
            digest = hashlib.sha256(f"{self.seed}|{kind}|{text}".encode()).digest()
            generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big") % (2**63 - 1))
            value = torch.randn(self.dim, generator=generator, dtype=torch.float32)
            self.cache[key] = value / value.norm().clamp_min(1e-8)
        return self.cache[key]


def episode_candidates(target: dict) -> list[str]:
    values = {"unknown"}
    for events in target["semantic_events"]:
        for event in events:
            if event["op"] == "assign":
                values.update(event["facts"].values())
    return sorted(values)


def batch_delta(assignment: dict, block_index: int, embeddings: Embeddings):
    import torch

    delta = torch.zeros((embeddings.dim, embeddings.dim), dtype=torch.float32)
    strength = 1. + .15 * block_index
    for entity, value in assignment["facts"].items():
        delta += strength * torch.outer(embeddings.get("value", value), embeddings.get("entity", entity))
    return delta


def clone_state(state: list[dict]) -> list[dict]:
    return [{"batch": row["batch"], "facts": dict(row["facts"]),
             "delta": row["delta"], "weight": row["weight"]} for row in state]


def apply_action(state: list[dict], action: str, assignment: dict, delta) -> list[dict]:
    output = clone_state(state)
    if action == "half":
        for row in output:
            row["weight"] *= .5
    elif action == "global":
        output = []
    elif action == "local":
        target = assignment["old_refs"][0] if assignment["old_refs"] else None
        output = [row for row in output if row["batch"] != target]
    elif action != "retain":
        raise ValueError("unknown low-rank forgetting action")
    output.append({"batch": assignment["batch"], "facts": dict(assignment["facts"]),
                   "delta": delta, "weight": 1.})
    return output


def matrix(state: list[dict], dim: int):
    import torch

    output = torch.zeros((dim, dim), dtype=torch.float32)
    for row in state:
        output += row["weight"] * row["delta"]
    return output


def query_metrics(state: list[dict], query: dict, candidates: list[str], embeddings: Embeddings,
                  temperature: float = .2, unknown_raw: float = .35) -> dict:
    import torch

    memory = matrix(state, embeddings.dim)
    output = memory @ embeddings.get("entity", query["key"])
    logits = []
    for value in candidates:
        raw = unknown_raw if value == "unknown" else float(torch.dot(embeddings.get("value", value), output))
        logits.append(raw / temperature)
    log_probs = torch.log_softmax(torch.tensor(logits), dim=-1)
    truth = candidates.index(query["answer"])
    prediction = int(torch.tensor(logits).argmax())
    return {"nll": float(-log_probs[truth]), "correct": int(prediction == truth),
            "prediction": candidates[prediction]}


def action_outcomes(state, assignment, delta, truth, candidates, embeddings,
                    allowed=ACTIONS) -> dict:
    outcomes = {}
    base_mass = sum(row["weight"] for row in state)
    target = assignment["old_refs"][0] if assignment["old_refs"] else None
    target_mass = sum(row["weight"] for row in state if row["batch"] == target)
    costs = {"retain": 0., "half": .5 * base_mass, "local": target_mass,
             "global": base_mass}
    for action in allowed:
        candidate = apply_action(state, action, assignment, delta)
        metrics = [query_metrics(candidate, row, candidates, embeddings) for row in truth]
        outcomes[action] = {"nll": statistics.fmean(row["nll"] for row in metrics),
                            "accuracy": statistics.fmean(row["correct"] for row in metrics),
                            "cost": costs[action]}
    best_nll = min(row["nll"] for row in outcomes.values())
    near = [action for action in allowed if outcomes[action]["nll"] <= best_nll + 1e-6]
    label = min(near, key=lambda action: (outcomes[action]["cost"], allowed.index(action)))
    return {"label": label, "actions": outcomes, "tied": len(near) > 1}


def features(state, assignment, delta, block_index: int, history: bool = True) -> list[float]:
    import torch

    dim = delta.shape[0]
    memory = matrix(state, dim)
    memory_norm, delta_norm = float(memory.norm()), float(delta.norm())
    alignment = float(torch.sum(memory * delta) / (memory.norm() * delta.norm()).clamp_min(1e-8))
    target = assignment["old_refs"][0] if assignment["old_refs"] else None
    position = next((index for index, row in enumerate(state) if row["batch"] == target), None)
    target_row = None if position is None else state[position]
    age = 0. if position is None else (len(state) - position) / 12.
    target_ratio = 0. if target_row is None else float(
        (target_row["weight"] * target_row["delta"].norm()) / memory.norm().clamp_min(1e-8))
    old_facts = {key: value for row in state for key, value in row["facts"].items()}
    overlap = sum(key in old_facts for key in assignment["facts"]) / len(assignment["facts"])
    duplicate = sum(old_facts.get(key) == value for key, value in assignment["facts"].items()) / len(assignment["facts"])
    reference = [float(target is not None), float(position is not None), age, target_ratio]
    if not history:
        reference = [0., 0., 0., 0.]
    return [math.log1p(memory_norm) / 5., math.log1p(delta_norm) / 5., alignment,
            *reference, min(assignment["prelude_lines"], 2) / 2., block_index / 8.,
            len(state) / 8., len(old_facts) / 20., overlap, duplicate]


def build_examples(pairs, embeddings: Embeddings, allowed=ACTIONS, history=True):
    rows, ties = [], 0
    for inp, target in pairs:
        state, candidates = [], episode_candidates(target)
        boundaries = set(inp["decision_boundaries"])
        for block in inp["blocks"]:
            assignment = parse_block(block["text"], state)
            delta = batch_delta(assignment, block["index"], embeddings)
            if block["index"] in boundaries:
                truth = boundary_truth(inp, target, block["index"])
                outcome = action_outcomes(state, assignment, delta, truth, candidates, embeddings, allowed)
                ties += outcome["tied"]
                rows.append({"group_id": inp["group_id"], "episode_id": inp["id"],
                             "block_index": block["index"],
                             "features": features(state, assignment, delta, block["index"], history),
                             "label": allowed.index(outcome["label"]), "action": outcome["label"],
                             "outcomes": outcome["actions"]})
                state = apply_action(state, outcome["label"], assignment, delta)
            else:
                state = apply_action(state, "retain", assignment, delta)
    counts = {action: sum(row["action"] == action for row in rows) for action in allowed}
    return rows, {"examples": len(rows), "ties": ties, "label_counts": counts}


def train_controller(examples, actions, seed: int, epochs: int):
    import torch

    x = torch.tensor([row["features"] for row in examples], dtype=torch.float32)
    y = torch.tensor([row["label"] for row in examples], dtype=torch.long)
    mean, std = x.mean(0), x.std(0).clamp_min(1e-5)
    x = (x - mean) / std
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 32), torch.nn.Tanh(),
                                torch.nn.Linear(32, len(actions)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=1e-4)
    curve = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True); logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y); loss.backward(); optimizer.step()
        if epoch in (1, 10, 50, 100, 250, epochs):
            curve.append({"epoch": epoch, "loss": float(loss.detach()),
                          "accuracy": float((logits.detach().argmax(-1) == y).float().mean())})
    return {"model": model.eval(), "mean": mean, "std": std, "actions": tuple(actions)}, curve


def controller_action(controller, values) -> str:
    import torch

    x = (torch.tensor([values], dtype=torch.float32) - controller["mean"]) / controller["std"]
    with torch.no_grad():
        return controller["actions"][int(controller["model"](x).argmax(-1))]


def select_action(policy, state, assignment, delta, block_index, truth, candidates,
                  embeddings, controllers, episode_id):
    if policy.startswith("fixed_"):
        return policy.removeprefix("fixed_")
    if policy == "random":
        digest = hashlib.sha256(f"{embeddings.seed}|{episode_id}|{block_index}".encode()).hexdigest()
        return ACTIONS[int(digest[:8], 16) % len(ACTIONS)]
    if policy == "physics":
        return "local" if assignment["old_refs"] else "global" if assignment["prelude_lines"] else "retain"
    if policy == "oracle":
        return action_outcomes(state, assignment, delta, truth, candidates, embeddings)["label"]
    if policy == "learned":
        return controller_action(controllers["learned"], features(state, assignment, delta, block_index, True))
    if policy == "reference_blind":
        return controller_action(controllers["reference_blind"], features(state, assignment, delta, block_index, False))
    if policy == "global_learned":
        return controller_action(controllers["global_learned"], features(state, assignment, delta, block_index, True))
    raise ValueError("unknown low-rank policy")


def evaluate_policy(pairs, policy, embeddings, controllers):
    records = []
    for inp, target in pairs:
        state, queries, decisions = [], [], []
        candidates, boundaries = episode_candidates(target), set(inp["decision_boundaries"])
        for block in inp["blocks"]:
            assignment = parse_block(block["text"], state)
            delta = batch_delta(assignment, block["index"], embeddings)
            if block["index"] in boundaries:
                truth = boundary_truth(inp, target, block["index"])
                action = select_action(policy, state, assignment, delta, block["index"], truth,
                                       candidates, embeddings, controllers, inp["id"])
                oracle = action_outcomes(state, assignment, delta, truth, candidates, embeddings)["label"]
                state = apply_action(state, action, assignment, delta)
                decisions.append({"block": block["index"], "selected": action, "oracle": oracle})
                for row in truth:
                    queries.append({**row, **query_metrics(state, row, candidates, embeddings)})
            else:
                state = apply_action(state, "retain", assignment, delta)
        records.append({"episode_id": inp["id"], "group_id": inp["group_id"],
                        "family": target["family"], "policy": policy,
                        "queries": queries, "decisions": decisions})
    return records


def aggregate(records):
    queries = [(row, query) for row in records for query in row["queries"]]
    decisions = [item for row in records for item in row["decisions"]]
    groups = {}
    for group in sorted({row["group_id"] for row in records}):
        subset = [query for row, query in queries if row["group_id"] == group]
        groups[group] = {"accuracy": statistics.fmean(row["correct"] for row in subset),
                         "nll": statistics.fmean(row["nll"] for row in subset)}
    return {"episodes": len(records), "queries": len(queries), "decisions": len(decisions),
            "accuracy": statistics.fmean(row["correct"] for _, row in queries),
            "nll": statistics.fmean(row["nll"] for _, row in queries),
            "action_accuracy": statistics.fmean(row["selected"] == row["oracle"] for row in decisions),
            "groups": groups,
            "by_family": {family: {"accuracy": statistics.fmean(q["correct"] for r, q in queries if r["family"] == family),
                                    "nll": statistics.fmean(q["nll"] for r, q in queries if r["family"] == family)}
                          for family in sorted({row["family"] for row in records})}}


def multi_seed_summary(per_seed: dict) -> dict:
    policies = next(iter(per_seed.values())).keys()
    output = {}
    for policy in policies:
        output[policy] = {
            "mean_accuracy": statistics.fmean(per_seed[str(seed)][policy]["accuracy"] for seed in SEEDS),
            "mean_nll": statistics.fmean(per_seed[str(seed)][policy]["nll"] for seed in SEEDS),
            "min_seed_accuracy": min(per_seed[str(seed)][policy]["accuracy"] for seed in SEEDS),
            "per_seed": {str(seed): per_seed[str(seed)][policy] for seed in SEEDS},
        }
    return output


def compare(primary: dict, baseline: dict, seed: int) -> dict:
    groups = sorted(next(iter(primary["per_seed"].values()))["groups"])
    gains = []
    nll = []
    for group in groups:
        gains.append(statistics.fmean(primary["per_seed"][str(s)]["groups"][group]["accuracy"] -
                                      baseline["per_seed"][str(s)]["groups"][group]["accuracy"] for s in SEEDS))
        nll.append(statistics.fmean(baseline["per_seed"][str(s)]["groups"][group]["nll"] -
                                    primary["per_seed"][str(s)]["groups"][group]["nll"] for s in SEEDS))
    return {"accuracy_gain": statistics.fmean(gains), "accuracy_bootstrap_95ci": _bootstrap(gains, seed),
            "nll_gain": statistics.fmean(nll), "nll_bootstrap_95ci": _bootstrap(nll, seed + 1)}


def development_gate(summary):
    fixed = ("fixed_retain", "fixed_half", "fixed_local", "fixed_global")
    best_fixed = max(fixed, key=lambda name: summary[name]["mean_accuracy"])
    stronger = max((best_fixed, "global_learned"), key=lambda name: summary[name]["mean_accuracy"])
    learned, oracle = summary["learned"], summary["oracle"]
    gain = compare(learned, summary[stronger], 1731)
    checks = {
        "oracle_readable": oracle["mean_accuracy"] >= .90,
        "learned_accuracy": learned["mean_accuracy"] >= .88,
        "all_seeds": learned["min_seed_accuracy"] >= .85,
        "near_oracle": oracle["mean_accuracy"] - learned["mean_accuracy"] <= .03,
        "control_margin": gain["accuracy_gain"] >= .08,
        "control_confidence": gain["accuracy_bootstrap_95ci"][0] > .05,
    }
    return {"checks": checks, "best_fixed": best_fixed, "stronger_control": stronger,
            "comparison": gain, "passed": all(checks.values())}


def serialize_controller(controller):
    return {"state_dict": controller["model"].state_dict(), "mean": controller["mean"],
            "std": controller["std"], "actions": controller["actions"]}


def run(args):
    import torch

    root, data_root = Path(args.output), Path(args.data_root); root.mkdir(parents=True)
    manifest = json.loads((data_root / "manifest.json").read_text())
    if manifest["group_counts"] != {"train": 128, "dev": 32, "test": 32}:
        raise ValueError("low-rank experiment requires registered 128/32/32 corpus")
    train_pairs, dev_pairs = load_split(data_root, "train"), load_split(data_root, "dev")
    all_controllers, training, dev_seed = {}, {}, {}
    start = time.perf_counter()
    for memory_seed in SEEDS:
        embeddings = Embeddings(memory_seed, args.dim)
        full, full_meta = build_examples(train_pairs, embeddings, ACTIONS, True)
        blind, blind_meta = build_examples(train_pairs, embeddings, ACTIONS, False)
        global_rows, global_meta = build_examples(train_pairs, embeddings, GLOBAL_ACTIONS, True)
        learned, curve = train_controller(full, ACTIONS, args.seed + memory_seed, args.epochs)
        reference_blind, blind_curve = train_controller(blind, ACTIONS, args.seed + memory_seed, args.epochs)
        global_learned, global_curve = train_controller(global_rows, GLOBAL_ACTIONS,
                                                         args.seed + memory_seed, args.epochs)
        controllers = {"learned": learned, "reference_blind": reference_blind,
                       "global_learned": global_learned}
        all_controllers[str(memory_seed)] = {name: serialize_controller(value)
                                             for name, value in controllers.items()}
        training[str(memory_seed)] = {"full": full_meta, "reference_blind": blind_meta,
                                      "global_only": global_meta, "curve": curve,
                                      "reference_blind_curve": blind_curve, "global_curve": global_curve}
        dev_seed[str(memory_seed)] = {}
        for policy in POLICIES:
            records = evaluate_policy(dev_pairs, policy, embeddings, controllers)
            dev_seed[str(memory_seed)][policy] = aggregate(records)
            path = root / f"dev_seed{memory_seed}_{policy}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
    dev = multi_seed_summary(dev_seed); gate = development_gate(dev)
    test_scored, test_seed, test = False, None, None
    if gate["passed"]:
        test_pairs = load_split(data_root, "test"); test_seed = {}
        for memory_seed in SEEDS:
            embeddings = Embeddings(memory_seed, args.dim)
            controllers = restore_controllers(all_controllers[str(memory_seed)], args.dim)
            test_seed[str(memory_seed)] = {policy: aggregate(evaluate_policy(
                test_pairs, policy, embeddings, controllers)) for policy in POLICIES}
        test = multi_seed_summary(test_seed); test_scored = True
    checkpoint = root / "controllers.pt"
    torch.save({"controllers": all_controllers, "dim": args.dim, "seeds": SEEDS,
                "epochs": args.epochs, "seed": args.seed}, checkpoint)
    training_result = {"seconds": time.perf_counter() - start, "seeds": training,
                       "checkpoint_sha256": sha(checkpoint)}
    write_json(root / "training.json", training_result)
    run_manifest = {"protocol": "lowrank_forgetting_v1", "data_manifest_sha256": sha(data_root / "manifest.json"),
                    "group_counts": manifest["group_counts"], "memory_seeds": SEEDS,
                    "dim": args.dim, "actions": ACTIONS, "global_grid": [0., .5, 1.],
                    "test_release_rule": "score once only after aggregate development gate", "test_scored": test_scored}
    write_json(root / "training_manifest.json", run_manifest)
    result = {"protocol": "lowrank_forgetting_v1", "dev": dev, "dev_gate": gate,
              "test": test, "test_scored": test_scored, "stage_passed": gate["passed"],
              "training_seconds": training_result["seconds"], "terminal_status": "completed_lowrank_forgetting"}
    write_json(root / "summary.json", result); return result


def restore_controllers(saved, dim):
    import torch

    output = {}
    for name, row in saved.items():
        actions = tuple(row["actions"])
        model = torch.nn.Sequential(torch.nn.Linear(len(row["mean"]), 32), torch.nn.Tanh(),
                                    torch.nn.Linear(32, len(actions)))
        model.load_state_dict(row["state_dict"])
        output[name] = {"model": model.eval(), "mean": row["mean"], "std": row["std"], "actions": actions}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--dim", type=int, default=64); parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1721)
    args = parser.parse_args(); print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
