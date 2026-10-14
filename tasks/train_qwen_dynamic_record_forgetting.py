"""Train the preregistered dynamic record-level forgetting controller."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from tasks.build_qwen_dynamic_record_forgetting import (
    DATA_SEED,
    FAMILIES,
    PROTOCOL as DATA_PROTOCOL,
    load_split,
    parse_visible_block,
)
from tasks.diagnose_qwen_sparse_reader import (
    apply_retention as apply_cohort_retention,
    clone_state,
    query_metrics,
)
from tasks.pack_cbf_event_warmup import sha
from tasks.pilot_qwen_dynamic_record_forgetting import (
    ACTIONS,
    GRID,
    PROTOCOL as RESOURCE_PROTOCOL,
    READER,
    _assignment,
    apply_dynamic_action,
    candidate_indices,
    lexicographic_label,
    score_action,
    valid_actions,
)
from tasks.train_cbf_event_writer import write_json
from tasks.train_cbf_lowrank_forgetting import episode_candidates
from tasks.train_cbf_qwen_lowrank_forgetting import SEEDS
from tasks.train_cbf_structured_forgetting import boundary_truth
from tasks.train_qwen_cohort_forgetting import (
    FEATURE_NAMES as COHORT_FEATURE_NAMES,
    GEOMETRY_HASHES,
    aggregate as base_aggregate,
    batch_delta,
    comparison,
    controller_retention,
    extract_embeddings,
    features as cohort_features,
    move_embeddings,
    physics_retention,
    restore_controllers,
)


PROTOCOL = "qwen_dynamic_record_forgetting_development_v1"
CONTROLLER_SEEDS = {2301: 5101, 2302: 5102, 2303: 5103}
GLOBAL_FEATURES = (
    "memory_log_norm",
    "delta_log_norm",
    "memory_delta_cosine",
    "block_fraction",
    "record_fraction",
    "active_fraction",
    "reference_fraction",
    "reset",
    "current_fact_fraction",
    "key_overlap_fraction",
    "exact_duplicate_fraction",
    "slot_fraction",
)
SLOT_FEATURES = (
    "present",
    "referenced",
    "age_fraction",
    "log_age",
    "weight",
    "fact_fraction",
    "key_overlap_fraction",
    "exact_duplicate_fraction",
    "max_key_cosine",
    "mean_key_cosine",
    "overlap_value_cosine",
    "delta_cosine",
    "norm_ratio",
    "active_source_fraction",
)
FEATURE_DIM = len(GLOBAL_FEATURES) + 3 * len(SLOT_FEATURES)
PRESENT_INDICES = tuple(
    len(GLOBAL_FEATURES) + slot * len(SLOT_FEATURES) for slot in range(3)
)
POLICIES = (
    "fixed_retain",
    "deterministic_random",
    "old_cohort",
    "cohort_physics",
    "exact_reference",
    "dynamic_rules",
    "slot_mlp",
    "set_unconstrained",
    "primary",
    "oracle",
)


def _cosine(left, right):
    import torch
    return float(torch.sum(left * right) / (left.norm() * right.norm()).clamp_min(1e-8))


def dynamic_features(state, assignment, delta, block_index, slots, embeddings):
    import torch

    active = [row for row in state if row["weight"] > 1e-12]
    memory = torch.zeros_like(delta)
    for row in state:
        memory += row["weight"] * row["delta"]
    memory_norm, delta_norm = memory.norm(), delta.norm()
    current_keys = list(assignment["facts"])
    old_facts = {key: value for row in active for key, value in row["facts"].items()}
    overlap = sum(key in old_facts for key in current_keys) / len(current_keys)
    duplicate = sum(
        old_facts.get(key) == value for key, value in assignment["facts"].items()
    ) / len(current_keys)
    global_values = [
        math.log1p(float(memory_norm)) / 5.0,
        math.log1p(float(delta_norm)) / 5.0,
        _cosine(memory, delta) if float(memory_norm) else 0.0,
        block_index / 16.0,
        len(state) / 16.0,
        len(active) / 16.0,
        len(assignment["old_refs"]) / 2.0,
        float(assignment["reset"]),
        len(current_keys) / 4.0,
        overlap,
        duplicate,
        len(slots) / 3.0,
    ]
    newest_source = {}
    for index, row in enumerate(state):
        if row["weight"] > 1e-12:
            for key in row["facts"]:
                newest_source[key] = index
    slot_values = []
    for slot in range(3):
        if slot >= len(slots):
            slot_values.extend([0.0] * len(SLOT_FEATURES))
            continue
        index, row = slots[slot], state[slots[slot]]
        age = len(state) - index
        keys = list(row["facts"])
        overlapping = set(keys).intersection(current_keys)
        key_cosines = [
            float(__import__("torch").dot(
                embeddings.get("entity", old), embeddings.get("entity", current)
            ))
            for old in keys for current in current_keys
        ]
        value_cosines = [
            float(__import__("torch").dot(
                embeddings.get("value", row["facts"][key]),
                embeddings.get("value", assignment["facts"][key]),
            ))
            for key in overlapping
        ]
        active_source = sum(newest_source.get(key) == index for key in keys) / len(keys)
        row_norm = row["delta"].norm()
        slot_values.extend([
            1.0,
            float(row["batch"] in assignment["old_refs"]),
            age / 16.0,
            math.log1p(age) / math.log(17.0),
            float(row["weight"]),
            len(keys) / 4.0,
            len(overlapping) / len(current_keys),
            sum(
                key in assignment["facts"]
                and assignment["facts"][key] == value
                for key, value in row["facts"].items()
            ) / len(current_keys),
            max(key_cosines),
            statistics.fmean(key_cosines),
            statistics.fmean(value_cosines) if value_cosines else 0.0,
            _cosine(row["delta"], delta),
            float(row_norm / memory_norm.clamp_min(1e-8)),
            active_source,
        ])
    values = global_values + slot_values
    if len(values) != FEATURE_DIM:
        raise RuntimeError("dynamic feature dimension mismatch")
    return values


def build_examples(pairs, embeddings):
    rows = []
    for inp, target in pairs:
        state, candidates = [], episode_candidates(target)
        boundaries = set(inp["decision_boundaries"])
        for block in inp["blocks"]:
            assignment = _assignment(parse_visible_block(block["text"], inp["split"]))
            delta = batch_delta(assignment, block["index"], embeddings)
            slots = candidate_indices(state, assignment, embeddings)
            if block["index"] not in boundaries:
                state = apply_dynamic_action(
                    state, assignment, delta, block["index"], slots,
                    (0, (1.0, 1.0, 1.0)),
                )
                continue
            truth = boundary_truth(inp, target, block["index"])
            actions = valid_actions(assignment, slots)
            outcomes = {
                action: score_action(
                    state, assignment, delta, block["index"], slots, action,
                    truth, candidates, embeddings,
                )
                for action in actions
            }
            label = lexicographic_label(outcomes, actions)
            referenced = set(assignment["old_refs"])
            selected = {state[index]["batch"] for index in slots}
            rows.append({
                "group_id": inp["group_id"],
                "episode_id": inp["id"],
                "twin_key": inp["id"].rsplit(".", 1)[0] + f".{block['index']}",
                "family": target["family"],
                "block_index": block["index"],
                "features": dynamic_features(
                    state, assignment, delta, block["index"], slots, embeddings
                ),
                "label": ACTIONS.index(label),
                "action": [label[0], list(label[1])],
                "valid_mask": [action in actions for action in ACTIONS],
                "selection_recall": (
                    1.0 if not referenced else len(referenced & selected) / len(referenced)
                ),
            })
            state = apply_dynamic_action(
                state, assignment, delta, block["index"], slots, label
            )
    return rows, {
        "examples": len(rows),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "interior_fraction": statistics.fmean(
            any(value == 0.5 for value in row["action"][1]) for row in rows
        ),
        "selection_recall": statistics.fmean(row["selection_recall"] for row in rows),
    }


def twin_pairs(rows):
    buckets = {}
    for index, row in enumerate(rows):
        buckets.setdefault(row["twin_key"], []).append(index)
    pairs = []
    for values in buckets.values():
        if len(values) != 2:
            raise ValueError("training twin key does not contain two examples")
        pairs.append(values)
    return pairs


def _set_model():
    import torch

    class RecordSetController(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.slot = torch.nn.Sequential(
                torch.nn.Linear(len(SLOT_FEATURES), 48),
                torch.nn.Tanh(),
                torch.nn.Linear(48, 32),
                torch.nn.Tanh(),
            )
            self.global_encoder = torch.nn.Sequential(
                torch.nn.Linear(len(GLOBAL_FEATURES), 32), torch.nn.Tanh()
            )
            self.context = torch.nn.Sequential(
                torch.nn.Linear(32 + 32 + 32, 96),
                torch.nn.Tanh(),
                torch.nn.Linear(96, 64),
                torch.nn.Tanh(),
            )
            self.clear_head = torch.nn.Linear(64, 2)
            self.state_head = torch.nn.Sequential(
                torch.nn.Linear(32 + 64, 48),
                torch.nn.Tanh(),
                torch.nn.Linear(48, 3),
            )
            self.register_buffer(
                "clear_indices",
                torch.tensor([action[0] for action in ACTIONS], dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "state_indices",
                torch.tensor([
                    [int(round(value * 2.0)) for value in action[1]]
                    for action in ACTIONS
                ], dtype=torch.long),
                persistent=False,
            )

        def forward(self, values):
            global_values = values[:, : len(GLOBAL_FEATURES)]
            slot_values = values[:, len(GLOBAL_FEATURES) :].reshape(
                -1, 3, len(SLOT_FEATURES)
            )
            mask = slot_values[:, :, :1]
            encoded = self.slot(slot_values) * mask
            pooled_mean = (encoded * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            pooled_max = encoded.masked_fill(mask == 0, -1e4).max(1).values
            pooled_max = __import__("torch").where(
                mask.sum(1) > 0, pooled_max, __import__("torch").zeros_like(pooled_max)
            )
            context = self.context(__import__("torch").cat([
                self.global_encoder(global_values), pooled_mean, pooled_max,
            ], dim=1))
            shared_context = context[:, None, :].expand(-1, 3, -1)
            state_scores = self.state_head(__import__("torch").cat([
                encoded, shared_context,
            ], dim=2))
            logits = self.clear_head(context)[:, self.clear_indices]
            # Each slot uses the same state scorer.  Permuting records therefore
            # permutes the corresponding action coordinates without changing the
            # physical decision function.
            for slot in range(3):
                logits = logits + state_scores[:, slot, :][:, self.state_indices[:, slot]]
            return logits

    return RecordSetController()


def _flat_model():
    import torch
    return torch.nn.Sequential(
        torch.nn.Linear(FEATURE_DIM, 96), torch.nn.Tanh(),
        torch.nn.Linear(96, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, len(ACTIONS)),
    )


def train_controller(rows, seed, epochs, kind, constrained):
    import torch

    x = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.long)
    masks = torch.tensor([row["valid_mask"] for row in rows], dtype=torch.bool)
    mean, std = x.mean(0), x.std(0).clamp_min(1e-5)
    normalized = (x - mean) / std
    normalized[:, list(PRESENT_INDICES)] = x[:, list(PRESENT_INDICES)]
    torch.manual_seed(seed)
    model = _set_model() if kind == "set" else _flat_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=1e-4)
    pairs = twin_pairs(rows)
    left = torch.tensor([row[0] for row in pairs]); right = torch.tensor([row[1] for row in pairs])
    curve = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        raw = model(normalized)
        logits = raw.masked_fill(~masks, -1e9) if constrained else raw
        classification = torch.nn.functional.cross_entropy(logits, labels)
        probabilities = torch.softmax(logits, dim=1)
        consistency = torch.nn.functional.mse_loss(
            probabilities[left], probabilities[right]
        )
        loss = classification + 0.1 * consistency
        loss.backward(); optimizer.step()
        if epoch in (1, 10, 50, 100, 200, epochs):
            curve.append({
                "epoch": epoch,
                "loss": float(loss.detach()),
                "classification": float(classification.detach()),
                "consistency": float(consistency.detach()),
                "accuracy": float((logits.detach().argmax(1) == labels).float().mean()),
            })
    return {
        "kind": kind,
        "constrained": constrained,
        "model": model.eval(),
        "mean": mean,
        "std": std,
    }, curve


def predict(controller, features, valid_mask):
    import torch
    raw = torch.tensor([features], dtype=torch.float32)
    values = (raw - controller["mean"]) / controller["std"]
    values[:, list(PRESENT_INDICES)] = raw[:, list(PRESENT_INDICES)]
    with torch.no_grad():
        logits = controller["model"](values)
    if controller["constrained"]:
        mask = torch.tensor([valid_mask], dtype=torch.bool)
        logits = logits.masked_fill(~mask, -1e9)
    return ACTIONS[int(logits.argmax(1))]


def serialize_controller(controller):
    return {
        "kind": controller["kind"],
        "constrained": controller["constrained"],
        "mean": controller["mean"],
        "std": controller["std"],
        "state": controller["model"].state_dict(),
    }


def restore_controller(saved):
    model = _set_model() if saved["kind"] == "set" else _flat_model()
    model.load_state_dict(saved["state"])
    return {
        "kind": saved["kind"], "constrained": saved["constrained"],
        "mean": saved["mean"], "std": saved["std"], "model": model.eval(),
    }


def exact_state(state, assignment, delta, block_index):
    output = clone_state(state)
    if assignment["reset"]:
        for row in output:
            row["weight"] = 0.0
    for reference in assignment["old_refs"]:
        for row in output:
            if row["batch"] == reference:
                row["weight"] = 0.0
    output.append({
        "batch": assignment["batch"], "facts": dict(assignment["facts"]),
        "delta": delta, "weight": 1.0, "block_index": int(block_index),
    })
    return output


def rule_action(state, assignment, slots, actions):
    if assignment["reset"]:
        return min((action for action in actions if action[0]), key=ACTIONS.index)
    retention = [1.0, 1.0, 1.0]
    references = set(assignment["old_refs"])
    for slot, index in enumerate(slots):
        if state[index]["batch"] in references:
            retention[slot] = 0.0
    action = (0, tuple(retention))
    if action not in actions:
        raise ValueError("dynamic physical rule produced an invalid action")
    return action


def evaluate_policy(pairs, policy, embeddings, controllers, old_controller):
    records = []
    for inp, target in pairs:
        state, queries, decisions = [], [], []
        candidates, boundaries = episode_candidates(target), set(inp["decision_boundaries"])
        for block in inp["blocks"]:
            assignment = _assignment(parse_visible_block(block["text"], inp["split"]))
            delta = batch_delta(assignment, block["index"], embeddings)
            slots = candidate_indices(state, assignment, embeddings)
            if block["index"] not in boundaries:
                state = apply_dynamic_action(
                    state, assignment, delta, block["index"], slots,
                    (0, (1.0, 1.0, 1.0)),
                )
                continue
            truth = boundary_truth(inp, target, block["index"])
            actions = valid_actions(assignment, slots)
            outcomes = {
                action: score_action(
                    state, assignment, delta, block["index"], slots, action,
                    truth, candidates, embeddings,
                )
                for action in actions
            }
            oracle = lexicographic_label(outcomes, actions)
            selected_batches = {state[index]["batch"] for index in slots}
            direct, action = None, None
            if policy == "fixed_retain":
                action = (0, (1.0, 1.0, 1.0))
            elif policy == "deterministic_random":
                value = int(hashlib.sha256(
                    f"{embeddings.seed}|{inp['id']}|{block['index']}".encode()
                ).hexdigest()[:12], 16)
                action = actions[value % len(actions)]
            elif policy == "dynamic_rules":
                action = rule_action(state, assignment, slots, actions)
            elif policy == "exact_reference":
                direct = exact_state(state, assignment, delta, block["index"])
            elif policy == "cohort_physics":
                retention = physics_retention(state, assignment)
                direct = apply_cohort_retention(
                    state, retention, assignment, delta, block["index"]
                )
            elif policy == "old_cohort":
                retention = controller_retention(
                    old_controller,
                    cohort_features(state, assignment, delta, block["index"], True),
                )
                direct = apply_cohort_retention(
                    state, retention, assignment, delta, block["index"]
                )
            elif policy == "oracle":
                action = oracle
            elif policy in controllers:
                feature_values = dynamic_features(
                    state, assignment, delta, block["index"], slots, embeddings
                )
                action = predict(
                    controllers[policy], feature_values,
                    [candidate in actions for candidate in ACTIONS],
                )
            else:
                raise ValueError(f"unknown policy: {policy}")
            violation = int(action is not None and action not in actions)
            if direct is None:
                state = apply_dynamic_action(
                    state, assignment, delta, block["index"], slots, action
                )
            else:
                state = direct
            references = set(assignment["old_refs"])
            recall = (
                1.0 if not references else
                len(references & selected_batches) / len(references)
            )
            comparable = action is not None
            decisions.append({
                "block": block["index"],
                "retention": None if action is None else [action[0], *action[1]],
                "oracle_retention": [oracle[0], *oracle[1]],
                "retention_mae": (
                    statistics.fmean(abs(a - b) for a, b in zip(
                        [action[0], *action[1]], [oracle[0], *oracle[1]]
                    )) if comparable else None
                ),
                "grid_action_match": int(action == oracle) if comparable else None,
                "retained_mass": sum(row["weight"] for row in state),
                "selection_recall": recall,
                "constraint_violation": violation,
                "oracle_interior": any(value == 0.5 for value in oracle[1]),
            })
            for row in truth:
                queries.append({
                    **row,
                    **query_metrics(state, row, candidates, embeddings, READER),
                })
        records.append({
            "episode_id": inp["id"], "group_id": inp["group_id"],
            "family": target["family"], "policy": policy, "reader": READER,
            "queries": queries, "decisions": decisions,
        })
    return records


def aggregate(records):
    output = base_aggregate(records)
    decisions = [row for episode in records for row in episode["decisions"]]
    output["selection_recall"] = statistics.fmean(
        row["selection_recall"] for row in decisions
    )
    output["constraint_violations"] = sum(
        row["constraint_violation"] for row in decisions
    )
    for interior in (False, True):
        bucket = [
            row for row in decisions
            if row["oracle_interior"] is interior and row["grid_action_match"] is not None
        ]
        output[("interior" if interior else "endpoint") + "_action_accuracy"] = (
            statistics.fmean(row["grid_action_match"] for row in bucket) if bucket else None
        )
        output[("interior" if interior else "endpoint") + "_decisions"] = len(bucket)
    return output


def verify_inputs(data_root, geometry_root, resource_audit, confirmation_root):
    manifest = json.loads((Path(data_root) / "manifest.json").read_text())
    if (
        manifest.get("protocol") != DATA_PROTOCOL
        or manifest.get("phase") != "development"
        or manifest.get("seed") != DATA_SEED
        or manifest.get("group_counts") != {"train": 512, "dev": 64}
        or manifest.get("sealed_test_generated") is not False
        or manifest.get("external_ood_generated") is not False
    ):
        raise ValueError("unexpected dynamic-record development corpus")
    audit = json.loads(Path(resource_audit).read_text())
    if (
        audit.get("protocol") != "audit_qwen_dynamic_record_resource_pilot_v1"
        or audit.get("audit_passed") is not True
        or audit.get("summary", {}).get("selected_formal_train_groups") != 512
    ):
        raise ValueError("dynamic resource pilot did not release 512 groups")
    observed = {name: sha(Path(geometry_root) / name) for name in GEOMETRY_HASHES}
    if observed != GEOMETRY_HASHES:
        raise ValueError("frozen contextual geometry dependency changed")
    confirmation_root = Path(confirmation_root)
    confirmation_manifest = json.loads(
        (confirmation_root / "result" / "manifest.json").read_text()
    )
    confirmation_summary = json.loads(
        (confirmation_root / "result" / "summary.json").read_text()
    )
    confirmation_audit = json.loads(
        (confirmation_root / "execution_audit.json").read_text()
    )
    if (
        confirmation_manifest.get("protocol") != "qwen_discrete_cohort_confirmation_v1"
        or confirmation_manifest.get("reader") != READER
        or confirmation_manifest.get("projection_seeds") != list(SEEDS)
        or confirmation_summary.get("stage_passed") is not True
        or confirmation_audit.get("protocol")
        != "audit_qwen_discrete_cohort_confirmation_v1"
        or confirmation_audit.get("audit_passed") is not True
    ):
        raise ValueError("old cohort comparator is not the audited passed confirmation")
    checkpoint = confirmation_root / "result" / "controllers.pt"
    if not checkpoint.exists():
        raise ValueError("old cohort comparator checkpoint is missing")
    if sha(checkpoint) != confirmation_audit.get("checkpoint_sha256"):
        raise ValueError("old cohort comparator checkpoint hash mismatch")
    return manifest, audit, observed, checkpoint


def run(args):
    import torch

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    data_manifest, resource_audit, geometry_hashes, old_checkpoint = verify_inputs(
        args.data_root, args.geometry_root, args.resource_audit, args.confirmation_root
    )
    requested = tuple(int(value) for value in args.projection_seeds.split(",") if value)
    if not requested or any(seed not in SEEDS for seed in requested):
        raise ValueError("projection seeds must be a nonempty registered subset")
    output.mkdir(parents=True)
    write_json(output / "progress.json", {"stage": "embedding_extraction"})
    train_pairs = load_split(args.data_root, "train")
    dev_pairs = load_split(args.data_root, "dev")
    transform_artifact = torch.load(
        Path(args.geometry_root) / "contextual_transforms.pt",
        map_location="cpu", weights_only=False,
    )
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    embeddings, embedding_hashes, extraction, model_meta = extract_embeddings(
        train_pairs + dev_pairs, transform_artifact["transforms"],
        args.model, args.tokenizer, args.batch_size,
    )
    extraction_peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    extraction_peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    old_saved = torch.load(old_checkpoint, map_location="cpu", weights_only=True)
    results, training_meta, checkpoint_rows, raw_hashes = {}, {}, {}, {}
    for seed in requested:
        write_json(output / "progress.json", {"stage": "labels", "seed": seed})
        memory = move_embeddings(embeddings[str(seed)], args.device)
        train_rows, meta = build_examples(train_pairs, memory)
        write_json(output / "progress.json", {"stage": "training", "seed": seed})
        primary, primary_curve = train_controller(
            train_rows, CONTROLLER_SEEDS[seed], args.epochs, "set", True
        )
        unconstrained, unconstrained_curve = train_controller(
            train_rows, CONTROLLER_SEEDS[seed] + 10, args.epochs, "set", False
        )
        flat, flat_curve = train_controller(
            train_rows, CONTROLLER_SEEDS[seed] + 20, args.epochs, "flat", True
        )
        controllers = {
            "primary": primary, "set_unconstrained": unconstrained, "slot_mlp": flat,
        }
        old = restore_controllers(
            old_saved["controllers"][str(seed)], len(COHORT_FEATURE_NAMES)
        )["grid_classifier"]
        results[str(seed)] = {}
        for policy in POLICIES:
            write_json(
                output / "progress.json",
                {"stage": "development", "seed": seed, "policy": policy},
            )
            records = evaluate_policy(dev_pairs, policy, memory, controllers, old)
            results[str(seed)][policy] = aggregate(records)
            path = output / f"dev_seed{seed}_{policy}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            raw_hashes[path.name] = sha(path)
        checkpoint_rows[str(seed)] = {
            name: serialize_controller(controller) for name, controller in controllers.items()
        }
        training_meta[str(seed)] = {
            **meta,
            "curves": {
                "primary": primary_curve,
                "set_unconstrained": unconstrained_curve,
                "slot_mlp": flat_curve,
            },
        }
        del memory
        torch.cuda.empty_cache()
    checkpoint_path = output / "controllers.pt"
    torch.save({
        "protocol": PROTOCOL,
        "projection_seeds": list(requested),
        "controller_seeds": CONTROLLER_SEEDS,
        "features": {"global": GLOBAL_FEATURES, "slot": SLOT_FEATURES},
        "actions": ACTIONS,
        "controllers": checkpoint_rows,
    }, checkpoint_path)
    training = {
        "seconds": time.perf_counter() - started,
        "extraction_peak_allocated_gib": extraction_peak_allocated,
        "extraction_peak_reserved_gib": extraction_peak_reserved,
        "overall_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "overall_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "embedding_hashes": {
            str(seed): embedding_hashes[str(seed)] for seed in requested
        },
        "embedding_extraction": extraction,
        "training": training_meta,
        "checkpoint_sha256": sha(checkpoint_path),
        "raw_rows_sha256": raw_hashes,
    }
    manifest = {
        "protocol": PROTOCOL,
        "data_manifest_sha256": sha(Path(args.data_root) / "manifest.json"),
        "resource_audit_sha256": sha(args.resource_audit),
        "geometry_hashes": geometry_hashes,
        "old_cohort_checkpoint_sha256": sha(old_checkpoint),
        "projection_seeds": list(requested),
        "controller_seeds": CONTROLLER_SEEDS,
        "epochs": args.epochs,
        "policies": POLICIES,
        "reader": READER,
        "actions": [[row[0], list(row[1])] for row in ACTIONS],
        "model": args.model,
        "tokenizer": args.tokenizer,
        "loading_info": model_meta["loading_info"],
        "test_generated": False,
        "test_scored": False,
        "data_audit": data_manifest["audit"],
        "resource_gate": resource_audit["summary"],
    }
    write_json(output / "seed_results.json", results)
    write_json(output / "training.json", training)
    write_json(output / "manifest.json", manifest)
    write_json(output / "progress.json", {"stage": "completed", "seeds": list(requested)})
    return {"seeds": list(requested), "results": results, "training": training}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--geometry-root", required=True)
    parser.add_argument("--resource-audit", required=True)
    parser.add_argument("--confirmation-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--projection-seeds", default="2301,2302,2303")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
