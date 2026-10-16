"""Layerwise native fast-weight admission headroom diagnostic (V2)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.build_native_support_write_gate_v1 import sha256
from tasks.cbf_checkpoint_native import NativeSession, load_model, native_digest
from tasks.cbf_checkpoint_readout import parameter_digest
from tasks.native_support_write_gate_v1 import (
    branch_boundary,
    fast_weight_digest,
    fresh_from_weights,
    load_inputs,
    score_queries,
)


PROTOCOL = "native_layerwise_fastweight_admission_v2"
PLAN = "NATIVE_LAYERWISE_WRITE_GATE_V2_PLAN.md"
LAYERS = (0, 6, 12, 18, 24, 30, 35)
SINGLETON_ACTIONS = tuple(f"admit_layer_{layer}" for layer in LAYERS)
ALL_ACTIONS = ("reject_all", *SINGLETON_ACTIONS, "admit_all")
BENEFIT_EPS = 1e-4
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2026101603


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def tensor_hash(value) -> str:
    import torch
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def layer_hashes(session: NativeSession) -> dict[str, str]:
    result = {}
    for layer in session.model.config.ttt_layers:
        weight = session.cache.ttt_states[layer][2]
        if weight is None:
            weight = session.model.model.layers[layer].mlp.down_proj.weight
        result[str(layer)] = tensor_hash(weight)
    return result


def compose_singleton(reject: NativeSession, admit: NativeSession, selected: int):
    """Create a fresh-KV state with exactly one admitted layer update."""
    if selected not in reject.model.config.ttt_layers:
        raise ValueError(f"not a TTT layer: {selected}")
    branch = NativeSession(reject.model)
    sources = {}
    for layer in reject.model.config.ttt_layers:
        donor = admit if layer == selected else reject
        weight = donor.cache.ttt_states[layer][2]
        if weight is None:
            weight = donor.model.model.layers[layer].mlp.down_proj.weight
        branch.cache.TTT_update((None, None, weight.detach().clone()), layer)
        sources[str(layer)] = "admit" if layer == selected else "reject"
    return branch, sources


def _load_rows(paths: list[Path]) -> list[dict]:
    return [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]


def load_v1(v1_root: Path, stage: str) -> tuple[dict, dict, dict]:
    audit_path = v1_root / f"{stage}_independent_audit.json"
    audit = json.loads(audit_path.read_text())
    if not audit.get("passed") or audit.get("stage") != stage:
        raise ValueError("V1 dependency did not pass independent audit")
    labels = _load_rows(sorted(v1_root.glob(f"{stage}_shard*_labels.jsonl")))
    features = _load_rows(sorted(v1_root.glob(f"{stage}_shard*_features.jsonl")))
    label_map = {(row["group_id"], row["boundary"]): row for row in labels}
    feature_map = {(row["group_id"], row["boundary"]): row for row in features}
    if len(label_map) != len(labels) or set(label_map) != set(feature_map):
        raise ValueError("V1 feature/label dependency mismatch")
    for key, label in label_map.items():
        if canonical_hash(feature_map[key]) != label["feature_sha256"]:
            raise ValueError("V1 feature join hash mismatch")
    return label_map, feature_map, {
        "audit_sha256": sha256(audit_path),
        "summary_sha256": sha256(v1_root / f"{stage}_summary.json"),
        "data_manifest_sha256": audit["data_manifest_sha256"],
        "plan_sha256": audit["plan_sha256"],
    }


def _measure(fn, device):
    import torch
    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter(); result = fn(); torch.cuda.synchronize(device)
    return result, {
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }


def collect(args) -> dict:
    import torch
    data_root, v1_root = Path(args.data_root), Path(args.v1_root)
    rows, manifest = load_inputs(data_root)
    if tuple(manifest["mechanisms"].values()) != (8,) * 6:
        raise ValueError("unexpected V1 source matrix")
    if args.stage == "smoke":
        smoke = set(manifest["smoke_groups"]); rows = [row for row in rows if row["group_id"] in smoke]
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard]
    if not rows: raise ValueError("empty shard")
    v1_labels, _, dependency = load_v1(v1_root, args.stage)
    output = Path(args.output)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026101603); torch.backends.cuda.matmul.allow_tf32 = False
    model, model_meta = load_model(args.model, "step81381_native")
    if tuple(model.config.ttt_layers) != LAYERS or model.config.ttt_chunk != manifest["chunk_size"]:
        raise ValueError("model layer/chunk contract changed")
    before = parameter_digest(model); versions = [parameter._version for parameter in model.parameters()]
    device = next(model.parameters()).device; resources = []
    with output.open("x") as sink:
        for group_index, row in enumerate(rows):
            torch.cuda.empty_cache()
            parent = NativeSession(model); parent._forward(row["setup_chunk_ids"])
            parent = fresh_from_weights(parent)
            for boundary in row["boundaries"]:
                key = (row["group_id"], boundary["boundary"])
                source = v1_labels[key]

                def label_one():
                    nonlocal parent
                    reject, admit, _, _, branch_audit = branch_boundary(
                        parent, boundary["chunk_ids"], collect_support=False
                    )
                    for name in ("parent_fast_sha256", "reject_fast_sha256", "admit_fast_sha256"):
                        if branch_audit[name] != source["branch_audit"][name]:
                            raise RuntimeError(f"V1 native branch replay mismatch: {name}")
                    reject_hashes, admit_hashes = layer_hashes(reject), layer_hashes(admit)
                    action_scores = {
                        "reject_all": source["actions"]["reject"],
                        "admit_all": source["actions"]["admit"],
                    }
                    composition = {}
                    for layer, action in zip(LAYERS, SINGLETON_ACTIONS):
                        singleton, sources = compose_singleton(reject, admit, layer)
                        hashes = layer_hashes(singleton)
                        if any(hashes[str(item)] != (admit_hashes if item == layer else reject_hashes)[str(item)]
                               for item in LAYERS):
                            raise RuntimeError("singleton composition mismatch")
                        action_scores[action] = score_queries(singleton, boundary["queries"], args.eos_id)
                        composition[action] = {"sources": sources, "layer_sha256": hashes}
                    parent = fresh_from_weights(admit)
                    return {
                        "protocol": PROTOCOL, "stage": args.stage,
                        "group_id": row["group_id"], "mechanism_hash": row["mechanism_hash"],
                        "boundary": boundary["boundary"], "feature_sha256": source["feature_sha256"],
                        "v1_label_sha256": canonical_hash(source), "branch_audit": branch_audit,
                        "base_layer_sha256": {"reject": reject_hashes, "admit": admit_hashes},
                        "composition": composition, "actions": action_scores,
                    }

                result, resource = _measure(label_one, device)
                result["resource"] = resource; resources.append(resource)
                sink.write(json.dumps(result) + "\n"); sink.flush()
                print(json.dumps({"group": row["group_id"], "boundary": boundary["boundary"], **resource}), flush=True)
            print(json.dumps({"completed_groups": group_index + 1, "total_groups": len(rows)}), flush=True)
    if before != parameter_digest(model) or versions != [parameter._version for parameter in model.parameters()]:
        raise RuntimeError("frozen model parameters changed")
    sidecar = {
        "protocol": PROTOCOL, "stage": args.stage, "shard": args.shard,
        "num_shards": args.num_shards, "groups": len(rows), "labels": len(resources),
        "rows_sha256": sha256(output), "v1_dependency": dependency,
        "data_manifest_sha256": sha256(data_root / "manifest.json"),
        "plan_sha256": sha256(Path(args.plan)), "model": model_meta,
        "weights_sha256": before, "backbone_unchanged": True,
        "mean_label_seconds": statistics.fmean(item["seconds"] for item in resources),
        "peak_allocated_gib": max(item["peak_allocated_gib"] for item in resources),
        "peak_reserved_gib": max(item["peak_reserved_gib"] for item in resources),
    }
    Path(str(output) + ".audit.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return sidecar


def bootstrap(values: list[float], draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    rng = random.Random(seed); count = len(values)
    estimates = sorted(statistics.fmean(values[rng.randrange(count)] for _ in range(count))
                       for _ in range(draws))
    return [estimates[int(.025 * draws)], estimates[min(draws - 1, int(.975 * draws))]]


def safe_singleton(row: dict) -> str:
    reject = row["actions"]["reject_all"]
    eligible = [name for name in SINGLETON_ACTIONS
                if row["actions"][name]["protected_accuracy"] >= reject["protected_accuracy"]
                and row["actions"][name]["J"] < reject["J"]]
    return min(eligible, key=lambda name: row["actions"][name]["J"]) if eligible else "reject_all"


def summarize(inputs: list[Path], output: Path) -> dict:
    rows = _load_rows(inputs)
    if len({(row["group_id"], row["boundary"]) for row in rows}) != len(rows):
        raise ValueError("duplicate rows")
    expected = 36 if rows[0]["stage"] == "smoke" else 144
    if len(rows) != expected or any(set(row["actions"]) != set(ALL_ACTIONS) for row in rows):
        raise ValueError("incomplete action matrix")
    headroom = [max(row["actions"]["reject_all"]["J"] - row["actions"][name]["J"]
                    for name in SINGLETON_ACTIONS) for row in rows]
    choices = [safe_singleton(row) for row in rows]
    groups = sorted({row["group_id"] for row in rows})
    group_gains = [statistics.fmean(
        row["actions"]["reject_all"]["J"] - row["actions"][choice]["J"]
        for row, choice in zip(rows, choices) if row["group_id"] == group
    ) for group in groups]
    interval = bootstrap(group_gains)
    def action_mean(key, names):
        return statistics.fmean(row["actions"][name][key] for row, name in zip(rows, names))
    reject_names = ["reject_all"] * len(rows)
    checks = {
        "singleton_headroom_fraction_at_least_15pct": sum(value > BENEFIT_EPS for value in headroom) / len(rows) >= .15,
        "safe_oracle_mean_gain_positive": statistics.fmean(group_gains) > 0,
        "safe_oracle_bootstrap_lower_positive": interval[0] > 0,
        "protected_accuracy_safe": action_mean("protected_accuracy", choices) >= action_mean("protected_accuracy", reject_names),
        "protected_nll_safe": action_mean("protected_nll", choices) - action_mean("protected_nll", reject_names) <= .01,
        "eos_safe": action_mean("eos_rate", choices) - action_mean("eos_rate", reject_names) <= .01,
        "branch_replay_and_composition": all(
            row["branch_audit"]["parent_fast_sha256"] == row["branch_audit"]["reject_fast_sha256"]
            and all(sum(source == "admit" for source in item["sources"].values()) == 1
                    for item in row["composition"].values()) for row in rows
        ),
    }
    result = {
        "protocol": PROTOCOL, "stage": rows[0]["stage"], "groups": len(groups), "labels": len(rows),
        "actions": list(ALL_ACTIONS), "headroom_fraction": sum(value > BENEFIT_EPS for value in headroom) / len(rows),
        "mean_max_singleton_benefit": statistics.fmean(headroom),
        "mean_J": {name: statistics.fmean(row["actions"][name]["J"] for row in rows) for name in ALL_ACTIONS},
        "safe_oracle_J": action_mean("J", choices),
        "safe_oracle_gain_vs_reject": statistics.fmean(group_gains),
        "safe_oracle_gain_group_bootstrap_95ci": interval,
        "safe_oracle_actions": {name: choices.count(name) for name in ALL_ACTIONS if choices.count(name)},
        "per_layer": {str(layer): {
            "positive_boundaries": sum(row["actions"]["reject_all"]["J"] - row["actions"][action]["J"] > BENEFIT_EPS for row in rows),
            "mean_benefit": statistics.fmean(row["actions"]["reject_all"]["J"] - row["actions"][action]["J"] for row in rows),
        } for layer, action in zip(LAYERS, SINGLETON_ACTIONS)},
        "checks": checks, "passed": all(checks.values()),
        "next_stage": "freeze_fresh_source_confirmation" if all(checks.values()) else "stop_layerwise_gate",
    }
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--data-root", required=True); collect_parser.add_argument("--v1-root", required=True)
    collect_parser.add_argument("--model", required=True); collect_parser.add_argument("--plan", default=PLAN)
    collect_parser.add_argument("--stage", choices=("smoke", "pilot"), required=True)
    collect_parser.add_argument("--shard", type=int, default=0); collect_parser.add_argument("--num-shards", type=int, default=1)
    collect_parser.add_argument("--eos-id", type=int, default=151645); collect_parser.add_argument("--output", required=True)
    summary_parser = sub.add_parser("summarize"); summary_parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect": print(json.dumps(collect(args), indent=2))
    else: print(json.dumps(summarize(args.inputs, args.output), indent=2))


if __name__ == "__main__": main()
