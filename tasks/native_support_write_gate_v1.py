"""Native reject/admit counterfactual collector for support-aware gate V1."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from tasks.build_native_support_write_gate_v1 import PROTOCOL, sha256, stable_hash
from tasks.cbf_checkpoint_native import NativeSession, load_model, native_digest
from tasks.cbf_checkpoint_readout import cache_digest, parameter_digest


PROJECTION_DIM = 128
PROJECTION_SEED = 2026101601
BENEFIT_EPS = 1e-4
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2026101602


def _tensor_digest(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def _fast_weights(session: NativeSession):
    return {layer: session.cache.ttt_states[layer][2] for layer in session.model.config.ttt_layers}


def fast_weight_digest(session: NativeSession) -> str:
    digest = hashlib.sha256()
    for layer, weight in _fast_weights(session).items():
        if weight is None:
            weight = session.model.model.layers[layer].mlp.down_proj.weight
        digest.update(str(layer).encode())
        digest.update(bytes.fromhex(_tensor_digest(weight)))
    return digest.hexdigest()


def fresh_from_weights(session: NativeSession) -> NativeSession:
    """Copy native W_fast into a new cache with no plaintext/KV history."""
    branch = NativeSession(session.model)
    for layer, weight in _fast_weights(session).items():
        if weight is not None:
            branch.cache.TTT_update((None, None, weight.detach().clone()), layer)
    return branch


def _signed_projection(x, layer: int):
    """Fixed signed modulo CountSketch; returns [batch, tokens, 128]."""
    import torch
    width = x.shape[-1]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(PROJECTION_SEED + 1009 * int(layer) + width)
    signs = torch.randint(0, 2, (width,), generator=generator, dtype=torch.int8)
    signs = signs.to(device=x.device, dtype=x.dtype).mul_(2).sub_(1)
    padding = (-width) % PROJECTION_DIM
    signed = x * signs
    if padding:
        signed = torch.nn.functional.pad(signed, (0, padding))
    groups = signed.shape[-1] // PROJECTION_DIM
    return signed.reshape(*signed.shape[:-1], groups, PROJECTION_DIM).sum(-2) / math.sqrt(groups)


class ActivationSupport:
    """Read-only hooks that retain compact numeric sufficient statistics only."""
    def __init__(self, model):
        self.model = model
        self.current = {}
        self.handles = []

    def _hook(self, layer):
        def collect(module, args):
            import torch
            x = args[0]
            with torch.inference_mode():
                projected = _signed_projection(x.float(), layer)
                self.current[layer] = {
                    "centroid": projected.mean((0, 1)).cpu(),
                    "variance": projected.var((0, 1), unbiased=False).cpu(),
                }
        return collect

    def __enter__(self):
        for layer in self.model.config.ttt_layers:
            mlp = self.model.model.layers[layer].mlp
            self.handles.append(mlp.register_forward_pre_hook(self._hook(layer)))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def support_features(current: dict, recent: dict | None, layers: list[int]) -> tuple[list[float], dict]:
    import torch
    features, state = [], {}
    for layer in layers:
        centroid = current[layer]["centroid"].float()
        variance = current[layer]["variance"].float().clamp_min(0)
        if recent is None or layer not in recent:
            cosine, likelihood, has_recent = 0.0, 0.0, 0.0
        else:
            old_mean = torch.tensor(recent[layer]["centroid"], dtype=torch.float32)
            old_var = torch.tensor(recent[layer]["variance"], dtype=torch.float32).clamp_min(1e-6)
            cosine = float(torch.nn.functional.cosine_similarity(centroid, old_mean, dim=0))
            likelihood = float(-0.5 * (((centroid - old_mean).square() / old_var) + old_var.log()).mean())
            has_recent = 1.0
        features.extend(centroid.tolist())
        features.extend(torch.log1p(variance).tolist())
        features.extend((cosine, likelihood, has_recent))
        state[layer] = {"centroid": centroid.tolist(), "variance": variance.tolist()}
    return features, state


def prediction_statistics(model, hidden, labels, block: int = 32) -> dict:
    import torch
    import torch.nn.functional as F
    total_nll = total_entropy = total_top1 = total_margin = 0.0
    count = labels.numel()
    with torch.inference_mode():
        for start in range(0, count, block):
            logits = model.lm_head(hidden[:, start:start + block]).float().reshape(-1, model.config.vocab_size)
            logp = logits.log_softmax(-1); probs = logp.exp()
            target = labels[start:start + logits.shape[0]]
            total_nll += F.nll_loss(logp, target, reduction="sum").item()
            total_entropy += float(-(probs * logp).sum())
            top = probs.topk(2, dim=-1).values
            total_top1 += float(top[:, 0].sum())
            total_margin += float((top[:, 0] - top[:, 1]).sum())
    return {"current_nll": total_nll / count, "predictive_entropy": total_entropy / count,
            "top1_probability": total_top1 / count, "top1_top2_margin": total_margin / count}


def geometry_features(model, before: NativeSession, after: NativeSession) -> tuple[list[float], dict]:
    import torch
    flat, audit = [], {}
    for layer in model.config.ttt_layers:
        base = model.model.layers[layer].mlp.down_proj.weight.float()
        old = before.cache.ttt_states[layer][2]
        old = base if old is None else old.float()
        new = after.cache.ttt_states[layer][2].float()
        delta = new - old
        base_norm = base.norm().clamp_min(1e-12)
        old_delta = old - base
        old_norm, delta_norm = old_delta.norm(), delta.norm()
        cosine = float((old_delta * delta).sum() / (old_norm * delta_norm + 1e-12))
        # Deterministic power iteration yields online geometry proxies without SVD.
        vector = torch.ones(delta.shape[1], device=delta.device, dtype=delta.dtype)
        vector /= vector.norm().clamp_min(1e-12)
        for _ in range(3):
            left = delta @ vector; left /= left.norm().clamp_min(1e-12)
            vector = delta.transpose(0, 1) @ left; vector /= vector.norm().clamp_min(1e-12)
        spectral = (delta @ vector).norm().clamp_min(1e-12)
        stable_rank = delta_norm.square() / spectral.square()
        singular_ratio = spectral / (delta_norm / math.sqrt(min(delta.shape))).clamp_min(1e-12)
        values = [float(torch.log1p(old_norm / base_norm)), float(torch.log1p(delta_norm / base_norm)),
                  cosine, float(stable_rank), float(singular_ratio)]
        flat.extend(values); audit[str(layer)] = values
    return flat, audit


@contextmanager
def native_lr(model, value: float):
    old = {layer: model.model.layers[layer].mlp.ttt_lr for layer in model.config.ttt_layers}
    try:
        for layer in old:
            model.model.layers[layer].mlp.ttt_lr = value
        yield
    finally:
        for layer, lr in old.items():
            model.model.layers[layer].mlp.ttt_lr = lr


def branch_boundary(parent: NativeSession, chunk_ids: list[int], collect_support: bool = True):
    import torch
    root_hash = native_digest(parent.cache); root_fast = fast_weight_digest(parent)
    reject = parent.clone()
    with native_lr(parent.model, 0.0):
        reject_hidden = reject._forward(chunk_ids)
    support = ActivationSupport(parent.model)
    admit = parent.clone()
    with support if collect_support else nullcontext():
        admit_hidden = admit._forward(chunk_ids)
    if native_digest(parent.cache) != root_hash or fast_weight_digest(parent) != root_fast:
        raise RuntimeError("counterfactual branching mutated the parent")
    if fast_weight_digest(reject) != root_fast:
        raise RuntimeError("reject changed native fast weight")
    if fast_weight_digest(admit) == root_fast:
        raise RuntimeError("admit produced no native fast-weight update")
    if reject.cache.get_seq_length() != admit.cache.get_seq_length():
        raise RuntimeError("branch KV lengths differ")
    hidden_error = float((reject_hidden.float() - admit_hidden.float()).abs().max())
    if hidden_error > 1e-5:
        raise RuntimeError(f"current hidden differs before commit: {hidden_error}")
    return reject, admit, admit_hidden, support.current, {
        "parent_native_sha256": root_hash, "parent_fast_sha256": root_fast,
        "reject_fast_sha256": fast_weight_digest(reject),
        "admit_fast_sha256": fast_weight_digest(admit),
        "current_hidden_max_abs_error": hidden_error,
        "kv_seq_length": admit.cache.get_seq_length(),
    }


def _candidate_score(session: NativeSession, prompt: list[int], answer: list[int]) -> float:
    import torch
    import torch.nn.functional as F
    branch = fresh_from_weights(session)
    hidden = branch._forward(prompt + answer[:-1])
    start = len(prompt) - 1
    logits = branch.model.lm_head(hidden[:, start:start + len(answer)]).float()
    labels = torch.tensor(answer, dtype=torch.long, device=logits.device)
    return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels, reduction="mean"))


def score_queries(session: NativeSession, queries: list[dict], eos_id: int) -> dict:
    import torch
    outcomes = []
    for query in queries:
        scores = [_candidate_score(session, query["prompt_ids"], ids) for ids in query["candidate_ids"]]
        probe = fresh_from_weights(session)
        hidden = probe._forward(query["prompt_ids"])
        greedy = int(probe.model.lm_head(hidden[:, -1:]).argmax(-1).item())
        outcomes.append({"query_id": query["query_id"], "role": query["role"],
                         "candidate_nll": scores, "label": query["label"],
                         "prediction": min(range(len(scores)), key=scores.__getitem__),
                         "correct": int(min(range(len(scores)), key=scores.__getitem__) == query["label"]),
                         "answer_nll": scores[query["label"]], "greedy_id": greedy,
                         "eos": int(greedy == eos_id)})
    def mean(role, key):
        values = [row[key] for row in outcomes if row["role"] == role]
        if not values: raise ValueError(f"no {role} queries")
        return statistics.fmean(values)
    related, protected = mean("related", "answer_nll"), mean("protected", "answer_nll")
    return {"queries": outcomes, "J": .5 * related + .5 * protected,
            "accuracy": statistics.fmean(row["correct"] for row in outcomes),
            "protected_accuracy": mean("protected", "correct"),
            "protected_nll": protected,
            "eos_rate": statistics.fmean(row["eos"] for row in outcomes)}


def _measure_cuda(fn, device):
    import torch
    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter(); value = fn(); torch.cuda.synchronize(device)
    return value, {"seconds": time.perf_counter() - started,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}


def load_inputs(data_root: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((data_root / "manifest.json").read_text())
    path = data_root / "model_inputs.jsonl"
    if manifest["protocol"] != PROTOCOL or manifest["files_sha256"][path.name] != sha256(path):
        raise ValueError("data manifest/hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows, manifest


def split_artifacts(result: dict) -> tuple[dict, dict]:
    """Physically separate online-decision features from counterfactual labels."""
    identity = {key: result[key] for key in ("protocol", "stage", "group_id", "boundary")}
    feature = {**identity, "features": result["features"],
               "feature_schema": result["feature_schema"],
               "online_memory": result["online_memory"],
               "uncertainty": result["uncertainty"], "geometry": result["geometry"]}
    feature_sha = hashlib.sha256(json.dumps(feature, sort_keys=True).encode()).hexdigest()
    label = {**identity, "mechanism_hash": result["mechanism_hash"],
             "branch_audit": result["branch_audit"], "actions": result["actions"],
             "J0": result["J0"], "J1": result["J1"], "B": result["B"],
             "resource": result["resource"], "feature_sha256": feature_sha}
    return feature, label


def collect(args) -> dict:
    import torch
    rows, manifest = load_inputs(Path(args.data_root))
    if args.stage == "smoke":
        selected_ids = set(manifest["smoke_groups"])
        rows = [row for row in rows if row["group_id"] in selected_ids]
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard]
    if not rows: raise ValueError("empty shard")
    output, features_output = Path(args.output), Path(args.features_output)
    if output.exists() or features_output.exists(): raise FileExistsError("label/feature output exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(PROJECTION_SEED); torch.backends.cuda.matmul.allow_tf32 = False
    model, model_meta = load_model(args.model, "step81381_native")
    if model.config.ttt_chunk != manifest["chunk_size"]:
        raise ValueError("data/checkpoint chunk mismatch")
    before = parameter_digest(model); versions = [p._version for p in model.parameters()]
    device = next(model.parameters()).device; all_resources = []
    with output.open("x") as sink, features_output.open("x") as feature_sink:
        for group_index, row in enumerate(rows):
            torch.cuda.empty_cache()
            parent = NativeSession(model); parent._forward(row["setup_chunk_ids"])
            parent = fresh_from_weights(parent); recent = None
            for boundary in row["boundaries"]:
                def label():
                    nonlocal parent, recent
                    reject, admit, hidden, current, branch_audit = branch_boundary(parent, boundary["chunk_ids"])
                    labels = torch.tensor(boundary["chunk_ids"][1:], dtype=torch.long, device=device)
                    uncertainty = prediction_statistics(model, hidden[:, :-1], labels)
                    support, recent_next = support_features(current, recent, list(model.config.ttt_layers))
                    geometry, geometry_audit = geometry_features(model, parent, admit)
                    reject_scores = score_queries(reject, boundary["queries"], args.eos_id)
                    admit_scores = score_queries(admit, boundary["queries"], args.eos_id)
                    benefit = reject_scores["J"] - admit_scores["J"]
                    feature = [*uncertainty.values(), *geometry, *support,
                               boundary["boundary"] / 2.0, 1.0]
                    parent = fresh_from_weights(admit); recent = recent_next
                    return {"protocol": PROTOCOL, "stage": args.stage,
                            "group_id": row["group_id"], "mechanism_hash": row["mechanism_hash"],
                            "boundary": boundary["boundary"], "features": feature,
                            "feature_schema": {"uncertainty": list(uncertainty),
                                               "geometry_per_layer": 5,
                                               "support_per_layer": 259,
                                               "tail": ["boundary_fraction", "bias"]},
                            "online_memory": recent_next, "uncertainty": uncertainty,
                            "geometry": geometry_audit, "branch_audit": branch_audit,
                            "actions": {"reject": reject_scores, "admit": admit_scores},
                            "J0": reject_scores["J"], "J1": admit_scores["J"], "B": benefit}
                result, resource = _measure_cuda(label, device)
                result["resource"] = resource; all_resources.append(resource)
                feature, label_row = split_artifacts(result)
                feature_sink.write(json.dumps(feature) + "\n"); feature_sink.flush()
                sink.write(json.dumps(label_row) + "\n"); sink.flush()
                print(json.dumps({"group": row["group_id"], "boundary": boundary["boundary"], **resource}), flush=True)
            print(json.dumps({"completed_groups": group_index + 1, "total_groups": len(rows)}), flush=True)
    if before != parameter_digest(model) or versions != [p._version for p in model.parameters()]:
        raise RuntimeError("frozen model parameters changed")
    audit = {"protocol": PROTOCOL, "stage": args.stage, "shard": args.shard,
             "num_shards": args.num_shards, "groups": len(rows), "labels": len(all_resources),
             "rows_sha256": sha256(output), "data_manifest_sha256": sha256(Path(args.data_root)/"manifest.json"),
             "features_sha256": sha256(features_output), "features_path": str(features_output),
             "model": model_meta, "weights_sha256": before, "backbone_unchanged": True,
             "mean_label_seconds": statistics.fmean(x["seconds"] for x in all_resources),
             "peak_allocated_gib": max(x["peak_allocated_gib"] for x in all_resources),
             "peak_reserved_gib": max(x["peak_reserved_gib"] for x in all_resources)}
    Path(str(output) + ".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def _bootstrap(values: list[float], draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    rng = random.Random(seed); size = len(values)
    samples = sorted(statistics.fmean(values[rng.randrange(size)] for _ in range(size)) for _ in range(draws))
    return [samples[int(.025 * draws)], samples[min(draws - 1, int(.975 * draws))]]


def summarize(inputs: list[Path], output: Path) -> dict:
    rows = [json.loads(line) for path in inputs for line in path.read_text().splitlines() if line.strip()]
    keys = {(row["group_id"], row["boundary"]) for row in rows}
    if len(keys) != len(rows): raise ValueError("duplicate rows")
    def safe_action(row):
        return ("admit" if row["J1"] < row["J0"]
                and row["actions"]["admit"]["protected_accuracy"]
                >= row["actions"]["reject"]["protected_accuracy"] else "reject")

    by_group = {}
    for group in sorted({row["group_id"] for row in rows}):
        part = [row for row in rows if row["group_id"] == group]
        by_group[group] = {
            "always_J": statistics.fmean(r["J1"] for r in part),
            "never_J": statistics.fmean(r["J0"] for r in part),
            "oracle_J": statistics.fmean(r["actions"][safe_action(r)]["J"] for r in part),
        }
    always = statistics.fmean(v["always_J"] for v in by_group.values())
    never = statistics.fmean(v["never_J"] for v in by_group.values())
    stronger_name = "always_write" if always <= never else "never_write"
    stronger_key = "always_J" if always <= never else "never_J"
    gains = [v[stronger_key] - v["oracle_J"] for v in by_group.values()]
    positives = sum(row["B"] > BENEFIT_EPS for row in rows) / len(rows)
    negatives = sum(row["B"] < -BENEFIT_EPS for row in rows) / len(rows)
    oracle_actions = [safe_action(row) for row in rows]
    def action_metric(action, key):
        return statistics.fmean(row["actions"][action][key] for row in rows)
    oracle_protected_accuracy = statistics.fmean(row["actions"][action]["protected_accuracy"] for row, action in zip(rows, oracle_actions))
    oracle_protected_nll = statistics.fmean(row["actions"][action]["protected_nll"] for row, action in zip(rows, oracle_actions))
    oracle_eos = statistics.fmean(row["actions"][action]["eos_rate"] for row, action in zip(rows, oracle_actions))
    baseline_action = "admit" if stronger_name == "always_write" else "reject"
    interval = _bootstrap(gains)
    checks = {
        "positive_fraction_at_least_15pct": positives >= .15,
        "negative_fraction_at_least_15pct": negatives >= .15,
        "oracle_mean_gain_positive": statistics.fmean(gains) > 0,
        "oracle_group_bootstrap_lower_positive": interval[0] > 0,
        "protected_accuracy_safe": oracle_protected_accuracy >= action_metric(baseline_action, "protected_accuracy"),
        "protected_nll_safe": oracle_protected_nll - action_metric(baseline_action, "protected_nll") <= .01,
        "eos_safe": oracle_eos - action_metric(baseline_action, "eos_rate") <= .01,
        "branch_invariants": all(r["branch_audit"]["current_hidden_max_abs_error"] <= 1e-5 and
                                 r["branch_audit"]["parent_fast_sha256"] == r["branch_audit"]["reject_fast_sha256"]
                                 for r in rows),
    }
    result = {"protocol": PROTOCOL, "stage": rows[0]["stage"], "groups": len(by_group),
              "labels": len(rows), "benefit_epsilon": BENEFIT_EPS,
              "positive_fraction": positives, "negative_fraction": negatives,
              "tie_fraction": 1 - positives - negatives,
              "mean_J": {"always_write": always, "never_write": never,
                         "oracle": statistics.fmean(v["oracle_J"] for v in by_group.values())},
              "stronger_baseline": stronger_name, "oracle_gain": statistics.fmean(gains),
              "oracle_gain_group_bootstrap_95ci": interval,
              "checks": checks, "passed": all(checks.values()),
              "next_stage": "formal_train_dev" if all(checks.values()) else "stop_before_controller_and_test"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--data-root", required=True); collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--stage", choices=("smoke", "pilot"), required=True)
    collect_parser.add_argument("--shard", type=int, default=0); collect_parser.add_argument("--num-shards", type=int, default=1)
    collect_parser.add_argument("--eos-id", type=int, default=151645); collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--features-output", required=True)
    summary_parser = sub.add_parser("summarize"); summary_parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect": print(json.dumps(collect(args), indent=2))
    else: print(json.dumps(summarize(args.inputs, args.output), indent=2))


if __name__ == "__main__": main()
