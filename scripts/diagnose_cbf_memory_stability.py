"""Frozen pilot-only dissection of candidate arithmetic and fast-weight storage."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from opt_einsum import contract

from cbf_ttt.runtime import CBFSession, cbf_forward_mlp
from tasks.cbf_selective import measure
from tasks.cbf_ttt import _load_model


MODES = ("native", "cbf", "clear", "none", "native_ops_m", "native_ops_w",
         "fp32_m", "native_ops_m_fp32", "native_ops_w_fp32")


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def native_delta(mlp, h, target_conv):
    if mlp.ttt_proj is None:
        return contract("c h, c d -> d h", h[0], target_conv[0]) * mlp.ttt_lr
    return contract("c h, c d, d e -> e h", h[0], target_conv[0], mlp.ttt_proj.weight) * mlp.ttt_lr


def cbf_delta(mlp, h, target_conv):
    value = target_conv[0].T @ h[0]
    if mlp.ttt_proj is not None:
        value = mlp.ttt_proj.weight.T @ value
    return value * mlp.ttt_lr


def effective(model, cache, layer):
    base = model.model.layers[layer].mlp.down_proj.weight
    mode = cache.stability_mode
    if mode == "native":
        value = cache.ttt_states[layer][2]
        return base if value is None else value
    if mode in ("native_ops_w", "native_ops_w_fp32"):
        return cache.stability_weights.get(layer, base).to(base.dtype)
    memory = cache.cbf_memory.get(layer)
    return base if memory is None else (base.to(memory.dtype)+memory).to(base.dtype)


def candidate_probe(mlp, h, target_conv, staged):
    reference = target_conv[0].float().T @ h[0].float()
    if mlp.ttt_proj is not None:
        reference = mlp.ttt_proj.weight.float().T @ reference
    reference *= mlp.ttt_lr
    other = native_delta(mlp, h, target_conv)
    denom = reference.norm().clamp_min(1e-12)
    return {"cbf_relative_error_vs_fp32": float((staged.float()-reference).norm()/denom),
            "native_relative_error_vs_fp32": float((other.float()-reference).norm()/denom),
            "native_cbf_relative_difference": float((other.float()-staged.float()).norm()/denom),
            "changed_fraction": float((other != staged).float().mean())}


def diagnostic_forward(mlp, x, target, cache, layer_idx):
    mode = cache.stability_mode
    if mode in ("cbf", "clear", "none"):
        result = cbf_forward_mlp(mlp, x, target, cache, layer_idx)
        if mode == "cbf" and cache.cbf_collect and cache.stability_probe:
            h = mlp.act_fn(mlp.gate_proj(x))*mlp.up_proj(x)
            conv = mlp.ttt_conv(target.transpose(1, 2)).transpose(1, 2)
            cache.stability_probe_results[layer_idx] = candidate_probe(
                mlp, h, conv, cache.cbf_candidates[layer_idx])
        return result
    h = mlp.act_fn(mlp.gate_proj(x))*mlp.up_proj(x)
    base = mlp.down_proj.weight
    if mode in ("native_ops_w", "native_ops_w_fp32"):
        weight = cache.stability_weights.get(layer_idx, base).to(base.dtype)
    else:
        memory = cache.cbf_memory.get(layer_idx)
        weight = base if memory is None else (base.to(memory.dtype)+memory).to(base.dtype)
    native_ops = mode.startswith("native_ops")
    if cache.cbf_collect:
        if x.shape[0] != 1 or x.shape[1] != mlp.ttt_chunk:
            raise ValueError("one complete chunk is required")
        conv = mlp.ttt_conv(target.transpose(1, 2)).transpose(1, 2)
        cache.cbf_candidates[layer_idx] = (native_delta if native_ops else cbf_delta)(mlp, h, conv)
        if native_ops:
            # Match the original contraction and output allocation/layout.
            output = torch.zeros_like(conv.reshape(1, 1, x.shape[1], mlp.hidden_size))
            output[0, 0] = contract("d h, c h -> c d", weight, h[0])
            return output.reshape(1, x.shape[1], mlp.hidden_size)
    return F.linear(h, weight, mlp.down_proj.bias)


def new_session(model, mode):
    if mode not in MODES:
        raise ValueError(mode)
    session = CBFSession(model)
    session.cache.stability_mode = mode
    session.cache.stability_weights = {}
    session.cache.stability_probe = False
    session.cache.stability_probe_results = {}
    if mode == "native":
        session.cache.cbf_enabled = False
    return session


@torch.inference_mode()
def commit(session):
    mode = session.cache.stability_mode
    if mode == "native":
        return
    if mode in ("cbf", "clear", "none"):
        session.commit_both(0. if mode == "clear" else 1., 0. if mode == "none" else 1.)
        return
    if set(session.cache.cbf_candidates) != set(session.layers):
        raise RuntimeError("incomplete candidate")
    direct = mode in ("native_ops_w", "native_ops_w_fp32")
    state = session.cache.stability_weights if direct else session.cache.cbf_memory
    for layer in session.layers:
        base = session.model.model.layers[layer].mlp.down_proj.weight
        dtype = torch.float32 if "fp32" in mode else base.dtype
        delta = session.cache.cbf_candidates[layer].to(dtype)
        old = state.get(layer)
        if old is None:
            state[layer] = base.to(dtype)+delta if direct else delta.clone()
        else:
            state[layer] = old+delta
    session.cache.cbf_candidates = {}
    session.cache.cbf_collect = False


def layer_metrics(session, before, candidates):
    values = []
    for layer in session.layers:
        base = session.model.model.layers[layer].mlp.down_proj.weight.float()
        after = effective(session.model, session.cache, layer).float()
        memory = session.cache.cbf_memory.get(layer)
        if session.cache.stability_mode in ("native_ops_w", "native_ops_w_fp32"):
            represented = session.cache.stability_weights[layer].float()-base
        else:
            represented = after-base if memory is None else memory.float()
        delta = candidates.get(layer)
        prior = before[layer].float()-base
        denom = base.norm().clamp_min(1e-12)
        row = {"layer": layer, "memory_ratio": float(represented.norm()/denom),
               "effective_memory_ratio": float((after-base).norm()/denom),
               "effective_change_ratio": float((after-before[layer].float()).norm()/denom),
               "candidate_ratio": None if delta is None else float(delta.float().norm()/denom),
               "prior_effective_memory_candidate_cosine": None if delta is None else
                   float((prior*delta.float()).sum()/(prior.norm()*delta.float().norm()).clamp_min(1e-12))}
        if not all(v is None or math.isfinite(v) for v in row.values()):
            raise RuntimeError("nonfinite memory statistic")
        values.append(row)
    return values


@torch.inference_mode()
def trajectory(model, scene, mode):
    session = new_session(model, mode)
    chunks = scene["prefix"]+[f["ids"] for f in scene["future"]]
    steps = []
    for index, ids in enumerate(chunks):
        before = {i: effective(model, session.cache, i).clone() for i in session.layers}
        session.cache.stability_probe = index == 0
        hidden = session._forward(ids, collect=mode != "native")
        candidates = dict(session.cache.cbf_candidates)
        commit(session)
        stats = layer_metrics(session, before, candidates)
        step = {"chunk": index+1, "layers": stats}
        if mode in ("native", "native_ops_w"):
            step["hidden_sha256"] = tensor_digest(hidden)
            step["effective_weight_sha256"] = {str(i): tensor_digest(effective(model, session.cache, i))
                                                 for i in session.layers}
        del hidden, before, candidates
        query, answer = ((chunks[index+1][:32], chunks[index+1][32:160]) if index < len(chunks)-1 else
                         (scene["future"][-1]["query_ids"], scene["future"][-1]["answer_ids"]))
        length = session.cache.get_seq_length()
        step["nll"] = session.score_answer(query, answer)
        if length != session.cache.get_seq_length() or not math.isfinite(step["nll"]):
            raise RuntimeError("invalid or state-mutating query")
        steps.append(step)
    return {"scene": scene["id"], "group": scene["group"], "mode": mode, "steps": steps,
            "same_input_candidate_probe": session.cache.stability_probe_results}


def collect(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    raw = Path(args.data).read_bytes()
    scenes = [json.loads(line) for line in raw.splitlines()]
    scenes = [s for s in scenes if s["split"] == "pilot" and s["regime"] == "stable"]
    if len(scenes) != 4 or {s["group"] for s in scenes} != set(range(4)):
        raise ValueError("expected the four pilot stable scenes")
    torch.manual_seed(108)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = _load_model(args.model, "cuda", "bfloat16")
    if model.config.model_type != "qwen3" or any(model.model.layers[i].mlp.down_proj.bias is not None
                                                for i in model.config.ttt_layers):
        raise ValueError("the diagnostic requires Qwen3 with unbiased down projections")
    if any(s["chunk_size"] != model.config.ttt_chunk for s in scenes):
        raise ValueError("chunk mismatch")
    versions = [p._version for p in model.parameters()]
    bases = {i: tensor_digest(model.model.layers[i].mlp.down_proj.weight) for i in model.config.ttt_layers}
    with patch("inference_model.hf_qwen3.modeling_qwen3.cbf_forward_mlp", diagnostic_forward), output.open("x") as sink:
        for scene in scenes:
            if scene["group"] % 2 != args.shard:
                continue
            for mode in MODES:
                torch.cuda.empty_cache()
                row, profile = measure(lambda: trajectory(model, scene, mode))
                row.update({"data_sha256": hashlib.sha256(raw).hexdigest(), "model": args.model, **profile})
                sink.write(json.dumps(row)+"\n")
                sink.flush()
                print(json.dumps({"completed_scene": scene["id"], "mode": mode, **profile}), flush=True)
    if versions != [p._version for p in model.parameters()] or bases != {
            i: tensor_digest(model.model.layers[i].mlp.down_proj.weight) for i in model.config.ttt_layers}:
        raise RuntimeError("backbone was modified")
    Path(str(output)+".audit.json").write_text(json.dumps({"backbone_unchanged": True, "trajectories": 18})+"\n")


def summarize(args):
    rows = [json.loads(line) for file in args.inputs for line in Path(file).read_text().splitlines()]
    if len(rows) != 36 or {(r["group"], r["mode"]) for r in rows} != {
            (g, mode) for g in range(4) for mode in MODES}:
        raise ValueError("incomplete or duplicate trajectories")
    for file in args.inputs:
        if not json.loads(Path(file+".audit.json").read_text())["backbone_unchanged"]:
            raise ValueError("backbone audit failed")
    refs = {r["id"]: r for file in args.reference for r in
            (json.loads(line) for line in Path(file).read_text().splitlines())}
    hashes = {r["data_sha256"] for r in rows}
    if len(hashes) != 1 or len({r["model"] for r in rows}) != 1:
        raise ValueError("mixed data/checkpoints")
    lookup = {(r["group"], r["mode"]): r for r in rows}
    max_reference_error = 0.
    bridge_equal = True
    for row in rows:
        if [s["chunk"] for s in row["steps"]] != list(range(1, 8)):
            raise ValueError("invalid chunk sequence")
        ref = refs[row["scene"]]
        if row["data_sha256"] != ref["data_sha256"] or row["model"] != ref["model"]:
            raise ValueError("reference mismatch")
        if row["mode"] in ("cbf", "clear"):
            name = "cohort_1_1_1" if row["mode"] == "cbf" else "fixed_global_clear"
            errors = [abs(a["nll"]-b) for a, b in zip(row["steps"][-3:], ref["results"][name]["losses"])]
            max_reference_error = max(max_reference_error, *errors)
    for group in range(4):
        for a, b in zip(lookup[group, "native"]["steps"], lookup[group, "native_ops_w"]["steps"]):
            bridge_equal &= (a["hidden_sha256"] == b["hidden_sha256"] and
                             a["effective_weight_sha256"] == b["effective_weight_sha256"] and a["nll"] == b["nll"])
    if max_reference_error > 1e-5:
        raise ValueError(f"archived CBF/clear reference mismatch: {max_reference_error}")
    output = {"protocol": "memory_stability_v1", "groups": 4, "trajectories": 36,
              "data_sha256": next(iter(hashes)), "reference_max_nll_error": max_reference_error,
              "native_bridge_bytewise_equal_all_steps": bridge_equal, "backbone_unchanged": True,
              "modes": {}, "candidate_probe": {}}
    for mode in MODES:
        selected = [r for r in rows if r["mode"] == mode]
        output["modes"][mode] = {
            "nll_by_chunk": [statistics.fmean(r["steps"][i]["nll"] for r in selected) for i in range(7)],
            "last3_mean_nll": statistics.fmean(s["nll"] for r in selected for s in r["steps"][-3:]),
            "seconds_mean": statistics.fmean(r["seconds"] for r in selected),
            "max_peak_allocated_gib": max(r["peak_allocated_gib"] for r in selected),
            "max_peak_reserved_gib": max(r["peak_reserved_gib"] for r in selected)}
        for metric in ("memory_ratio", "effective_memory_ratio", "effective_change_ratio", "candidate_ratio"):
            output["modes"][mode][metric+"_mean_by_chunk"] = [
                None if mode == "native" and metric == "candidate_ratio" else statistics.fmean(
                    layer[metric] for r in selected for layer in r["steps"][i]["layers"]) for i in range(7)]
        output["modes"][mode]["memory_ratio_max_layer_by_chunk"] = [
            max(layer["memory_ratio"] for r in selected for layer in r["steps"][i]["layers"]) for i in range(7)]
    probes = [p for row in rows if row["mode"] == "cbf" for p in row["same_input_candidate_probe"].values()]
    output["candidate_probe"] = {key: statistics.fmean(p[key] for p in probes) for key in probes[0]}
    output["summed_trajectory_seconds"] = sum(r["seconds"] for r in rows)
    Path(args.output).write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("collect")
    run.add_argument("--data", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--shard", type=int, choices=(0, 1), required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--inputs", nargs="+", required=True)
    summary.add_argument("--reference", nargs="+", required=True)
    summary.add_argument("--output", required=True)
    args = parser.parse_args()
    (collect if args.command == "collect" else summarize)(args)


if __name__ == "__main__":
    main()
