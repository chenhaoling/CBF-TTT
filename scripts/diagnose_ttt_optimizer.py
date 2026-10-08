"""Audit fused CE gradients and actual writer optimizer displacements."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.probe_ttt_writer_training import training_tokens, writer_parameters
from tasks.cbf_selective import measure


def standard_head_gradient(hidden, weight, labels, block_size=128):
    """Exact token weighting with bounded logits; return d(mean CE)/d(hidden).

    Each block uses an independent leaf, so gradients are assembled, not repeatedly
    rounded by accumulation into a shared BF16 tensor. The frozen head is not trained.
    """
    if weight.requires_grad or hidden.shape[:2] != labels.shape or block_size < 1:
        raise ValueError("requires a frozen head, aligned labels, and positive block size")
    targets = F.pad(labels, (0, 1), value=-100)[..., 1:].reshape(-1)
    count = int((targets != -100).sum())
    if not count:
        raise ValueError("no valid next-token targets")
    flat = hidden.detach().reshape(-1, hidden.shape[-1])
    gradient = torch.zeros_like(flat)
    losses = []
    for start in range(0, flat.shape[0], block_size):
        end = min(start+block_size, flat.shape[0])
        leaf = flat[start:end].clone().requires_grad_(True)
        logits = F.linear(leaf, weight).float()
        loss = F.cross_entropy(logits, targets[start:end], ignore_index=-100, reduction="sum")/count
        loss.backward()
        gradient[start:end] = leaf.grad.detach()
        losses.append(float(loss.detach()))
    return math.fsum(losses), gradient.reshape_as(hidden)


def difference(reference, actual):
    a, b = reference.float(), actual.float()
    an, bn = a.norm(), b.norm()
    return {"reference_norm": float(an), "actual_norm": float(bn),
            "relative_l2": float((b-a).norm()/an.clamp_min(1e-30)),
            "cosine": float((a*b).sum()/(an*bn).clamp_min(1e-30)),
            "max_abs": float((b-a).abs().max())}


def gradient_comparison(reference, actual):
    layers = {name: difference(reference[name], actual[name]) for name in reference}
    denom = math.fsum(v["reference_norm"]**2 for v in layers.values())
    other = math.fsum(v["actual_norm"]**2 for v in layers.values())
    error = math.fsum((v["relative_l2"]*v["reference_norm"])**2 for v in layers.values())
    dot = math.fsum(v["cosine"]*v["reference_norm"]*v["actual_norm"] for v in layers.values())
    return {"layers": layers, "global_relative_l2": math.sqrt(error/max(denom, 1e-60)),
            "global_cosine": dot/math.sqrt(max(denom*other, 1e-60))}


def displacement(before, after):
    return {n: {"before_norm": float(before[n].norm()), "after_norm": float(after[n].detach().norm()),
                "update_norm": float((after[n].detach()-before[n]).norm()),
                "relative_update": float((after[n].detach()-before[n]).norm()/before[n].norm().clamp_min(1e-30))}
            for n in before}


def intervention_weights(before, after, name, target):
    if name not in ("full", "without_last_conv", "only_last_conv", "half_last_conv"):
        raise ValueError(name)
    if target not in before or before.keys() != after.keys():
        raise ValueError("invalid writer tensors")
    return {n: (before[n] if name == "without_last_conv" and n == target else
                before[n] if name == "only_last_conv" and n != target else
                (before[n]+after[n])*.5 if name == "half_last_conv" and n == target else after[n])
            for n in before}


def fused_loss(model, hidden, labels):
    from liger_kernel.transformers.model.qwen3 import LigerForCausalLMLoss, unpack_cross_entropy_result
    result = LigerForCausalLMLoss(hidden_states=hidden, lm_head_weight=model.lm_head.weight,
                                labels=labels, hidden_size=model.config.hidden_size)
    return unpack_cross_entropy_result(result)[0]


@torch.no_grad()
def score(model, inputs):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return float(model(input_ids=inputs, labels=inputs, use_cache=False).loss)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("model", "training-data", "tokenizer", "output", "reference"):
        parser.add_argument("--"+flag, required=True)
    parser.add_argument("--length", type=int, choices=(6144, 12288), required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    from hf_models.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from transformers import AutoTokenizer
    torch.manual_seed(108)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids, hashes = training_tokens(args.training_data, AutoTokenizer.from_pretrained(args.tokenizer), args.length)
    digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    reference = json.loads(Path(args.reference).read_text())
    if digest != reference["input_ids_sha256"] or args.model != reference["model"]:
        raise ValueError("resource probe source/model mismatch")
    inputs = torch.tensor([ids], device="cuda")
    model, info = Qwen3ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                attn_implementation="sdpa", output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"checkpoint mismatch: {info}")
    model = model.to("cuda").train()
    selected = writer_parameters(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(list(selected.values()), lr=1e-5, weight_decay=.01)
    frozen = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    result = {"protocol": "optimizer_audit_v2", "model": args.model, "length": args.length,
              "input_ids_sha256": digest, "packed_training_row_sha256": hashes,
              "reference": args.reference, "lr": 1e-5, "weight_decay": .01, "clip": 1.,
              "ce_block_size": 128, "measurements": [], "interventions": {}, "baseline_measurements": [],
              "target_tensor": "model.layers.35.mlp.ttt_conv.weight"}
    target = result["target_tensor"]
    if target not in selected:
        raise ValueError("expected layer 35 writer")
    # Complete the ordinary optimizer trajectory FIRST. All subsequent probes use
    # fixed snapshots and cannot change later optimization states.
    snapshots = []
    for step in range(3):
        before = {n: p.detach().cpu().clone() for n, p in selected.items()}
        def ordinary_step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=inputs, labels=inputs, use_cache=False).loss
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in selected.values()):
                raise RuntimeError("invalid ordinary gradient")
            norm = float(torch.nn.utils.clip_grad_norm_(list(selected.values()), 1.0))
            optimizer.step()
            return {"step": step+1, "loss": float(loss.detach()), "unclipped_grad_norm": norm}
        row, profile = measure(ordinary_step)
        row.update(profile)
        after = {n: p.detach().cpu().clone() for n, p in selected.items()}
        snapshots.append((before, after))
        result["baseline_measurements"].append(row)
        print(json.dumps({"ordinary_step": step+1, **row}), flush=True)
    del optimizer

    def restore(weights):
        with torch.no_grad():
            for n, p in selected.items():
                p.copy_(weights[n])
        if not all(torch.equal(p.detach().cpu(), weights[n]) for n, p in selected.items()):
            raise RuntimeError("writer snapshot restoration failed")

    for step, (before, after) in enumerate(snapshots):
        restore(before)
        def audit_step():
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.model(input_ids=inputs, use_cache=False).last_hidden_state
                hidden.retain_grad()
                loss = fused_loss(model, hidden, inputs)
            loss.backward(retain_graph=True)
            fg = {n: p.grad.detach().clone() for n, p in selected.items()}
            hg = hidden.grad.detach().clone()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                standard_loss, upstream = standard_head_gradient(hidden, model.lm_head.weight, inputs)
            head_diff = difference(upstream, hg)
            model.zero_grad(set_to_none=True)
            hidden.grad = None
            hidden.backward(upstream)
            sg = {n: p.grad for n, p in selected.items()}
            if any(g is None or not torch.isfinite(g).all() for g in list(fg.values())+list(sg.values())):
                raise RuntimeError("missing/nonfinite gradient")
            comparison = gradient_comparison(sg, fg)
            row = {"step": step+1, "fused_loss": float(loss.detach()), "standard_loss": standard_loss,
                   "loss_abs_difference": abs(float(loss.detach())-standard_loss),
                   "reference_loss_abs_difference": abs(result["baseline_measurements"][step]["loss"]-reference["measurements"][step]["loss"]),
                   "state_replay_loss_abs_difference": abs(float(loss.detach())-result["baseline_measurements"][step]["loss"]),
                   "head_gradient": head_diff, "writer_gradients": comparison,
                   "unclipped_grad_norm": result["baseline_measurements"][step]["unclipped_grad_norm"],
                   "displacements": displacement(before, after)}
            if step == 1:
                try:
                    for name in ("full", "without_last_conv", "only_last_conv", "half_last_conv"):
                        restore(intervention_weights(before, after, name, target))
                        result["interventions"][name] = score(model, inputs)
                finally:
                    restore(before)
                result["interventions"]["before_second_update"] = row["fused_loss"]
            return row
        row, profile = measure(audit_step)
        row.update(profile)
        result["measurements"].append(row)
        print(json.dumps({"step": row["step"], "loss": row["fused_loss"],
                          "ce_loss_gap": row["loss_abs_difference"],
                          "writer_gradient_relative_l2": row["writer_gradients"]["global_relative_l2"], **profile}), flush=True)
    restore(snapshots[-1][1])
    if frozen != {n: p._version for n, p in model.named_parameters() if not p.requires_grad}:
        raise RuntimeError("frozen backbone changed")
    result["reference_max_loss_error"] = max(r["reference_loss_abs_difference"] for r in result["measurements"])
    result["frozen_backbone_unchanged"] = True
    result["checkpoint_saved"] = False
    result["gradient_tripwire"] = any(r["loss_abs_difference"] > .005 or
        r["writer_gradients"]["global_relative_l2"] > .05 or
        r["writer_gradients"]["global_cosine"] < .99 or
        any(v["relative_l2"] > .10 for v in r["writer_gradients"]["layers"].values())
        for r in result["measurements"])
    result["state_replay_max_loss_error"] = max(r["state_replay_loss_abs_difference"] for r in result["measurements"])
    result["intervention_restore_loss_error"] = abs(result["interventions"]["full"]-result["baseline_measurements"][2]["loss"])
    result["snapshot_restore_bytewise_equal"] = True
    result["passed_replay_audit"] = result["state_replay_max_loss_error"] <= 1e-5 and result["intervention_restore_loss_error"] <= 1e-5
    # Keep a failure artifact as well as the log; do not silently discard evidence.
    output.write_text(json.dumps(result, indent=2)+"\n")
    if not result["passed_replay_audit"]:
        raise RuntimeError("fixed-state forward replay did not reproduce ordinary trajectory states")


if __name__ == "__main__":
    main()
