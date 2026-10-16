"""Distil a readable contextual residual into the native dense TTT writer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from tasks.cbf_qwen_lowrank_bridge import unit_value
from tasks.cbf_selective import measure
from tasks.train_cbf_event_writer import write_json
from tasks.train_final_writer_binding import (
    FINAL_TRAIN_GROUPS, FORMAT, load, load_final_model, paired_objective,
    schedule, tensor_digest,
)


PROTOCOL = "native_writer_distillation_v16"


def query_hidden(model, memory, query_ids):
    from cbf_ttt.event_writer import cache_for, forward

    return forward(model, query_ids, cache_for(model, memory))[:, -1].float().reshape(-1)


def distillation_terms(left_hidden, right_hidden, zero_hidden,
                       left_target, right_target, scale: float,
                       margin: float = 0.1):
    """Correct states approach their teacher and beat the swapped memory."""
    import torch
    import torch.nn.functional as F

    if scale <= 0:
        raise ValueError("teacher scale must be positive")
    denominator = (left_target - zero_hidden).square().mean().clamp_min(1e-8)
    left_distance = (left_hidden - left_target).square().mean() / denominator
    right_distance = (right_hidden - right_target).square().mean() / denominator
    left_wrong = (right_hidden - left_target).square().mean() / denominator
    right_wrong = (left_hidden - right_target).square().mean() / denominator
    hidden_mse = (left_distance + right_distance) / 2
    hidden_contrast = (
        F.softplus(left_distance - left_wrong + margin)
        + F.softplus(right_distance - right_wrong + margin)
    ) / 2
    return hidden_mse, hidden_contrast, {
        "left_distance": left_distance, "right_distance": right_distance,
        "left_wrong_distance": left_wrong, "right_wrong_distance": right_wrong,
    }


def train(args):
    import torch
    from cbf_ttt.event_writer import backbone_digest, write_memory_after_prefix

    rows, design = load(args.root)
    groups = tuple(map(int, args.groups.split(",")))
    if groups != FINAL_TRAIN_GROUPS:
        raise ValueError("V16 pilot requires frozen groups 0--7")
    selected = {(row["group"], row["label"]): row for row in rows
                if row["group"] in groups}
    if len(selected) != len(groups) * 8:
        raise ValueError("incomplete V16 training matrix")

    torch.manual_seed(1730)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, params, model_meta = load_final_model(args.model)
    initial_writer_sha = tensor_digest(params)
    backbone_sha = backbone_digest(model)
    frozen_versions = {name: parameter._version for name, parameter in model.named_parameters()
                       if not parameter.requires_grad}
    optimizer = torch.optim.AdamW(list(params.values()), lr=args.lr, weight_decay=0.)
    batches = schedule(groups, args.rounds)
    output = Path(args.output)
    output.mkdir(parents=True)
    manifest = {
        "protocol": PROTOCOL, "arm": args.arm, "model": args.model,
        "groups": list(groups), "rounds": args.rounds, "updates": len(batches),
        "teacher_scale": args.teacher_scale, "lr": args.lr,
        "distill_weight": args.distill_weight,
        "hidden_contrast_weight": args.hidden_contrast_weight,
        "memory_cap": design["memory_cap"], "seed": 1730, "order_seed": 1729,
        "memory_substrate": "native_dense_ttt_fast_weight",
        "candidate_update_source": "native_ttt_proj_ttt_conv",
        "initial_writer_sha256": initial_writer_sha,
        "backbone_sha256": backbone_sha,
        "source_data_sha256": design["source_data_sha256"],
        "episodes_sha256": design["episodes_sha256"],
        "loading_info": model_meta["loading_info"],
        "trainable_parameters": sum(parameter.numel() for parameter in params.values()),
    }
    write_json(output / "training_manifest.json", manifest)

    records = []
    with (output / "steps.jsonl").open("x") as sink:
        for update, (group, left_label, right_label) in enumerate(batches, 1):
            left, right = selected[group, left_label], selected[group, right_label]

            def one_step():
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    left_memory = write_memory_after_prefix(
                        model, left["chunks"][0], left["chunks"][1], design["memory_cap"])
                    right_memory = write_memory_after_prefix(
                        model, right["chunks"][0], right["chunks"][1], design["memory_cap"])
                    left_hidden = query_hidden(model, left_memory, left["query_ids"])
                    right_hidden = query_hidden(model, right_memory, right["query_ids"])
                    with torch.no_grad():
                        zero_hidden = query_hidden(model, {}, left["query_ids"])
                        left_target = zero_hidden + args.teacher_scale * unit_value(
                            model, left["answer_id"])
                        right_target = zero_hidden + args.teacher_scale * unit_value(
                            model, right["answer_id"])
                        zero_logits = model.lm_head(zero_hidden.to(model.lm_head.weight.dtype)).float()
                    left_logits = model.lm_head(left_hidden.to(model.lm_head.weight.dtype)).float()
                    right_logits = model.lm_head(right_hidden.to(model.lm_head.weight.dtype)).float()
                    base_loss, base = paired_objective(
                        left_logits, right_logits, zero_logits,
                        left_label, right_label, left["choice_ids"],
                        margin=design["margin"],
                        full_vocab_weight=design["full_vocab_weight"],
                        swap_weight=design["swap_weight"],
                        zero_weight=design["zero_weight"],
                    )
                    hidden_mse, hidden_contrast, distances = distillation_terms(
                        left_hidden, right_hidden, zero_hidden, left_target,
                        right_target, args.teacher_scale)
                    loss = (base_loss + args.distill_weight * hidden_mse
                            + args.hidden_contrast_weight * hidden_contrast)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite V16 loss")
                loss.backward()
                if any(parameter.grad is None or not torch.isfinite(parameter.grad).all()
                       for parameter in params.values()):
                    raise RuntimeError("disconnected or nonfinite V16 writer gradient")
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                    list(params.values()), 1.0, error_if_nonfinite=True))
                optimizer.step()
                return {
                    "update": update, "group": group,
                    "labels": [left_label, right_label], "loss": float(loss.detach()),
                    "base_loss": float(base_loss.detach()),
                    "hidden_mse": float(hidden_mse.detach()),
                    "hidden_contrast": float(hidden_contrast.detach()),
                    "candidate_accuracy": float(base["accuracy"].detach()),
                    "unclipped_gradient_norm": gradient_norm,
                    **{key: float(value.detach()) for key, value in distances.items()},
                }

            record, profile = measure(one_step)
            record.update(profile); records.append(record)
            sink.write(json.dumps(record) + "\n"); sink.flush()
            if update == 1 or update % 16 == 0:
                print(json.dumps({key: record[key] for key in (
                    "update", "loss", "base_loss", "hidden_mse",
                    "hidden_contrast", "candidate_accuracy", "seconds")}), flush=True)

    if frozen_versions != {name: parameter._version for name, parameter in model.named_parameters()
                           if not parameter.requires_grad} or backbone_digest(model) != backbone_sha:
        raise RuntimeError("V16 frozen backbone changed")
    final_writer_sha = tensor_digest(params)
    if final_writer_sha == initial_writer_sha:
        raise RuntimeError("V16 writer did not update")
    checkpoint = output / "writer.pt"
    torch.save({"writer": {name: parameter.detach().cpu()
                            for name, parameter in params.items()},
                "manifest": manifest}, checkpoint)
    tail = records[-min(32, len(records)):]
    complete = {
        "protocol": PROTOCOL, "updates": len(records),
        "last_loss": statistics.fmean(row["loss"] for row in tail),
        "last_hidden_mse": statistics.fmean(row["hidden_mse"] for row in tail),
        "last_candidate_accuracy": statistics.fmean(
            row["candidate_accuracy"] for row in tail),
        "mean_update_seconds": statistics.fmean(row["seconds"] for row in records),
        "peak_allocated_gib": max(row["peak_allocated_gib"] for row in records),
        "peak_reserved_gib": max(row["peak_reserved_gib"] for row in records),
        "initial_writer_sha256": initial_writer_sha,
        "final_writer_sha256": final_writer_sha,
        "backbone_unchanged": True,
    }
    write_json(output / "training_complete.json", complete)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--groups", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-7)
    parser.add_argument("--teacher-scale", type=float, choices=(4.0, 8.0), required=True)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--hidden-contrast-weight", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
