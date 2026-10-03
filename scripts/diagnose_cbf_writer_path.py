"""Frozen dev-only probes of twin information through writer and numeric readout."""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from cbf_ttt.runtime import CBFSession
from tasks.build_cbf_paired_facts import validate
from tasks.cbf_paired_writer import load_selected
from tasks.cbf_ttt import _load_model
from tasks.cbf_writer import candidate_for, digest, end_profile, start_profile, write_json


def relative_difference(a, b):
    return float((a.float()-b.float()).norm()/((a.float().norm()+b.float().norm())/2).clamp_min(1e-12))


def amplified_pair(a, b, bounds_a, bounds_b, scale):
    if scale == 1:
        return a, b
    left, right = {}, {}
    for layer in a:
        center, difference = (a[layer]+b[layer])/2, (a[layer]-b[layer])/2
        if scale == 0:
            # A true shared-memory control must remain identical despite unequal norm bounds.
            bound = torch.minimum(bounds_a[layer].float().norm(), bounds_b[layer].float().norm())
            left[layer] = center * (bound/center.norm().clamp_min(1e-12)).clamp(max=1.)
            right[layer] = left[layer]
            continue
        for output, value, bound in ((left, center+scale*difference, bounds_a[layer]),
                                     (right, center-scale*difference, bounds_b[layer])):
            output[layer] = value * (bound.float().norm()/value.norm().clamp_min(1e-12)).clamp(max=1.)
    return left, right


def pair_metrics(a, b, index_a, index_b):
    return {"mean_nll": (a[index_a]+b[index_b])/2,
            "accuracy": (int(min(range(len(a)), key=a.__getitem__) == index_a)+
                         int(min(range(len(b)), key=b.__getitem__) == index_b))/2,
            "matched_gain_vs_twin": ((b[index_a]+a[index_b])-(a[index_a]+b[index_b]))/2,
            "choice_distribution_l1": float((torch.softmax(-torch.tensor(a), 0)-
                                              torch.softmax(-torch.tensor(b), 0)).abs().sum())}


@torch.no_grad()
def read_choices(model, memories, query, choices, head_weight=None):
    session = CBFSession(model)
    dtype = next(model.parameters()).dtype
    session.cache.cbf_memory = {layer: value.to(dtype) for layer, value in memories.items()}
    ids = torch.tensor([query], device=session.device)
    hidden = model.model(input_ids=ids, past_key_values=session.cache, use_cache=True).last_hidden_state[:, -1:]
    logits = {"native": model.lm_head(hidden).float()}
    if head_weight is not None:
        logits["head_fp32"] = F.linear(hidden.float(), head_weight,
                                       model.lm_head.bias.float() if model.lm_head.bias is not None else None)
    return {name: (-value.log_softmax(-1)[0, 0, choices]).cpu().tolist() for name, value in logits.items()}, hidden.float().cpu()


@torch.no_grad()
def layer_metrics(model, writer, raw_a, raw_b, a, b):
    result = []
    for layer in a:
        base = model.model.layers[layer].mlp.down_proj.weight
        before_a = raw_a[layer].float()+writer.b[str(layer)]@(writer.a[str(layer)]@raw_a[layer].float())
        before_b = raw_b[layer].float()+writer.b[str(layer)]@(writer.a[str(layer)]@raw_b[layer].float())
        ma, mb = a[layer].to(torch.bfloat16), b[layer].to(torch.bfloat16)
        effective_a, effective_b = base.to(torch.bfloat16)+ma, base.to(torch.bfloat16)+mb
        signal = (a[layer]-b[layer]).norm().clamp_min(1e-12)
        result.append({"layer": layer, "raw_relative_difference": relative_difference(raw_a[layer], raw_b[layer]),
                       "before_cap_relative_difference": relative_difference(before_a, before_b),
                       "writer_relative_difference": relative_difference(a[layer], b[layer]),
                       "cap_factor_a": min(1., float(raw_a[layer].float().norm()/before_a.norm().clamp_min(1e-12))),
                       "cap_factor_b": min(1., float(raw_b[layer].float().norm()/before_b.norm().clamp_min(1e-12))),
                       "bf16_memory_difference_ratio": float((ma.float()-mb.float()).norm()/signal),
                       "bf16_effective_difference_ratio": float((effective_a.float()-effective_b.float()).norm()/signal),
                       "bf16_effective_changed_fraction": float((effective_a != effective_b).float().mean())})
    return result


def run(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError("diagnostic output directory must be new")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    episodes = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    validate(episodes)
    dev = [r for r in episodes if r["split"] == "dev"]
    groups = sorted({r["group_id"] for r in dev})
    if len(groups) != 8 or len(dev) != 16:
        raise ValueError("expected fixed eight dev groups")
    model = _load_model(args.model, args.device, args.dtype)
    device = next(model.parameters()).device
    writers = {name: load_selected(Path(args.root)/f"train_{name}", args, device)
               for name in ("paired", "nll")}
    if any(selection["best"]["gate"] != 1. for _, selection in writers.values()):
        raise ValueError("this diagnostic requires the frozen g=1 checkpoints")
    reference = {name: {r["id"]: r for r in json.loads((Path(args.root)/f"train_{name}/selected_dev.json").read_text())}
                 for name in ("paired", "nll")}
    head_weight = model.lm_head.weight.float() if args.dtype == "bfloat16" else None
    output.mkdir(parents=True)
    records, signals = [], []
    max_reference_error = 0.
    with torch.no_grad(), (output/"rows.jsonl").open("w") as sink:
        for group in groups:
            started = start_profile(device)
            ra, rb = sorted((r for r in dev if r["group_id"] == group), key=lambda r: r["id"])
            da = candidate_for(ra, args.features, args.model, device)
            db = candidate_for(rb, args.features, args.model, device)
            pair_results = {}

            def probe(name, a, b, reference_name=None, reference_policy=None):
                nonlocal max_reference_error
                va, ha = read_choices(model, a, ra["query_ids"], ra["choice_ids"], head_weight)
                vb, hb = read_choices(model, b, rb["query_ids"], rb["choice_ids"], head_weight)
                for precision in va:
                    label = ("float32" if args.dtype == "float32" else "bfloat16") if precision == "native" else precision
                    pair_results[f"{name}/{label}"] = {**pair_metrics(va[precision], vb[precision], ra["answer_index"], rb["answer_index"]),
                                                       "hidden_relative_difference": relative_difference(ha, hb)}
                if args.dtype == "bfloat16" and reference_name:
                    for row, values in ((ra, va["native"]), (rb, vb["native"])):
                        expected = reference[reference_name][row["id"]]["policies"][reference_policy]
                        error = abs(values[row["answer_index"]]-expected["nll"])
                        max_reference_error = max(max_reference_error, error)
                        if error > 1e-5 or int(min(range(8), key=values.__getitem__) == row["answer_index"]) != expected["correct"]:
                            raise ValueError("native forward does not reproduce archived dev")

            probe("none", {}, {}, "paired", "none")
            probe("raw_g0.5", {k: v*.5 for k, v in da.items()}, {k: v*.5 for k, v in db.items()}, "paired", "raw_0.5")
            for name, (writer, selection) in writers.items():
                a, b = writer(da), writer(db)
                probe(name, a, b, name, "writer_1.0")
                if args.dtype == "bfloat16":
                    signals.extend({"group_id": group, "writer": name, **row} for row in layer_metrics(model, writer, da, db, a, b))
                if name == "paired":
                    for scale in (0, 8, 32):
                        left, right = amplified_pair(a, b, da, db, scale)
                        probe(f"paired_scale{scale}", left, right)
                        del left, right
                del a, b
            record = {"group_id": group, "split": "dev", "policies": pair_results, **end_profile(device, started)}
            sink.write(json.dumps(record)+"\n")
            sink.flush()
            records.append(record)
            del da, db
            print(json.dumps({"groups_complete": len(records), "dtype": args.dtype}), flush=True)
    summary = {"protocol": "writer_path_diagnostic_v1", "dtype": args.dtype, "groups": len(records),
               "split": "dev", "test_scored": False, "episodes_sha256": digest(args.data),
               "checkpoint_sha256": {name: selection["checkpoint_sha256"] for name, (_, selection) in writers.items()},
               "native_reference_max_error": max_reference_error if args.dtype == "bfloat16" else None,
               "tf32": False, "policies": {p: {metric: statistics.mean(r["policies"][p][metric] for r in records)
                                                for metric in records[0]["policies"][p]} for p in records[0]["policies"]},
               "mean_group_time_s": statistics.mean(r["time_s"] for r in records),
               "max_peak_allocated_gib": max((r["peak_allocated_gib"] or 0) for r in records),
               "max_peak_reserved_gib": max((r["peak_reserved_gib"] or 0) for r in records)}
    if signals:
        (output/"signals.jsonl").write_text("".join(json.dumps(r)+"\n" for r in signals))
        summary["signal_medians"] = {name: {key: statistics.median(r[key] for r in signals if r["writer"] == name)
                                             for key in signals[0] if key not in ("group_id", "writer", "layer")}
                                     for name in writers}
    write_json(output/"summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.data = str(Path(args.root)/"episodes.jsonl")
    args.features = str(Path(args.root)/"features")
    run(args)


if __name__ == "__main__":
    main()
