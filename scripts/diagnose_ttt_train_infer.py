"""Compare repository training and native inference forwards on frozen pilot text."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F

from tasks.cbf_selective import measure


@contextmanager
def update_scale(model, enabled):
    layers = [m for m in model.modules() if hasattr(m, "ttt_lr")]
    old = [m.ttt_lr for m in layers]
    try:
        if not enabled:
            for m in layers:
                m.ttt_lr = 0.
        yield
    finally:
        for m, value in zip(layers, old):
            m.ttt_lr = value


def hidden_forward(model, ids, stream=False):
    if not stream:
        return model.model(input_ids=ids, use_cache=False).last_hidden_state
    from inference_model.hf_qwen3.modeling_qwen3 import TTTDynamicCache
    cache = TTTDynamicCache(config=model.config)
    values = []
    for part in ids.split(model.config.ttt_chunk, dim=1):
        values.append(model.model(input_ids=part, past_key_values=cache, use_cache=True).last_hidden_state)
    if cache.get_seq_length() != ids.shape[1]:
        raise RuntimeError("stream cache length mismatch")
    return torch.cat(values, dim=1)


def relative_difference(a, b):
    a, b = a.float(), b.float()
    return {"relative_l2": float((a-b).norm()/a.norm().clamp_min(1e-12)),
            "max_abs": float((a-b).abs().max()), "equal_fraction": float((a == b).float().mean())}


def answer_nll(model, hidden, ids):
    # Last 128 next-token targets; identical positions across all three paths.
    logits = model.lm_head(hidden[:, -129:-1]).float()
    return float(F.cross_entropy(logits.flatten(0, 1), ids[:, -128:].flatten()))


@torch.inference_mode()
def compare(train, native, ids, enabled):
    with update_scale(train, enabled), update_scale(native, enabled):
        a = hidden_forward(train, ids)
        b = hidden_forward(native, ids)
        c = hidden_forward(native, ids, stream=True)
        output = {"nll": {name: answer_nll(model, hidden, ids) for name, model, hidden in
                           (("train_full", train, a), ("native_full", native, b), ("native_stream", native, c))},
                  "hidden_train_vs_native_full": relative_difference(a, b),
                  "hidden_train_vs_native_stream": relative_difference(a, c),
                  "hidden_native_full_vs_stream": relative_difference(b, c)}
        if not all(math.isfinite(v) for v in output["nll"].values()):
            raise RuntimeError("nonfinite NLL")
        return output


def load_pair(path):
    # Explicit classes avoid AutoModel registration order switching the implementation.
    from hf_models.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM as TrainModel
    from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM as NativeModel
    models = []
    for cls in (TrainModel, NativeModel):
        model, info = cls.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa",
                                          output_loading_info=True)
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"checkpoint loading mismatch: {info}")
        models.append(model.to("cuda").eval().requires_grad_(False))
    a, b = (m.state_dict() for m in models)
    if a.keys() != b.keys() or not all(torch.equal(a[k], b[k]) for k in a):
        raise RuntimeError("training and inference model tensors differ")
    return models


def collect(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    raw = Path(args.data).read_bytes()
    scenes = [json.loads(line) for line in raw.splitlines()]
    scenes = [s for s in scenes if s["split"] == "pilot" and s["regime"] == "stable"]
    if len(scenes) != 4 or {s["group"] for s in scenes} != set(range(4)):
        raise ValueError("expected four pilot stable scenes")
    torch.manual_seed(108)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train, native = load_pair(args.model)
    if train.config.ttt_chunk != 4096 or native.config.ttt_chunk != 4096:
        raise ValueError("expected chunk 4096")
    versions = [[p._version for p in m.parameters()] for m in (train, native)]
    with output.open("x") as sink:
        for scene in scenes:
            if scene["group"] % 2 != args.shard:
                continue
            tokens = [t for chunk in scene["prefix"] for t in chunk]
            for length in (6144, 12288):
                ids = torch.tensor([tokens[:length]], device="cuda")
                if ids.shape[1] != length:
                    raise ValueError("insufficient source length")
                for enabled in (False, True):
                    torch.cuda.empty_cache()
                    row, profile = measure(lambda: compare(train, native, ids, enabled))
                    row.update({"group": scene["group"], "scene": scene["id"], "length": length,
                                "updates_enabled": enabled, "model": args.model,
                                "data_sha256": hashlib.sha256(raw).hexdigest(), **profile})
                    sink.write(json.dumps(row)+"\n")
                    sink.flush()
                    print(json.dumps(row), flush=True)
    if versions != [[p._version for p in m.parameters()] for m in (train, native)]:
        raise RuntimeError("checkpoint modified")
    Path(str(output)+".audit.json").write_text(json.dumps({
        "checkpoint_tensors_equal": True, "checkpoint_unchanged": True, "rows": 8,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__})+"\n")


def summarize(args):
    rows = [json.loads(line) for file in args.inputs for line in Path(file).read_text().splitlines()]
    expected = {(g, length, enabled) for g in range(4) for length in (6144, 12288) for enabled in (False, True)}
    if len(rows) != 16 or {(r["group"], r["length"], r["updates_enabled"]) for r in rows} != expected:
        raise ValueError("incomplete/duplicate comparisons")
    audits = [json.loads(Path(f+".audit.json").read_text()) for f in args.inputs]
    if not all(a["checkpoint_tensors_equal"] and a["checkpoint_unchanged"] for a in audits):
        raise ValueError("checkpoint audit failed")
    if len({r["data_sha256"] for r in rows}) != 1 or len({r["model"] for r in rows}) != 1:
        raise ValueError("mixed sources/checkpoints")
    groups = {}
    for length in (6144, 12288):
        for enabled in (False, True):
            selected = [r for r in rows if r["length"] == length and r["updates_enabled"] == enabled]
            groups[f"{length}_{'ttt' if enabled else 'zero_lr'}"] = {
                "nll_mean": {k: statistics.fmean(r["nll"][k] for r in selected) for k in selected[0]["nll"]},
                "max_abs_paired_nll_gap_train_stream": max(abs(r["nll"]["train_full"]-r["nll"]["native_stream"]) for r in selected),
                "hidden_comparison_means": {k: {metric: statistics.fmean(r[k][metric] for r in selected)
                                                 for metric in selected[0][k]}
                                             for k in selected[0] if k.startswith("hidden_")}}
    result = {"protocol": "train_infer_parity_v1", "source_groups": 4, "comparisons": 16, "forward_paths": 48,
              "data_sha256": rows[0]["data_sha256"], "model": rows[0]["model"], "audits": audits, "groups": groups,
              "max_peak_allocated_gib": max(r["peak_allocated_gib"] for r in rows),
              "max_peak_reserved_gib": max(r["peak_reserved_gib"] for r in rows),
              "summed_comparison_seconds": sum(r["seconds"] for r in rows)}
    # Diagnostic tripwires, not significance/equivalence tests.
    result["needs_mismatch_investigation"] = any(
        value["max_abs_paired_nll_gap_train_stream"] > (.1 if key.startswith("6144") else .5)
        for key, value in groups.items())
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("collect")
    for flag in ("data", "model", "output"):
        run.add_argument("--"+flag, required=True)
    run.add_argument("--shard", type=int, choices=(0, 1), required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--inputs", nargs="+", required=True)
    summary.add_argument("--output", required=True)
    args = parser.parse_args()
    (collect if args.command == "collect" else summarize)(args)


if __name__ == "__main__":
    main()
