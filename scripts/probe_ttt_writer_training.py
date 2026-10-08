"""Bounded resource/gradient probe; never saves or promotes an adapted checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from tasks.cbf_selective import measure


def writer_parameters(model):
    model.requires_grad_(False)
    selected = {}
    for name, parameter in model.named_parameters():
        if name.endswith((".mlp.ttt_conv.weight", ".mlp.ttt_proj.weight")):
            # FP32 master parameters/moments avoid sub-BF16 optimizer steps vanishing.
            parameter.data = parameter.data.float()
            parameter.requires_grad_(True)
            selected[name] = parameter
    if not selected:
        raise ValueError("no existing TTT writer parameters")
    return selected


def training_tokens(path, tokenizer, length):
    tokens, row_hashes = [], []
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            text = row.get("content_split", row.get("text"))
            if not isinstance(text, str):
                raise ValueError("expected packed training text")
            tokens.extend(tokenizer.encode(text, add_special_tokens=False))
            tokens.append(tokenizer.eos_token_id)
            row_hashes.append(hashlib.sha256(line.encode()).hexdigest())
            if len(tokens) >= length:
                break
    if len(tokens) < length:
        raise ValueError("insufficient training tokens")
    return tokens[:length], row_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--length", type=int, choices=(6144, 12288), required=True)
    parser.add_argument("--output", required=True)
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
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # The same packed prefix is used by both lengths; this is a resource check,
    # not a natural long-document learning comparison.
    ids, row_hashes = training_tokens(args.training_data, tokenizer, args.length)
    inputs = torch.tensor([ids], device="cuda")
    model, info = Qwen3ForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                attn_implementation="sdpa", output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"checkpoint mismatch: {info}")
    model = model.to("cuda").train()
    selected = writer_parameters(model)
    if len(selected) != 2*len(model.config.ttt_layers):
        raise ValueError("expected conv and projection for every TTT layer")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(list(selected.values()), lr=1e-5, weight_decay=.01)
    frozen = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
    initial = {n: p.detach().clone() for n, p in selected.items()}
    result = {"protocol": "writer_resource_probe_v1", "model": args.model,
              "length": args.length, "steps": 3, "training_data": args.training_data,
              "packed_training_row_sha256": row_hashes,
              "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
              "trainable_parameters": sum(p.numel() for p in selected.values()),
              "trainable_names": list(selected), "writer_dtype": "float32", "backbone_dtype": "bfloat16",
              "compute_autocast": "bfloat16", "optimizer": "AdamW", "lr": 1e-5, "weight_decay": .01,
              "gradient_checkpointing": "nonreentrant", "measurements": []}
    for step in range(3):
        def one_step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=inputs, labels=inputs, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            grads = {n: float(p.grad.float().norm()) if p.grad is not None else None for n, p in selected.items()}
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in selected.values()):
                raise RuntimeError("missing/nonfinite writer gradient")
            if any(g == 0 for g in grads.values()):
                raise RuntimeError("zero writer gradient")
            norm = float(torch.nn.utils.clip_grad_norm_(list(selected.values()), 1.0))
            optimizer.step()
            return {"step": step+1, "loss": float(loss.detach()), "unclipped_grad_norm": norm,
                    "writer_grad_norms": grads}
        row, profile = measure(one_step)
        row.update(profile)
        result["measurements"].append(row)
        print(json.dumps(row), flush=True)
    if frozen != {n: p._version for n, p in model.named_parameters() if not p.requires_grad}:
        raise RuntimeError("frozen backbone changed")
    changes = {n: float((p.detach()-initial[n]).norm()) for n, p in selected.items()}
    if any(v <= 0 for v in changes.values()):
        raise RuntimeError("writer failed to update")
    result.update({"frozen_backbone_unchanged": True, "writer_update_norms": changes,
                   "optimizer_state_fp32": all(v.dtype == torch.float32 for state in optimizer.state.values()
                                                for v in state.values() if isinstance(v, torch.Tensor)),
                   "checkpoint_saved": False})
    output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
