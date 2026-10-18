#!/usr/bin/env python3
"""Preflight Qwen3 Liger loss and VeOmni fused AdamW on one hku-gpu2 GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_qwen3_liger_fused_adamw_gpu_preflight_v1"
EXPECTED_UPSTREAM = "be2324829b0e91c8fd10a74d4b43714fde6676e1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if platform.node() != args.expected_hostname:
        raise RuntimeError("GPU preflight is restricted to hku-gpu2")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("GPU preflight requires physical GPU 0 only")

    root = args.upstream_root.expanduser().resolve()
    commit = git(root, "rev-parse", "HEAD")
    clean = git(root, "status", "--porcelain") == ""
    if commit != EXPECTED_UPSTREAM or not clean:
        raise RuntimeError("upstream checkout identity or clean state differs")

    os.environ["MODELING_BACKEND"] = "hf"
    sys.path.insert(0, str(root))
    import torch
    import hf_models  # noqa: F401
    from hf_models.hf_qwen3.configuration_qwen3 import Qwen3Config
    from hf_models.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from liger_kernel.transformers.model.qwen3 import lce_forward
    from veomni.optim import build_optimizer

    import hf_models.hf_qwen3.configuration_qwen3 as configuration_module
    import hf_models.hf_qwen3.modeling_qwen3 as modeling_module
    import liger_kernel
    import veomni

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible CUDA device")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    if "RTX 5090" not in gpu_name or free_before < 28 * (1 << 30):
        raise RuntimeError("GPU identity or free-memory gate failed")

    configuration_path = Path(configuration_module.__file__).resolve()
    modeling_path = Path(modeling_module.__file__).resolve()
    if root not in configuration_path.parents or root not in modeling_path.parents:
        raise RuntimeError("Qwen3 modules did not resolve from registered upstream")

    torch.manual_seed(456)
    torch.cuda.manual_seed_all(456)
    torch.cuda.reset_peak_memory_stats(device)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        ttt_mode=True,
        ttt_layers=[0],
        ttt_proj=True,
        ttt_lr=3.0,
        ttt_chunk=4,
        ttt_target="hidden_states",
    )
    model = Qwen3ForCausalLM(config).to(device=device, dtype=torch.bfloat16).train()
    selected = model.model.layers[0].mlp
    conv_name = "model.layers.0.mlp.ttt_conv.weight"
    projection_name = "model.layers.0.mlp.ttt_proj.weight"
    named_parameters = dict(model.named_parameters())
    optimizer = build_optimizer(
        model,
        lr=5e-6,
        weight_decay=0.1,
        fused=True,
        optimizer_type="adamw",
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=device)

    def backward_row() -> dict[str, Any]:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        loss.backward()
        conv_grad = selected.ttt_conv.weight.grad.detach()
        projection_grad = selected.ttt_proj.weight.grad.detach()
        return {
            "loss_tensor": loss,
            "loss": float(loss.detach()),
            "conv_gradient_norm": float(conv_grad.float().norm()),
            "projection_gradient_norm": float(projection_grad.float().norm()),
            "writer_gradients_finite": bool(
                torch.isfinite(conv_grad).all() and torch.isfinite(projection_grad).all()
            ),
        }

    conv_before = selected.ttt_conv.weight.detach().clone()
    first = backward_row()
    total_norm1 = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    torch.cuda.synchronize(device)
    conv_displacement1 = float(
        (selected.ttt_conv.weight.detach() - conv_before).float().norm()
    )
    writer_finite1 = bool(
        torch.isfinite(selected.ttt_conv.weight).all()
        and torch.isfinite(selected.ttt_proj.weight).all()
    )

    second = backward_row()
    total_norm2 = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    torch.cuda.synchronize(device)
    writer_finite2 = bool(
        torch.isfinite(selected.ttt_conv.weight).all()
        and torch.isfinite(selected.ttt_proj.weight).all()
    )

    active_liger_forward = Qwen3ForCausalLM.forward is lce_forward
    optimizer_fused = bool(optimizer.defaults.get("fused"))
    checks = {
        "upstream_identity": commit == EXPECTED_UPSTREAM and clean,
        "module_resolution": (
            root in configuration_path.parents and root in modeling_path.parents
        ),
        "single_rtx5090_and_free_memory": (
            torch.cuda.device_count() == 1
            and "RTX 5090" in gpu_name
            and free_before >= 28 * (1 << 30)
        ),
        "liger_forward_active": active_liger_forward,
        "veomni_fused_adamw_active": (
            isinstance(optimizer, torch.optim.AdamW)
            and optimizer_fused
            and optimizer.defaults["lr"] == 5e-6
        ),
        "writer_parameters_in_optimizer": (
            id(named_parameters[conv_name]) in optimizer_ids
            and id(named_parameters[projection_name]) in optimizer_ids
        ),
        "first_fused_loss_finite": bool(torch.isfinite(first["loss_tensor"])),
        "first_writer_gradients_finite": first["writer_gradients_finite"],
        "first_conv_gradient_positive": first["conv_gradient_norm"] > 0,
        "first_projection_gradient_zero": first["projection_gradient_norm"] == 0,
        "first_total_gradient_norm_finite": bool(torch.isfinite(torch.tensor(total_norm1))),
        "first_fused_step_changed_conv": conv_displacement1 > 0,
        "first_fused_step_writer_finite": writer_finite1,
        "second_fused_loss_finite": bool(torch.isfinite(second["loss_tensor"])),
        "second_writer_gradients_finite": second["writer_gradients_finite"],
        "second_conv_gradient_positive": second["conv_gradient_norm"] > 0,
        "second_projection_gradient_positive": second["projection_gradient_norm"] > 0,
        "second_total_gradient_norm_finite": bool(torch.isfinite(torch.tensor(total_norm2))),
        "second_fused_step_writer_finite": writer_finite2,
    }
    result = {
        "protocol": PROTOCOL,
        "passed": all(checks.values()),
        "checks": checks,
        "runtime": {
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": gpu_name,
            "visible_device_count": torch.cuda.device_count(),
            "free_memory_before_bytes": free_before,
            "total_memory_bytes": total_memory,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "wall_seconds": time.perf_counter() - started,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "veomni_version": getattr(veomni, "__version__", None),
            "liger_kernel_version": getattr(liger_kernel, "__version__", None),
        },
        "upstream": {
            "root": str(root),
            "commit": commit,
            "clean": clean,
            "configuration_sha256": sha256(configuration_path),
            "modeling_sha256": sha256(modeling_path),
        },
        "optimizer": {
            "class": f"{optimizer.__class__.__module__}.{optimizer.__class__.__name__}",
            "fused": optimizer_fused,
            "learning_rate": optimizer.defaults["lr"],
            "weight_decay_groups": [group["weight_decay"] for group in optimizer.param_groups],
            "writer_parameters_in_groups": True,
            "clip_implementation": "torch.nn.utils.clip_grad_norm_for_non_distributed_fixture",
            "clip_max_norm": 1.0,
        },
        "fixture": {
            "hidden_size": 32,
            "layers": 2,
            "ttt_layers": [0],
            "ttt_chunk": 4,
            "ttt_lr": 3.0,
            "serialized_input_ids": False,
            "serialized_logits": False,
            "serialized_gradients": False,
            "first_loss": first["loss"],
            "first_conv_gradient_norm": first["conv_gradient_norm"],
            "first_projection_gradient_norm": first["projection_gradient_norm"],
            "first_total_gradient_norm": total_norm1,
            "first_conv_displacement_norm": conv_displacement1,
            "second_loss": second["loss"],
            "second_conv_gradient_norm": second["conv_gradient_norm"],
            "second_projection_gradient_norm": second["projection_gradient_norm"],
            "second_total_gradient_norm": total_norm2,
        },
        "claim_boundary": (
            "tiny single-GPU Liger and fused-AdamW environment preflight only; "
            "FSDP2, VeOmni distributed clipping, corpus training and RULER remain required"
        ),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-hostname", default="user-Super-Server")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run(args)
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(partial, output)
    print(json.dumps({"output": str(output), "passed": result["passed"]}, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
