#!/usr/bin/env python3
"""Audit the released Qwen3 native writer's two-step autograd reachability."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_qwen3_native_writer_autograd_audit_v1"
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
    if platform.node() != args.expected_hostname:
        raise RuntimeError("writer autograd audit is restricted to hku-gpu2")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("writer autograd audit must hide all GPUs")

    root = args.upstream_root.expanduser().resolve()
    commit = git(root, "rev-parse", "HEAD")
    clean = git(root, "status", "--porcelain") == ""
    if commit != EXPECTED_UPSTREAM or not clean:
        raise RuntimeError("upstream checkout identity or clean state differs")

    sys.path.insert(0, str(root))
    import torch
    import torch.nn.functional as functional
    from hf_models.hf_qwen3.configuration_qwen3 import Qwen3Config
    from hf_models.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM

    import hf_models.hf_qwen3.configuration_qwen3 as configuration_module
    import hf_models.hf_qwen3.modeling_qwen3 as modeling_module

    configuration_path = Path(configuration_module.__file__).resolve()
    modeling_path = Path(modeling_module.__file__).resolve()
    if root not in configuration_path.parents or root not in modeling_path.parents:
        raise RuntimeError("Qwen3 modules did not resolve from registered upstream")

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
    torch.manual_seed(456)
    model = Qwen3ForCausalLM(config)
    model.train()
    selected = model.model.layers[0].mlp
    parameter_names = dict(model.named_parameters())
    conv_name = "model.layers.0.mlp.ttt_conv.weight"
    projection_name = "model.layers.0.mlp.ttt_proj.weight"
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

    def next_token_loss():
        logits = model(input_ids=input_ids, use_cache=False).logits
        return functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, config.vocab_size),
            input_ids[:, 1:].reshape(-1),
        )

    loss1 = next_token_loss()
    loss1.backward()
    conv_grad1 = selected.ttt_conv.weight.grad.detach().clone()
    projection_grad1 = selected.ttt_proj.weight.grad.detach().clone()
    with torch.no_grad():
        selected.ttt_conv.weight.add_(conv_grad1, alpha=-1e-3)
    conv_nonzero_after_nudge = int(torch.count_nonzero(selected.ttt_conv.weight))

    model.zero_grad(set_to_none=True)
    loss2 = next_token_loss()
    loss2.backward()
    conv_grad2 = selected.ttt_conv.weight.grad.detach().clone()
    projection_grad2 = selected.ttt_proj.weight.grad.detach().clone()

    conv_grad1_norm = float(conv_grad1.norm())
    projection_grad1_norm = float(projection_grad1.norm())
    conv_grad2_norm = float(conv_grad2.norm())
    projection_grad2_norm = float(projection_grad2.norm())
    checks = {
        "upstream_identity": commit == EXPECTED_UPSTREAM and clean,
        "module_resolution": (
            root in configuration_path.parents and root in modeling_path.parents
        ),
        "writer_parameters_registered_trainable": (
            conv_name in parameter_names
            and projection_name in parameter_names
            and parameter_names[conv_name].requires_grad
            and parameter_names[projection_name].requires_grad
        ),
        "first_loss_finite": torch.isfinite(loss1).item(),
        "first_conv_gradient_finite_positive": (
            torch.isfinite(conv_grad1).all().item() and conv_grad1_norm > 0
        ),
        "first_projection_gradient_finite_zero": (
            torch.isfinite(projection_grad1).all().item()
            and projection_grad1_norm == 0
        ),
        "diagnostic_nudge_made_conv_nonzero": conv_nonzero_after_nudge > 0,
        "second_loss_finite": torch.isfinite(loss2).item(),
        "second_conv_gradient_finite_positive": (
            torch.isfinite(conv_grad2).all().item() and conv_grad2_norm > 0
        ),
        "second_projection_gradient_finite_positive": (
            torch.isfinite(projection_grad2).all().item()
            and projection_grad2_norm > 0
        ),
    }
    return {
        "protocol": PROTOCOL,
        "passed": all(checks.values()),
        "checks": checks,
        "runtime": {
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
            "loss_implementation": "torch.nn.functional.cross_entropy_on_official_logits",
        },
        "upstream": {
            "root": str(root),
            "commit": commit,
            "clean": clean,
            "configuration_path": str(configuration_path),
            "configuration_sha256": sha256(configuration_path),
            "modeling_path": str(modeling_path),
            "modeling_sha256": sha256(modeling_path),
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
            "first_loss": float(loss1.detach()),
            "first_conv_gradient_norm": conv_grad1_norm,
            "first_projection_gradient_norm": projection_grad1_norm,
            "conv_nonzero_after_nudge": conv_nonzero_after_nudge,
            "diagnostic_nudge_scale": -1e-3,
            "second_loss": float(loss2.detach()),
            "second_conv_gradient_norm": conv_grad2_norm,
            "second_projection_gradient_norm": projection_grad2_norm,
        },
        "claim_boundary": (
            "tiny CPU native-writer autograd audit using standard CE on official "
            "logits; Liger CUDA loss, AdamW/FSDP2 and full-model training remain required"
        ),
    }


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
