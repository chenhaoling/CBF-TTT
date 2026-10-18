#!/usr/bin/env python3
"""Audit released Qwen3 native-TTT initialization without exposing a GPU."""

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


PROTOCOL = "in_place_ttt_qwen3_native_initialization_audit_v1"
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
        raise RuntimeError("initialization audit is restricted to hku-gpu2")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("initialization audit must hide all GPUs")

    root = args.upstream_root.expanduser().resolve()
    commit = git(root, "rev-parse", "HEAD")
    clean = git(root, "status", "--porcelain") == ""
    if commit != EXPECTED_UPSTREAM or not clean:
        raise RuntimeError("upstream checkout identity or clean state differs")

    sys.path.insert(0, str(root))
    import torch
    from hf_models.hf_qwen3.configuration_qwen3 import Qwen3Config
    from hf_models.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM

    import hf_models.hf_qwen3.configuration_qwen3 as configuration_module
    import hf_models.hf_qwen3.modeling_qwen3 as modeling_module

    configuration_path = Path(configuration_module.__file__).resolve()
    modeling_path = Path(modeling_module.__file__).resolve()
    if root not in configuration_path.parents or root not in modeling_path.parents:
        raise RuntimeError("Qwen3 modules did not resolve from registered upstream")

    common = {
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
    }
    torch.manual_seed(123)
    baseline = Qwen3ForCausalLM(Qwen3Config(**common, ttt_mode=False))
    baseline_state = {
        key: value.detach().clone() for key, value in baseline.state_dict().items()
    }

    torch.manual_seed(456)
    ttt = Qwen3ForCausalLM(Qwen3Config(
        **common,
        ttt_mode=True,
        ttt_layers=[0],
        ttt_proj=True,
        ttt_lr=3.0,
        ttt_chunk=4,
        ttt_target="hidden_states",
    ))
    incompatible = ttt.load_state_dict(baseline_state, strict=False)
    selected = ttt.model.layers[0].mlp
    unselected = ttt.model.layers[1].mlp

    projection = selected.ttt_proj.weight.detach()
    diagonal = torch.diagonal(projection)
    off_diagonal = projection - torch.diag(diagonal)
    convolution = selected.ttt_conv
    target = torch.arange(32 * 8, dtype=torch.float32).reshape(1, 32, 8)
    target_output = convolution(target)

    baseline.eval()
    ttt.eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    with torch.no_grad():
        baseline_logits = baseline(input_ids=input_ids, use_cache=False).logits
        ttt_logits = ttt(input_ids=input_ids, use_cache=False).logits

    expected_missing = [
        "model.layers.0.mlp.ttt_conv.weight",
        "model.layers.0.mlp.ttt_proj.weight",
    ]
    checks = {
        "upstream_identity": commit == EXPECTED_UPSTREAM and clean,
        "module_resolution": (
            root in configuration_path.parents and root in modeling_path.parents
        ),
        "missing_keys_exact": sorted(incompatible.missing_keys) == expected_missing,
        "unexpected_keys_absent": not incompatible.unexpected_keys,
        "shared_slow_weights_bitwise_equal": all(
            torch.equal(ttt.state_dict()[key], value)
            for key, value in baseline_state.items()
        ),
        "conv_depthwise_kernel5_bias_free": (
            convolution.kernel_size == (5,)
            and convolution.groups == common["hidden_size"]
            and convolution.in_channels == common["hidden_size"]
            and convolution.out_channels == common["hidden_size"]
            and convolution.bias is None
        ),
        "conv_zero_initialized": torch.count_nonzero(convolution.weight).item() == 0,
        "projection_square_bias_free": (
            tuple(projection.shape) == (32, 32) and selected.ttt_proj.bias is None
        ),
        "projection_off_diagonal_zero": torch.count_nonzero(off_diagonal).item() == 0,
        "projection_diagonal_finite_nonzero": (
            torch.isfinite(diagonal).all().item()
            and torch.count_nonzero(diagonal).item() == diagonal.numel()
        ),
        "initial_target_exact_zero": torch.count_nonzero(target_output).item() == 0,
        "initial_logits_bitwise_equal": torch.equal(baseline_logits, ttt_logits),
        "unselected_layer_has_no_ttt_parameters": (
            not hasattr(unselected, "ttt_conv") and not hasattr(unselected, "ttt_proj")
        ),
    }
    result = {
        "protocol": PROTOCOL,
        "passed": all(checks.values()),
        "checks": checks,
        "runtime": {
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
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
            "missing_keys": sorted(incompatible.missing_keys),
            "unexpected_keys": sorted(incompatible.unexpected_keys),
            "conv_weight_shape": list(convolution.weight.shape),
            "conv_nonzero": int(torch.count_nonzero(convolution.weight)),
            "projection_shape": list(projection.shape),
            "projection_diagonal_nonzero": int(torch.count_nonzero(diagonal)),
            "projection_off_diagonal_nonzero": int(torch.count_nonzero(off_diagonal)),
            "target_output_nonzero": int(torch.count_nonzero(target_output)),
            "max_abs_logit_difference": float(
                (baseline_logits - ttt_logits).abs().max()
            ),
        },
        "claim_boundary": (
            "tiny CPU initialization-semantics audit only; full Qwen3-4B "
            "resource, training and evaluation audits remain required"
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
