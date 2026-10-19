#!/usr/bin/env python3
"""Audit the exact Qwen3-4B-Base native writer on one hku-gpu2 GPU."""

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


PROTOCOL = "in_place_ttt_qwen3_4b_full_checkpoint_writer_preflight_v1"
EXPECTED_UPSTREAM = "be2324829b0e91c8fd10a74d4b43714fde6676e1"
EXPECTED_REPOSITORY = "Qwen/Qwen3-4B-Base"
EXPECTED_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
TTT_LAYERS = [0, 6, 12, 18, 24, 30]


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


def verify_model_identity(model_root: Path, identity_path: Path) -> dict[str, Any]:
    identity = json.loads(identity_path.read_text())
    if not identity.get("passed") or not all(identity.get("checks", {}).values()):
        raise RuntimeError("model identity artifact is not passing")
    if (
        identity.get("repository") != EXPECTED_REPOSITORY
        or identity.get("revision") != EXPECTED_REVISION
    ):
        raise RuntimeError("model repository or revision differs")
    files = identity.get("files", {})
    if not files:
        raise RuntimeError("model identity has no file inventory")
    for name, expected in files.items():
        path = model_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"model file identity differs: {name}")
    return identity


def writer_name(layer: int, kind: str) -> str:
    return f"model.layers.{layer}.mlp.{kind}.weight"


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if platform.node() != args.expected_hostname:
        raise RuntimeError("full-checkpoint preflight is restricted to hku-gpu2")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("full-checkpoint preflight requires physical GPU 0 only")

    upstream = args.upstream_root.expanduser().resolve()
    model_root = args.model.expanduser().resolve()
    identity_path = args.model_identity.expanduser().resolve()
    upstream_commit = git(upstream, "rev-parse", "HEAD")
    upstream_clean = git(upstream, "status", "--porcelain") == ""
    if upstream_commit != EXPECTED_UPSTREAM or not upstream_clean:
        raise RuntimeError("upstream checkout identity or clean state differs")
    identity = verify_model_identity(model_root, identity_path)

    os.environ["MODELING_BACKEND"] = "hf"
    sys.path.insert(0, str(upstream))
    import torch
    import hf_models  # noqa: F401
    from hf_models.hf_qwen3.configuration_qwen3 import Qwen3Config
    from hf_models.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from liger_kernel.transformers.model.qwen3 import lce_forward

    import hf_models.hf_qwen3.configuration_qwen3 as configuration_module
    import hf_models.hf_qwen3.modeling_qwen3 as modeling_module
    import liger_kernel

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible CUDA device")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    if "RTX 5090" not in gpu_name or free_before < 28 * (1 << 30):
        raise RuntimeError("GPU identity or free-memory gate failed")

    configuration_path = Path(configuration_module.__file__).resolve()
    modeling_path = Path(modeling_module.__file__).resolve()
    if upstream not in configuration_path.parents or upstream not in modeling_path.parents:
        raise RuntimeError("Qwen3 modules did not resolve from registered upstream")

    torch.manual_seed(6404)
    torch.cuda.manual_seed_all(6404)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)

    config = Qwen3Config.from_pretrained(model_root, local_files_only=True)
    config.ttt_mode = True
    config.ttt_layers = TTT_LAYERS
    config.ttt_proj = True
    config.ttt_target = "hidden_states"
    config.ttt_lr = 3.0
    config.ttt_chunk = 16
    config.use_cache = False
    model, loading = Qwen3ForCausalLM.from_pretrained(
        model_root,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        output_loading_info=True,
    )
    model = model.to(device)

    expected_missing = sorted(
        [writer_name(layer, kind) for layer in TTT_LAYERS for kind in ("ttt_conv", "ttt_proj")]
    )
    missing = sorted(loading.get("missing_keys", []))
    unexpected = sorted(loading.get("unexpected_keys", []))
    mismatched = loading.get("mismatched_keys", [])
    errors = loading.get("error_msgs", [])

    named_parameters = dict(model.named_parameters())
    writer_names = expected_missing
    if any(name not in named_parameters for name in writer_names):
        raise RuntimeError("one or more expected writer parameters are absent")
    model.requires_grad_(False)
    for name in writer_names:
        named_parameters[name].requires_grad_(True)
    slow_versions = {
        name: parameter._version
        for name, parameter in model.named_parameters()
        if name not in writer_names
    }

    initialization_rows = []
    initialization_ok = True
    for layer in TTT_LAYERS:
        mlp = model.model.layers[layer].mlp
        conv = mlp.ttt_conv
        projection = mlp.ttt_proj
        diagonal = projection.weight.detach().diagonal()
        conv_nonzero = int(torch.count_nonzero(conv.weight.detach()))
        projection_nonzero = int(torch.count_nonzero(projection.weight.detach()))
        diagonal_nonzero = int(torch.count_nonzero(diagonal))
        row_ok = bool(
            conv.bias is None
            and conv.kernel_size == (5,)
            and conv.groups == config.hidden_size
            and conv.in_channels == config.hidden_size
            and conv.out_channels == config.hidden_size
            and conv_nonzero == 0
            and projection.bias is None
            and tuple(projection.weight.shape) == (config.hidden_size, config.hidden_size)
            and torch.isfinite(diagonal).all()
            and diagonal_nonzero == config.hidden_size
            and projection_nonzero == diagonal_nonzero
        )
        initialization_ok = initialization_ok and row_ok
        initialization_rows.append(
            {
                "layer": layer,
                "conv_shape": list(conv.weight.shape),
                "conv_nonzero": conv_nonzero,
                "projection_shape": list(projection.weight.shape),
                "projection_nonzero": projection_nonzero,
                "projection_diagonal_nonzero": diagonal_nonzero,
                "passed": row_ok,
            }
        )

    ids = [1 + ((index * 7919 + 17) % (config.vocab_size - 2)) for index in range(64)]
    input_sha256 = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    input_ids = torch.tensor([ids], device=device)

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        writer_logits = model(input_ids=input_ids, use_cache=False).logits
    branch_states = {layer: model.model.layers[layer].is_ttt_layer for layer in TTT_LAYERS}
    try:
        for layer in TTT_LAYERS:
            model.model.layers[layer].is_ttt_layer = False
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            bypass_logits = model(input_ids=input_ids, use_cache=False).logits
    finally:
        for layer, state in branch_states.items():
            model.model.layers[layer].is_ttt_layer = state
    initial_max_abs_logit_difference = float(
        (writer_logits.float() - bypass_logits.float()).abs().max()
    )
    initial_logits_finite = bool(
        torch.isfinite(writer_logits).all() and torch.isfinite(bypass_logits).all()
    )
    initial_logits_close = bool(
        torch.allclose(writer_logits.float(), bypass_logits.float(), rtol=1e-3, atol=1e-3)
    )
    del writer_logits, bypass_logits

    def backward_row() -> dict[str, Any]:
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        loss.backward()
        conv_norms: dict[str, float] = {}
        projection_norms: dict[str, float] = {}
        finite = True
        for layer in TTT_LAYERS:
            conv_grad = model.model.layers[layer].mlp.ttt_conv.weight.grad
            projection_grad = model.model.layers[layer].mlp.ttt_proj.weight.grad
            if conv_grad is None or projection_grad is None:
                raise RuntimeError(f"missing writer gradient at layer {layer}")
            finite = finite and bool(
                torch.isfinite(conv_grad).all() and torch.isfinite(projection_grad).all()
            )
            conv_norms[str(layer)] = float(conv_grad.float().norm())
            projection_norms[str(layer)] = float(projection_grad.float().norm())
        return {
            "loss_tensor": loss,
            "loss": float(loss.detach()),
            "conv_gradient_norms": conv_norms,
            "projection_gradient_norms": projection_norms,
            "writer_gradients_finite": finite,
        }

    model.train()
    first = backward_row()
    with torch.no_grad():
        for layer in TTT_LAYERS:
            conv = model.model.layers[layer].mlp.ttt_conv.weight
            conv.add_(conv.grad, alpha=-1e-3)
    conv_nonzero_after_nudge = {
        str(layer): int(torch.count_nonzero(model.model.layers[layer].mlp.ttt_conv.weight))
        for layer in TTT_LAYERS
    }
    second = backward_row()
    torch.cuda.synchronize(device)

    slow_unchanged = slow_versions == {
        name: parameter._version
        for name, parameter in model.named_parameters()
        if name not in writer_names
    }
    writer_finite = all(
        bool(torch.isfinite(named_parameters[name]).all()) for name in writer_names
    )
    active_liger_forward = Qwen3ForCausalLM.forward is lce_forward
    checks = {
        "upstream_identity": upstream_commit == EXPECTED_UPSTREAM and upstream_clean,
        "model_identity": (
            identity["repository"] == EXPECTED_REPOSITORY
            and identity["revision"] == EXPECTED_REVISION
        ),
        "module_resolution": upstream in configuration_path.parents and upstream in modeling_path.parents,
        "single_rtx5090_and_free_memory": (
            torch.cuda.device_count() == 1
            and "RTX 5090" in gpu_name
            and free_before >= 28 * (1 << 30)
        ),
        "full_model_shape": (
            config.num_hidden_layers == 36
            and config.hidden_size == 2560
            and config.vocab_size == 151936
            and len(model.model.layers) == 36
        ),
        "ttt_layer_set_exact": [index for index, layer in enumerate(model.model.layers) if layer.is_ttt_layer] == TTT_LAYERS,
        "loading_missing_keys_exact": missing == expected_missing,
        "loading_has_no_other_errors": not unexpected and not mismatched and not errors,
        "writer_initialization_exact": initialization_ok,
        "only_writer_trainable": sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad) == writer_names,
        "initial_logits_finite": initial_logits_finite,
        "initial_logits_match_bypass": initial_logits_close,
        "liger_forward_active": active_liger_forward,
        "first_fused_loss_finite": bool(torch.isfinite(first["loss_tensor"])),
        "first_writer_gradients_finite": first["writer_gradients_finite"],
        "first_all_conv_gradients_positive": all(value > 0 for value in first["conv_gradient_norms"].values()),
        "first_all_projection_gradients_zero": all(value == 0 for value in first["projection_gradient_norms"].values()),
        "diagnostic_nudge_all_conv_nonzero": all(value > 0 for value in conv_nonzero_after_nudge.values()),
        "second_fused_loss_finite": bool(torch.isfinite(second["loss_tensor"])),
        "second_writer_gradients_finite": second["writer_gradients_finite"],
        "second_all_conv_gradients_positive": all(value > 0 for value in second["conv_gradient_norms"].values()),
        "second_all_projection_gradients_positive": all(value > 0 for value in second["projection_gradient_norms"].values()),
        "slow_weights_unchanged": slow_unchanged,
        "writer_weights_finite": writer_finite,
    }
    return {
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
            "liger_kernel_version": getattr(liger_kernel, "__version__", None),
        },
        "upstream": {
            "root": str(upstream),
            "commit": upstream_commit,
            "clean": upstream_clean,
            "configuration_sha256": sha256(configuration_path),
            "modeling_sha256": sha256(modeling_path),
        },
        "model": {
            "root": str(model_root),
            "identity_path": str(identity_path),
            "identity_sha256": sha256(identity_path),
            "repository": identity["repository"],
            "revision": identity["revision"],
            "layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "dtype": "bfloat16",
            "attention": "sdpa",
        },
        "fixture": {
            "ttt_layers": TTT_LAYERS,
            "ttt_chunk": 16,
            "ttt_lr": 3.0,
            "sequence_tokens": 64,
            "input_ids_sha256": input_sha256,
            "serialized_input_ids": False,
            "serialized_logits": False,
            "serialized_gradients": False,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "initialization": initialization_rows,
            "initial_max_abs_logit_difference": initial_max_abs_logit_difference,
            "first_loss": first["loss"],
            "first_conv_gradient_norms": first["conv_gradient_norms"],
            "first_projection_gradient_norms": first["projection_gradient_norms"],
            "conv_nonzero_after_nudge": conv_nonzero_after_nudge,
            "second_loss": second["loss"],
            "second_conv_gradient_norms": second["conv_gradient_norms"],
            "second_projection_gradient_norms": second["projection_gradient_norms"],
            "checkpoint_saved": False,
        },
        "claim_boundary": (
            "exact Qwen3-4B-Base single-GPU writer integration preflight only; "
            "FSDP2, distributed clipping, formal chunks, corpus training and RULER remain required"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
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
