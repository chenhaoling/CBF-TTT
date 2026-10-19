#!/usr/bin/env python3
"""Audit exact Qwen3-4B-Base native-writer autograd through VeOmni FSDP2."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_qwen3_4b_fsdp2_autograd_audit_v1"
EXPECTED_UPSTREAM = "be2324829b0e91c8fd10a74d4b43714fde6676e1"
EXPECTED_REPOSITORY = "Qwen/Qwen3-4B-Base"
EXPECTED_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
TTT_LAYERS = [0, 6, 12, 18, 24, 30]
SEQUENCE_TOKENS = 8192
TTT_CHUNK = 4096
DIAGNOSTIC_NUDGE = -1e-3


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-hostname", default="user-Super-Server")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if platform.node() != args.expected_hostname:
        raise RuntimeError("FSDP2 autograd audit is restricted to hku-gpu2")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("FSDP2 autograd audit requires physical GPUs 0 and 1")

    os.environ["MODELING_BACKEND"] = "hf"
    upstream = args.upstream_root.expanduser().resolve()
    model_root = args.model.expanduser().resolve()
    identity_path = args.model_identity.expanduser().resolve()
    plan_path = args.plan.expanduser().resolve()
    output = args.output.expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    tool_root = Path(__file__).resolve().parents[1]
    for path in (identity_path, plan_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(upstream))
    import torch
    import torch.distributed as dist
    import hf_models  # noqa: F401
    from liger_kernel.transformers.model.qwen3 import lce_forward
    from torch.distributed.tensor import DTensor
    from veomni.distributed.offloading import build_activation_offloading_context
    from veomni.distributed.parallel_state import init_parallel_state
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models import build_foundation_model

    import hf_models.hf_qwen3.configuration_qwen3 as configuration_module
    import hf_models.hf_qwen3.modeling_qwen3 as modeling_module
    import liger_kernel
    import veomni

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if world_size != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError("expected exactly two ranks and two visible GPUs")
    gpu_name = torch.cuda.get_device_name(device)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    if "RTX 5090" not in gpu_name or free_before < 28 * (1 << 30):
        raise RuntimeError("GPU identity or free-memory gate failed")
    torch.cuda.reset_peak_memory_stats(device)

    output_exists: list[Any] = [None]
    if rank == 0:
        output_exists[0] = output.exists() or partial.exists()
    dist.broadcast_object_list(output_exists, src=0)
    if output_exists[0]:
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")

    identity_error = None
    identity: dict[str, Any] | None = None
    if rank == 0:
        try:
            identity = verify_model_identity(model_root, identity_path)
        except Exception as exc:
            identity_error = f"{type(exc).__name__}: {exc}"
    identity_payload: list[Any] = [identity_error, identity]
    dist.broadcast_object_list(identity_payload, src=0)
    identity_error, identity = identity_payload
    if identity_error is not None or identity is None:
        raise RuntimeError(f"model identity validation failed: {identity_error}")

    upstream_commit = git(upstream, "rev-parse", "HEAD")
    upstream_clean = git(upstream, "status", "--porcelain") == ""
    tool_commit = git(tool_root, "rev-parse", "HEAD")
    tool_clean = git(tool_root, "status", "--porcelain") == ""
    if upstream_commit != EXPECTED_UPSTREAM or not upstream_clean or not tool_clean:
        raise RuntimeError("source checkout identity or clean state differs")

    torch.manual_seed(6404 + rank)
    torch.cuda.manual_seed_all(6404 + rank)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    init_parallel_state(
        dp_size=2,
        dp_replicate_size=1,
        dp_shard_size=2,
        tp_size=1,
        ep_size=1,
        pp_size=1,
        cp_size=1,
        ulysses_size=1,
        dp_mode="fsdp2",
    )
    foundation = {
        "max_position_embeddings": 32768,
        "ttt_layers": TTT_LAYERS,
        "ttt_mode": True,
        "ttt_proj": True,
        "ttt_target": "hidden_states",
        "ttt_lr": 3.0,
        "ttt_chunk": TTT_CHUNK,
    }
    model = build_foundation_model(
        config_path=str(model_root),
        weights_path=str(model_root),
        torch_dtype="float32",
        attn_implementation="sdpa",
        init_device="meta",
        config_kwargs=foundation,
    )
    model = build_parallelize_model(
        model,
        init_device="meta",
        weights_path=str(model_root),
        enable_full_shard=True,
        enable_mixed_precision=True,
        enable_gradient_checkpointing=True,
        enable_fsdp_offload=False,
        basic_modules=model._no_split_modules,
    )
    model.train()
    torch.cuda.synchronize(device)

    writer_names = sorted(
        f"model.layers.{layer}.mlp.{kind}.weight"
        for layer in TTT_LAYERS
        for kind in ("ttt_conv", "ttt_proj")
    )
    named_parameters = dict(model.named_parameters())
    actual_writer_names = sorted(
        name
        for name in named_parameters
        if ".ttt_conv." in name or ".ttt_proj." in name
    )
    writer_dtensors = all(
        isinstance(named_parameters[name], DTensor) for name in writer_names
    )
    if actual_writer_names != writer_names or not writer_dtensors:
        raise RuntimeError("writer parameter set or DTensor storage differs")

    input_ids_list = [
        1 + ((index * 7919 + 17 + rank * 104729) % (model.config.vocab_size - 2))
        for index in range(SEQUENCE_TOKENS)
    ]
    input_sha256 = hashlib.sha256(
        json.dumps(input_ids_list, separators=(",", ":")).encode()
    ).hexdigest()
    input_ids = torch.tensor([input_ids_list], device=device, dtype=torch.long)
    labels = input_ids.clone()
    del input_ids_list

    forward_context, backward_context = build_activation_offloading_context(
        True, True, 0.0
    )

    def reduced_norm(tensor: Any) -> float:
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        square = local.detach().float().square().sum()
        dist.all_reduce(square, op=dist.ReduceOp.SUM)
        return math.sqrt(float(square))

    def reduced_nonzero(tensor: Any) -> int:
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        count = torch.count_nonzero(local.detach()).to(torch.int64)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        return int(count)

    def backward_pass() -> dict[str, Any]:
        model.zero_grad(set_to_none=True)
        with forward_context:
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("released fused forward returned no loss")
        with backward_context:
            loss.backward()
        torch.cuda.synchronize(device)
        conv_norms: dict[str, float] = {}
        projection_norms: dict[str, float] = {}
        local_gradients_finite = True
        for layer in TTT_LAYERS:
            mlp = model.model.layers[layer].mlp
            conv_grad = mlp.ttt_conv.weight.grad
            projection_grad = mlp.ttt_proj.weight.grad
            if conv_grad is None or projection_grad is None:
                raise RuntimeError(f"missing writer gradient at layer {layer}")
            for gradient in (conv_grad, projection_grad):
                local = gradient.to_local() if isinstance(gradient, DTensor) else gradient
                local_gradients_finite = (
                    local_gradients_finite and bool(torch.isfinite(local).all())
                )
            conv_norms[str(layer)] = reduced_norm(conv_grad)
            projection_norms[str(layer)] = reduced_norm(projection_grad)
        finite_tensor = torch.tensor(
            [int(local_gradients_finite)], device=device, dtype=torch.int64
        )
        dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
        losses: list[Any] = [None for _ in range(world_size)]
        dist.all_gather_object(losses, float(loss.detach()))
        return {
            "losses_by_rank": losses,
            "losses_finite": all(math.isfinite(value) for value in losses),
            "writer_gradients_finite": bool(finite_tensor.item()),
            "conv_gradient_norms": conv_norms,
            "projection_gradient_norms": projection_norms,
        }

    first = backward_pass()
    with torch.no_grad():
        for layer in TTT_LAYERS:
            conv = model.model.layers[layer].mlp.ttt_conv.weight
            conv.add_(conv.grad, alpha=DIAGNOSTIC_NUDGE)
    conv_nonzero_after_nudge = {
        str(layer): reduced_nonzero(model.model.layers[layer].mlp.ttt_conv.weight)
        for layer in TTT_LAYERS
    }
    second = backward_pass()
    torch.cuda.synchronize(device)

    rank_result = {
        "rank": rank,
        "local_rank": local_rank,
        "gpu_name": gpu_name,
        "free_memory_before_bytes": free_before,
        "total_memory_bytes": total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "input_ids_sha256": input_sha256,
        "input_ids_serialized": False,
        "first": first,
        "conv_nonzero_after_nudge": conv_nonzero_after_nudge,
        "second": second,
    }
    rank_results: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(rank_results, rank_result)

    configuration_path = Path(configuration_module.__file__).resolve()
    modeling_path = Path(modeling_module.__file__).resolve()
    foundation_path = Path(inspect.getsourcefile(build_foundation_model)).resolve()
    parallelize_path = Path(inspect.getsourcefile(build_parallelize_model)).resolve()
    offloading_path = Path(
        inspect.getsourcefile(build_activation_offloading_context)
    ).resolve()
    module_resolution = (
        upstream in configuration_path.parents
        and upstream in modeling_path.parents
    )
    offload_context_types = {
        "forward": type(forward_context).__name__,
        "backward": type(backward_context).__name__,
    }
    layer_set = [
        index for index, layer in enumerate(model.model.layers) if layer.is_ttt_layer
    ]
    first_identical = all(
        row["first"]["conv_gradient_norms"]
        == rank_results[0]["first"]["conv_gradient_norms"]
        and row["first"]["projection_gradient_norms"]
        == rank_results[0]["first"]["projection_gradient_norms"]
        for row in rank_results
    )
    second_identical = all(
        row["second"]["conv_gradient_norms"]
        == rank_results[0]["second"]["conv_gradient_norms"]
        and row["second"]["projection_gradient_norms"]
        == rank_results[0]["second"]["projection_gradient_norms"]
        for row in rank_results
    )
    checks = {
        "upstream_identity": upstream_commit == EXPECTED_UPSTREAM and upstream_clean,
        "tool_checkout_clean": tool_clean,
        "model_identity": (
            identity["repository"] == EXPECTED_REPOSITORY
            and identity["revision"] == EXPECTED_REVISION
        ),
        "module_resolution": module_resolution,
        "two_rank_two_rtx5090": (
            world_size == 2
            and all("RTX 5090" in row["gpu_name"] for row in rank_results)
            and all(
                row["free_memory_before_bytes"] >= 28 * (1 << 30)
                for row in rank_results
            )
        ),
        "model_shape": (
            model.config.num_hidden_layers == 36
            and model.config.hidden_size == 2560
            and model.config.vocab_size == 151936
            and len(model.model.layers) == 36
        ),
        "ttt_configuration_exact": (
            model.config.ttt_mode is True
            and model.config.ttt_layers == TTT_LAYERS
            and model.config.ttt_proj is True
            and model.config.ttt_target == "hidden_states"
            and model.config.ttt_lr == 3.0
            and model.config.ttt_chunk == TTT_CHUNK
            and layer_set == TTT_LAYERS
        ),
        "writer_parameter_set_exact": actual_writer_names == writer_names,
        "writer_parameters_are_dtensors": writer_dtensors,
        "gradient_checkpointing_active": bool(model.is_gradient_checkpointing),
        "liger_fused_forward_active": type(model).forward is lce_forward,
        "activation_offload_contexts_active": (
            forward_context is not None
            and backward_context is not None
            and offload_context_types["forward"] != "nullcontext"
            and offload_context_types["backward"] != "nullcontext"
        ),
        "first_losses_finite": first["losses_finite"],
        "first_writer_gradients_finite": first["writer_gradients_finite"],
        "first_all_conv_gradients_positive": all(
            value > 0 for value in first["conv_gradient_norms"].values()
        ),
        "first_all_projection_gradients_zero": all(
            value == 0 for value in first["projection_gradient_norms"].values()
        ),
        "diagnostic_nudge_all_conv_nonzero": all(
            value > 0 for value in conv_nonzero_after_nudge.values()
        ),
        "second_losses_finite": second["losses_finite"],
        "second_writer_gradients_finite": second["writer_gradients_finite"],
        "second_all_conv_gradients_positive": all(
            value > 0 for value in second["conv_gradient_norms"].values()
        ),
        "second_all_projection_gradients_positive": all(
            value > 0 for value in second["projection_gradient_norms"].values()
        ),
        "rank_reduced_results_agree": first_identical and second_identical,
        "no_optimizer_or_checkpoint": True,
    }

    if rank == 0:
        result = {
            "protocol": PROTOCOL,
            "passed": all(checks.values()),
            "checks": checks,
            "runtime": {
                "hostname": platform.node(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "world_size": world_size,
                "wall_seconds": time.perf_counter() - started,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "veomni_version": getattr(veomni, "__version__", None),
                "liger_kernel_version": getattr(liger_kernel, "__version__", None),
                "ranks": rank_results,
            },
            "source": {
                "upstream_root": str(upstream),
                "upstream_commit": upstream_commit,
                "upstream_clean": upstream_clean,
                "tool_root": str(tool_root),
                "tool_commit": tool_commit,
                "tool_clean": tool_clean,
                "plan_sha256": sha256(plan_path),
                "configuration_sha256": sha256(configuration_path),
                "modeling_sha256": sha256(modeling_path),
                "foundation_builder_sha256": sha256(foundation_path),
                "torch_parallelize_sha256": sha256(parallelize_path),
                "activation_offloading_sha256": sha256(offloading_path),
            },
            "model": {
                "root": str(model_root),
                "identity_path": str(identity_path),
                "identity_sha256": sha256(identity_path),
                "repository": identity["repository"],
                "revision": identity["revision"],
            },
            "fixture": {
                "sequence_tokens_per_rank": SEQUENCE_TOKENS,
                "ttt_chunk": TTT_CHUNK,
                "chunks": SEQUENCE_TOKENS // TTT_CHUNK,
                "ttt_layers": TTT_LAYERS,
                "ttt_lr": 3.0,
                "diagnostic_nudge_scale": DIAGNOSTIC_NUDGE,
                "activation_offload_context_types": offload_context_types,
                "serialized_input_ids": False,
                "serialized_logits": False,
                "serialized_activations": False,
                "serialized_gradients": False,
                "serialized_model_tensors": False,
            },
            "optimizer_constructed": False,
            "checkpoint_saved": False,
            "claim_boundary": (
                "exact Qwen3-4B-Base two-rank FSDP2 8K native-writer autograd "
                "diagnostic only; optimizer capacity, full-corpus training, "
                "conversion and RULER remain required"
            ),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(partial, output)
        print(json.dumps({"output": str(output), "passed": result["passed"]}, sort_keys=True))

    pass_tensor = torch.tensor([int(all(checks.values()))], device=device)
    dist.all_reduce(pass_tensor, op=dist.ReduceOp.MIN)
    dist.barrier()
    passed = bool(pass_tensor.item())
    dist.destroy_process_group()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
