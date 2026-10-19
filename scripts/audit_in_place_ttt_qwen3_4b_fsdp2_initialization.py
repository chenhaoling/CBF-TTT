#!/usr/bin/env python3
"""Audit exact Qwen3-4B-Base writer initialization through VeOmni FSDP2."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_qwen3_4b_fsdp2_initialization_audit_v1"
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
    for name, expected in identity.get("files", {}).items():
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
        raise RuntimeError("FSDP2 initialization audit is restricted to hku-gpu2")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("FSDP2 initialization audit requires physical GPUs 0 and 1")

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
    from torch.distributed.tensor import DTensor
    from veomni.distributed.parallel_state import init_parallel_state
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models import build_foundation_model

    import hf_models.hf_qwen3.configuration_qwen3 as configuration_module
    import hf_models.hf_qwen3.modeling_qwen3 as modeling_module
    import veomni
    import veomni.distributed.torch_parallelize as parallelize_module

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
        except Exception as exc:  # broadcast the same terminal error to every rank
            identity_error = f"{type(exc).__name__}: {exc}"
    payload: list[Any] = [identity_error, identity]
    dist.broadcast_object_list(payload, src=0)
    identity_error, identity = payload
    if identity_error is not None or identity is None:
        raise RuntimeError(f"model identity validation failed: {identity_error}")

    upstream_commit = git(upstream, "rev-parse", "HEAD")
    upstream_clean = git(upstream, "status", "--porcelain") == ""
    tool_commit = git(tool_root, "rev-parse", "HEAD")
    tool_clean = git(tool_root, "status", "--porcelain") == ""
    if upstream_commit != EXPECTED_UPSTREAM or not upstream_clean or not tool_clean:
        raise RuntimeError("source checkout identity or clean state differs")

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
        "ttt_chunk": 4096,
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
        enable_gradient_checkpointing=False,
        enable_fsdp_offload=False,
        basic_modules=model._no_split_modules,
    )
    torch.cuda.synchronize(device)

    writer_names = sorted(
        f"model.layers.{layer}.mlp.{kind}.weight"
        for layer in TTT_LAYERS
        for kind in ("ttt_conv", "ttt_proj")
    )
    actual_writer_names = sorted(
        name
        for name, _ in model.named_parameters()
        if ".ttt_conv." in name or ".ttt_proj." in name
    )
    rows = []
    structure_ok = True
    dtensor_ok = True
    dtype_device_ok = True
    for layer in TTT_LAYERS:
        mlp = model.model.layers[layer].mlp
        layer_row: dict[str, Any] = {"layer": layer}
        for kind in ("ttt_conv", "ttt_proj"):
            parameter = getattr(mlp, kind).weight.detach()
            is_dtensor = isinstance(parameter, DTensor)
            dtensor_ok = dtensor_ok and is_dtensor
            if not is_dtensor:
                raise RuntimeError(f"{kind} at layer {layer} is not a DTensor")
            local = parameter.to_local()
            full = parameter.full_tensor()
            local_ok = local.device.type == "cuda" and local.dtype == torch.float32
            dtype_device_ok = dtype_device_ok and local_ok
            if kind == "ttt_conv":
                global_nonzero = int(torch.count_nonzero(full))
                tensor_ok = bool(
                    tuple(full.shape) == (2560, 1, 5)
                    and getattr(mlp, kind).bias is None
                    and global_nonzero == 0
                    and torch.isfinite(full).all()
                )
                layer_row[kind] = {
                    "is_dtensor": is_dtensor,
                    "global_shape": list(full.shape),
                    "local_shape": list(local.shape),
                    "local_dtype": str(local.dtype),
                    "local_device_type": local.device.type,
                    "global_nonzero": global_nonzero,
                    "passed": tensor_ok and local_ok,
                }
            else:
                diagonal = full.diagonal()
                global_nonzero = int(torch.count_nonzero(full))
                diagonal_nonzero = int(torch.count_nonzero(diagonal))
                tensor_ok = bool(
                    tuple(full.shape) == (2560, 2560)
                    and getattr(mlp, kind).bias is None
                    and torch.isfinite(full).all()
                    and diagonal_nonzero == 2560
                    and global_nonzero == diagonal_nonzero
                )
                layer_row[kind] = {
                    "is_dtensor": is_dtensor,
                    "global_shape": list(full.shape),
                    "local_shape": list(local.shape),
                    "local_dtype": str(local.dtype),
                    "local_device_type": local.device.type,
                    "global_nonzero": global_nonzero,
                    "diagonal_nonzero": diagonal_nonzero,
                    "off_diagonal_nonzero": global_nonzero - diagonal_nonzero,
                    "passed": tensor_ok and local_ok,
                }
            structure_ok = structure_ok and tensor_ok
            del full
        rows.append(layer_row)

    local_result = {
        "rank": rank,
        "local_rank": local_rank,
        "gpu_name": gpu_name,
        "free_memory_before_bytes": free_before,
        "total_memory_bytes": total_memory,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "writer_rows": rows,
        "structure_ok": structure_ok,
        "dtensor_ok": dtensor_ok,
        "dtype_device_ok": dtype_device_ok,
    }
    rank_results: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(rank_results, local_result)
    checks = {
        "upstream_identity": upstream_commit == EXPECTED_UPSTREAM and upstream_clean,
        "tool_checkout_clean": tool_clean,
        "model_identity": (
            identity["repository"] == EXPECTED_REPOSITORY
            and identity["revision"] == EXPECTED_REVISION
        ),
        "two_rank_two_rtx5090": (
            world_size == 2
            and all("RTX 5090" in row["gpu_name"] for row in rank_results)
            and all(row["free_memory_before_bytes"] >= 28 * (1 << 30) for row in rank_results)
        ),
        "model_shape": (
            model.config.num_hidden_layers == 36
            and model.config.hidden_size == 2560
            and len(model.model.layers) == 36
        ),
        "ttt_configuration_exact": (
            model.config.ttt_mode is True
            and model.config.ttt_layers == TTT_LAYERS
            and model.config.ttt_proj is True
            and model.config.ttt_target == "hidden_states"
            and model.config.ttt_lr == 3.0
            and model.config.ttt_chunk == 4096
        ),
        "writer_parameter_set_exact": actual_writer_names == writer_names,
        "writer_parameters_are_dtensors": all(row["dtensor_ok"] for row in rank_results),
        "writer_local_storage_cuda_fp32": all(row["dtype_device_ok"] for row in rank_results),
        "writer_global_structure_exact": all(row["structure_ok"] for row in rank_results),
        "rank_reconstructions_agree": rank_results[0]["writer_rows"] == rank_results[1]["writer_rows"],
        "no_training_or_checkpoint": True,
    }

    if rank == 0:
        configuration_path = Path(configuration_module.__file__).resolve()
        modeling_path = Path(modeling_module.__file__).resolve()
        parallelize_path = Path(inspect.getsourcefile(build_parallelize_model)).resolve()
        foundation_path = Path(inspect.getsourcefile(build_foundation_model)).resolve()
        loader = parallelize_module.load_model_weights
        while hasattr(loader, "__wrapped__"):
            loader = loader.__wrapped__
        postprocess = loader.__globals__["post_process_after_weight_loading"]
        module_utils_path = Path(inspect.getsourcefile(postprocess)).resolve()
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
                "module_utils_sha256": sha256(module_utils_path),
            },
            "model": {
                "root": str(model_root),
                "identity_path": str(identity_path),
                "identity_sha256": sha256(identity_path),
                "repository": identity["repository"],
                "revision": identity["revision"],
            },
            "serialized_model_tensors": False,
            "forward_executed": False,
            "optimizer_constructed": False,
            "checkpoint_saved": False,
            "claim_boundary": (
                "exact Qwen3-4B-Base FSDP2/meta writer initialization only; "
                "forward, optimizer, corpus training, conversion and RULER remain required"
            ),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(partial, output)
        print(json.dumps({"output": str(output), "passed": result["passed"]}, sort_keys=True))

    dist.barrier()
    passed_tensor = torch.tensor([int(all(checks.values()))], device=device)
    dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN)
    dist.destroy_process_group()
    if not bool(passed_tensor.item()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
