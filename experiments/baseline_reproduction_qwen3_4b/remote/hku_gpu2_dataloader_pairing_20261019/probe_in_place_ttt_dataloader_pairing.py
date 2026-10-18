from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from veomni.data import build_dataloader, build_dataset
from veomni.data.data_transform import process_pretrain_example
from veomni.distributed.parallel_state import init_parallel_state
from veomni.models import build_tokenizer
from veomni.utils import helper


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().numpy().astype("<u4", copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    helper.set_seed(args.seed)
    init_parallel_state(
        dp_size=world_size,
        dp_shard_size=world_size,
        dp_mode="fsdp2",
        device_type="cpu",
    )

    tokenizer = build_tokenizer(str(args.tokenizer))
    transform = partial(
        process_pretrain_example,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        text_keys="content_split",
    )
    dataset = build_dataset(
        dataset_name="iterable",
        train_path=str(args.train_path),
        transform=transform,
        namespace="train",
        seed=args.seed,
    )
    per_rank_batch = args.global_batch_size // world_size
    loader = build_dataloader(
        dataloader_type="native",
        dataset=dataset,
        micro_batch_size=1,
        global_batch_size=args.global_batch_size,
        dataloader_batch_size=per_rank_batch,
        max_seq_len=args.max_seq_len,
        train_steps=1,
        rmpad=False,
        rmpad_with_pos_ids=False,
        num_workers=2,
        drop_last=True,
        pin_memory=False,
        prefetch_factor=2,
        seed=args.seed,
    )
    loader.set_epoch(0)
    micro_batches = next(iter(loader))
    record = {
        "rank": rank,
        "micro_batch_count": len(micro_batches),
        "input_shapes": [list(batch["input_ids"].shape) for batch in micro_batches],
        "ordered_token_hashes": [token_hash(batch["input_ids"]) for batch in micro_batches],
        "labels_equal_inputs": all(
            torch.equal(batch["labels"], batch["input_ids"]) for batch in micro_batches
        ),
        "attention_all_one": all(
            bool(torch.all(batch["attention_mask"] == 1)) for batch in micro_batches
        ),
        "terminal_is_eos": all(
            int(batch["input_ids"][0, -1]) == tokenizer.eos_token_id
            for batch in micro_batches
        ),
    }
    gathered: list[dict | None] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(record, gathered, dst=0)

    if rank == 0:
        data_transform = Path(
            "/home/ctj/miniconda3/envs/in_place_ttt_official_py311/lib/python3.11/"
            "site-packages/veomni/data/data_transform.py"
        )
        dataset_source = data_transform.with_name("dataset.py")
        loader_source = data_transform.with_name("data_loader.py")
        output = {
            "protocol": "in_place_ttt_native_dataloader_pairing_probe_v1",
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "seed": args.seed,
            "world_size": world_size,
            "max_seq_len": args.max_seq_len,
            "global_batch_size": args.global_batch_size,
            "num_workers": 2,
            "train_path": str(args.train_path),
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "datasets": importlib.metadata.version("datasets"),
                "transformers": importlib.metadata.version("transformers"),
                "veomni": importlib.metadata.version("veomni"),
            },
            "bindings": {
                "probe_sha256": sha256(Path(__file__)),
                "train_path_sha256": sha256(args.train_path),
                "data_transform_sha256": sha256(data_transform),
                "dataset_sha256": sha256(dataset_source),
                "data_loader_sha256": sha256(loader_source),
                "upstream_commit": subprocess.check_output(
                    [
                        "git",
                        "-C",
                        "/home/ctj/in_place_ttt_official_be232482_resource_clean",
                        "rev-parse",
                        "HEAD",
                    ],
                    text=True,
                ).strip(),
            },
            "ranks": gathered,
        }
        output["passed"] = all(
            row is not None
            and row["micro_batch_count"] == per_rank_batch
            and all(shape == [1, args.max_seq_len] for shape in row["input_shapes"])
            and row["labels_equal_inputs"]
            and row["attention_all_one"]
            and row["terminal_is_eos"]
            for row in gathered
        )
        partial_path = args.output.with_suffix(args.output.suffix + ".partial")
        partial_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        os.replace(partial_path, args.output)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
