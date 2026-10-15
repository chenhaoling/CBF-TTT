"""Train the preregistered Qwen3-4B LoRA structured fact comparator S1."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

from tasks.cbf_checkpoint_readout import load_plain
from tasks.pack_cbf_event_warmup import sha


PROTOCOL = "qwen_structured_fact_comparator_s1_train_v1"
DATA_PROTOCOL = "qwen_structured_fact_comparator_s1"
FIELDS = ("same_entity", "same_relation", "same_value")
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
CHECKPOINT_EPOCHS = (0.25, 0.5, 1.0, 2.0)


def prompt(row: dict) -> str:
    return (
        "Compare the two marked facts. Decide whether they have the same subject entity, "
        "the same semantic relation, and the same value. The operation is duplicate only "
        "when all three are the same, update only when subject and relation are the same "
        "but value differs, and retain otherwise. Return one JSON object with keys "
        "same_entity, same_relation, same_value, operation in that order and no other text.\n"
        f"Fact A: {row['left_text']}\nFact B: {row['right_text']}\nAnswer:"
    )


def canonical_answer(row: dict) -> str:
    values = ",".join(f'"{field}":{str(bool(row[field])).lower()}' for field in FIELDS)
    return "{" + values + f',"operation":"{row["operation"]}"' + "}"


def load_data(root: str | Path) -> tuple[list[dict], dict]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    if (manifest["protocol"] != DATA_PROTOCOL or audit.get("passed") is not True
            or audit.get("test_rows_parsed") != 0
            or manifest.get("semeval_test_parsed") is not False
            or manifest.get("semeval_test_examples_written") is not False):
        raise ValueError("S1 training requires audited sealed data")
    path = root / "train.jsonl"
    if sha(path) != manifest["files_sha256"][path.name]:
        raise ValueError("S1 training file hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != 48000 or any(row["split"] != "train" for row in rows):
        raise ValueError("unexpected S1 training rows")
    return rows, {"manifest_sha256": sha(root / "manifest.json"),
                  "audit_sha256": sha(root / "audit.json"),
                  "train_sha256": sha(path)}


class TokenizedDataset:
    def __init__(self, rows, tokenizer, max_length):
        self.items, self.lengths = [], []
        for row in rows:
            prefix = tokenizer(prompt(row), add_special_tokens=True)["input_ids"]
            target = tokenizer(canonical_answer(row) + tokenizer.eos_token,
                               add_special_tokens=False)["input_ids"]
            if len(prefix) + len(target) > max_length:
                raise ValueError(f"S1 example exceeds max length: {row['pair_id']}")
            ids = prefix + target
            self.items.append({"input_ids": ids, "labels": [-100] * len(prefix) + target,
                               "pair_id": row["pair_id"]})
            self.lengths.append(len(ids))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(items, pad_token_id):
    import torch
    width = max(len(item["input_ids"]) for item in items)
    input_ids, labels, masks = [], [], []
    for item in items:
        padding = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        labels.append(item["labels"] + [-100] * padding)
        masks.append([1] * len(item["input_ids"]) + [0] * padding)
    return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)}


def cosine_schedule(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def checkpoint_steps(steps_per_epoch: int) -> dict[int, float]:
    return {max(1, round(epoch * steps_per_epoch)): epoch for epoch in CHECKPOINT_EPOCHS}


def train(args):
    import accelerate
    import peft
    import torch
    import torch.distributed as dist
    import transformers
    from peft import LoraConfig, get_peft_model
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler
    from transformers import AutoTokenizer

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != 2:
        raise ValueError("formal S1 training requires exactly two processes/GPUs")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()

    output = Path(args.output)
    if rank == 0:
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
    dist.barrier()
    rows, data_meta = load_data(args.data_root)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = TokenizedDataset(rows, tokenizer, args.max_length)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                 seed=args.seed, drop_last=False)
    loader = DataLoader(dataset, batch_size=args.per_device_batch, sampler=sampler,
                        num_workers=0, pin_memory=True,
                        collate_fn=lambda items: collate(items, tokenizer.pad_token_id))
    if len(loader) % args.grad_accum:
        raise ValueError("microbatch count must divide gradient accumulation")
    steps_per_epoch = len(loader) // args.grad_accum
    formal_steps = steps_per_epoch * 2
    total_steps = args.smoke_steps or formal_steps
    save_steps = {} if args.smoke_steps else checkpoint_steps(steps_per_epoch)
    if not args.smoke_steps and max(save_steps) != formal_steps:
        raise ValueError("registered checkpoint schedule mismatch")

    model, base_meta = load_plain(args.model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=list(TARGET_MODULES))
    model = get_peft_model(model, lora)
    trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("unexpected S1 trainable parameter set")
    model.train()
    wrapped = DistributedDataParallel(model, device_ids=[local_rank],
                                      output_device=local_rank, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(trainable.values(), lr=2e-4, weight_decay=0.01)
    warmup = max(1, round(0.03 * formal_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda value: cosine_schedule(value, formal_steps, warmup))
    optimizer.zero_grad(set_to_none=True)
    started, global_step, micro_step = time.perf_counter(), 0, 0
    loss_sum, loss_count = 0.0, 0
    log_path = output / f"train_rank{rank}.jsonl"
    with log_path.open("x") as log:
        for epoch in range(2):
            sampler.set_epoch(epoch)
            for batch in loader:
                micro_step += 1
                final_micro = micro_step % args.grad_accum == 0
                sync = contextlib.nullcontext() if final_micro else wrapped.no_sync()
                with sync:
                    batch = {key: value.cuda(local_rank, non_blocking=True)
                             for key, value in batch.items()}
                    result = wrapped(**batch, use_cache=False)
                    loss = result.loss / args.grad_accum
                    loss.backward()
                loss_sum += float(loss.detach()) * args.grad_accum
                loss_count += 1
                if not final_micro:
                    continue
                torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 10 == 0 or global_step == 1:
                    mean_loss = loss_sum / max(1, loss_count)
                    row = {"rank": rank, "step": global_step,
                           "epoch": global_step / steps_per_epoch,
                           "loss": mean_loss, "lr": scheduler.get_last_lr()[0],
                           "seconds": time.perf_counter() - started}
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    loss_sum = 0.0
                    loss_count = 0
                    if rank == 0:
                        print(json.dumps(row), flush=True)
                if global_step in save_steps:
                    dist.barrier()
                    if rank == 0:
                        checkpoint = output / f"checkpoint_epoch_{save_steps[global_step]:.2f}"
                        model.save_pretrained(checkpoint, safe_serialization=True)
                        (checkpoint / "checkpoint_meta.json").write_text(json.dumps({
                            "protocol": PROTOCOL, "global_step": global_step,
                            "epoch": save_steps[global_step], "seed": args.seed,
                            "data": data_meta, "code_commit": args.code_commit,
                        }, indent=2, sort_keys=True) + "\n")
                    dist.barrier()
                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break
    dist.barrier()
    elapsed = time.perf_counter() - started
    profile = {"rank": rank, "seconds": elapsed,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    profiles = [None] * world
    dist.all_gather_object(profiles, profile)
    if rank == 0:
        manifest = {
            "protocol": PROTOCOL,
            "terminal_status": "completed_smoke" if args.smoke_steps
            else "completed_training_before_calibration",
            "formal": not bool(args.smoke_steps), "seed": args.seed,
            "world_size": world, "per_device_batch": args.per_device_batch,
            "gradient_accumulation": args.grad_accum,
            "effective_batch": args.per_device_batch * world * args.grad_accum,
            "steps_per_epoch": steps_per_epoch, "completed_steps": global_step,
            "checkpoint_steps": save_steps, "max_length": args.max_length,
            "length_min": min(dataset.lengths), "length_max": max(dataset.lengths),
            "length_mean": sum(dataset.lengths) / len(dataset.lengths),
            "lora": {"rank": 16, "alpha": 32, "dropout": 0.05,
                     "target_modules": TARGET_MODULES,
                     "trainable_parameters": sum(value.numel() for value in trainable.values())},
            "optimizer": {"name": "AdamW", "lr": 2e-4, "weight_decay": 0.01,
                          "warmup_steps": warmup, "schedule": "cosine", "grad_clip": 1.0},
            "base_model": base_meta, "data": data_meta, "profiles": profiles,
            "dependencies": {"torch": torch.__version__, "transformers": transformers.__version__,
                             "peft": peft.__version__, "accelerate": accelerate.__version__},
            "code_commit": args.code_commit, "calibration_loaded": False,
            "development_loaded": False, "semeval_test_loaded": False,
        }
        (output / "training_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": manifest["terminal_status"], "steps": global_step,
                          "seconds": elapsed, "profiles": profiles}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=24117)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--per-device-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--smoke-steps", type=int, default=0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
