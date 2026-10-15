"""Train the preregistered S2 binary natural-relation LoRA reranker."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time
from pathlib import Path

from tasks.cbf_checkpoint_readout import load_plain
from tasks.pack_cbf_event_warmup import sha
from tasks.train_structured_fact_comparator_s1 import collate, cosine_schedule


PROTOCOL = "qwen_natural_relation_reranker_s2_train_v1"
DATA_PROTOCOL = "qwen_natural_relation_reranker_s2"
DATA_AUDIT_PROTOCOL = "qwen_natural_relation_reranker_s2_data_audit_v1"
SEED = 25117
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
CHECKPOINT_EPOCHS = (0.25, 0.5, 1.0)


def prompt(row: dict) -> str:
    return (
        "Decide whether the two natural statements express the same semantic relation. "
        "[SUBJECT] and [VALUE] replace the entity strings, so compare the relation meaning "
        "of the surrounding language. Return exactly one JSON object with the single key "
        "same_relation and no other text.\n"
        f"Statement A: {row['left_text']}\nStatement B: {row['right_text']}\nAnswer:"
    )


def canonical_answer(value: dict | bool) -> str:
    relation = bool(value["same_relation"]) if isinstance(value, dict) else bool(value)
    return '{"same_relation":' + str(relation).lower() + "}"


def checkpoint_steps(steps_per_epoch: int) -> dict[int, float]:
    return {max(1, round(epoch * steps_per_epoch)): epoch for epoch in CHECKPOINT_EPOCHS}


def load_data(root: str | Path) -> tuple[list[dict], dict]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    audit = json.loads((root / "data_audit.json").read_text())
    if (manifest.get("protocol") != DATA_PROTOCOL
            or audit.get("protocol") != DATA_AUDIT_PROTOCOL or audit.get("passed") is not True
            or audit.get("semeval_test_rows_parsed") != 0
            or audit.get("development_model_outputs_loaded") is not False
            or manifest.get("semeval_test_parsed") is not False
            or manifest.get("model_outputs_loaded") is not False):
        raise ValueError("S2 training requires audited shortcut-free sealed data")
    path = root / "train_relation_pairs.jsonl"
    if sha(path) != manifest["files_sha256"][path.name]:
        raise ValueError("S2 training file hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    expected = manifest["counts"]["train"]["relation_pairs"]
    if len(rows) != expected or any(row["split"] != "train" for row in rows):
        raise ValueError("unexpected S2 training rows")
    if sum(bool(row["same_relation"]) for row in rows) * 2 != len(rows):
        raise ValueError("S2 relation training must be class-balanced")
    return rows, {"manifest_sha256": sha(root / "manifest.json"),
                  "audit_sha256": sha(root / "data_audit.json"),
                  "train_sha256": sha(path)}


class TokenizedDataset:
    def __init__(self, rows, tokenizer, max_length):
        self.items, self.lengths = [], []
        for row in rows:
            prefix = tokenizer(prompt(row), add_special_tokens=True)["input_ids"]
            target = tokenizer(canonical_answer(row) + tokenizer.eos_token,
                               add_special_tokens=False)["input_ids"]
            if len(prefix) + len(target) > max_length:
                raise ValueError(f"S2 example exceeds max length: {row['pair_id']}")
            ids = prefix + target
            self.items.append({"input_ids": ids, "labels": [-100] * len(prefix) + target,
                               "pair_id": row["pair_id"]})
            self.lengths.append(len(ids))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


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
        raise ValueError("formal S2 training requires exactly two processes/GPUs")
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
    formal_steps = steps_per_epoch
    total_steps = args.smoke_steps or formal_steps
    save_steps = {} if args.smoke_steps else checkpoint_steps(steps_per_epoch)
    if not args.smoke_steps and max(save_steps) != formal_steps:
        raise ValueError("registered S2 checkpoint schedule mismatch")

    model, base_meta = load_plain(args.model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=list(TARGET_MODULES))
    model = get_peft_model(model, lora)
    trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("unexpected S2 trainable parameter set")
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
        sampler.set_epoch(0)
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
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % 10 == 0 or global_step == 1:
                item = {"rank": rank, "step": global_step,
                        "epoch": global_step / steps_per_epoch,
                        "loss": loss_sum / max(1, loss_count),
                        "lr": scheduler.get_last_lr()[0],
                        "seconds": time.perf_counter() - started}
                log.write(json.dumps(item) + "\n"); log.flush()
                loss_sum, loss_count = 0.0, 0
                if rank == 0:
                    print(json.dumps(item), flush=True)
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
            "development_model_outputs_loaded": False, "semeval_test_loaded": False,
        }
        (output / "training_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": manifest["terminal_status"], "steps": global_step,
                          "seconds": elapsed, "profiles": profiles}), flush=True)
    dist.barrier(); dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--per-device-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--smoke-steps", type=int, default=0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
