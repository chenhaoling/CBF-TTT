# In-Place TTT Level-B matched-training execution addendum

Status: **frozen before any Level-B optimizer step**<br>
Freeze date: 2026-10-18 (Asia/Shanghai)<br>
Parent public-corpus plan SHA256: `c5019a28b9fa7fa27052ddf1d69eb029a2ec5b3b6446f1527e9ceb7db1a12bab`

## 1. Purpose and scope

This addendum closes execution details discovered while inspecting the exact
VeOmni environment selected for the public-data matched replication. It does
not change the model, native writer, optimizer, token budgets, stage lengths,
source mixture, RULER matrix or claim boundary in the parent plan. It does not
authorize any update-admission or forgetting controller.

No Level-B optimizer step had been run when this addendum was frozen.

## 2. Official iterable-data behavior

The pinned VeOmni `build_iterable_dataset` implementation loads the registered
JSONL as a Hugging Face streaming dataset, then unconditionally executes:

```python
dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
dataset = split_dataset_by_node(dataset, dp_rank, dp_size)
```

Disabling this shuffle would modify the released training path. Therefore the
phrase “same ordered JSONL” in the parent plan means that both arms bind the
same byte-identical ordered artifact and use the same released deterministic
loader. Actual optimizer consumption follows VeOmni's buffer-10,000 schedule,
then DP-rank splitting. It is not claimed to be the file's linear record order.

The following values are now explicit in all four formal configs rather than
being inherited as defaults:

- `train.seed=42`;
- `data.num_workers=2`;
- `data.prefetch_factor=2`;
- `data.pin_memory=true`.

Both arms in a stage must use the same config values, VeOmni package, world
size, DP layout and input file hash. Any shuffle-buffer, seed, worker,
prefetch, DP-size or dataset-code change requires a new addendum.

Pinned implementation evidence from the official training environment:

| File | SHA256 |
|---|---|
| `veomni/data/dataset.py` | `52a1a7361d3da5214b83ee2948272c700fbbe9394fca7b6077477ea700004f69` |
| `veomni/data/data_loader.py` | `8703db9609006855d1fc1bd61896052012d4bfab4413fd53c25639ae959f1c42` |

## 3. Frozen configuration bindings

| Stage/arm | Config SHA256 |
|---|---|
| Stage 1 no-TTT | `dda27e98de2abb1f6554a764dd3639f193e1fe203a5b4fbe50bfb16a737e490e` |
| Stage 1 TTT | `56d7a996c1eee100238e86b4b066702abefcbf6337cc7a1fd7a44c4edc80328b` |
| Stage 2 no-TTT | `fa898898706fa1b7f641f40cda50a5b78900e0e6b1d07f288eae22446a6db548` |
| Stage 2 TTT | `e9a75da30489f8b55b9273d3a7cb4a85788043acdbe0ddc468d70bdc68845fb4` |

The configs retain GB64, microbatch 1, FSDP2, AdamW `5e-6`, weight decay
`0.1`, 2% warmup, cosine decay, bf16 mixed precision, gradient checkpointing,
no CPU offload, save interval 500 and a final epoch save. Stage 2 receives the
corresponding arm's converted Stage-1 HF weights and starts a fresh optimizer
and scheduler.

The preflight runner SHA256 is
`4c8984d0945e5d6272ab955318484b4c0cc0c6d42abff50969024861e3f47668`;
its independent auditor SHA256 is
`257d36d9b9d09a9f94bb5ab54b3d134ea024ec90a3331de5f3d1c71167cbae22`.

## 4. Host and storage gate

The matched preflight requires exactly eight explicitly selected NVIDIA GPUs.
Every selected device must expose at least 80,000 MiB total memory, use no more
than 512 MiB before launch and have no compute process. Both one-step arms must
finish with finite loss and gradient norm and retain at least 5% free memory on
every device.

Measured official one-step DCP sizes are 49,830,704,053 bytes for no-TTT and
50,303,642,036 bytes for TTT. With saves every 500 steps plus the exact final
step, Stage 1 retains 20 DCPs per arm and Stage 2 retains 4 per arm. A shared
training filesystem must therefore have at least 3 TiB free before both formal
arms are scheduled. A per-arm Stage-1 filesystem must have at least 1.25 TiB
free. These limits include margin for the public corpus, HF conversions, logs
and metadata; they do not assume checkpoint pruning.

## 5. Execution and stop rules

1. The full public corpus and its independent audit must complete first. A
   `.partial` directory is never a training input.
2. Run the matched 32K one-step preflight. Failure of either arm, any OOM,
   non-finite metric, dependency mismatch or less than 5% device headroom
   stops before formal Stage 1.
3. Train both Stage-1 arms from byte-identical Qwen3-4B files and the same
   audited `stage1_32k.jsonl`. Retain all registered checkpoints.
4. Independently audit exact final step, finite logs, config/data/model hashes,
   checkpoint structure, resource trace and deterministic loader bindings.
5. Convert each final Stage-1 DCP through the official path. Run the matched
   128K preflight from the corresponding converted weights.
6. Only if both 128K preflights pass, train and audit both Stage-2 arms on the
   same audited `stage2_128k.jsonl` with fresh optimizer/scheduler state.
7. Convert final checkpoints and run the frozen 13-task × 100-example RULER
   matrix at 4K, 8K, 16K, 32K, 64K, 128K and 256K.

No config, world-size, context-parallel, offload, optimizer, checkpoint
retention, dataset shuffle or writer change may be introduced to make a failed
gate pass without a separately frozen addendum.

## 6. Claim boundary

Passing this protocol yields a **public-data matched replication** of the
released Qwen3-4B path. It does not prove reproduction of the unavailable
private corpus or missing internal recipe. Update-admission experiments remain
sealed until both arms and the complete RULER matrix finish and are audited.
