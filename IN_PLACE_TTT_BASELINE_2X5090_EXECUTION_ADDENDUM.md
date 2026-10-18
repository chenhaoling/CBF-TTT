# In-Place TTT baseline 2xRTX5090 execution addendum

Status: **frozen before any activation-offload optimizer step**

Freeze date: 2026-10-18 (Asia/Shanghai)

Parent public-corpus plan SHA256: `c5019a28b9fa7fa27052ddf1d69eb029a2ec5b3b6446f1527e9ceb7db1a12bab`

Parent Level-B training addendum SHA256: `c33310cae2c196b06a15eca2462d009d9ca34674adc11f55a167048dd6d8d03b`

## 1. Scope and explicit hardware override

The user requires every new GPU run to use only `hku-gpu2`, which has two
32,607-MiB RTX 5090 devices and about 62 GiB host RAM. This addendum replaces
the Level-B 8x80-GB execution gate with a resource-gated two-GPU path. It does
not change the Qwen3-4B backbone, native `ttt_conv`/`ttt_proj` writer, public
FineWeb-Edu/LongCrawl64 corpus, source ratio, 32K-to-128K curriculum, optimizer,
global batch, tokenizer, seed or RULER tasks.

The result is a **2x5090 resource-constrained public-data replication**. It is
not the registered 35B Level-B result unless the complete 35B schedule is
actually run. It cannot be called an exact reproduction of the paper's
private corpus or hardware. Update-admission and forgetting gates remain
sealed until a matched no-TTT/native-TTT baseline tier finishes and is audited.

## 2. Evidence motivating the new path

The clean, exclusive native FSDP2 8K probes completed one optimizer step but
peaked at 31,620/31,594 MiB for no-TTT and 31,920/31,894 MiB for TTT. Their
steady optimizer-step times were 3.89 and 4.24 seconds for 16,384 tokens,
respectively. Even assuming no long-context slowdown, 35B tokens would take
about 96.2 days for no-TTT and 104.8 days for TTT. The actual 32K/128K schedule
will be slower, so a full two-arm launch is forbidden before a measured
long-context throughput projection.

FSDP2 parameter offload remains unavailable in the pinned VeOmni version, and
the measured FSDP1 parameter-offload probe exhausted host memory. The pinned
training entry point does, however, implement activation offload with
`torch.autograd.graph.saved_tensors_hooks`. This is a real, released-code path:
with gradient checkpointing enabled it offloads forward saved tensors and
uses `activation_gpu_limit` for backward saved tensors.

## 3. Frozen systems change

The two-GPU path retains FSDP2 full sharding and sets:

```yaml
enable_gradient_checkpointing: true
enable_activation_offload: true
activation_gpu_limit: 0.0
enable_fsdp_offload: false
```

It also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Activations
may move to the CPU; parameters, gradients and optimizer state are not CPU
offloaded. `ulysses_parallel_size` remains 1 because using both devices for
context parallelism would remove the two-rank FSDP data-parallel shard and
replicate the 4B optimizer state on each 32-GB device.

No training probe may run while the full corpus builder or corpus auditor is
active on the same 62-GiB host. At probe launch both GPUs must have no compute
process and use at most 128 MiB; host `MemAvailable` must be at least 48 GiB.

## 4. Sparse resource ladder

Each boundary forks from the same original model/config within an arm, uses
global batch 2 and microbatch 1, and executes exactly one optimizer step.
Run no-TTT before TTT at each boundary:

1. 8K activation-offload validation;
2. 16K boundary;
3. registered 32K Stage-1 boundary;
4. after Stage 1 conversion, 64K and registered 128K Stage-2 boundaries from
   the corresponding arm checkpoint.

Stop the ladder at the first OOM, nonzero exit, nonfinite loss/gradient,
device free fraction below 5%, host available memory below 8 GiB, swap use
above 3.5 GiB, dependency mismatch or missing checkpoint. Record elapsed
time, optimizer-step time, peak memory per GPU, minimum host available memory,
peak swap and checkpoint bytes. Resource checkpoints may be removed only
after their file manifest, size, logs, summary and independent audit have been
saved; training checkpoints are retained.

## 5. First end-to-end tier

Only if both 32K arms pass the resource gate, run the opt-in 100M tier:

| Stage | Length | GB64 steps | Exact tokens |
|---|---:|---:|---:|
| Stage 1 | 32,768 | 27 | 56,623,104 |
| Stage 2 | 131,072 | 5 | 41,943,040 |
| total | — | 32 | 98,566,144 |

This is the nearest small whole-step tier preserving the paper's approximate
20:15 stage ratio. Both arms bind the same complete audited corpus artifacts,
released deterministic buffer-10,000 shuffle, seed 42, world size 2 and GB64.
Stage 2 starts from that arm's officially converted Stage-1 HF weights with a
fresh AdamW optimizer/scheduler, matching the parent public-data assumption.

To fit local storage, each stage saves only its exact final DCP and converted
HF checkpoint (`save_steps=max_steps`, `save_epochs=0`). This changes retention
only; it does not change optimizer steps or model values. No checkpoint is
overwritten, and all four final DCPs remain until the tier audit and evaluation
finish.

The four registered configs are:

- `qwen3_4b_2x5090_100m_stage1_no_ttt.yaml`;
- `qwen3_4b_2x5090_100m_stage1_ttt.yaml`;
- `qwen3_4b_2x5090_100m_stage2_no_ttt.yaml`;
- `qwen3_4b_2x5090_100m_stage2_ttt.yaml`.

Before the 100M launch, a multi-step no-checkpoint timing pilot must project
the two-arm wall time. If the projection is nonfinite, either registered
length cannot execute, or the host gates fail, stop without training. After
100M, expansion to the approximately 1B tier (272 Stage-1 steps and 51
Stage-2 steps, 998,244,352 exact tokens) requires a new frozen addendum and is
allowed only if 100M completes and its matched RULER evaluation is audited.

## 6. Evaluation and claim boundary

After both 100M arms finish, run the same 13 RULER tasks with 100 examples at
4K, 8K, 16K, 32K, 64K and 128K. Attempt 256K only through a separately audited
two-GPU inference configuration; if it cannot fit, report the registered
resource failure and do not substitute fewer examples. Report every task,
context and `TTT - no-TTT` difference. No positive baseline or gate claim is
allowed from a single arm, a shorter context subset or smoke samples.
