# Corrected Qwen3-4B-Base Stage-2 timing plan

Status: **frozen before corrected 128K multi-step timing**

Freeze date: 2026-10-19 (Asia/Shanghai)

## Scope

Measure the released Stage-2 training loop at sequence length 131,072 and
global batch 64 before formal Stage-2 training. This is timing and resource
evidence only; all updates are discarded. All GPU execution is restricted to
`hku-gpu2` and its two RTX 5090 devices.

## Prerequisites

The runner must stop before CUDA initialization unless:

- corrected formal Qwen3-4B-Base Stage-1 completed for both arms and both its
  underlying and wrapper audits pass;
- the corrected 64K/128K Stage-2 resource ladder completed all four children
  and both its underlying and wrapper audits pass;
- the complete Base-tokenizer public corpus is atomically published and its
  canonical audit passes;
- the arm-specific Stage-1 HF starts, clean upstream/tool commits and exact
  Base identity are unchanged;
- no corpus process or CUDA process is active, both GPUs use no more than
  128 MiB, and host `MemAvailable` is at least 48 GiB.

## Registered timing

Run no-TTT before native TTT. Each arm starts from its own unchanged,
independently audited formal Stage-1 HF checkpoint and consumes the same
audited `stage2_128k.jsonl`. Use seed 42, world size 2, microbatch 1, global
batch 64, sequence length 131,072 and exactly three optimizer steps
(25,165,824 tokens per arm). Keep the frozen BF16 FSDP2, activation offload,
gradient checkpointing, fresh Stage-2 optimizer/scheduler, YaRN configuration,
data loader and native writer. Disable every checkpoint and HF save.

The first step is warm-up. The primary statistic is the median individual
optimizer-step duration of steps 2 and 3. Record all loss/gradient/time values,
raw GPU/host traces and command wall time. Project the registered five-step
formal Stage-2 duration for each arm and their sequential sum.

## Acceptance and release boundary

Both arms require exit zero, exactly three finite monotonic progress records,
positive warm-excluded timing, no checkpoint artifacts, at least 5% free GPU
memory, at least 8 GiB host memory and no more than 3.5 GiB swap use. The
existing independent timing audit and the corrected Base wrapper audit must
both pass.

This timing watcher never creates a formal release automatically. Formal
Stage-2 training remains sealed until a separate accepted release binds both
timing audits and records the measured projection and reviewer decision.
RULER, 256K extrapolation and update-admission experiments remain sealed.
