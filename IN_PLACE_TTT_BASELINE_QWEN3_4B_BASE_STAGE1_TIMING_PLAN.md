# Corrected Qwen3-4B-Base Stage-1 timing plan

Status: **frozen before corrected 32K multi-step timing**
Freeze date: 2026-10-19 (Asia/Shanghai)

## Scope

Measure the released training loop at the registered 32K Stage-1 length and
global batch before formal 100M training. The run is timing/resource evidence;
its updates are discarded. All GPU execution is restricted to `hku-gpu2`.

## Prerequisites

The runner must stop before CUDA initialization unless:

- corrected Qwen3-4B-Base Level A passed its terminal audit;
- the corrected 16K/32K Stage-1 resource ladder completed all four children
  and its independent audit passed;
- the complete Base-tokenizer public corpus is atomically published and its
  canonical audit passes;
- the clean upstream/tool commits and exact Base identity are unchanged;
- no corpus process or CUDA process is active, both GPUs use no more than
  128 MiB, and host `MemAvailable` is at least 48 GiB.

## Registered timing

Run no-TTT before native TTT. Both arms start from the same exact
`Qwen/Qwen3-4B-Base` checkpoint and consume the same audited Stage-1 stream.
Use seed 42, world size 2, microbatch 1, global batch 64, sequence length
32,768 and exactly three optimizer steps (6,291,456 tokens per arm). Keep the
frozen BF16 FSDP2, activation offload, gradient checkpointing, AdamW `5e-6`,
weight decay `0.1`, data loader and writer settings. Disable every checkpoint
and HF save.

The first step is warm-up. The primary statistic is the median individual
optimizer-step duration of steps 2 and 3. Record all loss/gradient/time values,
raw GPU/host traces and command wall time. Project the registered 27-step
Stage-1 duration for each arm and their sequential sum.

## Acceptance and release boundary

Both arms require exit zero, exactly three finite monotonic progress records,
positive warm-excluded timing, no checkpoint artifacts, at least 5% free GPU
memory, at least 8 GiB host memory and no more than 3.5 GiB swap use. The
existing independent timing audit and the corrected Base wrapper audit must
both pass.

The timing result never creates a formal release automatically. Formal
Stage-1 training remains sealed until a separate accepted release records the
measured projection and reviewer decision. Stage 2, RULER and update-admission
experiments remain sealed.
