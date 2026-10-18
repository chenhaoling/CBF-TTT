# In-Place TTT 2xRTX5090 Stage-1 timing plan

Status: **frozen before the first 32K multi-step timing run**

Freeze date: 2026-10-18 (Asia/Shanghai)

Parent execution addendum SHA256:
`0b70a098f8ee5cadf12f8d371dcc0020124069f5365aa5279d706cde4322e5c6`

Parent baseline plan SHA256:
`fbce7e0e52c495467c5b6383242a7350cf16042d61af8795309b9d6eda021fd3`

## 1. Purpose and scope

This pilot measures the released training loop at the registered Stage-1
length and global batch before either 98,566,144-token arm is launched. It is
a timing and resource gate only. Its weights, losses and updates are discarded
and cannot be used as efficacy evidence or as a formal-training checkpoint.

All GPU execution is restricted to `hku-gpu2` (`user-Super-Server`) and its two
RTX 5090 devices. The Qwen3-4B backbone, native `ttt_proj`/`ttt_conv`, FSDP2,
activation offload, optimizer, seed, shuffle and data loader remain those of
the registered 100M Stage-1 configurations.

## 2. Prerequisites

The runner must stop before CUDA initialization unless all of the following
are true:

1. the complete public corpus has been atomically published and its independent
   audit passes the registered full counts and hashes;
2. the 8K/16K/32K no-TTT and TTT Stage-1 resource ladder is complete and its
   independent audit passes;
3. the official checkout is clean at commit
   `be2324829b0e91c8fd10a74d4b43714fde6676e1`, and the tool checkout is clean
   at the explicitly supplied revision;
4. no public-corpus builder/auditor or CUDA compute process is active;
5. both GPUs use at most 128 MiB, and host `MemAvailable` is at least 48 GiB.

## 3. Registered timing run

Run no-TTT before TTT. Each arm starts from the same Qwen3-4B model and reads
the same audited `stage1_32k.jsonl` corpus with seed 42, buffer-10,000 shuffle,
world size 2, microbatch 1 and global batch 64. Execute exactly three optimizer
steps:

- sequence length: 32,768;
- tokens per step: 2,097,152;
- total per arm: 6,291,456 tokens;
- activation offload: enabled, GPU limit 0.0;
- parameter offload: disabled;
- checkpoint/HF saving: disabled (`save_steps=0`, `save_epochs=0`,
  `save_hf_weights=false`).

The first optimizer step is warm-up. The primary timing statistic is the
median of individual optimizer-step wall times for steps 2 and 3, derived from
the progress trace's cumulative elapsed times. Record all three cumulative and
individual times, command wall time, GPU/host traces, losses and gradient
norms. Project the 27-step formal Stage-1 time separately for each arm and
their sequential sum. Do not extrapolate the unmeasured 128K Stage-2 time from
this pilot.

## 4. Acceptance and stopping rules

Each arm must have exit code zero, exactly three finite loss/gradient records,
three monotonic progress observations, a finite positive warm-excluded median,
no OOM/nonfinite marker, no checkpoint files, at least 5% free memory on both
GPUs, at least 8 GiB minimum host memory and no more than 3.5 GiB peak swap.
Both arms must pass independent audit before formal Stage 1 is released.

Stop without formal Stage-1 training if either arm fails, if the projected
time is absent/nonfinite, or if any dependency/hash/prerequisite differs.
There is no performance threshold selected after observing the timing result;
the measured finite projection is reported and the user decides whether its
wall time is acceptable.

## 5. Stage-2 boundary

This plan does not authorize 128K timing or training. After both formal
Stage-1 arms complete and their HF conversions pass audit, a separate frozen
Stage-2 plan must bind the arm-specific checkpoints, run the registered 64K
and 128K one-step boundaries, and measure multi-step 128K timing before the
five-step Stage-2 runs.
