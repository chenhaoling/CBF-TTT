# In-Place TTT 2xRTX5090 Stage-2 timing plan

Status: **frozen before the first 128K multi-step timing run**

Freeze date: 2026-10-18 (Asia/Shanghai)

Parent execution addendum SHA256:
`0b70a098f8ee5cadf12f8d371dcc0020124069f5365aa5279d706cde4322e5c6`

Stage-2 resource plan SHA256:
`d16636a1c053735c3f55731e073100f3605b4eccc39ca84dd8b793c4b548be9d`

## 1. Purpose and scope

This pilot measures the registered 128K Stage-2 training loop at global batch
64 after the arm-specific 64K/128K resource ladder passes. It is a timing and
resource gate only: all updates are discarded, no checkpoint is saved, and it
cannot support an efficacy claim.

All GPU execution is restricted to `hku-gpu2`. Each arm starts from its own
independently audited formal Stage-1 HF checkpoint. The Stage-2 YaRN config,
fresh optimizer/scheduler, Qwen3-4B backbone, native `ttt_proj`/`ttt_conv`,
activation offload, seed and data loader remain those of the registered 100M
tier.

## 2. Registered timing run

Run no-TTT before TTT. Each arm reads the same independently audited full
`stage2_128k.jsonl` corpus and executes exactly:

- sequence length: 131,072;
- global batch: 64; microbatch: 1;
- optimizer steps: 3;
- tokens per step: 8,388,608;
- total per arm: 25,165,824 tokens;
- checkpoint/HF saving disabled.

Step 1 is warm-up. The primary statistic is the median of reconstructed
individual step-2 and step-3 times. Multiply it by five to project the formal
Stage-2 wall time separately per arm and report their sequential sum. Record
all progress observations, command wall time, GPU/host traces, losses and
gradient norms.

## 3. Prerequisites and stopping rules

Before CUDA initialization require the paired formal Stage-1 audit, complete
arm-specific Stage-1 HF manifests, the four-boundary Stage-2 ladder audit,
full public-corpus audit, exact clean revisions, idle GPUs, no corpus process
and 48 GiB host `MemAvailable`.

Each arm must exit zero with exactly three finite, monotonic optimizer records,
a finite positive warm-excluded median, no checkpoint files, at least 5% free
GPU memory, at least 8 GiB minimum host memory and at most 3.5 GiB peak swap.
Stop before formal Stage 2 on any failure, nonfinite projection, OOM, hash
drift or arm/model mismatch.

## 4. Formal Stage-2 release boundary

This plan does not automatically launch formal Stage 2. After the paired
timing audit passes, the measured five-step projection must be reviewed and a
separate release record must bind its SHA256. Formal Stage 2 then uses each
arm's unchanged Stage-1 HF checkpoint with a fresh optimizer, runs exactly
five GB64 steps (41,943,040 tokens per arm), and saves only the final DCP/HF.

RULER and any update-admission experiment remain sealed until both complete
Stage-2 arms and their independent audit pass.
