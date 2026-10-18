# In-Place TTT 2xRTX5090 Stage-2 resource-ladder plan

Status: **frozen before any post-Stage-1 64K/128K optimizer step**

Freeze date: 2026-10-18 (Asia/Shanghai)

Parent execution addendum SHA256:
`0b70a098f8ee5cadf12f8d371dcc0020124069f5365aa5279d706cde4322e5c6`

Formal Stage-1 plan SHA256:
`f6ac1c4355b48d5344af00d85e7ad61c841b0589d9423fe188db67fbb38e01ea`

Exact resource-input addendum SHA256:
`5b65319ceeae3f25e7338928290ba9e59ecad22a1152a9209845b2ba0b68647c`

## 1. Scope

This plan measures whether the registered Stage-2 configuration fits and
produces finite updates on `hku-gpu2` after formal Stage 1. It does not train
formal Stage 2 and does not release RULER or update-admission experiments.

Each arm must start from its own independently audited formal Stage-1 HF
checkpoint. A no-TTT Stage-2 probe may not start from the TTT checkpoint, and
vice versa. Stage-2 optimizer/scheduler state is fresh, matching the baseline
curriculum. The model uses the registered 131,072-position YaRN configuration;
native `ttt_proj`/`ttt_conv` and all optimizer settings remain unchanged.

## 2. Registered sparse boundaries

Run no-TTT before TTT at each length:

1. 65,536 tokens, global batch 2, one optimizer step;
2. 131,072 tokens, global batch 2, one optimizer step.

Every input contains exactly one FineWeb and one LongCrawl record derived from
the independently audited continuous-corpus pilot. Inputs, manifest and input
audit are those frozen by the exact resource-input addendum. Each boundary
starts from the corresponding unchanged Stage-1 HF checkpoint rather than the
previous resource probe.

## 3. Prerequisites and stopping rules

Before CUDA initialization require:

- paired formal Stage-1 state `completed_100m_stage1` and a passing independent
  audit;
- current Stage-1 HF file manifests matching that audit;
- exact clean official/tool revisions and frozen plans/config audit;
- no corpus builder/auditor or CUDA compute process;
- both GPUs at no more than 128 MiB and host `MemAvailable` at least 48 GiB.

Stop the ladder at the first nonzero exit, OOM/nonfinite marker, missing DCP,
dependency mismatch, less than 5% free GPU memory, less than 8 GiB host memory
or more than 3.5 GiB swap. Preserve logs, monitors, manifests and failed
outputs. Do not advance from 64K to 128K unless both 64K arms pass.

## 4. Required evidence and next release

The independent ladder audit must rehash both Stage-1 HF manifests, exact
64K/128K inputs, configs, plans, runner, per-boundary logs and child audits.
It must show that all four children used the correct arm-specific model config
and produced a complete at-least-40-GB DCP with finite loss/gradient.

Only a passing four-boundary audit permits a separately frozen three-step
128K GB64 timing pilot. Formal five-step Stage 2 remains locked until that
timing result is finite and reviewed. No efficacy claim follows from this
resource ladder.
