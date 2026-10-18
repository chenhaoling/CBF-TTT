# Corrected Qwen3-4B-Base Stage-2 resource plan

Status: **frozen before corrected 64K/128K resource execution**

Freeze date: 2026-10-19 (Asia/Shanghai)

## Scope

Measure one optimizer step for the matched no-TTT and native-TTT arms at the
two Stage-2 sequence lengths before any 128K timing or formal Stage-2 run.
All GPU execution is restricted to `hku-gpu2`. This plan produces resource
evidence only and does not release RULER or update-admission experiments.

Each arm starts from its own independently audited corrected formal Stage-1
HF checkpoint. The no-TTT checkpoint may never initialize the TTT arm or vice
versa. Stage-2 optimizer and scheduler state is fresh. Keep the registered
131,072-position YaRN configuration, activation offload, gradient
checkpointing, native writer and optimizer settings.

## Registered order and inputs

Use the independently audited Base full-corpus sparse inputs. Run exactly:

1. 65,536 no-TTT, then 65,536 native TTT;
2. 131,072 no-TTT, then 131,072 native TTT.

Every child uses global batch 2 for one optimizer step and starts again from
the corresponding unchanged Stage-1 HF checkpoint. Do not continue to 128K
unless both 64K children and their audits pass.

## Prerequisites and stopping rules

Before CUDA initialization require the corrected formal Stage-1 wrapper and
underlying paired audit to pass; current HF file manifests must match both
records. Also require the exact Base identity, full corpus and resource-input
audits, clean revisions, no corpus/CUDA process, no more than 128 MiB used per
GPU and at least 48 GiB host memory.

Stop at the first nonzero exit, OOM/nonfinite marker, incomplete update,
missing DCP, stale checkpoint/input/config binding, less than 5% free GPU
memory, less than 8 GiB host memory or more than 3.5 GiB swap. Preserve every
partial child and log; never treat one arm as paired evidence.

## Release boundary

The terminal wrapper audit must re-hash corrected Stage-1 evidence, both
arm-specific HF manifests, Base identity/tokenizer, full corpus inputs, four
child traces, plans, configs and existing runner/auditor. Only all four
passing boundaries may release a separately frozen three-step 128K timing
pilot. Formal Stage 2, RULER and gating remain sealed.
