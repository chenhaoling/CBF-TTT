# Corrected Qwen3-4B-Base scale-feasibility conclusion plan

Status: frozen before corrected Stage-1 or Stage-2 timing outcomes exist.

Freeze date: 2026-10-19 (Asia/Shanghai)

## Purpose

The registered 100M-tier run is an end-to-end resource milestone, not the
paper's approximately 35B-token continual-pretraining run.  Use the audited
three-step 32K and 128K steady-state timings to quantify the gap before any
gate exploration is described as following a paper-scale reproduction.

## Calculation

For each arm, divide the exact target token count by its audited steady tokens
per second:

- Stage 1: 20,000,000,000 tokens per arm at 32K;
- Stage 2: 15,000,000,000 tokens per arm at 128K.

Both measured configurations use both RTX 5090 GPUs, so the two arms cannot be
run concurrently on the user-restricted host.  Sum the two arm projections in
each stage and then both stages.  The paper-scale comparison therefore covers
70B processed tokens across the matched pair, while each model sees 35B.

Independently verify that each measured 100M-tier per-arm projection equals
its registered selected-token count divided by steady throughput: 56,623,104
tokens in Stage 1 and 41,943,040 tokens in Stage 2.  Report seconds, hours and
days, plus the speedup required to finish the sequential pair within 30 and
90 days.

## Resource and claim rules

Carry forward the measured GPU peak/minimum-free fractions, host minimum
available memory and peak swap for each stage and arm.  Do not extrapolate
memory with token count because model, batch and sequence shapes remain fixed.
Do not call projected execution completed.  The output must say
`paper_scale_executed=false`, `projection_only=true`, and
`gate_experiment_started=false`.  It may document whether the projected time
fits 30 or 90 days, but it imposes no efficacy threshold and does not launch
training or gate work.
