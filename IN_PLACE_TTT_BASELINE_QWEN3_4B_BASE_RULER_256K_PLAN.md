# Corrected Qwen3-4B-Base RULER-256K extrapolation plan

Status: frozen before any corrected-Base 256K resource run or score.

Freeze date: 2026-10-19 (Asia/Shanghai)

## Scope and ordering

The paper reports 256K as an extrapolation beyond the 131,072-position
continual-training curriculum.  Register it separately from the primary
4K--128K mean.  The one-sample resource boundary may start only after the
paired primary matrix and both its underlying and corrected Base audits pass.

Use only the two final arm-specific HF checkpoints bound by corrected formal
Stage 2, the exact Qwen3-4B-Base tokenizer revision
`906bfd4b4dc7f14ee4320094d8b41684abff8539`, and the already frozen 13-task,
seed-42, deterministic OpenCompass protocol.  Every paper-facing native TTT
update must use the independently validated Frobenius `1e-5` clipping overlay.

All GPU work is restricted to `hku-gpu2`: physical GPU 0 runs no-TTT and
physical GPU 1 runs native TTT.  No other GPU host is an authorized fallback.

## Resource boundary

At 262,144 tokens, evaluate exactly one example from every registered task in
each arm.  Discard all scores.  Require byte-identical paired prompts and
references, all 13 task artifacts, zero exit status, passing child and paired
audits, exact model/tokenizer/tool bindings and at least 5% free memory on each
GPU at peak.  Stop on OOM, dependency drift, tokenizer warning, input mismatch,
missing artifact, nonzero exit or failed audit.  Do not shorten inputs, remove
tasks, quantize, move to another host or alter the native writer.

## Formal boundary

A passing corrected resource wrapper establishes feasibility only.  Formal
256K scoring remains sealed until a separate non-authorizing review packet is
inspected and a human explicitly supplies acceptance, reviewer identity and a
decision note.  The formal run must retain all 13 tasks and 100 examples per
task for both arms, and report its delta against the paper's 41.7/43.9 values.

The 256K result remains a public-data YaRN extrapolation.  It is never included
in the primary six-context mean and cannot change a public-data result into an
exact reproduction of the private paper corpus.  Update-admission exploration
remains governed by the primary corrected baseline gate.
