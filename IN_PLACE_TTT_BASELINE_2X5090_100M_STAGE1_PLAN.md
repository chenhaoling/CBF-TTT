# In-Place TTT 2xRTX5090 100M tier: formal Stage-1 plan

Status: **frozen before formal Stage-1 training**

Freeze date: 2026-10-18 (Asia/Shanghai)

Parent execution addendum SHA256:
`0b70a098f8ee5cadf12f8d371dcc0020124069f5365aa5279d706cde4322e5c6`

Stage-1 timing plan SHA256:
`bf47735a04f0aa2f5b3b27e4627975be22652d1bade783a3eb6970982593d991`

## 1. Scope

This plan executes the formal 32K half of the registered 98,566,144-token
resource-constrained public-data tier. It does not authorize Stage 2, RULER or
any update-admission experiment. All GPU work is restricted to `hku-gpu2` and
its two RTX 5090 devices.

Run no-TTT before TTT. Both arms start from the same Qwen3-4B checkpoint and
consume the same independently audited full `stage1_32k.jsonl` artifact with
the registered seed and data-loader behavior. Each arm executes exactly:

- sequence length 32,768;
- global batch 64 and microbatch 1;
- 27 optimizer steps;
- 56,623,104 training tokens;
- AdamW, lr 5e-6, cosine schedule and weight decay 0.1;
- FSDP2 with gradient checkpointing and activation offload;
- native no-TTT or native `ttt_proj`/`ttt_conv`, differing only by `ttt_mode`.

Only the exact final `global_step_27` DCP and its converted HF checkpoint are
saved. No intermediate checkpoint may be substituted for the final model.

## 2. Release prerequisites

Formal Stage 1 remains locked until all of the following are true:

1. the complete public corpus and independent full audit pass;
2. the 8K/16K/32K paired resource ladder and independent audit pass;
3. the paired three-step 32K GB64 timing pilot and independent audit pass;
4. the measured per-arm and sequential 27-step projections are finite;
5. a release JSON created after reviewing the measured timing explicitly sets
   `accepted=true` and binds the exact timing-audit SHA256;
6. the official and tool checkouts are clean at their registered revisions;
7. no corpus process or CUDA process is active, each GPU uses at most 128 MiB,
   host `MemAvailable` is at least 48 GiB, and at least 160 GB disk is free
   before the pair begins.

The release record controls scheduling only. It cannot waive any resource,
integrity or independent-audit gate.

## 3. Runtime and stop rules

Before each arm, recheck the exact host, both idle GPUs, no corpus process,
48-GiB host availability and at least 80 GB free disk. Stop immediately on a
nonzero exit, OOM/nonfinite marker, fewer than 27 finite optimizer records,
less than 5% free GPU memory, less than 8 GiB host memory, more than 3.5 GiB
swap, missing or undersized DCP/HF output, dependency drift or audit failure.

The no-TTT output is retained if the later TTT arm fails, but neither arm is a
matched baseline result until the paired root independently passes. Existing
checkpoints are never overwritten or deleted by this runner.

## 4. Required evidence

Record plan/config/model/corpus/timing/release/tool/upstream hashes, exact GPU
and host preflights, one-second resource traces, the complete logs and exit
codes, all 27 progress records, final DCP/HF manifests and an independent
paired audit. Publish the paired output atomically only after both arms pass.

Stage 2 requires a separate frozen plan using each arm's audited Stage-1 HF
checkpoint, followed by the registered 64K/128K resource ladder and 128K
multi-step timing measurement. No RULER score or gate experiment is unsealed
by Stage 1 alone.
