# In-Place TTT 2xRTX5090 100M-tier formal Stage-2 plan

Status: **frozen before formal Stage-2 execution**

Freeze date: 2026-10-18 (Asia/Shanghai)

Parent execution addendum SHA256:
`0b70a098f8ee5cadf12f8d371dcc0020124069f5365aa5279d706cde4322e5c6`

Stage-2 timing plan SHA256:
`9898d283923a7a8ad45448249541901435087a45e8f89174b3a8283d894ec294`

## 1. Registered run

Run only on `hku-gpu2`, in the fixed order no-TTT then native TTT. Each arm
starts directly from its corresponding independently audited formal Stage-1
HF checkpoint and creates a fresh optimizer and scheduler. Both arms consume
the same independently audited full `stage2_128k.jsonl` artifact and execute:

- sequence length 131,072 with registered YaRN factor 4;
- global batch 64 and microbatch 1;
- exactly 5 optimizer steps;
- exactly 41,943,040 tokens per arm;
- final-only DCP and converted HF checkpoint at `global_step_5`.

No Stage-2 resource or timing child may be used as a starting checkpoint.
The original Stage-1 checkpoints remain unchanged and retained.

## 2. Release and stopping rules

Formal execution requires all of the following before CUDA initialization:

1. paired formal Stage-1 state and independent audit pass, with current
   complete HF manifests unchanged;
2. arm-specific 64K/128K Stage-2 resource ladder passes;
3. paired 3-step Stage-2 timing audit passes with a finite positive formal
   projection;
4. an explicit JSON release record uses protocol
   `in_place_ttt_2x5090_100m_stage2_release_v1`, sets `accepted=true`, and
   binds the exact timing-audit SHA256;
5. clean exact tool/upstream revisions, full public-corpus audit, at least
   160 GB free disk for the pair, two idle RTX 5090 GPUs, at least 48 GiB host
   `MemAvailable`, and no corpus builder/auditor process.

Stop at the first nonzero exit, OOM, nonfinite metric, dependency drift,
arm/start-model mismatch, GPU free fraction below 5%, host available memory
below 8 GiB, swap above 3.5 GiB, or invalid final checkpoint. A completed
no-TTT arm may remain in the partial directory if TTT fails, but the paired
root is published only after an independent audit passes both arms.

## 3. Audit and downstream boundary

The independent audit rehashes the corpus, plans, release, configs, runner,
auditor, formal Stage-1 audit, current complete Stage-1 HF manifests and all
run artifacts. Each arm must contain exactly five finite monotonic progress
records, only `global_step_5`, at least 40 GB of DCP files, at least 7 GB of HF
files, the correct `ttt_mode`, and 131,072 maximum positions.

Only the atomically published paired audit may release the frozen RULER matrix.
RULER must compare the two final Stage-2 HF checkpoints over all registered
tasks, contexts and examples. Update-admission work remains sealed until that
matched baseline evaluation is independently audited.
