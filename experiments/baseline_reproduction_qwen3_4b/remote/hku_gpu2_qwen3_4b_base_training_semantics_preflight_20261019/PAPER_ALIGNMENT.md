# Qwen3-4B-Base training-semantics preflight

Captured on 2026-10-19 before any corrected-Base training child started.
All remote checks ran on `hku-gpu2`; CUDA was hidden for the executable
configuration audit. This preflight does not authorize a training release.

## Result

The registered 100M-tier no-TTT/native-TTT pair is internally consistent and
implements the paper-disclosed training structure at reduced scale:

- exact `Qwen/Qwen3-4B-Base` revision `906bfd4b...`;
- 36 transformer layers with native TTT at layers
  `[0, 6, 12, 18, 24, 30]`, exactly every sixth layer;
- Stage 1 at 32,768 tokens and Stage 2 at 131,072 tokens with YaRN factor 4;
- AdamW, learning rate `5e-6`, and weight decay `0.1`;
- native upstream depthwise `ttt_conv` with kernel size 5;
- matched no-TTT and native-TTT arms using the same selected stream per stage;
- Stage 2 starts each arm from its corresponding audited Stage-1 checkpoint.

The executable configuration auditor passed with no errors. It bound all four
configuration hashes, `world_size=2`, 56,623,104 Stage-1 tokens per arm,
41,943,040 Stage-2 tokens per arm, and 98,566,144 tokens per completed arm.
All seven clean tool snapshots used by the registered Level-A, Stage-1 and
Stage-2 chains contain byte-identical copies of those four configurations and
the same auditor.

## Paper-disclosed requirements

The April 2026 paper specifies the following relevant settings:

| Requirement | Registered implementation | Status |
|---|---|---|
| Qwen3-4B-Base | Pinned Base revision `906bfd4b...` | matched |
| matched baseline and TTT curriculum | paired selected streams and sequential arms | matched |
| approximately 20B tokens at 32K | 56,623,104 tokens at 32K in the first executable tier | reduced scale |
| approximately 15B tokens at 128K | 41,943,040 tokens at 128K in the first executable tier | reduced scale |
| AdamW, LR `5e-6`, weight decay `0.1` | exact values in all four configurations | matched |
| Stage-2 YaRN | factor 4 from original 32K context | matched |
| convolution size 5 | upstream native depthwise `ttt_conv` hard-codes kernel 5 | matched |
| TTT every sixth layer | layers 0, 6, 12, 18, 24 and 30 of the 36-layer model | matched |

The 100M tier preserves the paper's approximate 20:15 stage ratio, but it is
not a paper-scale run. A matched pair processes 197,132,288 training tokens in
total because both arms must run. The paper-scale matched pair would process
approximately 70B tokens.

## Public-replication assumptions

The paper does not publish an executable record-level continual corpus or all
training-system choices. The following are therefore frozen replication
assumptions and must not be presented as paper-disclosed facts:

- token-balanced public FineWeb-Edu and LongCrawl64 replace the unreleased
  continual corpus;
- global batch size 64, micro batch size 1, seed 42, 2% warmup, cosine decay
  to a 0.9 learning-rate ratio, gradient clipping at 1.0, bf16, FSDP2,
  activation offload and gradient checkpointing;
- native writer settings `ttt_proj=true`, target `hidden_states`, `ttt_lr=3`
  and `ttt_chunk=4096` are frozen official-repository implementation choices;
- Stage 2 uses a fresh optimizer and scheduler after each arm's Stage-1
  checkpoint conversion;
- the first end-to-end tier is approximately 100M tokens per arm because the
  paper's approximately 35B-token curriculum is not feasible without a
  measured scale decision on the user-restricted hardware.

These assumptions preserve a fair paired comparison. They do not establish
exact reproduction of the paper's private training distribution or Table 1
numbers.

## Evidence and boundaries

- `remote_config_audit.json` is the executable zero-error configuration audit.
- `remote_bindings.txt` records the host/GPU policy, Base model identity, every
  live tool commit and clean state, byte-level configuration/auditor hashes,
  and the clean upstream writer implementation.
- `ARTIFACTS_SHA256.txt` binds this directory after documentation is complete.

This preflight establishes recipe-definition alignment only. It does not prove
runtime feasibility, completed optimization, checkpoint validity, model
quality, RULER accuracy, or paper-scale efficacy. Those claims remain gated by
the registered resource, timing, formal-training and independent-audit chain.
No update-admission experiment is released by this result.
