# In-Place TTT Qwen3-4B Baseline Reproduction Plan

Status: **frozen before baseline resource or metric scoring**  
Freeze date: 2026-10-16 (Asia/Shanghai)  
Upstream repository: `ByteDance-Seed/In-Place-TTT`  
Upstream commit: `be2324829b0e91c8fd10a74d4b43714fde6676e1`

## 1. Objective and ordering

Reproduce the In-Place TTT baseline before any further update-admission or
forgetting-gate experiment. Gate experiments remain paused until a matched
no-TTT versus In-Place-TTT baseline has completed the registered evaluation
matrix or an explicit feasibility report shows why the paper-scale run cannot
be executed with the available hardware/data.

The primary target is the paper's drop-in Qwen3-4B continual-pretraining
experiment, not the separate 120B-token from-scratch experiment.

## 2. What the paper actually reports

The primary Qwen3-4B experiment starts from Qwen3-4B-Base and trains two
matched arms with the same continual-pretraining curriculum:

- Stage 1: approximately 20B tokens, 32,768-token sequences, AdamW, learning
  rate `5e-6`, weight decay `0.1`, no RoPE extension;
- Stage 2: approximately 15B tokens, 131,072-token sequences, AdamW, learning
  rate `5e-6`, weight decay `0.1`, YaRN;
- In-Place TTT is inserted every sixth layer;
- depthwise causal Conv1D kernel size is 5;
- reported Qwen3-4B inference clips each fast-weight update to Frobenius norm
  `tau=1e-5`;
- the primary metric is OpenCompass RULER average accuracy at 4K, 8K, 16K,
  32K, 64K, 128K, and 256K.

Paper Table 1 reference values are:

| Arm | 4K | 8K | 16K | 32K | 64K | 128K | 256K |
|---|---:|---:|---:|---:|---:|---:|---:|
| matched baseline | 96.6 | 94.1 | 92.1 | 88.7 | 74.3 | 74.8 | 41.7 |
| In-Place TTT | 96.1 | 95.6 | 92.7 | 89.3 | 78.7 | 77.0 | 43.9 |

The paper's HellaSwag, ARC-E, ARC-C, MMLU, PIQA, and RULER-4K/8K/16K table
belongs to a different 4B model trained from scratch for 120B tokens. Those
numbers are not the primary acceptance target for the Qwen3-4B continual run.

## 3. Reproducibility gaps in the release

The paper uses a private ByteDance mixture. Its continual-pretraining data
contains a pretraining-like short-document portion plus books, repository
code, retrieval-augmented synthetic data, and long-context QA constructions,
organized as 32K and 128K subsets. The repository explicitly does not release
the data or preprocessing scripts.

The released `qwen3_longct.yaml` is a recommended recipe, not a complete
paper-exact recipe. It specifies 20B tokens, 65,536-token sequences, global
batch 64, 5,000 steps, `ttt_chunk=4096`, and `ttt_lr=3`; the paper states
32K/128K stages but does not report their global batch, number of steps,
`ttt_lr`, or chunk size in Appendix C. The release does not include the 128K
YaRN training config. The released inference source also does not expose the
paper's `tau=1e-5` update clipping as a configuration item.

Therefore an exact paper-number reproduction is blocked without additional
author data/configuration. This project must not call a public-data result an
exact reproduction.

## 4. Three evidence levels

### Level A: released-code reproduction

Use an isolated clean checkout at the frozen upstream commit. Verify the
official environment, one-step training, DCP-to-HF conversion, TTT-aware
inference, and OpenCompass RULER metric pipeline. No CBF-TTT task, modified
inference file, controller, or gate may enter this path.

### Level B: public-data matched replication

Use Qwen3-4B and a token-balanced public substitute:

- FineWeb-Edu `sample-10BT` for the short/general portion;
- LongCrawl64 for long natural documents.

The two arms must start from byte-identical Qwen weights and see an identical
ordered token stream:

1. matched baseline: native Qwen continual training with TTT disabled;
2. In-Place TTT: upstream writer and every-sixth-layer configuration enabled.

All optimizer, batch, length, token-budget, scheduler, seed, and checkpoint
settings must match across arms. The data manifest records source revision,
input hashes, tokenizer hash, packing code hash, record/token counts, document
boundaries, and ordered-stream hash.

### Level C: paper-number reproduction

This label is allowed only if the original 32K/128K mixture and missing
training/inference details are obtained and the full two-stage H800-scale
curriculum is executed. Level B results may be compared with Table 1 but may
only be described as a public-data replication.

## 5. Execution order and resource gate

1. Create an isolated upstream checkout and environment manifest.
2. Audit the existing Qwen3-4B assets and public source data without changing
   any checkpoint.
3. Run an inference/RULER 4K smoke on unmodified Qwen3-4B to validate the
   OpenCompass path and score extraction.
4. Run one optimizer-step resource pilots for both matched arms. Use the same
   two RTX 5090 GPUs and progress through 8K, 16K, and 32K sequence lengths.
   Attempt 64K only if 32K has at least 15% free device memory.
5. At every setting record wall time, tokens/second, peak allocated/reserved
   CUDA memory, host RAM, loss, gradient norm, checkpoint bytes, and whether
   both arms consume identical sample ids.
6. Stop scaling at the first OOM, non-finite value, token mismatch, or less
   than 5% free device memory. Do not silently change architecture or offload.
7. Use measured throughput to estimate 20B and 35B total wall time. Select the
   full execution host only after this report is written.
8. If feasible, build the full public replacement corpus at the chosen paper
   stage lengths and train both arms. If infeasible on 2x5090, transfer the
   frozen plan and data hashes to the 8-card NPU/H800-class host; do not replace
   the full reproduction with another unlabelled short run.
9. Convert both final DCP checkpoints and run the identical RULER 4K--256K
   matrix once. Keep raw predictions and per-task scores for paired analysis.
10. Only after the baseline report is complete may support/update gating resume.

The former 1B-token Qwen3-4B checkpoint is retained as an engineering pilot.
It is excluded from baseline acceptance because it used 6,144-token records,
global batch 2, `ttt_lr=0.3`, and no matched no-TTT arm.

## 6. Evaluation and reporting

Primary outputs for each arm and context length:

- RULER task-level accuracy;
- macro average accuracy using the official OpenCompass RULER summarizer;
- number of generated/evaluated/failed examples;
- exact model/config/tokenizer/data/checkpoint hashes;
- prefill wall time, peak CUDA memory, and failure logs.

The comparison reports `TTT - matched baseline` at each length and the mean
over 4K--128K; 256K is explicitly extrapolation. Unmodified base-Qwen results
are diagnostic and do not replace the matched continual-training baseline.

Secondary lm-evaluation-harness metrics are added only for a separately
registered from-scratch experiment. ZsRE, LongMemEval-S, LoCoMo, forgetting
corpora, and support-gate synthetic mechanisms are outside the paper baseline
matrix.

## 7. Acceptance and claim boundary

Level A passes when training, conversion, inference, and metric extraction all
complete from the clean upstream checkout with complete hashes.

Level B passes execution when both arms complete the same public token stream
and the full registered RULER matrix. Efficacy is reported numerically; no
directional threshold is imposed after seeing scores. A gain at long contexts
supports replication on the public substitute only.

No result may be described as reproducing the paper's data or exact Table 1
numbers while the private corpus and missing settings remain unavailable.

## 8. Isolation and compatibility

All baseline work is opt-in and stored outside prior CBF-TTT output roots.
Existing Qwen weights, the 1B checkpoint, V1--V5 gate artifacts, baseline
source files, and sealed test data are immutable. Partial V5 Phase B files are
marked canceled and are never summarized as efficacy evidence.

