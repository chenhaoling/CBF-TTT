# In-Place TTT preliminary 100M public-data replication addendum

Status: **frozen before constructing the preliminary corpus or producing a
preliminary training result**  
Freeze date: 2026-10-19 (Asia/Shanghai)

## 1. Decision and scope

The user has chosen an initial matched validation instead of waiting for the
approximately 35B-token public replacement corpus. This addendum authorizes a
98,566,144-token-per-arm experiment on `hku-gpu2` only. It reuses the frozen
FineWeb-Edu `sample-100BT` and LongCrawl64 inventories, Qwen3-4B-Base revision
`906bfd4b4dc7f14ee4320094d8b41684abff8539`, native `ttt_conv`/`ttt_proj`
writer, optimizer, scheduler, checkpoint conversion and RULER metric paths.

This is a **preliminary public-data matched replication**. It is neither the
paper's private data nor a paper-scale reproduction. A null or positive result
at this scale cannot establish the paper's final 35B-token result.

## 2. Frozen corpus profile

The builder profile is named `preliminary_100m`. FineWeb-Edu and LongCrawl64
each provide exactly half the training tokens in each stage. Source cursors and
pending tokens continue from Stage 1 into Stage 2 without rewind or replay.

| Stage | Sequence length | Records/source | Combined records | Tokens/source | Combined tokens | Global-batch-64 steps |
|---|---:|---:|---:|---:|---:|---:|
| Stage 1 | 32,768 | 864 | 1,728 | 28,311,552 | 56,623,104 | 27 |
| Stage 2 | 131,072 | 160 | 320 | 20,971,520 | 41,943,040 | 5 |
| total | — | 1,024 | 2,048 | 49,283,072 | 98,566,144 | 32 |

The resulting manifest must bind this plan, the builder, independently audited
source inventory, exact tokenizer assets, ordered token hashes, source cursor
states and all data/provenance artifact hashes. The existing independent
auditor must pass before the corpus can enter training.

## 3. Matched arms

The two arms consume the same ordered corpus and use the registered
`qwen3_4b_2x5090_100m_stage{1,2}_{no_ttt,ttt}.yaml` settings. Their only model
behavior difference is native TTT activation:

- no-TTT: `ttt_mode=false`;
- native TTT: `ttt_mode=true`, layers `[0,6,12,18,24,30]`, projection enabled,
  hidden-state target, chunk 4096 and TTT learning-rate multiplier 3.

Both use global batch 64, AdamW learning rate `5e-6`, weight decay `0.1`, BF16,
FSDP2 and gradient checkpointing. Stage 1 executes 27 optimizer steps. Each
arm's final Stage-1 DCP is converted through the registered converter and
starts its own Stage 2 with a fresh optimizer/scheduler for 5 steps. Initial
weights, data order, seed, software and hardware must match across arms.

## 4. Execution order and stop rules

1. Run focused corpus tests locally and in the frozen remote environment.
2. Build and independently audit `preliminary_100m` with CUDA hidden. Keep the
   already running full-corpus job intact until this reduced corpus passes.
3. Record construction wall time, throughput, peak host RSS and disk use.
4. Defer the full-corpus job without deleting its partial artifacts before any
   GPU training, because registered GPU resource gates require an idle host.
5. Run the corrected Base Level-A paired optimizer-step gate, including
   activation offload, checkpoint save/convert/reload and RULER smoke.
6. Run Stage 1 only after Level A passes; run Stage 2 only after paired Stage-1
   audit and transition checks pass.
7. Run the frozen official RULER matrix only after both final checkpoints pass
   independent audit.

Stop on any corpus hash/order/continuity failure, non-finite loss or gradient,
OOM, checkpoint mismatch, arm input mismatch, less than 5% free device memory,
less than 8 GiB available host memory, or more than 3.5 GiB host swap use. Do
not silently change sequence length, writer, optimizer, batch size or model.

## 5. Evaluation and claims

The efficacy metric remains the official 13-task OpenCompass RULER macro
accuracy with 100 examples per task, deterministic decoding and seed 42 at 4K,
8K, 16K, 32K, 64K and 128K. Report both arm scores and `TTT - no-TTT` at every
length and the 4K--128K mean. The registered 256K point remains a separate
resource-gated extrapolation.

The update-admission gate remains sealed until this matched preliminary pair,
checkpoint audits and RULER matrix finish. A useful preliminary result is an
auditable paired estimate and resource measurement; it is not required to
match the paper's absolute values. Any later 1B or 35B scale-up requires a new
pre-outcome addendum.
