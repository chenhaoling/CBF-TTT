# In-Place TTT Level-B public-corpus and matched-training plan

Status: **frozen before source inventory, full-corpus construction or Level-B training**  
Freeze date: 2026-10-18 (Asia/Shanghai)  
Parent plan SHA256: `fbce7e0e52c495467c5b6383242a7350cf16042d61af8795309b9d6eda021fd3`  
32K pilot addendum SHA256: `632ab39cdb015e57d082cb9f56d034f5484bd32316cec459f7988c776db2af83`

## 1. Scope and explicit override

This plan defines the public-data Level-B matched replication after the
8-record-per-source 32K construction pilot passed. It does not authorize an
update-admission or forgetting gate and does not claim use of the paper's
private corpus.

The parent plan named FineWeb-Edu `sample-10BT`. That sample contains only
about 10B GPT-2 tokens and cannot supply the two-stage, 17.5B-Qwen-token
FineWeb quota without replay. This plan explicitly replaces it with the
publisher's nested `sample-100BT` configuration at the same pinned revision.
No other source or mixture change is allowed.

## 2. Frozen source streams

| Source | Repository | Revision | Ordered publisher path |
|---|---|---|---|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | `sample/100BT/*.parquet` |
| LongCrawl64 | `manifestai/longcrawl64` | `8e7af027525b3597852746ee11e4baa0e98492d7` | `train/*.parquet` |

Before packing, an inventory tool must resolve each glob, sort repository
paths lexicographically, and save every publisher path, byte size, blob ID and
LFS SHA256. The inventory binds the repository revisions, inventory code,
this plan and the original Qwen3-4B tokenizer assets. Missing or duplicate
paths, empty hashes, revision drift or a non-monotonic file list stops the
run.

Within a source, traversal order is publisher file, parquet row index, then
Qwen token offset. Stage 2 continues the exact Stage-1 source cursor and
pending token buffer. It may split a document across the stage boundary, but
it must never rewind, replay or discard unconsumed source tokens.

## 3. Exact public token budget

Each stage is balanced 50/50 by Qwen training tokens. Counts are chosen as the
nearest equal-source whole-record totals that are divisible by global batch
64.

| Stage | Seq. length | Records/source | Combined records | Tokens/source | Combined tokens | GB64 steps |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 32,768 | 305,184 | 610,368 | 10,000,269,312 | 20,000,538,624 | 9,537 |
| 2 | 131,072 | 57,216 | 114,432 | 7,499,415,552 | 14,998,831,104 | 1,788 |
| total | — | 362,400 | 724,800 | 17,499,684,864 | 34,999,369,728 | 11,325 |

The small difference from decimal 20B/15B is solely whole-record and GB64
rounding. Reports must use the exact integer counts above.

## 4. Packing and ordered stream

1. Reuse the audited Qwen plaintext round-trip rule: source documents receive
   boundary EOS tokens; every emitted plaintext independently re-encodes to
   `seq_len - 1`; VeOmni adds the terminal EOS.
2. Pack FineWeb and LongCrawl independently while preserving their source
   cursors. Interleave final records strictly as
   `fineweb[0], longcrawl[0], fineweb[1], longcrawl[1], ...` in each stage.
3. Stage 1 and Stage 2 use separate JSONL/provenance files under one atomically
   published corpus root. The Stage-2 provenance begins from the Stage-1 final
   source cursor for both sources.
4. Provenance contains hashes and token spans only. It must contain no source
   `text` or `content_split` field. The training JSONL contains only
   `content_split`.
5. The manifest binds inventories, source positions, tokenizer assets, plan,
   builder, both outputs/provenance files, per-record hashes, stage ordered
   token hashes, the combined curriculum hash, record/token/document counts,
   synthetic EOS counts, wall time and peak RSS.
6. All files are written below a new `<root>.partial` directory and the root
   is renamed only after every invariant passes. Existing final or partial
   paths are never overwritten.

## 5. Required data-pipeline pilot

Before constructing 35B tokens, run the complete two-stage protocol with
continuous source cursors:

| Stage | Records/source | Combined tokens |
|---|---:|---:|
| 32K | 64 | 4,194,304 |
| 128K | 8 | 2,097,152 |

The pilot must use the frozen `sample-100BT` and LongCrawl inventories, record
per-source/per-stage throughput and peak RSS, and pass a separate auditor that
does not import builder code. Stop before the full corpus if any row length,
source cursor, alternation, document span, hash or no-replay check fails; if
peak RSS exceeds 16 GiB; or if any source-stage throughput is zero/non-finite.

Passing this pilot validates 128K packing and stage transition only. It does
not authorize training on the 2x5090 host.

## 6. Matched training configuration

Both arms use clean upstream commit
`be2324829b0e91c8fd10a74d4b43714fde6676e1`, the same Qwen3-4B-Base files,
the same ordered JSONL, seed, FSDP2 layout and checkpoint schedule.

Common settings:

- global batch 64, micro batch 1, `ulysses_parallel_size=1`;
- AdamW, learning rate `5e-6`, weight decay `0.1`, gradient clip `1.0`;
- cosine decay to 0.9 of the initial rate and 2% warmup;
- bf16 mixed precision, gradient checkpointing and full FSDP2 sharding;
- native plaintext transform and `content_split` key;
- save every 500 steps and at the exact final step;
- no CPU offload, allocator workaround, controller or modified writer.

The matched baseline sets `ttt_mode=false`. The In-Place-TTT arm sets
`ttt_mode=true`, `ttt_layers=[0,6,12,18,24,30]`, `ttt_proj=true`,
`ttt_target=hidden_states`, `ttt_chunk=4096` and `ttt_lr=3`.

Stage 1 uses the original RoPE configuration and 32,768 tokens. Stage 2 uses
Qwen's public 128K YaRN setting:

```yaml
max_position_embeddings: 131072
rope_scaling:
  rope_type: yarn
  factor: 4.0
  original_max_position_embeddings: 32768
```

The release cannot resume a DCP with a new dataloader and scheduler without
also restoring the old optimizer/dataloader state. Therefore the registered
Level-B assumption is: convert each arm's final Stage-1 DCP through the
official converter, initialize Stage 2 from those weights, and create a fresh
AdamW optimizer and scheduler. This is a public-replication assumption, not a
claim about the unreleased paper recipe.

## 7. Host and training gates

The measured exclusive 2xRTX5090 run has only 2.107% free memory at 8K. It is
forbidden for 32K/128K Level-B training. Formal execution requires a separately
recorded host preflight, preferably 8 GPUs with at least 80 GiB each, enough
local storage for the corpus plus two-arm checkpoints, and no unrelated GPU
process.

Before any full training, run one matched optimizer step at 32K on the selected
host. Only if both arms are finite and retain at least 5% free memory may the
Stage-1 run start. Before Stage 2, run one matched 128K step from converted
Stage-1-format weights. Stop at the first OOM, non-finite value, token/hash
mismatch, source-order mismatch or less than 5% free device memory. Any
context-parallel, offload, optimizer, batch or writer change requires a new
addendum and cannot be introduced silently.

## 8. Stage transition and audits

An independent audit must verify before Stage 1:

- source inventory, corpus, tokenizer, code and plan hashes;
- exact record/token counts, equal-source alternation and ordered stream;
- zero source replay between stages and continuous cursor/spans;
- byte-identical initial model files and matched configs except `ttt_mode` and
  its native writer fields.

After each stage it verifies checkpoints, step count, finite loss/gradient
logs, data IDs/hashes consumed by both arms and resource traces. After Stage 1
it also verifies official conversion and that Stage-2 inputs bind the correct
arm checkpoint while the optimizer/scheduler reset is explicit.

## 9. Evaluation and claim boundary

After both arms complete Stage 2 and official HF conversion, run the same 13
OpenCompass RULER tasks with 100 examples per task at 4K, 8K, 16K, 32K, 64K,
128K and 256K. Use the audited original Qwen tokenizer path and retain raw
predictions and task-level scores. Report `TTT - matched baseline` per context
length and the 4K--128K mean; 256K is extrapolation.

ZsRE, MMLU, LongMemEval-S, LoCoMo and all gate datasets remain outside this
primary Qwen3-4B paper-baseline matrix. They may be run later under separate
plans but cannot substitute for RULER acceptance.

Success means both arms consume the exact public stream, complete both stages,
convert, and complete the full registered RULER matrix. The result is called a
**public-data matched replication**. It must never be called an exact
reproduction of the private paper corpus or missing training recipe.
