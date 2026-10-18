# Corrected Qwen3-4B-Base Level-A execution addendum

Status: **frozen before corrected-Base GPU execution**  
Freeze date: 2026-10-19 (Asia/Shanghai)

## 1. Scope

This addendum replaces the Level-A evidence produced with the mistakenly
staged post-trained `Qwen/Qwen3-4B` checkpoint. It does not change the parent
baseline objective, the native writer, the model architecture or any formal
Level-B metric. It registers one corrected execution closure using the exact
paper starting checkpoint `Qwen/Qwen3-4B-Base`.

All GPU work is restricted to `hku-gpu2` (hostname `user-Super-Server`) and its
two 32,607-MiB RTX 5090 devices. No other GPU host is an eligible fallback.

## 2. Immutable prerequisites

The watcher must stop before CUDA execution unless all of the following hold:

1. the clean upstream checkout is exactly
   `be2324829b0e91c8fd10a74d4b43714fde6676e1`;
2. the tool checkout is an exact clean commit containing this addendum;
3. the local model is `Qwen/Qwen3-4B-Base` revision
   `906bfd4b4dc7f14ee4320094d8b41684abff8539` and every file matches the
   passing identity manifest;
4. the corrected Base-tokenizer full public corpus has atomically published
   724,800 records and 34,999,369,728 training tokens, has no `.partial`
   sibling and passes every canonical independent-audit check;
5. the corpus builder/auditor has exited, both GPUs are exclusively idle and
   at least 48 GiB host memory is available.

The former post-trained-tokenizer corpus and all old resource/checkpoint roots
are forbidden inputs.

## 3. Registered Level-A run

Derive the five sparse resource inputs (8K, 16K, 32K, 64K and 128K) from the
first FineWeb-Edu/LongCrawl64 pair in the corrected full corpus with the Base
tokenizer. The input manifest must bind the full-corpus manifest and audit,
contain no source text, and pass the independent resource-input audit. Only
the 8K input is consumed in this Level-A run; the larger inputs prepare the
unchanged resource ladder. Construction follows the separately frozen
`IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_RESOURCE_INPUT_ADDENDUM.md`.

Run the matched arms sequentially, always `no_ttt` before `ttt`, from the same
Base checkpoint and the same two-record 8K stream. Each arm performs exactly
one optimizer step with global batch 2, BF16 FSDP2, gradient checkpointing,
activation offload, AdamW learning rate `5e-6`, weight decay `0.1`, and the
released writer settings. The arms may differ only in native `ttt_mode`.

Each arm must finish with finite loss and gradient norm, a complete DCP
checkpoint, at least 5% free memory on both GPUs, at least 8 GiB available host
memory throughout, and no more than 3.5 GiB swap use. Any OOM, nonfinite value,
input mismatch, resource violation or failed independent audit stops the chain.

After both arms pass, convert each DCP checkpoint with the upstream converter.
The no-TTT checkpoint must contain no `ttt_proj`/`ttt_conv` tensors; the TTT
checkpoint must contain them. Then run exactly one example for each of the 13
official RULER-4K tasks using the frozen OpenCompass environment, deterministic
settings and the Base tokenizer with `fix_mistral_regex=false`. Both arms must
complete all tasks, have zero tokenizer-regex warnings, and match the official
group score under independent recomputation.

## 4. Evidence and claim boundary

The terminal audit must re-hash the identity, full corpus, derived input,
training, conversion and RULER artifacts; confirm the run order and matched
bindings; and write one compact result. Raw DCP/HF checkpoints and predictions
remain on `hku-gpu2`.

Passing this addendum proves released-code execution closure for the corrected
Base checkpoint. One RULER example per task is not an efficacy estimate and
does not release formal 100M training, the 4K--128K RULER matrix, 256K
extrapolation or update-admission experiments. Those remain governed by their
separate frozen gates.
