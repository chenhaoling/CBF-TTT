# In-Place TTT 2xRTX5090 formal RULER-256K plan

Status: **frozen before the 256K resource decision and any formal 256K score**

Freeze date: 2026-10-18 (Asia/Shanghai)

Parent extrapolation plan SHA256:
`a54341c923987437f6b26aa9aef12d8974bac04c8f82141cdb6ab6a68301b26b`

## 1. Release boundary

Formal 256K scoring is an extrapolation outside the registered 131,072 model
position limit. It requires the paired 4K--128K matrix and the separate paired
256K one-sample resource audit to pass cleanly. A human-reviewed release JSON
must set `accepted=true` and bind the exact resource-audit SHA256. No watcher
may create or accept that release automatically.

## 2. Registered formal run

Run only on `hku-gpu2`, with no-TTT on physical GPU 0 and native TTT on GPU 1.
Each arm uses its unchanged, independently audited formal Stage-2 HF
checkpoint and the original Qwen3-4B tokenizer. Evaluate all 13 RULER tasks at
262,144 tokens with exactly 100 examples per task, seed 42, deterministic
decoding, thinking disabled and at most 64 generated tokens.

Retain raw prompts, references, predictions, task scores, OpenCompass output,
GPU traces and complete dependency/model hashes. The two arms must use
byte-identical evaluation inputs. Report macro accuracy, TTT-minus-no-TTT,
and differences from the paper's matched-baseline/TTT values 41.7/43.9.

## 3. Stopping and claim rules

Stop on OOM, nonzero exit, less than 5% free GPU memory, tokenizer warning,
input mismatch, missing example, nonfinite score, dependency drift or audit
failure. Do not shorten inputs, change tasks, quantize, alter the writer or
reuse partial scores.

The 256K result is reported as a public-data YaRN extrapolation. It is never
merged into the registered 4K--128K mean and cannot turn the public substitute
into an exact reproduction of the paper's private-data experiment.
