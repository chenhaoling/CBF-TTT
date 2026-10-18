# In-Place TTT 2xRTX5090 RULER-256K extrapolation plan

Status: **frozen before any 256K resource run or score**

Freeze date: 2026-10-18 (Asia/Shanghai)

Parent RULER plan SHA256:
`fe41a68a1fc474c5d19a921623e1d17ea2011e1d7327b7679993b645f1ab3962`

## 1. Scope

The paper reports a 256K RULER point, while the registered public-data Stage-2
checkpoints have `max_position_embeddings=131072`. Therefore 256K is a YaRN
extrapolation, separate from the primary 4K--128K matrix. It cannot change the
primary matrix or release update-admission experiments.

Use only the two final HF checkpoints from the passing paired formal Stage-2
audit, the original Qwen3-4B tokenizer and the same 13 RULER tasks, seed 42,
deterministic decoding and 64-token output cap. All GPU work is restricted to
`hku-gpu2`, with no-TTT on GPU 0 and native TTT on GPU 1.

## 2. Resource decision

The resource run starts only after the independently audited 4K--128K matrix
completes. Each arm evaluates exactly one example from every task at 262,144
tokens. Scores are discarded. Both arms must exit zero, retain identical
prompts and references, pass the existing single-run resource audit, and keep
at least 5% GPU memory free. Any OOM, nonzero exit, tokenizer warning, input
mismatch, dependency drift or missing artifact stops the extrapolation.

## 3. Formal 256K boundary

A passing resource decision permits a separately frozen 100-example-per-task
formal 256K runner. It does not launch that runner automatically. The formal
result must report all task scores, paired input hashes, macro accuracy,
TTT-minus-no-TTT, and differences from the paper's 41.7/43.9 reference values.

If the resource decision fails, report that 256K was infeasible on two 32-GB
RTX 5090 GPUs. Do not shorten inputs, reduce the task set, quantize the model,
change the writer, or present a subset as the paper's 256K metric.
