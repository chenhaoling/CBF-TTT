# Qwen3-4B-Base two-RTX-5090 4K capacity retry addendum

Status: **frozen after the registered 8K failure and before 4K execution**

Date: 2026-10-19

Execution host: `hku-gpu2` only

## Motivation

The separately registered no-headroom 8K diagnostic completed its first
native-TTT optimizer step, then failed during the second backward pass while
requesting 152 MiB. GPU 0/1 had only 78/114 MiB free after the first step had
materialized persistent Adam state. This was a real CUDA OOM rather than a 5%
policy stop.

This pre-execution retry changes only the sequence length from 8,192 to 4,096.
It retains Qwen3-4B-Base, both RTX 5090 devices, the native writer, global batch
2, micro batch 1, activation offload, fused AdamW, ten consecutive steps, the
audited preliminary Stage-1 source stream, and one final DCP save. It does not
apply a GPU or host-memory headroom threshold.

## Decision rule and claim boundary

Pass requires ten completed optimizer steps, finite final loss and gradient
norm, exit code zero, no CUDA/OOM signature, and a nonempty step-10 DCP. All
resource minima remain descriptive. Failure of this retry means the current
two-card training stack needs optimizer/offload changes or more device memory.

A pass establishes only sustained 4K engineering capacity on two RTX 5090s.
It does not retroactively pass the frozen 8K Level-A gate, validate 8K or longer
training, release RULER, or establish TTT efficacy.
