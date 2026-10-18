# Qwen3 Liger and fused-AdamW GPU preflight addendum

Status: **frozen before the first GPU preflight execution**

Freeze date: 2026-10-19 (Asia/Shanghai)

## Purpose

CPU audits prove native Qwen initialization and writer autograd reachability,
but the official training entry replaces the causal-LM forward with Liger's
CUDA fused linear cross entropy and constructs CUDA fused AdamW through
VeOmni. This preflight catches an environment/kernel incompatibility before
the full corpus releases Level A.

It is an environment preflight, not an early Level-A result. It uses no public
corpus record, creates no checkpoint and cannot authorize formal training.

## Frozen execution

- run only on physical GPU 0 of `hku-gpu2`, with
  `CUDA_VISIBLE_DEVICES=0`;
- require exactly one visible RTX 5090 and at least 28 GiB free before model
  construction;
- use clean upstream commit
  `be2324829b0e91c8fd10a74d4b43714fde6676e1` and the official baseline Python
  environment;
- import `hf_models` before VeOmni, matching `tasks/train_torch.py`, and require
  the Qwen causal-LM forward to resolve to Liger `lce_forward`;
- instantiate the deterministic two-layer, hidden-size-32 Qwen TTT fixture in
  bfloat16 on CUDA, with one selected layer, chunk size 4 and TTT LR 3;
- create the optimizer through VeOmni `build_optimizer` with the formal
  learning rate `5e-6`, weight decay `0.1`, `fused=True` and AdamW.

The fixed internal eight-token input is never serialized. No dataset or model
artifact is read other than source and environment packages.

## Frozen checks

1. Host, visible-device restriction, GPU identity/free memory, clean upstream
   source and imported module hashes match the registration.
2. Liger's Qwen `lce_forward` is the active model forward.
3. VeOmni returns `torch.optim.AdamW` with fused mode and the registered
   learning rate; both writer tensors occur in optimizer parameter groups.
4. First fused loss and all writer gradients are finite. At the zero writer,
   convolution gradient norm is positive and projection gradient norm is zero.
5. Record the finite total gradient norm, clip with ordinary PyTorch norm 1.0
   only for this non-distributed fixture, and complete the fused optimizer
   step. Convolution displacement must be positive.
6. The second fused loss and writer gradients are finite; both convolution and
   projection gradient norms must be positive. Complete a second fused step.
7. Both optimizer steps leave every writer parameter finite.
8. Peak allocated and reserved CUDA memory, wall time, versions and hashes are
   recorded. The artifact contains no token IDs, logits or tensor payloads.

This preflight does not claim VeOmni FSDP2, distributed gradient clipping,
activation offload, real-sequence memory feasibility, training convergence or
RULER accuracy. Those remain mandatory in the registered Level-A and later
chain. A failure stops GPU training; a pass does not release it.
