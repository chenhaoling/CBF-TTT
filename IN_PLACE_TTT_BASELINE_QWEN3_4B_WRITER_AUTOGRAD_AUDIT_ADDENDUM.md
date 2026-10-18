# Qwen3 native-writer autograd audit addendum

Status: **frozen before the persistent formal autograd audit**

Freeze date: 2026-10-19 (Asia/Shanghai)

## Purpose

The initialization audit proves that the native TTT update begins at zero. It
does not by itself prove that the zero-initialized writer can learn. This
addendum verifies that the released Qwen training implementation preserves the
expected gradient path from language-model logits into `ttt_conv` and then
into `ttt_proj` once the convolution leaves zero.

Two earlier interactive CPU probes wrote no persistent artifact. The first
reached the environment's Liger Triton loss and correctly failed with no CUDA
driver visible. The second used standard cross entropy on the same official
logits and observed the expected gradients. Both are exploratory only; the
formal artifact below is authoritative.

## Frozen runtime and scope

- host `hku-gpu2` (`user-Super-Server`), with `CUDA_VISIBLE_DEVICES=`;
- clean official upstream commit
  `be2324829b0e91c8fd10a74d4b43714fde6676e1`;
- the same deterministic two-layer, hidden-size-32 official Qwen fixture used
  by the initialization audit;
- one selected TTT layer, `ttt_chunk=4`, `ttt_lr=3`, and an eight-token
  internal fixture that is never serialized;
- standard PyTorch next-token cross entropy computed from the official model
  logits.

This audit tests native-writer autograd only. It makes no claim about the Liger
fused CUDA loss, AdamW/FSDP2, throughput, convergence or Qwen3-4B quality.
Those remain covered by the real GPU Level-A and formal training chain.

## Frozen checks

1. Host, hidden-CUDA policy, clean upstream identity and imported module hashes
   match the registered baseline.
2. Both native TTT tensors exist, require gradients and are returned by
   `named_parameters()`.
3. Before any nudge, the next-token loss is finite.
4. The first backward pass yields a finite, strictly positive convolution
   gradient norm.
5. The first projection gradient is finite and exactly zero, as required when
   its input from the zero convolution is zero.
6. Apply only a deterministic diagnostic nudge
   `ttt_conv -= 1e-3 * first_conv_gradient`; this is not an optimizer step and
   is discarded with the fixture.
7. After the nudge, the convolution contains at least one nonzero value and a
   second next-token loss is finite.
8. The second backward pass yields finite, strictly positive gradient norms
   for both convolution and projection, proving the projection is reachable
   after the writer leaves the zero point.
9. The artifact contains only shapes, counts, norms, losses and hashes; it
   contains no input IDs, logits, gradients or corpus content.

Any failed check is a stop condition for corrected Base training. A pass does
not authorize a formal release or any update-admission experiment.
