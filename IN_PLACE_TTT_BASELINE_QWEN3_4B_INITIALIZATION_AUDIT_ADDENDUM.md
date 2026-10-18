# Qwen3-4B native TTT initialization audit addendum

Status: **frozen before the persistent formal initialization audit**

Freeze date: 2026-10-19 (Asia/Shanghai)

## Purpose

The paper requires continual training to begin from the original pretrained
behavior while newly added native TTT components start with negligible update.
The depthwise convolution must be zero initialized and the target projection
must be sparse diagonal. This addendum verifies the released Qwen training
implementation before the corrected full-corpus training chain starts.

An earlier interactive CPU smoke observed the expected behavior but wrote no
persistent result and measured no model score. It is exploratory evidence only.
The formal artifact defined here is the authoritative initialization audit.

## Frozen runtime

- host: `hku-gpu2` (`user-Super-Server`);
- GPUs hidden with `CUDA_VISIBLE_DEVICES=`;
- clean official upstream checkout at
  `be2324829b0e91c8fd10a74d4b43714fde6676e1`;
- official Python environment used by the baseline chain;
- a deterministic two-layer, hidden-size-32 Qwen fixture using the same
  `Qwen3Config`, `Qwen3ForCausalLM`, `post_init`, state loading and forward
  implementations as the full model.

The fixture is an initialization semantics test. It is not a substitute for
the full Qwen3-4B resource gate or training run.

## Frozen checks

1. The upstream checkout commit and clean state match the registered source.
2. The imported configuration and modeling modules resolve inside that exact
   checkout, and their SHA256 values are recorded.
3. Loading a no-TTT fixture state into a TTT fixture leaves exactly two missing
   tensors for the selected layer: `ttt_conv.weight` and `ttt_proj.weight`;
   there are no unexpected keys.
4. Every shared slow-weight tensor is bitwise identical after loading.
5. `ttt_conv` is depthwise, bias-free, kernel size 5 and contains no nonzero
   value.
6. `ttt_proj` is bias-free and square; every off-diagonal value is zero, every
   diagonal value is finite and every diagonal entry is nonzero.
7. A nonzero target passed through the convolution produces an exact zero
   tensor, so the initial native fast-weight delta is zero.
8. On the same token IDs with cache disabled, no-TTT and TTT logits are
   bitwise identical and have maximum absolute difference zero.
9. A non-selected layer has no TTT parameters.
10. The output contains only configuration, shapes, counts, hashes and numeric
    checks; it contains no corpus text or token sequence.

Any failed check is a stop condition for corrected Base training. The audit
does not authorize a formal release, change the native writer, evaluate RULER,
or open an update-admission experiment.
