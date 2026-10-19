# Qwen3-4B-Base full-checkpoint writer preflight

This directory archives the frozen full-checkpoint integration preflight run
only on physical GPU 0 of `hku-gpu2`. It used clean tool commit
`8fe8c0b6e041e9d97c2565776c5f007baefa325e`, clean official upstream
`be2324829b0e91c8fd10a74d4b43714fde6676e1`, and exact Qwen3-4B-Base revision
`906bfd4b4dc7f14ee4320094d8b41684abff8539`.

The frozen result is **failed: 22 of 24 checks passed**. It is not relabeled as
a pass. The two failed checks were:

- every BF16 projection diagonal entry had to be nonzero;
- initial writer-active logits had to match a branch-bypassed forward at
  `rtol=1e-3`, `atol=1e-3`.

All checkpoint-integration and gradient-path checks passed. The model loaded
exactly the expected twelve new writer tensors at layers
`[0, 6, 12, 18, 24, 30]`, with no other missing, unexpected or mismatched
weights. Every convolution was zero, every off-diagonal projection entry was
zero, all six first convolution gradients were positive, all six first
projection gradients were exactly zero, and all six second convolution and
projection gradients were positive after the frozen diagnostic nudge. The
pretrained parameter version counters did not change.

The post-failure `precision_probe.json` does not alter the result. It explains
the observed mechanisms:

- CPU BF16 normal initialization produced 4--24 exact zeros per 2560 values
  over 32 fixed seeds (mean 12.8125), matching the full model's 7--20 zeros per
  projection diagonal. CUDA BF16 sampling produced zero exact zeros for those
  seeds. `from_pretrained` initializes missing writer weights on CPU before
  moving the model to CUDA.
- For the real down-projection dimensions, BF16 `F.linear` and the native
  chunked `einsum` path differed by maximum `0.0625` even with identical
  operands. This supplies a direct numerical mechanism for the final six-layer
  maximum logit difference `0.25` despite exactly zero convolution weights.

The run measured 12.60 seconds, 4,766,092 KiB peak host RSS and
10,460,139,008/10,938,744,832 peak allocated/reserved CUDA bytes. No input
IDs, logits, gradients or checkpoint were serialized. `SHA256SUMS` was created
on the remote host and `ARCHIVE_SHA256.txt` binds the completed local archive.

This failure does not modify the baseline implementation or its formal
release gates. It shows that the tiny CPU initialization audit's bitwise and
all-diagonal-nonzero properties do not transfer unchanged to full-checkpoint
BF16 `from_pretrained` execution. FSDP2/meta-device initialization, formal
32K/128K training and RULER remain pending. Update admission remains sealed.
