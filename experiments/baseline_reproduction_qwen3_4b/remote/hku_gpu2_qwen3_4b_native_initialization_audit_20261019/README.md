# Native Qwen3 TTT initialization audit

This directory archives the persistent CPU-only initialization audit executed
on `hku-gpu2` from tool commit
`8b1f992628f41cacf9d2d0cbb79a887476213824` against the clean official
upstream commit `be2324829b0e91c8fd10a74d4b43714fde6676e1`.

The audit passed all registered checks:

- loading the baseline state left exactly the selected layer's
  `ttt_conv.weight` and `ttt_proj.weight` missing;
- every shared slow-weight tensor remained bitwise identical;
- the depthwise kernel-5 convolution contained zero nonzero values;
- the projection contained 32 finite, nonzero diagonal entries and zero
  nonzero off-diagonal values;
- a nonzero target produced an exact zero convolution output;
- initial no-TTT and TTT logits were bitwise identical, with maximum absolute
  difference `0.0`;
- the non-selected layer had no TTT parameters.

CUDA was hidden. Runtime was 3.16 seconds with 803,256 KiB maximum host RSS.
`SHA256SUMS` is the checksum file generated on the remote host;
`ARCHIVE_SHA256.txt` binds the complete local archive including this README.

This is a tiny-model initialization-semantics audit. It does not replace the
full Qwen3-4B resource gate, training, checkpoint, RULER or independent final
audits, and it releases no update-admission experiment.
