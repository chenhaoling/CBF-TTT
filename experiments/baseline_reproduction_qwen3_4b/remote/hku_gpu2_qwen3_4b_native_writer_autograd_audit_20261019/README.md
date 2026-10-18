# Native Qwen3 writer autograd audit

This directory archives the persistent CPU-only writer-gradient audit executed
on `hku-gpu2` from clean tool commit
`ad553231a441a0e118a0b377421ef1e242a0849e` against clean official upstream
`be2324829b0e91c8fd10a74d4b43714fde6676e1`.

All ten registered checks passed. At the zero-convolution start:

- the next-token loss was finite;
- `ttt_conv` had a finite positive gradient norm of
  `0.13500109314918518`;
- `ttt_proj` had an exact zero gradient norm, as expected because the
  convolution supplied a zero target.

After the frozen diagnostic nudge
`ttt_conv -= 1e-3 * first_conv_gradient`:

- 160 convolution values were nonzero;
- the second loss remained finite;
- the convolution gradient norm remained positive at
  `0.13500575721263885`;
- the projection gradient became positive at `0.0007111165905371308`.

CUDA was hidden. Standard PyTorch cross entropy was computed on the official
Qwen logits because the environment's Liger fused loss is CUDA-only. The audit
therefore proves native-writer autograd reachability, while the real Level-A
GPU run remains responsible for Liger, AdamW, FSDP2 and full-model closure.

The first persistent attempt at tool commit `26b12d8f...` completed model
computation but failed before writing JSON because three Python result fields
used lowercase `false`. Commit `ad553231...` corrected only those literals and
used a new output directory. `remote_identity.txt` records this supersession.

`SHA256SUMS` was generated on the remote host. `ARCHIVE_SHA256.txt` binds the
complete local archive. No input IDs, logits or gradient tensors are stored.
This audit authorizes neither formal training nor update admission.
