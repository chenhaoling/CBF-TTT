# Qwen3 Liger and fused-AdamW GPU preflight

This directory archives the frozen single-GPU environment preflight executed
only on physical GPU 0 of `hku-gpu2`. The run used clean tool commit
`2bd8d4b6543b728e800474d1ecdd90c4663180a2` and clean official upstream
`be2324829b0e91c8fd10a74d4b43714fde6676e1`.

All 19 registered checks passed. The tiny two-layer Qwen3 TTT fixture verified
that:

- the Hugging Face Qwen3 forward was replaced by the repository's Liger fused
  linear-cross-entropy forward;
- VeOmni constructed `torch.optim.AdamW` with `fused=true`, learning rate
  `5e-6` and writer weight decay `0.1`;
- both native `ttt_conv` and `ttt_proj` parameters were present in optimizer
  groups;
- the first fused-loss backward gave `ttt_conv` a positive gradient while the
  zero-input projection gradient remained exactly zero;
- the first fused optimizer step changed `ttt_conv` and kept both writer
  tensors finite;
- the second fused-loss backward gave both writer tensors positive finite
  gradients, and the second fused step remained finite.

The two fused losses were `5.3756909370` and `5.3651542664`. The run took
4.91 seconds by `/usr/bin/time`; the script measured 4.25 seconds. Peak
PyTorch allocated/reserved GPU memory was 17,367,552/23,068,672 bytes and peak
host RSS was 2,001,524 KiB. These resource values describe only the tiny
preflight fixture and must not be used as a full Qwen3-4B estimate.

`CUDA_VISIBLE_DEVICES=0` exposed exactly one RTX 5090. GPU 1 was not used. No
input IDs, logits or gradient tensors are serialized. `SHA256SUMS` was created
on the remote host, and `ARCHIVE_SHA256.txt` binds the completed local archive.

This result closes the local Liger/fused-AdamW environment check. It does not
validate FSDP2, VeOmni distributed gradient clipping, full-model memory,
continual training, checkpoints or RULER scores. It authorizes neither formal
release nor update-admission experiments.
