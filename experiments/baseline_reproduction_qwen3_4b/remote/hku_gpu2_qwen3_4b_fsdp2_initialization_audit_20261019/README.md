# Qwen3-4B-Base FSDP2 writer-initialization audit

This directory archives the initialization-only audit executed on both RTX
5090 GPUs of `hku-gpu2`. The run used clean tool commit
`252828fdf1211b73994f53d952e4ef5820e172f4`, clean official upstream
`be2324829b0e91c8fd10a74d4b43714fde6676e1`, and exact Qwen3-4B-Base revision
`906bfd4b4dc7f14ee4320094d8b41684abff8539`.

All 12 registered checks passed. The audit followed the same model
construction chain as `tasks/train_torch.py`: FP32 meta model construction,
two-rank FSDP2 full sharding, CUDA materialization, checkpoint loading and
VeOmni missing-key initialization through the official Qwen `_init_weights`.

For every selected layer `[0, 6, 12, 18, 24, 30]` on both ranks:

- `ttt_conv.weight` and `ttt_proj.weight` were CUDA FP32 DTensors;
- local shards had shapes `[1280,1,5]` and `[1280,2560]`;
- reconstructed convolutions had global shape `[2560,1,5]` and zero nonzero
  entries;
- reconstructed projections had global shape `[2560,2560]`, exactly 2560
  finite nonzero diagonal entries and zero nonzero off-diagonal entries.

The two ranks reconstructed bytewise-equivalent scalar descriptions. Each GPU
peaked at 10,514,637,312 allocated and 10,525,605,888 reserved CUDA bytes. The
script measured 20.61 seconds; `/usr/bin/time` measured 23.31 seconds and
5,393,824 KiB peak host RSS.

This resolves the scope of the preceding single-GPU failure: its CPU BF16
zero-diagonal behavior does not occur in the formal FP32-meta/CUDA-DTensor
initialization path. The single-GPU BF16 active-versus-bypass output difference
remains valid because this audit executes no forward pass.

No token, model tensor, activation, logit or gradient is serialized. No
optimizer was constructed and no checkpoint was saved. `SHA256SUMS` was
created remotely and `ARCHIVE_SHA256.txt` binds the completed local archive.

This result does not replace Level A, formal 32K/128K training, checkpoint
conversion or RULER. Update admission remains sealed.
