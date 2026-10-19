# hku-gpu2 exact Qwen3-4B-Base FSDP2 autograd audit

This directory preserves the first persistent execution of the frozen
`in_place_ttt_qwen3_4b_fsdp2_autograd_audit_v1` protocol. It ran only on the
two RTX 5090 devices of `hku-gpu2` from clean tool commit
`2b10fd17062687778bcb74b624368c7d5814eff4` and clean official commit
`be2324829b0e91c8fd10a74d4b43714fde6676e1`.

## Result

- Exit status: `0`
- Frozen checks: **23/23 passed**
- Model: exact `Qwen/Qwen3-4B-Base` revision `906bfd4b...`
- Runtime path: FP32 meta construction, two-rank FSDP2 full shard, BF16 mixed
  precision, gradient checkpointing, VeOmni activation offload and the
  released Liger fused language-model loss
- Writer configuration: layers `[0,6,12,18,24,30]`, `ttt_chunk=4096`, two
  chunks per 8192-token sequence, projection enabled and `ttt_lr=3`

The first backward produced finite positive global Conv5 gradient norms in all
six writers (range 14.28195--3695.65028) and exact-zero projection gradient
norms. After the frozen diagnostic update
`conv -= 1e-3 * first_gradient`, the second backward produced finite positive
Conv5 norms (0.14070--78.95646) and finite positive projection norms
(9.81846--3362.03093) in every writer. The two ranks reported identical
reduced gradient statistics while using different deterministic input hashes.
No token IDs, logits, activations, gradients or model tensors were serialized.

## Resources

- Audit wall time: 36.86 seconds
- `/usr/bin/time` wall time: 39.37 seconds
- Per-GPU peak allocated: 19,874,859,008 bytes (18.51 GiB)
- Per-GPU peak reserved: 25,769,803,776 bytes (24.00 GiB)
- Peak host RSS: 5,392,288 KiB
- Optimizer constructed: no
- Checkpoint written: no

The public-corpus builder remained alive and advanced during the audit. The
native Level-A watcher and every formal training gate were left unchanged.

## Evidence

- `audit.json`: machine-readable checks and scalar evidence
- `run.time.txt`: independent process resource record
- `gpu_before.csv`, `gpu_after.csv`: physical device snapshots
- `remote_identity.txt`: source, model and live-corpus identities
- `stdout.log`, `stderr.log`: raw execution logs
- `SHA256SUMS`: hashes generated on `hku-gpu2`
- `ARCHIVE_SHA256.txt`: hashes of the complete local archive

## Claim boundary

This establishes native-writer gradient reachability through the exact
full-model two-rank FSDP2 8K graph. It does not validate optimizer-state
capacity, clipping, an optimizer step, checkpoint conversion, 32K/128K
training, convergence or RULER quality. The corrected full-corpus Level-A and
all downstream baseline gates remain mandatory. Update admission remains
sealed.
