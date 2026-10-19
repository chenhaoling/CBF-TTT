# Qwen3-4B-Base full-checkpoint native-writer preflight addendum

Status: **frozen before persistent execution**

Freeze date: 2026-10-19 (Asia/Shanghai)

Parent: `IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md`

## 1. Purpose

The tiny initialization, autograd and Liger/fused-AdamW audits validate the
released code path but do not load the exact 36-layer Qwen3-4B-Base checkpoint
with all six registered native-TTT layers. Before the corrected full corpus
finishes, run one bounded integration preflight that verifies this exact
checkpoint can materialize the native writer and propagate the fused language
model loss to every writer layer.

This is a diagnostic preflight. It does not replace the registered two-GPU
FSDP2 Level-A run or authorize any downstream release.

## 2. Frozen identity and device policy

- Host: `hku-gpu2` only.
- Device: physical GPU 0 only, exposed as `CUDA_VISIBLE_DEVICES=0`.
- Require exactly one visible RTX 5090 and at least 28 GiB free before load.
- Official source: clean commit
  `be2324829b0e91c8fd10a74d4b43714fde6676e1`.
- Model: local exact `Qwen/Qwen3-4B-Base` revision
  `906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- Recompute every model file hash recorded by the passing identity artifact
  before model load.

No other GPU host or physical GPU may be used.

## 3. Frozen fixture

Load the exact Base checkpoint through the official Qwen3 implementation in
BF16 and SDPA. Add native TTT only at layers `[0, 6, 12, 18, 24, 30]`, with
`ttt_proj=true`, `ttt_target=hidden_states` and `ttt_lr=3.0`.

Use `ttt_chunk=16` and one deterministic 64-token row so that this bounded
fixture crosses multiple chunks. The reduced chunk and row length are
diagnostic choices and are not the formal Stage-1 recipe (`ttt_chunk=4096`,
32K records). Serialize only the SHA-256 of the token row, never token IDs,
logits or gradient tensors.

Freeze every pretrained parameter and enable gradients only for the twelve
new `ttt_conv`/`ttt_proj` tensors. Do not save or promote a checkpoint.

## 4. Required checks

The persistent result passes only if all checks below pass:

1. host, device, free-memory, upstream and model identities match exactly;
2. official configuration/model modules resolve from the clean upstream;
3. the loaded model has 36 layers, hidden size 2560 and exactly the six
   registered TTT layers;
4. loading reports exactly the twelve new writer weights as missing, with no
   unexpected, mismatched or error entries;
5. each convolution is bias-free depthwise Conv1d with kernel 5 and is exactly
   zero; each projection is bias-free square, has a finite nonzero diagonal
   and zero off-diagonal entries;
6. with the writer active at initialization, logits are finite and numerically
   equal to a same-model forward whose six TTT branches are temporarily
   bypassed (`allclose`, `rtol=1e-3`, `atol=1e-3`);
7. the official Liger fused forward is active and the first fused loss is
   finite;
8. every convolution has a finite positive first gradient and every
   projection has the expected exact-zero first gradient;
9. after the frozen diagnostic nudge
   `ttt_conv -= 1e-3 * first_conv_gradient`, every convolution is nonzero;
10. the second fused loss is finite and every convolution and projection has
    a finite positive gradient;
11. all pretrained parameters retain their original version counter, all
    writer tensors remain finite, and no checkpoint is written.

Any failed check is a failed preflight. Preserve the failure artifact and do
not weaken a numerical gate after observing the result.

## 5. Resources and artifacts

Record wall time, peak allocated/reserved CUDA bytes, peak host RSS, runtime
versions, scalar losses, per-layer scalar gradient norms, model/source hashes
and the no-TTT-bypass maximum absolute logit difference. Retain stdout,
stderr, `/usr/bin/time -v`, remote identity and SHA-256 manifests.

## 6. Claim boundary

A pass establishes exact-checkpoint integration of the released six-layer
native writer and fused-loss gradient path on one GPU. It does not validate
FSDP2 sharding, VeOmni distributed clipping, activation offload, formal
`ttt_chunk=4096`, 32K/128K memory, optimizer-state capacity, convergence,
checkpoint conversion or RULER. The full corpus and registered matched-arm
chain remain mandatory, and update admission remains sealed.
