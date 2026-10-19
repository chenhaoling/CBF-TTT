# Qwen3-4B-Base FSDP2 native-writer autograd audit addendum

Status: **frozen before persistent execution**

Freeze date: 2026-10-19 (Asia/Shanghai)

Parent: `IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md`

## 1. Purpose

The initialization audit proves that the formal FP32-meta/CUDA-DTensor path
creates the expected native writers. It does not execute the official fused
loss or prove that gradients cross the exact full-model FSDP2 graph. Run one
diagnostic two-backward audit before the full-corpus Level-A chain becomes
eligible.

This audit is diagnostic evidence only. It does not consume the partial public
corpus, perform an optimizer step, save a checkpoint, replace the corrected
Level-A gate or release any update-admission experiment.

## 2. Frozen identities and host

- Run only on `hku-gpu2`, hostname `user-Super-Server`.
- Expose exactly physical GPUs 0 and 1 as `CUDA_VISIBLE_DEVICES=0,1`.
- Require two RTX 5090 ranks and at least 28 GiB free per GPU before model
  construction.
- Require clean official commit
  `be2324829b0e91c8fd10a74d4b43714fde6676e1` and a clean committed tool
  checkout containing this frozen plan and the audit script.
- Revalidate every local model file against the passing identity for
  `Qwen/Qwen3-4B-Base` revision
  `906bfd4b4dc7f14ee4320094d8b41684abff8539`.

No other GPU host is permitted.

## 3. Frozen execution path

Use the same model construction and parallelization calls as
`tasks/train_torch.py`:

1. import the released `hf_models` registration before VeOmni;
2. initialize two-rank NCCL and VeOmni FSDP2 full sharding;
3. construct the model on meta with FP32 storage, mixed precision, gradient
   checkpointing and the exact checkpoint;
4. retain writer layers `[0,6,12,18,24,30]`, projection enabled, hidden-state
   target, `ttt_lr=3` and the formal `ttt_chunk=4096`;
5. run the released Liger fused causal-language-model loss inside VeOmni's
   activation-offload forward/backward contexts.

Each rank creates one deterministic 8192-token sequence from the vocabulary
size and rank. The two sequences differ across ranks. Record only SHA256
digests and scalar statistics; do not serialize token IDs, logits,
activations, gradients or model tensors.

## 4. Frozen two-backward diagnostic

The first backward starts from the exact loaded initialization. It must produce
a finite loss, finite positive global Conv5 gradient norm at all six writers,
and exactly zero global projection gradient norm at all six writers.

Then, under `no_grad`, update only each local Conv5 shard by
`conv -= 1e-3 * first_gradient`. This is a diagnostic perturbation rather than
an optimizer step. Clear gradients and repeat the same fused-loss backward.
The second pass must produce finite loss and finite positive global gradient
norms for both Conv5 and projection at every writer layer.

## 5. Pass conditions and stop rules

The audit passes only when all identity, source, device, model-shape, writer
configuration, Liger-registration, FSDP2-DTensor, gradient-checkpointing and
activation-offload conditions hold; both backward passes satisfy the frozen
gradient conditions; every rank agrees on the reduced scalar results; and no
optimizer or checkpoint is created. Any CUDA OOM, nonfinite value, missing
gradient, identity mismatch or failed condition remains a failure. Do not
reduce the sequence, change the chunk, weaken a threshold or retry under a new
configuration after observing the result.

## 6. Evidence and claim boundary

Record per-rank GPU identity, free and peak memory, wall time, two losses,
global writer-gradient norms, the number of nonzero Conv5 values after the
diagnostic perturbation, runtime versions and hashes of the exact source files.

A pass proves only full-model distributed autograd reachability for the native
writer under an 8K two-chunk diagnostic. It does not prove optimizer-state
capacity, gradient clipping, an optimizer step, checkpoint conversion,
32K/128K training, convergence or RULER quality. The corrected full-corpus
Level-A and every downstream baseline gate remain mandatory. Update admission
remains sealed.
