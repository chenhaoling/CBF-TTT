# Qwen3-4B-Base FSDP2 writer-initialization audit addendum

Status: **frozen before persistent execution**

Freeze date: 2026-10-19 (Asia/Shanghai)

Parent: `IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md`

## 1. Motivation

The single-GPU full-checkpoint preflight correctly remains failed because
BF16 `from_pretrained` on CPU generated a small number of exact-zero projection
diagonal entries and its active writer path was not numerically identical to a
direct-linear bypass. The formal trainer does not use that construction path:
it builds a FP32 meta model, applies FSDP2, materializes DTensors on CUDA, loads
the checkpoint and initializes the missing writer tensors through VeOmni's
post-load hook.

Run one exact-checkpoint, two-GPU initialization-only audit to determine the
runtime dtype, device and sparse structure produced by the actual formal
FSDP2/meta path. This audit does not revise or supersede the failed
single-GPU result.

## 2. Frozen host and identities

- Run only on `hku-gpu2`, hostname `user-Super-Server`.
- Expose exactly physical GPUs 0 and 1 as `CUDA_VISIBLE_DEVICES=0,1`.
- Require two visible RTX 5090 devices and at least 28 GiB free on each before
  model construction.
- Use clean official source commit
  `be2324829b0e91c8fd10a74d4b43714fde6676e1`.
- Use exact `Qwen/Qwen3-4B-Base` revision
  `906bfd4b4dc7f14ee4320094d8b41684abff8539` and revalidate every file against
  its passing identity manifest.

No other GPU host is permitted.

## 3. Frozen construction path

Use the same calls as `tasks/train_torch.py`:

1. set `MODELING_BACKEND=hf` and import `hf_models` before VeOmni;
2. initialize a two-rank NCCL process group and VeOmni FSDP2 parallel state;
3. call `build_foundation_model` with the exact model as config and weights,
   `torch_dtype=float32`, `init_device=meta`, and the registered writer fields;
4. call `build_parallelize_model` with FSDP2 full sharding, mixed precision,
   `init_device=meta`, and the same weights path;
5. inspect parameters immediately after VeOmni weight loading and missing-key
   initialization; do not run forward, backward or optimizer construction.

Writer fields are `[0,6,12,18,24,30]`, `ttt_proj=true`,
`ttt_target=hidden_states`, `ttt_lr=3`, and `ttt_chunk=4096`.

## 4. Pass conditions

The audit passes only when:

1. every host, device, source, tool and model identity check passes;
2. the model has 36 layers, hidden size 2560, and exactly the registered six
   TTT layers;
3. exactly twelve writer parameters exist and no unregistered layer contains
   a writer;
4. every writer parameter is a CUDA DTensor with FP32 local storage;
5. every global convolution has shape `[2560,1,5]`, is bias-free and exactly
   zero;
6. every global projection has shape `[2560,2560]`, is bias-free, contains
   exactly 2560 finite nonzero diagonal values and no nonzero off-diagonal
   value;
7. both ranks report finite resource measurements and the reconstructed global
   values agree;
8. no forward, optimizer step or checkpoint is executed or saved.

Any failure remains a failure. Do not change initialization or weaken a gate
after observing the result.

## 5. Evidence and claim boundary

Record per-rank GPU identity/free memory/peak memory, global and local writer
shapes, dtypes/devices, structural counts, runtime versions, source hashes and
wall time. Serialize no model tensor, token, activation, logit or gradient.

A pass proves only the exact formal FSDP2/meta initialization structure. It
does not erase the single-GPU BF16 computation-path difference, execute a
loss, prove optimizer capacity, validate activation offload, train on 32K or
128K data, convert a checkpoint or score RULER. The corrected Level-A run and
all downstream gates remain mandatory. Update admission remains sealed.
