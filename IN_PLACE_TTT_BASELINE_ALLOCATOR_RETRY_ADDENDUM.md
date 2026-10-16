# In-Place TTT baseline allocator retry addendum

Status: **frozen before allocator retry**  
Freeze date: 2026-10-16 (Asia/Shanghai)  
Parent plan: `IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md`  
Official source commit: `be2324829b0e91c8fd10a74d4b43714fde6676e1`

## 1. Registered reason

The exact 8K FSDP2 probes failed while allocating 20--48 MiB AdamW state. The
tracebacks reported roughly 166--176 MiB reserved but unallocated and explicitly
suggested `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` when fragmentation
is material. Since those runs, one unrelated 684-MiB process on physical GPU1
has ended. Physical GPU0 remains shared by three unrelated processes totaling
about 2,380 MiB; physical GPU1 retains one unrelated process at about 684 MiB.

This addendum permits one allocator-only retry. It changes no model, writer,
optimizer, scheduler, dtype, sharding mode, data, sequence length, batch size,
or checkpoint setting. It does not remove or alter other users' processes.

## 2. Frozen execution

Run the exact native FSDP2 8K no-TTT resource configuration from a new output
root with only:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Record the allocator setting in metadata and retain the same Qwen config hash,
packed data hash, global batch 2, microbatch 1, bf16, gradient checkpointing,
AdamW learning rate `5e-6`, weight decay `0.1`, and no CPU offload.

If and only if no-TTT completes one optimizer step with finite loss and
gradient norm, run the matched native-TTT arm with the same allocator setting.
The two arm configs must differ only in `ttt_mode`.

## 3. Gates

Success requires exit zero, finite loss and gradient norm, complete GPU/host
traces, exact dependency hashes, and a saved one-step checkpoint. Stop on the
first OOM, non-finite value, data/hash mismatch, or incomplete sidecar.

This addendum authorizes no longer sequence and no formal training. A pass
only restores the parent plan's 8K resource ladder; a failure confirms that
the remaining GPU occupancy or absolute memory demand, rather than allocator
fragmentation alone, prevents execution.

## 4. Claim boundary

Allocator behavior is a systems detail. Neither a pass nor a failure is an
efficacy result. Paper-data, full-curriculum, checkpoint-conversion, and RULER
acceptance requirements remain unchanged, and update-gate work remains paused.
