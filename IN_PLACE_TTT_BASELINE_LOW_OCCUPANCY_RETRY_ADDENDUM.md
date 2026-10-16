# In-Place TTT baseline low-occupancy retry addendum

Status: **frozen before low-occupancy retry**  
Freeze date: 2026-10-17 (Asia/Shanghai)  
Parent plan: `IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md`  
Prior allocator addendum: `IN_PLACE_TTT_BASELINE_ALLOCATOR_RETRY_ADDENDUM.md`  
Official source commit: `be2324829b0e91c8fd10a74d4b43714fde6676e1`

## 1. Registered reason

The allocator-only no-TTT retry completed forward and backward but failed on a
48-MiB AdamW `exp_avg_sq` allocation. At failure, unrelated processes occupied
2,380 MiB on physical GPU0 and 684 MiB on physical GPU1; the two training ranks
reported only 11.81 and 9.00 MiB free. Expandable segments reduced
reserved-but-unallocated memory to 29.17 and 26.54 MiB, establishing an
observed shared-capacity boundary rather than a remaining fragmentation fix.

At `2026-10-17 00:27:42 HKT`, a new read-only snapshot reported total device
usage of 812 and 670 MiB, with unrelated compute processes using 724 and
644 MiB. The roughly 1.6-GiB reduction on GPU0 is an external-state change and
justifies one new registered retry. No other user's process may be terminated,
paused, reniced, or modified.

## 2. Frozen preflight

Immediately before launch, save the full `nvidia-smi` device and compute-process
snapshot. Launch is allowed only when:

- total used memory is at most 1,024 MiB on each GPU;
- no process from an earlier baseline probe is alive;
- the official checkout is clean at
  `be2324829b0e91c8fd10a74d4b43714fde6676e1`;
- the parent-plan, model-config, packed-data and training-config hashes match
  the earlier allocator retry.

If occupancy rises above the threshold before launch, wait and take a new
snapshot; do not kill the unrelated process.

## 3. Frozen execution

Run the exact native FSDP2 8K no-TTT resource configuration from a new output
root with:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Retain the same Qwen files, packed data, global batch 2, microbatch 1, bf16,
gradient checkpointing, AdamW learning rate `5e-6`, weight decay `0.1`, FSDP2
full sharding, and no CPU offload. The external occupancy and output root are
the only changes relative to the failed allocator retry.

If and only if no-TTT completes one optimizer step with finite loss and
gradient norm, run the matched native-TTT arm from a second new output root.
The two configs must differ only in `ttt_mode`.

## 4. Gates and stopping

Each arm requires exit zero, finite loss and gradient norm, complete GPU/host
traces, exact dependency hashes, and a saved one-step checkpoint. Stop the arm
sequence on the first OOM, non-finite value, data/hash mismatch, missing
sidecar, or preflight violation.

This addendum authorizes no longer sequence and no formal pretraining. If both
arms pass, the next action is a separately frozen 16K resource step. If either
arm fails, do not tune model, optimizer, precision, batch size or sharding under
this addendum.

## 5. Claim boundary

This is a resource-feasibility test. It provides no efficacy result and does
not reproduce the paper's private data or reported RULER numbers. Gate work
remains paused until the baseline reproduction requirements are met.
