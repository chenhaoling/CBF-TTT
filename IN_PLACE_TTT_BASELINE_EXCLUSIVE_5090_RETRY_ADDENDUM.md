# In-Place TTT baseline exclusive-5090 retry addendum

Status: **frozen before watcher launch**  
Freeze date: 2026-10-17 (Asia/Shanghai)  
Parent plan: `IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md`  
Official source commit: `be2324829b0e91c8fd10a74d4b43714fde6676e1`

## 1. Registered reason

The low-occupancy 8K no-TTT retry still failed during the first FP32 AdamW
state initialization. Its preflight retained unrelated CARLA compute processes
using 724 and 644 MiB. At the two failures only 18.44 and 9.00 MiB remained,
so the experiment does not determine whether removing the last unrelated
compute allocations is sufficient.

The alternative `hku-a100` host is not available: its only A100-SXM4-80GB is
occupied by a long-running Qwen3-32B vLLM service under another account. No
foreign process on either host may be killed, paused, reniced, or modified.

## 2. Frozen watcher

One remote watcher may wait for at most 24 hours. It may launch only after two
checks separated by five seconds both establish:

- exactly two visible RTX 5090 devices;
- no process reported by `nvidia-smi --query-compute-apps`;
- total used memory at most 512 MiB on each device;
- no earlier output root for the arm;
- a clean official checkout at the registered commit.

The watcher polls once per minute, writes its latest timestamp, GPU state,
compute-process state and lifecycle status, and exits without training when the
24-hour limit expires. It never sends a signal to an existing process.

## 3. Frozen execution

On exclusive preflight, run the same native FSDP2 8K no-TTT configuration with
global batch 2, microbatch 1, bf16, gradient checkpointing, FP32 AdamW
(`lr=5e-6`, `weight_decay=0.1`), no CPU offload, the same Qwen files and packed
data, and:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

If and only if no-TTT exits zero with finite loss and gradient norm, wait for
the same exclusive condition again and run the matched native-TTT arm. The arm
configs differ only in `ttt_mode`; both use new output roots. Stop after the
first arm failure, missing sidecar, hash mismatch or non-finite result.

## 4. Gates and claim boundary

Each successful arm requires a complete optimizer step, exit zero, finite loss
and gradient norm, complete GPU/host traces, dependency hashes and a saved
one-step checkpoint. This addendum authorizes no 16K step or formal training.

The result is a resource-feasibility measurement. It cannot establish model
quality, exact paper-data reproduction or a RULER result. Update-admission and
forgetting-gate work remains paused.
