# In-Place TTT baseline asymmetric-idle retry addendum

Status: **frozen before asymmetric-idle retry**  
Freeze date: 2026-10-17 (Asia/Shanghai)  
Parent plan: `IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md`  
Official source commit: `be2324829b0e91c8fd10a74d4b43714fde6676e1`

## 1. Registered reason

The prior low-occupancy 8K no-TTT run started with 724/644 MiB of unrelated
compute memory and failed during first AdamW state initialization. Rank 0's
20-MiB request had 18.44 MiB free; rank 1's 48-MiB request had 9.00 MiB free.

At `2026-10-17 20:54:52 HKT`, the external state changed asymmetrically:
physical GPU0 retained one 642-MiB CARLA process, while physical GPU1 had no
compute process and only 18 MiB total use. Total use was 730/18 MiB. Relative
to the registered failure, this releases about 82 MiB on rank 0 and 652 MiB on
rank 1, enough to make one new resource measurement informative.

No foreign process may be killed, paused, reniced or modified. The active
exclusive watcher remains ineligible while the CARLA process exists and must
not launch concurrently.

## 2. Frozen preflight and execution

Immediately before launch, save device and compute-process snapshots. Launch
is permitted only if:

- exactly two RTX 5090 devices are visible;
- GPU0 total use is at most 768 MiB and has at most one unrelated compute
  process using at most 672 MiB;
- GPU1 has no compute process and total use is at most 128 MiB;
- no earlier baseline training rank is alive;
- the official checkout is clean at the registered commit;
- the arm output root does not exist.

Run the exact native FSDP2 8K no-TTT configuration with the same Qwen files,
packed data, global batch 2, microbatch 1, bf16, gradient checkpointing, FP32
AdamW (`lr=5e-6`, `weight_decay=0.1`), no CPU offload, and:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

If and only if no-TTT completes one optimizer step with finite loss and
gradient norm, repeat the preflight and run matched native TTT from a second
new output root. The configs differ only in `ttt_mode`.

## 3. Stop and claim boundary

Stop on the first OOM, non-finite value, preflight violation, data/hash
mismatch or missing sidecar. Do not change model, optimizer, precision,
sharding, batch size or sequence length, and do not run a longer sequence or
formal training under this addendum.

This is a resource-feasibility probe, not an efficacy result. It does not
reproduce the private paper corpus or reported RULER values. Update-admission
and forgetting-gate experiments remain paused.
