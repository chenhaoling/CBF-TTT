# In-Place TTT baseline FSDP1 CPU-offload feasibility addendum

Status: **frozen before FSDP1 execution**  
Freeze date: 2026-10-16 (Asia/Shanghai)  
Parent plan: `IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md`  
Prior addendum: `IN_PLACE_TTT_BASELINE_CPU_OFFLOAD_ADDENDUM.md`  
Official source commit: `be2324829b0e91c8fd10a74d4b43714fde6676e1`

## 1. Registered reason

The first CPU-offload probe set `enable_fsdp_offload=true` with the official
FSDP2 mode, but the pinned VeOmni FSDP2 implementation did not consume or apply
that keyword. The no-TTT run exactly repeated the prior GPU OOM boundary. An
independent source inspection records that `parallelize_model_fsdp2` contains
neither `enable_fsdp_offload` handling nor an offload policy, whereas the
released `parallelize_model_fsdp1` constructs
`CPUOffload(offload_params=True)` and states that it offloads parameters,
gradients, and optimizer states.

This addendum registers one final current-host feasibility attempt using that
existing, unmodified FSDP1 path. It is a systems substitution and is excluded
from paper-exact claims.

## 2. Frozen probe

First run one 8K no-TTT optimizer step with:

- `data_parallel_mode=fsdp1`;
- `enable_full_shard=true`;
- `enable_fsdp_offload=true`;
- the same Qwen3-4B files and packed data hash as all prior probes;
- sequence length 8,192, global batch 2, microbatch 1, bf16, gradient
  checkpointing, AdamW, learning rate `5e-6`, and weight decay `0.1`.

If and only if no-TTT completes with finite loss and gradient norm and passes
the host-memory gate, run the matched native-TTT arm. It differs only in
`ttt_mode=true`; TTT layer, projection, target, learning-rate and chunk
settings remain frozen.

## 3. Measurements and gates

Reuse the V2 resource monitor and hashes. Require:

- exit code zero and a completed optimizer step;
- finite loss and gradient norm;
- identical data/model/addendum hashes between arms;
- at least 8 GiB minimum available host memory;
- no more than 3.5 GiB peak used swap;
- complete GPU, host, log, config and checkpoint sidecars.

Stop at the first failure. If both arms pass, this addendum authorizes only a
matched 16K FSDP1-offload resource pair. It does not authorize formal training,
32K/128K scaling, or metric claims.

## 4. Claim boundary

A pass demonstrates only that the released code can execute a mathematically
matched public-data training step through FSDP1 CPU offload on this shared
host. FSDP version, placement, throughput, numerical order and hardware differ
from the paper. A failure establishes that the current host offers no released,
preregistered path to the baseline and that an external-state change is needed.
