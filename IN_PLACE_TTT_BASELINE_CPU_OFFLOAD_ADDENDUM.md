# In-Place TTT baseline CPU-offload feasibility addendum

Status: **frozen before CPU-offload execution**  
Freeze date: 2026-10-16 (Asia/Shanghai)  
Parent plan: `IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md`  
Official source commit: `be2324829b0e91c8fd10a74d4b43714fde6676e1`

## 1. Reason for this addendum

The preregistered, no-offload 8K probes reached full forward and backward for
both matched Qwen3-4B arms but OOMed during the first AdamW state allocation.
The shared two-RTX-5090 host also had unrelated CARLA processes using about
1.4--2.4 GiB. The parent plan correctly stopped the paper-oriented resource
ladder at this failure.

This addendum registers a separate engineering feasibility branch using
VeOmni's existing `enable_fsdp_offload=true`. CPU offload changes parameter and
gradient placement, wall time, and numerical execution details. It is not the
paper's reported systems configuration and cannot make a run paper-exact. It
does not overwrite or reinterpret the no-offload failure.

The purpose is narrower: determine whether the otherwise matched public-data
replication can execute on the currently available host while a larger,
exclusive GPU host is unavailable.

## 2. Frozen comparison

Run two one-optimizer-step 8K arms from the same base model and packed stream:

1. `cpu_offload_no_ttt`: native Qwen writer disabled;
2. `cpu_offload_ttt`: upstream native writer enabled every sixth layer.

The arms must keep identical model files, record order, data hash, sequence
length 8,192, global batch 2, microbatch 1, FSDP2 full sharding, bf16,
gradient checkpointing, AdamW, learning rate `5e-6`, weight decay `0.1`,
scheduler, seed, and save settings. The only algorithmic arm difference is
`ttt_mode`; both set `enable_fsdp_offload=true`.

TTT settings remain `ttt_layers=[0,6,12,18,24,30]`, `ttt_proj=true`,
`ttt_target=hidden_states`, `ttt_lr=3`, and `ttt_chunk=4096`.

## 3. Measurements

For each arm save:

- exact plan, addendum, config, runner, summarizer, data, model-config, and
  upstream hashes;
- exit status, loss, gradient norm, wall time, nominal tokens/second, and
  checkpoint bytes;
- 0.5-second per-GPU memory/utilization trace;
- 0.5-second host used/available RAM and used swap trace;
- full stdout/stderr and exact failure stage.

The server is shared. Unrelated process occupancy is recorded and is not
subtracted from monitored device peaks.

## 4. Gates and stopping rules

The 8K feasibility gate passes only if both arms:

- complete one optimizer step with finite loss and gradient norm;
- consume the same model/data hashes and nominal token count;
- retain at least 8 GiB host available memory throughout;
- use no more than 3.5 GiB swap;
- save complete resource sidecars.

Stop immediately on OOM, non-finite values, host available memory below 8 GiB,
swap above 3.5 GiB, data/hash mismatch, or incomplete monitoring. Do not kill or
modify unrelated server processes.

If both 8K arms pass, report the measured CPU-offload overhead and then run a
newly packed 16K one-step pair. A 32K pair is allowed only if 16K retains at
least 15% free GPU memory and at least 16 GiB host available memory. No 64K or
formal token-budget training is authorized by this addendum.

## 5. Claim boundary

A pass shows only that a matched public-data replication is executable with
CPU offload on this host. It does not reproduce the paper's training hardware,
private data, 32K/128K curriculum, Table 1 values, or inference clipping.

A failure strengthens the resource-transfer requirement. It is not evidence
against In-Place TTT efficacy. Update-admission experiments remain paused in
either case.
