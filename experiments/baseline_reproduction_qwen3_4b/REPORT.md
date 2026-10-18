# In-Place TTT Qwen3-4B baseline reproduction report

Date: 2026-10-18<br>
Frozen plan: [`IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md`](../../IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md)  
Plan SHA256: `fbce7e0e52c495467c5b6383242a7350cf16042d61af8795309b9d6eda021fd3`  
Official source commit: `be2324829b0e91c8fd10a74d4b43714fde6676e1`

## 1. Conclusion

The released-code Level-A execution path is complete, but the paper-scale
baseline has **not** been reproduced yet.

- A clean official checkout and isolated official-version environment load on
  `hku-gpu2`.
- Under a truly exclusive 2xRTX 5090 allocation, both matched 8K arms complete
  one finite optimizer step, write DCP checkpoints, convert through the
  official DCP-to-HF script, reload with the official TTT-aware inference
  registration, and complete all 13 RULER-4K smoke tasks through the audited
  original-Qwen tokenizer path.
- no-TTT and TTT peak at 31,620/31,594 MiB and 31,920/31,894 MiB. The minimum
  remaining fraction is 2.107%, below the preregistered 5% resource gate, so
  the 2x5090 length ladder stops at 8K and does not attempt 16K.
- Two preregistered offload diagnostics also stop: the pinned VeOmni FSDP2
  path ignores `enable_fsdp_offload`, while the released FSDP1 path does
  offload but exhausts the 62-GiB host RAM and 4-GiB swap and is SIGKILLed.
- A final allocator-only retry with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` still OOMs at the first
  AdamW state allocation. The run peaks at 32,092/32,103 MiB and leaves only
  11.81/9.00 MiB free at the failing allocation, so fragmentation is not the
  remaining blocker on the shared host.
- A separately frozen low-occupancy retry also fails at the same AdamW state
  initialization. Its preflight records only 812/670 MiB total device use, but
  the run still peaks at 32,086/32,103 MiB.
- A final asymmetric-idle retry starts with GPU0/GPU1 at 730/18 MiB. GPU1 has
  no compute process, while GPU0 retains one 642-MiB CARLA process. The no-TTT
  rank on GPU0 still OOMs on a 48-MiB AdamW allocation with 40.44 MiB free.
  One idle peer device therefore cannot rescue a rank whose own device is at
  the capacity boundary; the later exclusive success confirms that the prior
  blocker was the final external allocation.
- The converted no-TTT and TTT checkpoints each score `76.9231` on the
  one-example-per-task smoke. This validates execution and score extraction
  only and is not an efficacy estimate comparable to the paper's 100-example
  result.
- A Transformers 4.57.3 false-positive Mistral-regex warning in converted
  local Qwen configs was diagnosed and closed. The replacement smokes bind
  the byte-identical original Qwen tokenizer, explicitly keep
  `fix_mistral_regex=false`, and emit zero warnings in both arms.
- A separately frozen 32K public-data construction pilot now passes for both
  FineWeb-Edu and LongCrawl64. Each source produces 8 records and 262,144
  VeOmni training tokens; independent audits re-tokenize every record and
  verify source, tokenizer, plan, code, provenance, artifact and ordered-token
  hashes. This validates the construction path only and starts no training.
- The formal Level-B public-corpus protocol is now frozen. A revision-pinned
  inventory resolves exactly 140 FineWeb-Edu `sample/100BT` files and all 256
  numerically ordered LongCrawl64 shards; its independent audit passes all
  seven checks.
- The required continuous two-stage pilot passes: 64 records/source at 32K
  followed from the same source cursor by 8 records/source at 128K. It contains
  6,291,456 training tokens, passes all ten independent audit checks, and peaks
  at 1,002.52 MiB host RSS. This closes the packing and no-replay gate.
- Four formal 32K/128K × no-TTT/TTT configs are implemented and statically
  audited. The full 34,999,369,728-token public corpus build is currently
  running in an isolated CPU-only tmux job; no formal training has started.
- The user now requires every GPU run to remain on `hku-gpu2`. A separately
  frozen two-5090 path enables the released VeOmni activation-offload hook,
  retains FSDP2 and the native writer, and registers an audited 8K→16K→32K
  resource ladder plus a matched 98,566,144-token end-to-end tier. No new
  optimizer step has run under this path while the corpus builder consumes
  host RAM.
- Inspection of the pinned VeOmni loader found its mandatory deterministic
  buffer-10,000 shuffle. A training addendum now freezes seed 42, worker and
  prefetch settings, implementation hashes and a measured 3-TiB paired-run
  storage gate before any Level-B optimizer step.
- The paper's exact continual-pretraining corpus and several stage settings
  are not public. FineWeb-Edu plus LongCrawl64 remains a declared public-data
  substitute and cannot be called the original paper dataset.

Consequently, Level A passes and both Level-B data-pipeline pilots pass. The
full Level-B corpus build is active, but matched training has not started.
The former 8×80-GB execution target remains documented, while the active
execution constraint is now the audited `hku-gpu2`-only path. Level C remains
unavailable because the paper corpus and parts of the recipe are private.
Update-admission and forgetting-gate exploration remains paused.

## 2. Paper target and claim boundary

The primary target is the paper's drop-in Qwen3-4B experiment:

| Stage | Tokens | Sequence length | Optimizer |
|---|---:|---:|---|
| 1 | about 20B | 32,768 | AdamW, lr `5e-6`, wd `0.1` |
| 2 | about 15B | 131,072 | same, with YaRN |

TTT is inserted every sixth layer, the causal convolution kernel is 5, and the
paper reports RULER at 4K through 256K. The release omits the private mixture,
preprocessing scripts, a complete 128K recipe, and a configurable
implementation of the paper's inference update clip `tau=1e-5`. Therefore the
current work can establish released-code reproduction and public-data
replication, but cannot establish exact paper-number reproduction.

## 3. Isolation and implementation

The baseline uses a detached, clean checkout at
`/home/ctj/in_place_ttt_official_be232482`. No CBF controller, modified Qwen
file, modified native writer, or prior gate checkpoint enters this path.

New opt-in files are:

- `configs/baseline_reproduction/qwen3_4b_resource_{no_ttt,ttt}.yaml`;
- `scripts/setup_in_place_ttt_official_env.sh`;
- `scripts/setup_in_place_ttt_official_eval_env.sh`;
- `scripts/run_in_place_ttt_baseline_resource_pilot.sh`;
- `scripts/run_in_place_ttt_baseline_asymmetric_idle_retry.sh`;
- `scripts/summarize_in_place_ttt_baseline_resource.py`;
- `scripts/convert_and_audit_in_place_ttt_baseline_checkpoints.sh`;
- `scripts/run_in_place_ttt_converted_checkpoint_smokes.sh`;
- `scripts/audit_in_place_ttt_baseline_exclusive.py`;
- `IN_PLACE_TTT_BASELINE_TOKENIZER_AUDIT_ADDENDUM.md`;
- `IN_PLACE_TTT_BASELINE_PUBLIC_DATA_PILOT_ADDENDUM.md`;
- `scripts/build_in_place_ttt_public_data_pilot.py`;
- `scripts/audit_in_place_ttt_public_data_pilot.py`;
- `IN_PLACE_TTT_BASELINE_PUBLIC_CORPUS_PLAN.md`;
- `scripts/inventory_in_place_ttt_public_corpus.py`;
- `scripts/audit_in_place_ttt_public_corpus_inventory.py`;
- `scripts/build_in_place_ttt_public_corpus.py`;
- `scripts/audit_in_place_ttt_public_corpus.py`;
- `configs/baseline_reproduction/qwen3_4b_public_stage{1,2}_{no_ttt,ttt}.yaml`;
- `IN_PLACE_TTT_BASELINE_LEVEL_B_RULER_ADDENDUM.md`;
- `eval_config/baseline_reproduction/qwen3_4b_formal_ruler.py`;
- `scripts/run_in_place_ttt_level_b_ruler_single.sh`;
- `scripts/summarize_in_place_ttt_level_b_ruler.py`;
- `scripts/audit_in_place_ttt_level_b_ruler.py`;
- `scripts/run_in_place_ttt_level_b_ruler_matrix.sh`;
- `scripts/summarize_in_place_ttt_level_b_ruler_matrix.py`;
- `scripts/audit_in_place_ttt_level_b_ruler_matrix.py`;
- `scripts/audit_in_place_ttt_level_b_configs.py`;
- `scripts/run_in_place_ttt_level_b_preflight.sh`;
- `scripts/audit_in_place_ttt_level_b_preflight.py`;
- `IN_PLACE_TTT_BASELINE_LEVEL_B_TRAINING_ADDENDUM.md`;
- `scripts/run_in_place_ttt_level_b_training.sh`;
- `scripts/audit_in_place_ttt_level_b_training.py`;
- `IN_PLACE_TTT_BASELINE_2X5090_EXECUTION_ADDENDUM.md`;
- `configs/baseline_reproduction/qwen3_4b_2x5090_100m_stage{1,2}_{no_ttt,ttt}.yaml`;
- `scripts/audit_in_place_ttt_2x5090_configs.py`;
- `scripts/audit_in_place_ttt_2x5090_resource.py`;
- `scripts/watch_and_run_in_place_ttt_2x5090_activation.sh`;
- `IN_PLACE_TTT_BASELINE_2X5090_RESOURCE_INPUT_ADDENDUM.md`;
- `scripts/build_in_place_ttt_2x5090_resource_inputs.py`;
- `scripts/audit_in_place_ttt_2x5090_resource_inputs.py`;
- `scripts/run_in_place_ttt_2x5090_stage1_ladder.sh`;
- `scripts/audit_in_place_ttt_2x5090_stage1_ladder.py`;
- `scripts/watch_and_run_in_place_ttt_2x5090_stage1_ladder.sh`;
- `IN_PLACE_TTT_BASELINE_2X5090_STAGE1_TIMING_PLAN.md`;
- `scripts/run_in_place_ttt_2x5090_stage1_timing.sh`;
- `scripts/audit_in_place_ttt_2x5090_stage1_timing.py`;
- `scripts/watch_and_run_in_place_ttt_2x5090_stage1_timing.sh`;
- `IN_PLACE_TTT_BASELINE_2X5090_100M_STAGE1_PLAN.md`;
- `scripts/run_in_place_ttt_2x5090_100m_stage1.sh`;
- `scripts/audit_in_place_ttt_2x5090_100m_stage1.py`;
- `IN_PLACE_TTT_BASELINE_2X5090_STAGE2_RESOURCE_PLAN.md`;
- `scripts/run_in_place_ttt_2x5090_stage2_ladder.sh`;
- `scripts/audit_in_place_ttt_2x5090_stage2_ladder.py`;
- `IN_PLACE_TTT_BASELINE_2X5090_STAGE2_TIMING_PLAN.md`;
- `scripts/run_in_place_ttt_2x5090_stage2_timing.sh`;
- `scripts/audit_in_place_ttt_2x5090_stage2_timing.py`;
- `IN_PLACE_TTT_BASELINE_2X5090_100M_STAGE2_PLAN.md`;
- `scripts/run_in_place_ttt_2x5090_100m_stage2.sh`;
- `scripts/audit_in_place_ttt_2x5090_100m_stage2.py`;
- `IN_PLACE_TTT_BASELINE_2X5090_RULER_PLAN.md`;
- `eval_config/baseline_reproduction/qwen3_4b_2x5090_ruler.py`;
- `scripts/run_in_place_ttt_2x5090_ruler_single.sh`;
- `scripts/summarize_in_place_ttt_2x5090_ruler.py`;
- `scripts/audit_in_place_ttt_2x5090_ruler.py`;
- `scripts/audit_in_place_ttt_2x5090_ruler_resource.py`;
- `scripts/run_in_place_ttt_2x5090_ruler_resource_ladder.sh`;
- `scripts/audit_in_place_ttt_2x5090_ruler_resource_ladder.py`;
- `scripts/run_in_place_ttt_2x5090_ruler_matrix.sh`;
- `scripts/summarize_in_place_ttt_2x5090_ruler_matrix.py`;
- `scripts/audit_in_place_ttt_2x5090_ruler_matrix.py`;
- `eval_config/baseline_reproduction/qwen3_4b_base_4k_smoke.py`;
- `scripts/run_in_place_ttt_ruler_smoke.sh`;
- `scripts/summarize_in_place_ttt_ruler_smoke.py`;
- the focused summary and public-data packing tests.

The two resource configurations differ only in `ttt_mode`. Both start from
the same Qwen files, consume the same packed pilot stream, use sequence length
8,192, global batch 2, FSDP2 full sharding, gradient checkpointing, bf16,
AdamW, lr `5e-6`, and weight decay `0.1`. CPU offload is disabled as frozen.

The environment is recorded in
[`environment_manifest.json`](environment_manifest.json). `byted-wandb` is not
published on public PyPI, so public `wandb==0.30.0` supplies the imported API.
OpenCompass uses the explicit compatible pair `numpy==1.26.4` and
`pyarrow==25.0.1`.

## 4. Training resource pilot

The packed pilot has 16 records, balanced 8/8 between FineWeb-Edu and
LongCrawl64, each exactly 8,192 tokenizer tokens including EOS. Both arms bind
the same data hash
`2eb03326ec87bc831ee1441e812bb5bef843f8e906656787cd2ddafc69d9d823` and
the same Qwen config hash
`8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a`.

| Arm | Exit | Failure stage | Wall time | GPU0 peak | GPU1 peak |
|---|---:|---|---:|---:|---:|
| no TTT | 1 | AdamW `exp_avg_sq` allocation | 25.47 s | 32,078 MiB | 32,102 MiB |
| native TTT | 1 | AdamW `exp_avg` allocation | 22.94 s | 32,076 MiB | 32,092 MiB |

Both arms completed forward and backward. No loss/throughput is reported
because neither completed an optimizer step. Unrelated CARLA processes used
about 1.4--2.4 GiB across the devices during the probes. Thus the strict result
is that the **current shared host configuration** is infeasible. Exclusive
GPUs could change the 8K boundary and would require a new registered rerun;
they would not justify extrapolating to 32K/128K without measurement.

The frozen rule stops the length ladder at the first OOM and forbids silently
enabling CPU offload. Therefore 16K, 32K, 64K and formal 20B/35B training were
not launched.

Compact evidence:

- [`remote/resource_8k_no_ttt/summary.json`](remote/resource_8k_no_ttt/summary.json)
- [`remote/resource_8k_ttt/summary.json`](remote/resource_8k_ttt/summary.json)
- full logs and 0.5-second GPU traces in the same directories.

### Registered offload diagnostics

After recording the parent-plan failure, two separate addenda tested whether
the released code offered a current-host engineering path. These results do
not alter the no-offload baseline conclusion.

| Profile | Result | GPU0/GPU1 peak | Host minimum available | Peak used swap |
|---|---|---:|---:|---:|
| FSDP2 + `enable_fsdp_offload=true` | flag is a no-op; repeated GPU OOM | 32,078 / 32,102 MiB | 46,800 MiB | 660 MiB |
| FSDP1 + CPU offload | rank SIGKILL during first step | 15,782 / 7,008 MiB | 358 MiB | 4,096 MiB |

The FSDP2 run logged the flag as true, but independent inspection of pinned
VeOmni file hash
`305ab376f337d1d06421ec1096935be2964e79b1e34bc2f16e9d2e4ef81dddef`
found that `parallelize_model_fsdp2` never consumes the keyword and creates no
offload policy. Its GPU peaks and allocation traceback reproduce the original
no-offload run.

The FSDP1 function does construct `CPUOffload(offload_params=True)` and the
runtime confirms that parameters, gradients and optimizer state were moved.
GPU memory fell substantially, but monitored host use peaked at 63,392 MiB,
available RAM fell to 358 MiB, all 4,096 MiB swap was consumed, and rank 1
received SIGKILL. This violates both frozen host gates. Per the addendum, the
matched TTT arm and all larger lengths were not launched.

Evidence:

- [`IN_PLACE_TTT_BASELINE_CPU_OFFLOAD_ADDENDUM.md`](../../IN_PLACE_TTT_BASELINE_CPU_OFFLOAD_ADDENDUM.md)
- [`IN_PLACE_TTT_BASELINE_FSDP1_OFFLOAD_ADDENDUM.md`](../../IN_PLACE_TTT_BASELINE_FSDP1_OFFLOAD_ADDENDUM.md)
- [`remote/resource_8k_no_ttt_cpu_offload_noop/summary.json`](remote/resource_8k_no_ttt_cpu_offload_noop/summary.json)
- [`remote/resource_8k_no_ttt_cpu_offload_noop/veomni_offload_diagnostic.json`](remote/resource_8k_no_ttt_cpu_offload_noop/veomni_offload_diagnostic.json)
- [`remote/resource_8k_no_ttt_fsdp1_cpu_offload/summary.json`](remote/resource_8k_no_ttt_fsdp1_cpu_offload/summary.json)

### Registered allocator-only retry

The allocator addendum permitted exactly one new native FSDP2 no-TTT run with
all model, data, optimizer, sharding, batch and sequence settings unchanged and
only `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` added. The run used a
new output root and bound the frozen addendum SHA256
`f648cc8bf627c98e88404d92a211c678da0939385db05335910275495a955c98`.

| Result | Failure stage | Wall time | GPU0/GPU1 peak | Free memory reported at OOM |
|---|---|---:|---:|---:|
| exit 1 | AdamW `exp_avg_sq`, 48-MiB allocation | 24.88 s | 32,092 / 32,103 MiB | 11.81 / 9.00 MiB |

The allocator reduced PyTorch's reserved-but-unallocated memory to
29.17/26.54 MiB, below the failed 48-MiB request. At that instant the training
rank used about 28.91 GiB on GPU0 and 29.97 GiB on GPU1, while unrelated CARLA
processes occupied 2,380 MiB on GPU0 and 684 MiB on GPU1. This is an absolute
capacity boundary under the observed shared occupancy. Per the addendum, the
matched TTT arm was not launched.

The post-exit `nvidia-smi` state of 2,479/710 MiB is not the training footprint:
the failed ranks released roughly 29--30 GiB before that snapshot. The retained
0.5-second GPU trace records the runtime peak.

Evidence:

- [`IN_PLACE_TTT_BASELINE_ALLOCATOR_RETRY_ADDENDUM.md`](../../IN_PLACE_TTT_BASELINE_ALLOCATOR_RETRY_ADDENDUM.md)
- [`remote/resource_8k_no_ttt_allocator_retry/summary.json`](remote/resource_8k_no_ttt_allocator_retry/summary.json)
- full traceback and GPU/host traces in the same directory.

### Registered low-occupancy retry

The next-day external state materially changed: GPU0's unrelated occupancy
fell from 2,380 MiB to 724 MiB, and GPU1 held 644 MiB. The separately frozen
[`IN_PLACE_TTT_BASELINE_LOW_OCCUPANCY_RETRY_ADDENDUM.md`](../../IN_PLACE_TTT_BASELINE_LOW_OCCUPANCY_RETRY_ADDENDUM.md)
therefore permitted one more identical no-TTT run. Its addendum SHA256 is
`0824292000f7e06b66dbed3861a7fa3426c47406bfdc811bf2acc0ef34950b10`.

The first preflight command stopped before launch because its `pgrep` pattern
matched the checking shell itself. No output root or tmux session was created.
The corrected self-excluding pattern then recorded 812/670 MiB total device
use, found no earlier baseline rank, and launched from a new output root.

| Result | Failure stage | Wall time | GPU0/GPU1 peak | Preflight total use |
|---|---|---:|---:|---:|
| exit 1 | AdamW state initialization | 22.69 s | 32,086 / 32,103 MiB | 812 / 670 MiB |

Rank 0 failed on a 20-MiB allocation with 18.44 MiB free; rank 1 failed on a
48-MiB allocation with 9.00 MiB free. Reserved-but-unallocated memory was only
21.54/19.04 MiB. The result independently rejects fragmentation as the main
remaining explanation and shows that the released configuration is infeasible
while even these small CARLA allocations remain. It does **not** prove that a
fully exclusive 32-GB device pair fails: removing 724/644 MiB could still move
the optimizer-state boundary. Per the frozen stop rule, matched TTT was not
launched.

Evidence:

- [`remote/low_occupancy_retry/preflight_gpu.csv`](remote/low_occupancy_retry/preflight_gpu.csv)
- [`remote/low_occupancy_retry/preflight_processes.csv`](remote/low_occupancy_retry/preflight_processes.csv)
- [`remote/low_occupancy_retry/resource_8k_no_ttt/summary.json`](remote/low_occupancy_retry/resource_8k_no_ttt/summary.json)
- full log and GPU/host traces in the same directory.

### Registered asymmetric-idle retry

At `2026-10-17 20:57:21 HKT`, physical GPU1 became compute-process-free and
used only 18 MiB, while GPU0 retained one 642-MiB CARLA process and used 730
MiB total. The frozen
[`IN_PLACE_TTT_BASELINE_ASYMMETRIC_IDLE_RETRY_ADDENDUM.md`](../../IN_PLACE_TTT_BASELINE_ASYMMETRIC_IDLE_RETRY_ADDENDUM.md)
permitted one identical no-TTT measurement under this materially different
state. Its SHA256 is
`3b4a112b55172fcf1b3e46dfe668866a05db0110d5eb7790b1ef6d9ba9d2a152`.

The atomic runner saved the launch snapshot, verified the official checkout,
and started at `2026-10-17 20:58:55 HKT`. It stopped after 21.83 seconds at the
first failing arm:

| Result | Failure stage | GPU0/GPU1 peak | Preflight total use |
|---|---|---:|---:|
| exit 1 | AdamW state initialization | 32,064 / 31,594 MiB | 730 / 18 MiB |

Rank 0 requested 48 MiB with only 40.44 MiB free. Its PyTorch
reserved-but-unallocated memory was 14.04 MiB, so allocator fragmentation is
again too small to explain the missing capacity. GPU1 did not contain an
unrelated compute process, but FSDP ranks allocate locally and cannot borrow
that device's remaining memory for rank 0. This proves that freeing only one
of the two cards is insufficient for the frozen run while the other card
retains 642 MiB. It does not prove that two fully exclusive 5090s fail. The
matched TTT arm was not launched, as required by the stop rule.

Evidence:

- [`remote/asymmetric_idle_retry/preflight_no_ttt_gpu.csv`](remote/asymmetric_idle_retry/preflight_no_ttt_gpu.csv)
- [`remote/asymmetric_idle_retry/preflight_no_ttt_processes.csv`](remote/asymmetric_idle_retry/preflight_no_ttt_processes.csv)
- [`remote/asymmetric_idle_retry/resource_8k_no_ttt/summary.json`](remote/asymmetric_idle_retry/resource_8k_no_ttt/summary.json)
- full log, metadata, and 0.5-second GPU/host traces in the same directory.

### Exclusive-5090 matched run (completed)

Because the low-occupancy run did not remove the final 724/644 MiB compute
allocations, a third addendum freezes the only remaining informative 5090
measurement: a truly compute-process-free retry. The opt-in watcher requires
two checks five seconds apart with no `nvidia-smi` compute process and at most
512 MiB total use on each of exactly two RTX 5090 devices. It polls every 60
seconds, waits at most 24 hours, and never signals another process.

The watcher started in remote tmux session
`baseline_exclusive_5090_watcher` at `2026-10-17 00:38:15 HKT`. At
`21:18:54 HKT`, the final CARLA process exited. Two consecutive preflights
recorded no compute process and 44/18 MiB total use, so the watcher ran no-TTT
and then repeated the exclusive preflight before matched TTT. The watcher
SHA256 is
`5e011820886b9c67814c7497d73b5658fe9e3722dd6f7adf07a9445f297aae5c`;
the frozen addendum SHA256 is
`dd492e41d5f27796a469f187ed4d37eba665cd0fa356c12b2c156ac055a35954`.

| Arm | Exit | Loss | Gradient norm | Wall | Tokens/s | GPU0/GPU1 peak |
|---|---:|---:|---:|---:|---:|---:|
| no TTT | 0 | 2.7364 | 11.2364 | 68.48 s | 239.25 | 31,620 / 31,594 MiB |
| native TTT | 0 | 2.7364 | 4,471.9336 | 87.36 s | 187.55 | 31,920 / 31,894 MiB |

Both DCP writes completed. The TTT arm is 21.61% slower by the one-step
end-to-end throughput ratio and has a 397.99x larger pre-clipping gradient
norm. `max_grad_norm=1.0` remains enabled, so this observation is a stability
risk for longer training rather than proof of divergence.

The raw trace independently gives a minimum free fraction of 2.107% at the
TTT peak. This is below the frozen 5% ladder condition. The run therefore
stops before 16K; neither 32K nor 64K is permitted on this host. The result
also resolves the earlier shared-host ambiguity: 8K is feasible only when the
two 32-GB devices are effectively exclusive.

The official converter produced five bfloat16 safetensors shards per arm.
no-TTT contains 399 model tensors (8.823 GB) and no writer tensors; TTT
contains 411 model tensors (8.902 GB), including the expected 12 native
`ttt_proj/ttt_conv` tensors for six valid TTT layers. A second full SHA256 pass
verified all 30 model-asset and weight files.

Evidence:

- [`remote/exclusive_5090_completed/independent_audit.json`](remote/exclusive_5090_completed/independent_audit.json)
- [`remote/exclusive_5090_completed/resource_8k_no_ttt/summary.json`](remote/exclusive_5090_completed/resource_8k_no_ttt/summary.json)
- [`remote/exclusive_5090_completed/resource_8k_ttt/summary.json`](remote/exclusive_5090_completed/resource_8k_ttt/summary.json)
- [`remote/exclusive_5090_completed/hf_integrity_check.txt`](remote/exclusive_5090_completed/hf_integrity_check.txt)
- full preflights, logs, traces, conversion audits, and weight manifests in
  [`remote/exclusive_5090_completed/`](remote/exclusive_5090_completed/).

### Public-data 32K construction pilot (completed)

Before implementation, the data-only protocol was frozen in
[`IN_PLACE_TTT_BASELINE_PUBLIC_DATA_PILOT_ADDENDUM.md`](../../IN_PLACE_TTT_BASELINE_PUBLIC_DATA_PILOT_ADDENDUM.md)
(commit `09741d39`, SHA256
`632ab39cdb015e57d082cb9f56d034f5484bd32316cec459f7988c776db2af83`).
It pins FineWeb-Edu revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, LongCrawl64 revision
`8e7af027525b3597852746ee11e4baa0e98492d7`, the original Qwen tokenizer,
8 records per source and 32,768 VeOmni tokens per record. Training is forbidden
in this pilot.

The opt-in builder writes `data.jsonl`, plaintext-free `provenance.jsonl` and
`manifest.json` below a new staging directory, then publishes the directory
atomically. It refuses existing final or partial paths. Each source document
boundary receives an EOS. Every candidate row is decoded and independently
re-encoded to exactly 32,767 content tokens; VeOmni adds the terminal EOS.
Per-record and ordered training-token hashes use the resulting IDs, while
provenance stores only document hashes, fixed source positions and consumed
token spans.

The first FineWeb attempt produced all 8 rows but then exited 134 during the
Datasets/PyArrow fragment reader's interpreter shutdown. Its runner correctly
stopped before audit and before LongCrawl. The failed directory and log are
retained and contribute no passing result. The repair replaces the Datasets
fragment path with a pinned Hugging Face parquet range reader backed by
`pyarrow.parquet.ParquetFile.iter_batches`, explicitly closes the source
generator, and binds the publisher LFS metadata. The repaired builder SHA256
is `e405d9b7e5b6a2c624b8cfe198bd93f0163d032e9cd902488127d1f29d9c320b`.

| Source | Exit/audit | Records | Training tokens | Documents | Synthetic EOS | Build wall | Peak host RSS |
|---|---|---:|---:|---:|---:|---:|---:|
| FineWeb-Edu | 0 / pass | 8 | 262,144 | 283 | 0 | 127.60 s | 790.46 MiB |
| LongCrawl64 | 0 / pass | 8 | 262,144 | 6 | 0 | 15.86 s | 783.07 MiB |

FineWeb binds publisher file `sample/10BT/000_00000.parquet`, size
2,152,819,114 bytes and LFS SHA256
`b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871`.
LongCrawl binds local publisher shard `train/0-of-256.parquet`, size
10,903,312,345 bytes and SHA256
`bd89e68512e0465957596711d1805e91d0bb1d097ecc44c7ddf5c81d636ba1d3`.
The independent auditor imports no builder code. For each source it re-hashes
the source binding, plan, builder and tokenizer assets; re-tokenizes every
output row; recomputes record, ordered-stream, document-stream and artifact
hashes; checks span continuity and rejects any `text` or `content_split` key
inside provenance. All 12 checks pass for both sources. GPU memory remained at
the idle 46/18 MiB because this stage is CPU-only.

Evidence:

- [`remote/public_data_32k_pilot/in_place_ttt_public_data_pilot_20261017_r2/summary.json`](remote/public_data_32k_pilot/in_place_ttt_public_data_pilot_20261017_r2/summary.json)
- [`remote/public_data_32k_pilot/in_place_ttt_public_data_pilot_20261017_r2/fineweb/independent_audit.json`](remote/public_data_32k_pilot/in_place_ttt_public_data_pilot_20261017_r2/fineweb/independent_audit.json)
- [`remote/public_data_32k_pilot/in_place_ttt_public_data_pilot_20261017_r2/longcrawl/independent_audit.json`](remote/public_data_32k_pilot/in_place_ttt_public_data_pilot_20261017_r2/longcrawl/independent_audit.json)
- [`remote/public_data_32k_pilot/local_archive_verification.json`](remote/public_data_32k_pilot/local_archive_verification.json)
- the failed attempt, repaired run, runners, logs, range probe, transfer archive
  and archive SHA are retained in
  [`remote/public_data_32k_pilot/`](remote/public_data_32k_pilot/).

This first pilot alone did not freeze the final 20B/15B public mixture or
validate 128K packing. Those two data-pipeline requirements are addressed by
the following frozen Level-B protocol; neither result reproduces the private
paper corpus.

### Formal public-corpus inventory and continuous 32K→128K pilot

Before inventory or full-scale construction, the protocol was frozen in
[`IN_PLACE_TTT_BASELINE_PUBLIC_CORPUS_PLAN.md`](../../IN_PLACE_TTT_BASELINE_PUBLIC_CORPUS_PLAN.md)
(current SHA256
`c5019a28b9fa7fa27052ddf1d69eb029a2ec5b3b6446f1527e9ceb7db1a12bab`).
It replaces the insufficient nested FineWeb `sample-10BT` pilot source with
the same publisher revision's `sample-100BT`, fixes exact whole-record budgets,
requires continuous source cursors between stages, and defines a public-data
matched replication rather than an exact private-corpus reproduction.

The inventory and its separate auditor resolve publisher metadata without
downloading the multi-terabyte source corpus:

| Source | Files | Ordered compressed bytes | Ordered-file-list SHA256 |
|---|---:|---:|---|
| FineWeb-Edu `sample/100BT` | 140 | 286,394,522,604 | `4f4e9dc42766053b0b35d1898e1279d980fc2ce813382e5a99f1c4bafb6d9381` |
| LongCrawl64 | 256 | 2,878,733,748,672 | `6577d3d3afe095572dc0893ba8077c9ca284a1ed5d5b522e70d053524016d1f0` |

Inventory SHA256 is
`c7e6761dc5d27013c09d6002de5c3e2b2185492b18100fba4b966c855debf7df`;
audit SHA256 is
`32daacb7eee2c9536ab8c06c01ac7c423d429f04cdf512a958f9ffd201487507`.
All seven inventory checks pass, including the exact file counts, revisions,
LFS/blob metadata, tokenizer binding and LongCrawl numeric order.

The continuous pilot then uses one persistent packer per source across both
stages. Stage 2 begins from the exact pending tokens and document cursor saved
after Stage 1; it does not rewind, discard or replay source text.

| Stage | Records/source | Combined records | Training tokens | Wall | FineWeb tok/s | LongCrawl tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 32K | 64 | 128 | 4,194,304 | 35.15 s | 103,034 | 144,540 |
| 128K | 8 | 16 | 2,097,152 | 7.31 s | 249,750 | 353,273 |

The complete builder takes 44.32 seconds and reports peak host RSS 1,002.52
MiB. The independent auditor re-tokenizes all 144 rows and passes all ten
checks: source order, record length, continuous spans, stage transition,
artifact/token/curriculum hashes, plaintext-free provenance, inventory,
tokenizer, code/plan and resource bindings. Synthetic EOS count is zero. GPUs
remain idle at 46/18 MiB. The copied 9,719,747-byte archive has SHA256
`29484d830dcc44ca225514adf998ef01e95d2217a9e564b0dcc10c6fdb431a1a`;
local verification re-hashes every output successfully.

Evidence:

- [`remote/public_corpus_inventory/`](remote/public_corpus_inventory/)
- [`remote/public_corpus_pilot/local_verification.json`](remote/public_corpus_pilot/local_verification.json)
- [`remote/public_corpus_pilot/in_place_ttt_public_corpus_pilot_20261018.tar.gz`](remote/public_corpus_pilot/in_place_ttt_public_corpus_pilot_20261018.tar.gz)

The full profile fixes 305,184 records/source at 32K and 57,216
records/source at 128K: 34,999,369,728 combined training tokens and 11,325
global-batch-64 steps. Its CPU-only build started on `hku-gpu2` at
`2026-10-18 00:51:37 HKT` under tmux `public_corpus_full_20261018`. At the
first recorded live check it had written 211 MiB, used about 1.0 GiB RSS and
91% CPU, and retained 914 GiB disk free. This is a running process, not a
completed artifact; the runner will launch the independent full audit only
after atomic publication.

Four opt-in formal configs encode the exact stages and two arms. Stage 1 uses
32,768 positions and 9,537 steps; Stage 2 uses Qwen YaRN factor 4,
`original_max_position_embeddings=32768`, 131,072 positions and 1,788 steps.
Both use GB64, microbatch 1, FSDP2, AdamW `5e-6`, cosine decay, 2% warmup,
bf16 mixed precision, gradient checkpointing and no offload. The arms differ
only in `ttt_mode` plus their runtime checkpoint/output paths. Stage 2 starts
from each arm's converted Stage-1 HF weights with a fresh optimizer and
scheduler. The dependency-free static auditor passes. A separate host-gated
runner requires an explicit set of eight unique GPUs, each with at least
80,000 MiB total memory, at most 512 MiB preflight use and no compute process.
It runs matched one-step arms sequentially; the independent auditor requires
finite loss/gradient values and at least 5% free memory on every selected GPU.
The runner has not executed because no currently reachable host meets that
gate.

Before implementing the formal runner, the exact installed VeOmni source was
inspected. Its iterable builder unconditionally applies
`shuffle(seed=seed, buffer_size=10000)` before `split_dataset_by_node`; turning
that off would modify the released path. The frozen
[`IN_PLACE_TTT_BASELINE_LEVEL_B_TRAINING_ADDENDUM.md`](../../IN_PLACE_TTT_BASELINE_LEVEL_B_TRAINING_ADDENDUM.md)
(commit `d366af6a`, SHA256
`c33310cae2c196b06a15eca2462d009d9ca34674adc11f55a167048dd6d8d03b`)
therefore defines the consumed schedule as the deterministic official shuffle
of the byte-identical ordered JSONL. All configs now explicitly set seed 42,
two workers, prefetch factor 2 and pinned memory. The pinned dataset and
dataloader implementation SHA256 values are `52a1a736...f69` and
`8703db96...c42`.

Real one-step DCPs measure 49,830,704,053 bytes for no-TTT and 50,303,642,036
bytes for TTT. With the registered interval and final save, Stage 1 retains 20
checkpoints per arm and Stage 2 retains four. The addendum consequently
requires at least 3 TiB free on a shared paired-run filesystem. The formal
single-arm runner additionally refuses insufficient per-arm storage, a host or
GPU selection different from the matched preflight, any dirty checkout, failed
corpus/config audit, or a pre-existing output. Its independent auditor checks
all expected checkpoint steps, finite logged metrics, DCP metadata/size, final
HF structure and arm mode, input hashes, resource traces and the 5% GPU
headroom gate. The combined focused suite passes 19 tests. This is implemented
readiness evidence; no formal optimizer step has run.

### Alternate host check

At `2026-10-16 23:13:47 HKT`, `hku-a100` exposed one
`NVIDIA A100-SXM4-80GB` (81,920 MiB). A `VLLM::EngineCore` process occupied
75,124 MiB, leaving 6,007 MiB, so the host was not available for this run.
Host RAM was 125 GiB total and 82 GiB available, while its 2-GiB swap was
fully used. If the A100 becomes exclusive, its 80-GiB capacity warrants a new
single-GPU resource pilot; it is not a drop-in execution of the frozen
two-GPU FSDP2 protocol.

Snapshot: [`remote/hku_a100_resource_snapshot.json`](remote/hku_a100_resource_snapshot.json).
The `2026-10-17 21:04 HKT` recheck is unchanged: the same vLLM process uses
75,124 MiB and only 6,007 MiB is free. The current snapshot is archived at
[`remote/hku_a100_resource_snapshot_20261017_2104.txt`](remote/hku_a100_resource_snapshot_20261017_2104.txt).
The `2026-10-18` recheck again exposes only that same single A100 and the same
75,124-MiB vLLM allocation, so it still cannot satisfy the formal 8-GPU gate.

## 5. RULER pipeline smoke

The final run used the clean official inference registration, unmodified
`/home/ctj/models/Qwen3-4B`, OpenCompass 0.5.4, context 4,096, batch 1, and one
example for each of the 13 official RULER subtasks.

| Item | Result |
|---|---:|
| Exit code | 0 |
| Completed tasks | 13/13 |
| Official RULER group score | 76.9230769 |
| Independent mean of task JSON scores | 76.9230769 |
| Wall clock | 2:34.87 |
| GPU0 monitored peak | 12,288 MiB |
| Missing/unexpected tasks | 0/0 |

The official group result and independent extraction agree exactly. Scores are
binary at sample size one, so this smoke must not be compared to the paper's
reported 96.6 baseline at 4K.

Compact evidence:

- [`remote/ruler_4k_base_smoke/summary_v2.json`](remote/ruler_4k_base_smoke/summary_v2.json)
- [`remote/ruler_4k_base_smoke/eval.log`](remote/ruler_4k_base_smoke/eval.log)
- [`remote/ruler_4k_base_smoke/gpu.csv`](remote/ruler_4k_base_smoke/gpu.csv)

Two earlier smoke attempts are retained only on the server: one exposed the
NumPy/PyArrow conflict before model loading; one exposed an unserializable
module in the external OpenCompass config. A third direct-SSH attempt was
interrupted by the Cloudflare tunnel. The successful run used tmux. None of
these failed attempts contributes a score.

The first converted-checkpoint smokes completed, but each log emitted 27
copies of the Transformers Mistral-regex warning:

| Checkpoint | Tasks | Official score | Independent mean | Wall | GPU0 peak |
|---|---:|---:|---:|---:|---:|
| no-TTT step 1 | 13/13 | 76.9230769 | 76.9230769 | 1:47.12 | 9,300 MiB |
| TTT step 1 | 13/13 | 76.9230769 | 76.9230769 | 1:42.51 | 9,916 MiB |

Read-only diagnosis found byte-identical `tokenizer.json` files in the
original and both converted checkpoints. The official conversion changes the
recorded Transformers version from 4.51.0 to 4.57.3, and Transformers 4.57.3
then misclassifies the local `qwen3` config as a Mistral tokenizer. Applying
`fix_mistral_regex=true` changes the converted pre-tokenizer hash and is
incorrect for Qwen. The frozen tokenizer addendum therefore uses the original
Qwen tokenizer directory for both OpenCompass model tokenization and RULER
data generation, checks that its `tokenizer.json` hash matches each converted
checkpoint, and explicitly keeps the Mistral fix false.

The replacement smokes use new output directories and bind the tokenizer,
model, parent plan and tokenizer-addendum hashes:

| Checkpoint | Tasks | Official score | Regex warnings | Wall | GPU0 peak |
|---|---:|---:|---:|---:|---:|
| no-TTT step 1 | 13/13 | 76.9230769 | 0 | 1:38.95 | 9,300 MiB |
| TTT step 1 | 13/13 | 76.9230769 | 0 | 1:39.41 | 9,916 MiB |

The audited tokenizer SHA256 is
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`
for the original, no-TTT and TTT copies. The per-task score vectors remain
identical. Three result hashes differ between the arms, confirming that the
TTT-aware path produces different raw outputs, but one binary item per task
still has no power to estimate efficacy.

Evidence:

- [`remote/exclusive_5090_completed/ruler_4k_no_ttt_step1_smoke/summary.json`](remote/exclusive_5090_completed/ruler_4k_no_ttt_step1_smoke/summary.json)
- [`remote/exclusive_5090_completed/ruler_4k_ttt_step1_smoke/summary.json`](remote/exclusive_5090_completed/ruler_4k_ttt_step1_smoke/summary.json)
- [`remote/exclusive_5090_completed/ruler_4k_no_ttt_step1_tokenizer_audited_smoke/summary.json`](remote/exclusive_5090_completed/ruler_4k_no_ttt_step1_tokenizer_audited_smoke/summary.json)
- [`remote/exclusive_5090_completed/ruler_4k_ttt_step1_tokenizer_audited_smoke/summary.json`](remote/exclusive_5090_completed/ruler_4k_ttt_step1_tokenizer_audited_smoke/summary.json)
- [`remote/exclusive_5090_completed/tokenizer_diagnostic.json`](remote/exclusive_5090_completed/tokenizer_diagnostic.json)
- raw predictions, task results, official summary files, GPU traces, and logs
  are stored below the same directories.

## 6. Verification

- CUDA bf16 tensor creation and FlashAttention forward passed on RTX 5090.
- Official source checkout commit and clean status were checked before every
  run.
- Plan, config, data, model-config, log, GPU trace, and result hashes are stored
  in the compact summaries.
- Focused resource and RULER summarizer tests: `3 passed`.
- Public-data packing/overwrite tests: local and remote `4 passed`.
- RULER task set is checked against an explicit 13-task allowlist.
- Official group score is independently recomputed from task result JSONs.
- Independent exclusive-run audit recomputes GPU peaks, resource headroom,
  input bindings, conversion structure, manifest bindings, and both smoke
  summaries. Version 2 additionally verifies the false-positive diagnosis,
  matching tokenizer hashes, frozen addendum, false Mistral-fix policy and
  zero-warning replacement smokes. It passes and requires stopping before
  16K.
- The independent public-data auditor re-hashes both source bindings and all
  emitted artifacts, re-tokenizes every 32K row, and recomputes ordered token
  and document streams. Both 8-record source audits pass all 12 checks; the
  copied archive passes a second local hash verification.
- The formal source inventory passes a separate seven-check audit. The
  continuous two-stage pilot passes ten checks after re-tokenizing all 144
  rows, and its archive passes local SHA256 verification.
- The four formal training configurations pass the dependency-free static
  auditor. The host-gated matched-preflight metric/resource checks and the
  formal-training checkpoint/resource fixture checks pass; the combined
  focused suite has 25 passing tests.
- The formal RULER addendum is frozen with SHA256
  `a5762933f8c6f9eee2863e42ba6823bbad922433fb00cf6e872630fe56543e7b`.
  The new single-run auditor requires the exact Stage-2 audit/model, package
  versions, ten generator-source hashes, three RULER source-data hashes,
  original tokenizer, 13 tasks, 100 ordered examples and matching official
  aggregate. The paired matrix audit also proves that `origin_prompt` and
  `references` are identical between no-TTT and TTT before computing deltas.
- The Stage-2 training audit now records the HF checkpoint under its final
  published root instead of the temporary `.partial` root. This closes a path
  binding bug before any formal training run; it does not change training.
- A fresh remote checkout at commit `a4159855d8f8d0a6f5a7a96d1365b23d0f89d8eb`
  loaded all seven configs in the pinned OpenCompass environment. Every config
  resolved 13 tasks with 100 samples and seed 42; the remote RULER/training
  audit suite passed 9/9. This validates tooling only and does not count as a
  formal model score. Compact evidence is in
  [`remote/level_b_ruler_tooling_smoke/summary.json`](remote/level_b_ruler_tooling_smoke/summary.json).

## 7. Commands

Environment setup:

```bash
bash scripts/setup_in_place_ttt_official_env.sh
bash scripts/setup_in_place_ttt_official_eval_env.sh
```

Build and independently audit the frozen 32K public-data pilot:

```bash
python scripts/build_in_place_ttt_public_data_pilot.py \
  --source fineweb --tokenizer /home/ctj/models/Qwen3-4B \
  --output-root /new/output/fineweb \
  --max-records 8 --seq-len 32768
python scripts/audit_in_place_ttt_public_data_pilot.py \
  --root /new/output/fineweb --tokenizer /home/ctj/models/Qwen3-4B

python scripts/build_in_place_ttt_public_data_pilot.py \
  --source longcrawl \
  --longcrawl-input /path/to/longcrawl64/train/0-of-256.parquet \
  --tokenizer /home/ctj/models/Qwen3-4B \
  --output-root /new/output/longcrawl \
  --max-records 8 --seq-len 32768
python scripts/audit_in_place_ttt_public_data_pilot.py \
  --root /new/output/longcrawl --tokenizer /home/ctj/models/Qwen3-4B
```

Inventory, build and audit the formal public corpus:

```bash
python scripts/inventory_in_place_ttt_public_corpus.py \
  --tokenizer /home/ctj/models/Qwen3-4B \
  --output-root /new/output/source_inventory
python scripts/audit_in_place_ttt_public_corpus_inventory.py \
  --root /new/output/source_inventory \
  --tokenizer /home/ctj/models/Qwen3-4B

python scripts/build_in_place_ttt_public_corpus.py \
  --inventory /new/output/source_inventory/inventory.json \
  --inventory-audit /new/output/source_inventory/independent_audit.json \
  --tokenizer /home/ctj/models/Qwen3-4B \
  --output-root /new/output/public_corpus --profile full
python scripts/audit_in_place_ttt_public_corpus.py \
  --root /new/output/public_corpus \
  --inventory /new/output/source_inventory/inventory.json \
  --inventory-audit /new/output/source_inventory/independent_audit.json \
  --tokenizer /home/ctj/models/Qwen3-4B

python scripts/audit_in_place_ttt_level_b_configs.py \
  --corpus-root /new/output/public_corpus \
  --output /new/output/level_b_config_audit.json
```

After the full audit passes, run a matched one-step preflight on an exclusive
8×80-GB-class NVIDIA host:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
UPSTREAM_REPO=/path/to/clean/in-place-ttt-be232482 \
TOOL_REPO=/path/to/CBF-TTT/In-Place-TTT \
PYTHON_BIN=/path/to/official-env/bin/python \
CORPUS_ROOT=/path/to/audited/public_corpus \
BASE_MODEL=/path/to/Qwen3-4B \
OUTPUT_ROOT=/new/output/level_b_stage1_preflight \
bash scripts/run_in_place_ttt_level_b_preflight.sh stage1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
UPSTREAM_REPO=/path/to/clean/in-place-ttt-be232482 \
TOOL_REPO=/path/to/CBF-TTT/In-Place-TTT \
PYTHON_BIN=/path/to/official-env/bin/python \
CORPUS_ROOT=/path/to/audited/public_corpus \
BASE_MODEL=/path/to/Qwen3-4B \
STAGE1_NO_TTT_MODEL=/path/to/stage1_no_ttt_hf \
STAGE1_TTT_MODEL=/path/to/stage1_ttt_hf \
OUTPUT_ROOT=/new/output/level_b_stage2_preflight \
bash scripts/run_in_place_ttt_level_b_preflight.sh stage2
```

Run one released formal stage/arm after its matched preflight passes. Stage 1
must run `no_ttt` first so its runner verifies the full 3-TiB paired budget:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
UPSTREAM_REPO=/path/to/clean/in-place-ttt-be232482 \
TOOL_REPO=/path/to/clean/CBF-TTT/In-Place-TTT \
PYTHON_BIN=/path/to/official-env/bin/python \
CORPUS_ROOT=/path/to/audited/public_corpus \
START_MODEL=/path/to/Qwen3-4B \
PREFLIGHT_ROOT=/path/to/level_b_stage1_preflight \
OUTPUT_ROOT=/new/output/stage1_no_ttt \
bash scripts/run_in_place_ttt_level_b_training.sh stage1 no_ttt
```

Use the same command with the matched config/model/output for `stage1 ttt`.
For Stage 2, set `START_MODEL` to that arm's audited Stage-1 `hf_ckpt` and use
the completed Stage-2 matched preflight.

After both Stage-2 arms pass their independent audits, run the frozen paired
RULER matrix on seven idle 80-GB GPUs. The runner evaluates all seven lengths
for no-TTT, then all seven for TTT, and publishes only after every single-run
and matrix audit passes:

```bash
UPSTREAM_REPO=/path/to/clean/in-place-ttt-be232482 \
TOOL_REPO=/path/to/clean/CBF-TTT/In-Place-TTT \
PYTHON_BIN=/path/to/official-eval-env/bin/python \
TOKENIZER=/path/to/original/Qwen3-4B \
NO_TTT_MODEL=/path/to/stage2_no_ttt/final/hf_ckpt \
TTT_MODEL=/path/to/stage2_ttt/final/hf_ckpt \
NO_TTT_TRAINING_ROOT=/path/to/stage2_no_ttt \
TTT_TRAINING_ROOT=/path/to/stage2_ttt \
RULER_GPUS=0,1,2,3,4,5,6 \
OUTPUT_ROOT=/new/output/level_b_ruler_matrix \
bash scripts/run_in_place_ttt_level_b_ruler_matrix.sh
```

Resource probe (change `ARM` and output path for the other arm):

```bash
UPSTREAM_REPO=/home/ctj/in_place_ttt_official_be232482 \
PYTHON_BIN=/home/ctj/miniconda3/envs/in_place_ttt_official_py311/bin/python \
MODEL=/home/ctj/models/Qwen3-4B \
DATA=/home/ctj/data/cbf_ttt_pilot/mixed_16x8192.jsonl \
OUTPUT_ROOT=/new/output/resource_8k_no_ttt \
ARM=no_ttt SEQ_LEN=8192 GLOBAL_BATCH_SIZE=2 \
PLAN=$PWD/IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md TOOL_REPO=$PWD \
bash scripts/run_in_place_ttt_baseline_resource_pilot.sh
```

Convert and audit both completed DCP checkpoints:

```bash
UPSTREAM_REPO=/home/ctj/in_place_ttt_official_be232482 \
PYTHON_BIN=/home/ctj/miniconda3/envs/in_place_ttt_official_py311/bin/python \
OUTPUT_ROOT=/path/to/completed_exclusive_run \
bash scripts/convert_and_audit_in_place_ttt_baseline_checkpoints.sh
```

RULER smoke:

```bash
UPSTREAM_REPO=/home/ctj/in_place_ttt_official_be232482 \
PYTHON_BIN=/home/ctj/miniconda3/envs/in_place_ttt_official_eval/bin/python \
MODEL=/home/ctj/models/Qwen3-4B OUTPUT_ROOT=/new/output/ruler_4k_smoke \
PLAN=$PWD/IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md TOOL_REPO=$PWD \
bash scripts/run_in_place_ttt_ruler_smoke.sh
```

Run both converted-checkpoint smokes:

```bash
UPSTREAM_REPO=/home/ctj/in_place_ttt_official_be232482 \
EVAL_PYTHON=/home/ctj/miniconda3/envs/in_place_ttt_official_eval/bin/python \
TOKENIZER=/home/ctj/models/Qwen3-4B \
OUTPUT_ROOT=/path/to/completed_exclusive_run \
PLAN=$PWD/IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md TOOL_REPO=$PWD \
TOKENIZER_AUDIT_ADDENDUM=$PWD/IN_PLACE_TTT_BASELINE_TOKENIZER_AUDIT_ADDENDUM.md \
SMOKE_SUFFIX=tokenizer_audited_smoke \
REQUIRE_ZERO_TOKENIZER_REGEX_WARNINGS=1 \
bash scripts/run_in_place_ttt_converted_checkpoint_smokes.sh
```

Independent audit:

```bash
python scripts/audit_in_place_ttt_baseline_exclusive.py \
  --root experiments/baseline_reproduction_qwen3_4b/remote/exclusive_5090_completed \
  --tokenizer-addendum IN_PLACE_TTT_BASELINE_TOKENIZER_AUDIT_ADDENDUM.md \
  --output experiments/baseline_reproduction_qwen3_4b/remote/exclusive_5090_completed/independent_audit.json
```

## 8. Next required experiment

1. Let the already running full public-corpus build finish, then require its
   independent audit to pass exact counts, source continuity and hashes. Do
   not treat the live partial directory as training input.
2. On `hku-gpu2` only, with the builder and auditor stopped, run matched
   no-TTT then TTT activation-offload resource probes at 8K, 16K and 32K.
   Every boundary must pass its independent audit, retain 5% device headroom,
   keep at least 8 GiB host memory available and use at most 3.5 GiB swap.
3. If both 32K arms pass, measure a multi-step throughput projection and run
   the matched 98,566,144-token tier: 27 GB64 steps at 32K followed by five
   GB64 steps at 128K. Stage 2 uses each arm's converted Stage-1 weights and a
   fresh optimizer/scheduler.
4. Convert both final DCP checkpoints and execute the 100-example RULER matrix
   at 4K--128K. Attempt 256K only through a separately audited two-GPU
   inference configuration; report a resource failure if it cannot fit.
5. Only after the matched 100M tier and RULER matrix pass may a new frozen
   addendum authorize the approximately 1B tier. Resume support/update gates
   only after an audited matched baseline tier is complete.

Exact paper reproduction additionally requires the authors' private mixture,
32K/128K stage recipe, and inference clipping details.

The released-code Level-A path is now closed. Native FSDP2 8K succeeds on
exclusive 2x5090 but leaves only 2.107% free memory; parameter offload is not
usable on this host. The next registered measurement uses the released
activation-offload implementation on the same two 5090s. This is a systems
feasibility test and cannot by itself support an efficacy claim.

## 9. Full-corpus audit scalability hardening (2026-10-18)

The active full-corpus builder remains live in tmux
`public_corpus_full_20261018`. A direct process check at
`2026-10-17 18:31:42 UTC` found PID 1397987 in `RUNNING_BUILD`, using about
88% CPU and 4.4 GiB RSS; the partial directory had grown to 7.7 GiB. The
partial directory remains invalid as training input.

Review of the independent auditor found that its original continuity checker
retained every past document entry and cursor. The pilot observed 2,919
FineWeb documents for 3,145,728 FineWeb training tokens; extrapolation to the
registered FineWeb budget is roughly 16.2 million retained entries. This is a
host-memory scaling risk for the audit after a successful build, not a corpus
construction error.

`IN_PLACE_TTT_BASELINE_LEVEL_B_CORPUS_AUDIT_SCALABILITY_ADDENDUM.md` freezes a
streaming representation before it may be used. The revised auditor retires a
fully consumed document after the next consecutive document begins, retains
prefetched pending documents, preserves the ordered document-stream hash and
all ten existing acceptance checks, and reports documents seen plus peak
active documents. A 10,000-document test requires at most two active entries;
a separate test proves that a prefetched snapshot cannot skip its first
document. The complete Level-B focused suite now passes 27/27 tests.

The active runner remains bound to its original builder and auditor hashes and
must not be modified or restarted. If its original post-build audit passes,
that result is accepted. The streaming auditor has now repassed the frozen
pilot from clean commit `a61bf944062cf0bc8685b0bbf2638ac8f0780ee6` under a
distinct output filename. All 10 checks passed; both stage observations, the
four registered total/hash fields and every binding other than the expected
auditor hash were identical to the original audit. It observed 2,919 FineWeb
and 63 LongCrawl documents while retaining at most two active documents per
source. The saved resource run took 13.93 seconds, peaked at 821,588 KiB RSS,
used no swap and exited zero.

The downloaded run, hashes and local comparison are archived in
`experiments/baseline_reproduction_qwen3_4b/remote/in_place_ttt_public_corpus_streaming_audit_smoke_20261018/`;
the streaming audit SHA256 is
`23eff15cab3f784d144494d0a369fc3f2bdd157882b813201779e6a59a789b6c`.
This version may now serve as a secondary audit, or as a fallback only if the
builder has atomically published the complete corpus and the original audit
fails specifically because of host memory. It cannot validate or rescue a
partial build.

## 10. `hku-gpu2`-only activation-offload execution path (2026-10-18)

The user explicitly restricted every subsequent GPU run to the two RTX 5090s
on SSH alias `hku-gpu2` (actual hostname `user-Super-Server`). Before any
activation-offload optimizer step, the new
`IN_PLACE_TTT_BASELINE_2X5090_EXECUTION_ADDENDUM.md` was frozen with SHA256
`0b70a098f8ee5cadf12f8d371dcc0020124069f5365aa5279d706cde4322e5c6`.
It keeps Qwen3-4B, FineWeb-Edu/LongCrawl64, the native `ttt_proj`/`ttt_conv`
writer, AdamW, GB64, the 32K→128K curriculum and matched no-TTT/TTT arms. The
systems-only change enables VeOmni's released activation offload, keeps FSDP2
parameter/optimizer sharding, disables parameter offload and enables the
expandable-segments allocator.

Four opt-in configs register a first complete tier with exact counts:

| Stage | Length | Steps | Tokens |
|---|---:|---:|---:|
| Stage 1 | 32,768 | 27 | 56,623,104 |
| Stage 2 | 131,072 | 5 | 41,943,040 |
| total | — | 32 | 98,566,144 |

`scripts/audit_in_place_ttt_2x5090_configs.py` verifies the four files as a
matched matrix and reports `world_size=2`, `total_tokens=98566144` and no
errors. `scripts/run_in_place_ttt_baseline_resource_pilot.sh` now supports an
opt-in `activation_offload` profile and refuses to launch unless the actual
hostname, device indices 0/1, RTX-5090 model, 32,607-MiB capacity, ≤128-MiB
idle use, empty compute-process list and ≥48-GiB available host RAM all match.
It also refuses to launch while either public-corpus builder or auditor is
active. It hashes the loaded VeOmni offloading implementation, configuration,
data, model config, plans, preflight snapshots, runner and summarizer.

The resource summary now records steady optimizer-step time separately from
checkpoint time, per-device minimum free fraction, host memory/swap and the
activation-offload release gate. The new independent resource auditor
recomputes all hashes, effective log settings, both GPU traces, preflight
state, finite metrics and the ≥40-GB DCP structure. The combined focused
baseline suite passes 34/34 tests; shell syntax, bytecode compilation,
configuration audit and `git diff --check` pass.

Commit `e9c1966fe5bd0f232657709e6394140f3494f1c3` was pushed and checked out
cleanly on `hku-gpu2`; the same 34/34 tests and config audit passed there. A
runtime review then found the corpus builder's official checkout intentionally
contains untracked build tools, so a separate clean official checkout at the
same `be232482` commit was created for training. Commit
`19e7f480b7fabe257d1ecde5ede785f961e95781` replaced `pgrep` conflict matching
with exact argv-basename inspection of `/proc`, preventing parent shell text
from appearing as a false builder process.

The corrected runner was executed once as a negative preflight while the
builder remained live. It exited 1 with `public-corpus builder/auditor is still
active`, identified exactly PIDs 1397986 and 1397987, found no CUDA compute
process, left GPU0/GPU1 at 46/18 MiB and created neither `run_metadata.json`
nor `train.log`. The downloaded snapshots and local hash verification are in
`remote/hku_gpu2_activation_preflight_rejection_20261018/`. This is evidence
that the guard prevents training; it is not an optimizer-step result.

An opt-in watcher now binds the exact tool/upstream commits, model, 8K pilot,
plans and configuration audit, then polls the same GPU, host and exact
`/proc` conflict gates. The first scheduler revision exposed a Bash control
flow issue: because `eligible` was called in an `if`, `set -e` did not stop the
function after its Python check returned 1. It therefore called the runner,
whose independent hard preflight rejected the live builder before metadata,
training log or CUDA work and left both GPUs at 46/18 MiB. Commit
`f14cb7152bc7607e0b8555d449bcf2b28853eb1c` explicitly propagates every
eligibility failure.

The repaired watcher runs in tmux
`baseline_2x5090_activation_8k_20261018` from immutable detached checkout
`/home/ctj/cbf_ttt_tool_f14cb715`. Its output root is
`/home/ctj/in_place_ttt_2x5090_activation_8k_f14cb715_20261018`. The first
authoritative snapshot remained `waiting_8k_no_ttt`, recorded the two real
builder PIDs, created no `resource_8k_no_ttt` child, and left both devices at
0% utilization. It may launch only after two consecutive eligible snapshots;
if either 8K arm or its independent audit fails, it stops before 16K.

No activation-offload GPU probe has run yet. At `2026-10-18T10:12:01+08:00`,
PID 1397987 and tmux `public_corpus_full_20261018` were still live after
9:20:23. The partial corpus occupied about 45 GiB, the builder used
13,477,044 KiB RSS, host `MemAvailable` was 47,875,532 KiB and swap use was
about 354 MiB. This state violates the frozen no-concurrency and 48-GiB launch
gates, so starting a GPU probe now would invalidate the resource evidence.

At the measured exclusive native 8K rates, a full 35B-token two-arm run has an
optimistic lower bound of roughly 201 days before long-context slowdown. The
100M tier is therefore the first honest end-to-end target. It remains a
resource-constrained public-data replication; it is not the paper-scale 35B
result, and no gate experiment is unsealed by these static checks.

## 11. Exact sparse-ladder inputs (2026-10-18)

Before any 16K/32K activation-offload probe, the resource-input addendum was
frozen with SHA256
`5b65319ceeae3f25e7338928290ba9e59ecad22a1152a9209845b2ba0b68647c`.
It prevents a 32K plaintext row from being mislabeled as a shorter measurement
through a CLI length override. The source is fixed to the already audited
continuous public-corpus pilot manifest
`3b2e7f4821692dfe75bae48e4cb52c6ac39f4b9b3b4c75f96aeb32ab7dec7206`.
Every boundary uses exactly its first FineWeb and first LongCrawl record.

The builder preserves 32K/128K parent text byte-for-byte. For 8K/16K/64K it
selects the longest stable Qwen-token prefix and pads only with tokenizer EOS
tokens when a decode/encode boundary requires it. The manifest contains no
plaintext and binds the source artifacts, tokenizer, plan, builder, record
hashes and ordered uint32 token stream. The independent auditor reimplements
the derivation rather than importing the builder.

The CPU-only build on `hku-gpu2` atomically published all five inputs in 4.83
seconds with 822,896 KiB peak RSS. Its first audit invocation exposed a CLI
name-mapping bug before reading any output; commit
`8019cc55b26d8f32f506cac1dddd72797103459b` fixed the mapping and added a CLI
regression test. The second audit completed in 4.95 seconds with 820,884 KiB
peak RSS and passed all seven checks. GPUs remained 46/18 MiB and 0% utilized.

| Length | Records | Training tokens | Artifact SHA256 |
|---:|---:|---:|---|
| 8K | 2 | 16,384 | `73679b13...9c57` |
| 16K | 2 | 32,768 | `cf6c1c1c...e084` |
| 32K | 2 | 65,536 | `d8ad5c06...bb30` |
| 64K | 2 | 131,072 | `15d06cf2...4398` |
| 128K | 2 | 262,144 | `17838e50...af42` |

The published manifest SHA256 is
`9bc2bb610f95dbf5b6541e9b9c1f42f5a1a0d067faa47924f82400340b5acdce`;
the independent audit SHA256 is
`ebf1b4c65145efb9921d6fbfd4b868a37ed211dbcdaef49ddc744f891d2006f8`.
The resource runner and independent resource auditor now require and bind this
manifest/audit for every activation-offload boundary above 8K. The plaintext
JSONL files remain on `hku-gpu2`; Git stores only manifests, audits, resource
logs and hashes under `remote/hku_gpu2_resource_inputs_20261018/`.

The complete focused baseline suite now passes 41/41 tests, including both
rejection of an unbound 16K input and acceptance of a correctly bound
manifest/audit fixture. This closes input
length ambiguity only. The live 8K watcher remains bound to its earlier exact
8K pilot, and GPU feasibility remains pending its resource run.

## 12. Audited Stage-1 resource-ladder orchestration (2026-10-18)

The opt-in `scripts/run_in_place_ttt_2x5090_stage1_ladder.sh` now encodes the
next admissible GPU action without starting it. The runner is restricted to
the `hku-gpu2` hostname `user-Super-Server`; it rejects reused output roots,
dirty or mismatched official/tool checkouts, missing frozen plans, and stale
exact-input manifests. It first requires the existing 8K watcher to have
completed and independently passed both no-TTT and TTT arms. Only then may it
run the fixed 16K no-TTT, 16K TTT, 32K no-TTT and 32K TTT boundaries in that
order. A failure or invalid child stops the ladder immediately.

The independent `scripts/audit_in_place_ttt_2x5090_stage1_ladder.py` does not
trust the orchestration state. It recomputes the 8K prerequisite, verifies
current log and monitor hashes, checks finite loss/gradient/step time, the 5%
per-GPU headroom gate, host-memory/swap limits and at least 40 GB of DCP
checkpoint files. For 16K and 32K it also rehashes the exact input artifacts
and their manifest/audit bindings, then checks all four child summaries and
per-boundary independent audits. Metadata binds the frozen addenda, config
audit, runner, auditor and immutable tool/upstream revisions.

Two focused tests cover rejection before the 8K prerequisite and a complete
six-child fixture, including re-audit after atomic final-state publication.
The complete focused baseline suite passes 43/43 tests; Bash syntax and
`git diff --check` also pass. No 16K/32K GPU process has been launched. The
formal corpus builder remains the current hard blocker, and all future GPU
execution is restricted to `hku-gpu2`.

An additional opt-in handoff watcher,
`scripts/watch_and_run_in_place_ttt_2x5090_stage1_ladder.sh`, can wait without
allocating CUDA memory for the existing 8K tmux handle. It records the live
tmux pane handle and upstream state on every poll. A missing handle, terminal
8K failure, timeout or independently invalid 8K result is terminal; only the
audited `completed_8k_both_arms` state launches the Stage-1 ladder. The watcher
and ladder both independently enforce the `hku-gpu2` hostname and immutable
clean checkouts, so the handoff cannot redirect work to another GPU host.

The watcher was deployed from clean detached commit `bae87e6c` in tmux
`baseline_2x5090_stage1_ladder_bae87e6c_20261018`. The same checkout passed
43/43 focused tests on `hku-gpu2`. At `2026-10-18T10:49:49+08:00` its state
was `waiting_for_audited_8k`, the recorded 8K pane was live, the future ladder
output did not exist, and GPU0/GPU1 used only 46/18 MiB at 0% utilization.
The public-corpus builder PID 1397987 was also directly observed live with a
48 GiB partial output. Deployment evidence is stored under
`remote/hku_gpu2_stage1_ladder_watcher_20261018/`.

## 13. Registered Stage-1 multi-step timing gate (2026-10-18)

Review of the 2x5090 path found that the frozen execution addendum required a
multi-step, no-checkpoint projection before formal 100M training, but the
repository only had one-step GB2 resource probes. The new frozen
`IN_PLACE_TTT_BASELINE_2X5090_STAGE1_TIMING_PLAN.md` closes that gap without
launching a GPU job. It registers three 32K, GB64 optimizer steps per arm on
the complete audited public corpus, always no-TTT before TTT. The first step
is warm-up; the median of individual steps 2 and 3 projects each arm's formal
27-step Stage-1 wall time. All pilot updates are discarded, and checkpoint/HF
saving is disabled.

`scripts/run_in_place_ttt_2x5090_stage1_timing.sh` refuses CUDA initialization
unless the full corpus and the complete Stage-1 resource ladder independently
pass, the exact official/tool revisions are clean, both GPUs are idle, no
corpus process exists and host memory is at least 48 GiB. It binds the corpus,
ladder, configs, model, plan, runner and auditor hashes and records per-arm
GPU/host traces. The runner is restricted to `hku-gpu2`.

`scripts/audit_in_place_ttt_2x5090_stage1_timing.py` rehashes the corpus and
all dependencies, validates the four ladder boundaries, parses the first
completed tqdm observation for each of the three steps, reconstructs
individual times from cumulative elapsed time, verifies finite metrics,
resource limits, effective activation-offload settings and absence of DCP/HF
checkpoints, then computes the paired sequential Stage-1 projection. It does
not extrapolate 128K timing. Stage 2 remains locked until arm-specific Stage-1
HF checkpoints exist and a separate 64K/128K resource and timing plan passes.

Two new tests cover progress-trace de-duplication and a complete paired audit
fixture. The full focused baseline suite passes 45/45 tests; Bash syntax,
Python compilation and `git diff --check` pass. The timing runner is prepared
but has not been launched because its corpus and ladder prerequisites are
still running.

The companion CPU-only timing watcher records the live Stage-1 ladder tmux
handle and state. It stops on a missing handle or any terminal ladder failure;
only `completed_stage1_ladder` plus a passing saved independent audit launches
the timing runner. The timing runner then repeats all corpus, resource, host
and revision gates, so the watcher does not weaken the release boundary.

The timing watcher was deployed from clean detached commit `56094b89` in tmux
`baseline_2x5090_stage1_timing_56094b89_20261018`. At
`2026-10-18T10:59:40+08:00` it was `waiting_for_audited_stage1_ladder`, the
ladder-watcher pane was live, no timing output existed and both GPUs remained
at 46/18 MiB and 0%. A separate negative preflight from implementation commit
`18c63932` exited 1 while the corpus/ladder were absent, created no output or
partial path and left GPU state unchanged. Evidence is under
`remote/hku_gpu2_stage1_timing_watcher_20261018/`.

## 14. Formal 100M-tier Stage-1 execution path (2026-10-18)

The 2x5090 configs previously specified the formal 27-step Stage-1 schedule,
but no paired runner independently enforced its timing release, disk budget,
final checkpoint structure or atomic publication. The new frozen
`IN_PLACE_TTT_BASELINE_2X5090_100M_STAGE1_PLAN.md` registers no-TTT followed by
TTT, each with 27 GB64 steps at 32K and exactly 56,623,104 tokens. Both arms
start from the same Qwen3-4B model and full audited corpus. Only
`global_step_27` DCP plus its converted HF checkpoint is retained.

Formal execution requires the full corpus audit, paired resource ladder and
paired three-step timing audit to pass. It also requires a separate release
JSON, created only after the measured projection is reviewed, which sets
`accepted=true` and binds the exact timing-audit SHA256. This prevents the
prepared code from automatically starting the formal run before its observed
wall time is known. The release cannot waive any runtime or audit gate.

`scripts/run_in_place_ttt_2x5090_100m_stage1.sh` is restricted to `hku-gpu2`,
requires 160 GB free disk for the pair and rechecks 80 GB, idle GPUs, 48 GiB
host memory and absence of corpus processes before each arm. It preserves a
successful no-TTT checkpoint if TTT later fails but publishes the paired root
only after both arms pass. `scripts/audit_in_place_ttt_2x5090_100m_stage1.py`
recomputes all dependency and corpus hashes, validates the release and timing
projection, reconstructs 27 finite step records per arm, checks GPU/host
limits, requires exactly one at-least-40-GB DCP and at-least-7-GB HF model,
validates arm-specific `ttt_mode`, and hashes complete DCP/HF manifests.

Two focused tests cover 27-step progress de-duplication and a complete paired
formal fixture. The complete focused baseline suite passes 47/47 tests; Bash
syntax, Python compilation and `git diff --check` pass. No formal Stage-1
training has been launched. Stage 2 and RULER remain locked behind a
successful paired Stage-1 audit.

Commit `8b9a56b4` was checked out cleanly on `hku-gpu2`; its focused formal
tests passed 2/2 and Bash syntax passed. A negative preflight with the future
corpus/timing/release paths exited 1, created no output or partial directory
and left both GPUs at 46/18 MiB and 0%. At
`2026-10-18T11:06:28+08:00`, 926,365,290,496 bytes of disk remained free and
the corpus PID plus all upstream watchers were still live. Evidence is stored
under `remote/hku_gpu2_formal_stage1_tooling_20261018/`.

## 15. Arm-specific Stage-2 resource ladder (2026-10-18)

The parent execution addendum requires 64K and registered 128K one-step
resource boundaries after Stage-1 conversion, but no runner previously proved
that each Stage-2 arm started from its own audited Stage-1 HF checkpoint. The
new frozen `IN_PLACE_TTT_BASELINE_2X5090_STAGE2_RESOURCE_PLAN.md` registers
64K no-TTT/TTT followed by 128K no-TTT/TTT, with GB2 and one optimizer step at
each boundary. Every child forks directly from its arm-specific Stage-1 HF
checkpoint with a fresh optimizer; no resource child feeds the next one.

`scripts/run_in_place_ttt_2x5090_stage2_ladder.sh` refuses to start unless the
paired formal Stage-1 state and independent audit pass. Before CUDA it rehashes
the complete HF manifests for both arms and verifies `ttt_mode` and 32K
Stage-1 model configuration. It then reuses the existing activation-offload
resource runner, exact 64K/128K inputs, Stage-2 YaRN configs, checkpoint
capture and per-boundary independent audit. The runner remains restricted to
`hku-gpu2` and stops at the first invalid boundary.

`scripts/audit_in_place_ttt_2x5090_stage2_ladder.py` independently reconstructs
both Stage-1 HF manifests, exact input bindings and all four resource children.
It checks that every child's model path and model-config hash match the correct
arm, verifies the Stage-2 config hash, and reapplies finite metric, DCP,
GPU/host and dependency gates through the independent boundary validator.
Two tests cover stale Stage-1 manifests and a complete arm-specific four-child
fixture. The full focused baseline suite passes 49/49 tests. This path is
prepared only; formal Stage 1 has not run, so no Stage-2 GPU work is released.

Commit `4860ea6e` was checked out cleanly on `hku-gpu2`; the focused Stage-2
tests passed 2/2 and Bash syntax passed. A negative preflight with a future
formal Stage-1 root exited 1, created no ladder output and left both GPUs at
46/18 MiB and 0%. The corpus PID and three upstream watcher states were still
live/waiting at `2026-10-18T11:11:47+08:00`. Evidence is stored under
`remote/hku_gpu2_stage2_ladder_tooling_20261018/`.

## 16. Arm-specific Stage-2 multi-step timing gate (2026-10-18)

The one-step 128K resource boundary cannot estimate steady Stage-2 training
time because initialization and the first optimizer step dominate it. The
frozen `IN_PLACE_TTT_BASELINE_2X5090_STAGE2_TIMING_PLAN.md` (SHA256
`9898d283923a7a8ad45448249541901435087a45e8f89174b3a8283d894ec294`)
therefore registers a separate three-step 128K, global-batch-64 pilot for each
arm. Step 1 is warm-up; the median reconstructed step-2/3 duration projects
the registered five-step formal Stage 2. All pilot updates are discarded and
checkpoint/HF saving is disabled.

`scripts/run_in_place_ttt_2x5090_stage2_timing.sh` is hard restricted to the
`user-Super-Server` host behind the `hku-gpu2` SSH alias. Before any CUDA
preflight it requires clean exact revisions, the paired formal Stage-1 audit,
the four-child Stage-2 ladder audit, the full public-corpus audit and unchanged
complete HF manifests for both arm-specific starting checkpoints. Each arm
then independently rechecks two idle RTX 5090 GPUs, available host memory and
absence of corpus processes. No other GPU server is an accepted execution
target.

`scripts/audit_in_place_ttt_2x5090_stage2_timing.py` rehashes all dependencies,
the full 128K corpus and both current start models; verifies three finite,
monotonic steps per arm; enforces GPU, host-memory and swap limits; rejects
all checkpoint files; and reports arm-specific throughput plus the sequential
five-step projection. Tests cover accidental checkpoint creation and a full
paired fixture with distinct no-TTT/TTT starts. The complete focused baseline
suite passes 51/51 tests; Bash syntax, direct auditor import, Python compile
and `git diff --check` pass.

This commit prepares and tests the timing gate only. It does not launch GPU
work: formal Stage 1 and the Stage-2 resource ladder have not completed. A
future formal Stage-2 runner must additionally consume a separately reviewed,
timing-audit-bound release record. RULER and update-admission experiments stay
sealed until the paired Stage-2 audit passes.

Commit `69ee323a` was checked out cleanly on `hku-gpu2`. The remote CPU-side
focused suite passed 51/51, Bash syntax and the auditor CLI passed, and a
negative run with unpublished formal prerequisites exited 1 before CUDA while
creating neither final nor partial output. At `2026-10-18T11:19:15+08:00`,
GPU 0/1 used 46/18 MiB at 0%; the corpus builder and all three upstream
watchers remained alive. Evidence is stored under
`remote/hku_gpu2_stage2_timing_tooling_20261018/`.

## 17. Paired formal 100M-tier Stage-2 entry (2026-10-18)

The Stage-2 resource and timing gates previously ended without an executable
formal path. The frozen `IN_PLACE_TTT_BASELINE_2X5090_100M_STAGE2_PLAN.md`
(SHA256 `0017abb1aade09e3298bcc3b36f5cd2b515db2444a7f0eee6d7562fcf9c24a6c`)
now registers the final 128K segment: five GB64 optimizer steps and 41,943,040
tokens per arm, no-TTT before TTT. Each arm starts from its corresponding
audited Stage-1 HF checkpoint with a fresh optimizer/scheduler; resource and
timing pilot checkpoints cannot become formal starts.

`scripts/run_in_place_ttt_2x5090_100m_stage2.sh` is restricted to `hku-gpu2`
and requires an explicit accepted release JSON bound to the completed Stage-2
timing audit. It rehashes the full Stage-2 corpus, paired formal Stage-1 audit,
both current complete Stage-1 HF manifests, configs, plans and release before
training. It requires 160 GB free disk for the pair and rechecks 80 GB, two
idle RTX 5090 GPUs, 48 GiB host memory and absence of corpus processes before
each arm. Only an independently audited pair is atomically published.

`scripts/audit_in_place_ttt_2x5090_100m_stage2.py` independently reconstructs
the arm-specific start models, timing/release and corpus bindings. It requires
five finite monotonic progress records per arm, resource limits, only
`global_step_5`, at least 40 GB DCP and 7 GB HF data, correct `ttt_mode`, and
131,072 maximum positions. Tests cover duplicate final progress output and a
complete paired fixture with distinct Stage-1 starts. The focused baseline
suite passes 53/53 tests; Bash syntax, Python compilation, direct auditor CLI
and `git diff --check` pass.

No formal Stage-2 GPU execution or release record has been created. The
entry remains sealed behind formal Stage 1, the Stage-2 ladder, measured
Stage-2 timing and explicit review. RULER and update-admission work remain
sealed behind the resulting paired formal Stage-2 audit.

Commit `d2bc98e8` was checked out cleanly on `hku-gpu2`. The remote focused
suite passed 53/53, Bash syntax and the direct auditor CLI passed, and a
negative run with unpublished formal prerequisites exited 1 before CUDA while
creating neither final nor partial output. At `2026-10-18T11:26:00+08:00`,
GPU 0/1 used 46/18 MiB at 0%; the corpus builder remained live with a 51-GB
partial directory and all upstream watchers remained waiting. Evidence is
stored under `remote/hku_gpu2_formal_stage2_tooling_20261018/`.

## 18. Paired 2x5090 100M-tier RULER path (2026-10-18)

The existing formal RULER scripts require seven concurrent 80-GB GPUs and
arm-specific Level-B training roots, so they cannot execute the registered
2x5090 tier. The frozen `IN_PLACE_TTT_BASELINE_2X5090_RULER_PLAN.md` (SHA256
`fe41a68a1fc474c5d19a921623e1d17ea2011e1d7327b7679993b645f1ab3962`)
keeps the same OpenCompass RULER sources, 13 tasks, deterministic decoding,
seed 42 and 100 examples while restricting all execution to `hku-gpu2`.

The opt-in config `qwen3_4b_2x5090_ruler.py` differs from the registered
Level-B config only by permitting either one sample for discarded resource
measurements or exactly 100 for formal scoring. The single-run entry rehashes
the paired formal Stage-2 audit and complete arm-specific HF manifest, original
tokenizer, plans, config, runner/auditor and OpenCompass data/code. It requires
physical GPU 0 or 1 to be an idle 32-GB RTX 5090, 32 GiB available host memory
and no public-corpus process. Large model hashes stream in fixed chunks.

Before scoring, `run_in_place_ttt_2x5090_ruler_resource_ladder.sh` pairs
no-TTT on GPU 0 with native TTT on GPU 1 at 4K, 32K, 64K and 128K, with one
example for every task. Its scores are explicitly discarded. Independent
single and ladder auditors require all 13 tasks, identical paired prompts and
references, exact final models, finite output, zero tokenizer warnings and at
least 5% device headroom. The ladder stops at the first failed boundary.

After the ladder passes, `run_in_place_ttt_2x5090_ruler_matrix.sh` evaluates
the paired arms concurrently at each context, then advances sequentially over
4K, 8K, 16K, 32K, 64K and 128K. Independent single and matrix audits recompute
all task/context scores and input hashes from raw predictions, verify every
dependency and report per-task deltas, context macros and the six-context
mean. A 256K run remains a separately labelled extrapolation resource decision
through the same single-run resource mode; it cannot alter the registered
matrix if it fails.

Six new tests cover formal single-run identity/model drift, one-sample resource
artifacts, paired resource inputs, six-context matrix recomputation and paired
formal-input mismatch. The full focused baseline suite passes 59/59; all new
Bash files parse, Python files compile and `git diff --check` passes. No RULER
GPU process has been launched because the paired formal Stage-2 checkpoints do
not yet exist.

Commit `52775cb3` was checked out cleanly on `hku-gpu2`. The remote focused
suite passed 59/59, all three runner scripts passed Bash syntax validation and
all four auditor CLIs loaded successfully. Negative resource-ladder and formal
matrix preflights both exited 1 at the absent paired formal Stage-2 prerequisite,
before CUDA and without creating final or partial output. At
`2026-10-18T11:47:50+08:00`, GPU 0/1 used 46/18 MiB at 0%; the public-corpus
builder remained live with a 52-GB partial directory. Evidence is stored under
`remote/hku_gpu2_ruler_tooling_20261018/`. All future GPU execution for this
baseline remains restricted to `hku-gpu2`.

## 19. Repeated resource-snapshot repair (2026-10-18)

The live 8K activation watcher exposed a polling defect before any GPU process
started. Its `snapshot()` function used `cp /proc/meminfo` to create
`last_no_ttt_host.txt`; Linux propagated the source mode `0444`, so every
later polling iteration failed to overwrite the watcher's own snapshot. The
watcher process remained live and its state misleadingly stayed at
`waiting_8k_no_ttt`.

The live file was changed from mode `0444` to owner-writable `0644`. On the
next scheduled poll at `2026-10-18T11:52:11+08:00`, both the host snapshot and
state timestamp advanced and the watcher again recorded the two active corpus
builder processes. GPU 0/1 remained at 46/18 MiB and 0% utilization, so this
repair did not start or alter training.

For subsequent clean runs, `watch_and_run_in_place_ttt_2x5090_activation.sh`
now writes `/proc/meminfo` through shell redirection. The destination therefore
uses the watcher umask and remains writable across polls. Other occurrences
copy one-shot preflight evidence and do not overwrite the same path in a loop.

Commit `40bedb659a104d3177584f5afc528c9cc955d1b8` was then checked out cleanly
on `hku-gpu2`. The complete focused baseline suite passed 59/59, Bash syntax
passed, and the checked script SHA256 matched the remotely reproduced
candidate. Evidence is stored under
`remote/hku_gpu2_activation_watcher_repair_20261018/`.

## 20. Explicit timing-bound formal release helper (2026-10-18)

The formal Stage-1 and Stage-2 runners require a post-measurement release JSON
but previously depended on hand-written records. The opt-in
`create_in_place_ttt_2x5090_timing_release.py` validates the exact timing-audit
protocol, clean pass state, both registered arms, positive finite steady and
formal projections, and equality between the paired projection and its two
arms. It records the complete timing-audit SHA256 and the reviewed projection.

Without `--accept`, the CLI prints the review summary, exits 2 and writes no
release. An accepted release additionally requires nonempty reviewer and
decision text, refuses overwrite, and is published atomically with the exact
Stage-1 or Stage-2 release protocol expected by the formal runner. Four tests
cover explicit Stage-1 acceptance, no-write review mode, inconsistent timing
rejection and the distinct Stage-2 protocol.

Commit `2ac3a890d664ada87801b2b1d15232853b6e1557` was fetched on `hku-gpu2`.
Because the SSH gateway repeatedly reset during full-worktree materialization,
the two additive files were extracted from the exact fetched Git object into
an isolated directory. Their 4/4 tests and direct CLI check passed. Combined
with the unchanged 59/59 suite from the preceding clean commit, the focused
baseline count is 63. No GPU process was launched. Evidence is stored under
`remote/hku_gpu2_timing_release_tooling_20261018/`.

## 21. Formal Stage-1 release watcher (2026-10-18)

`watch_and_run_in_place_ttt_2x5090_100m_stage1.sh` closes the persistent chain
after the existing resource-ladder and timing watchers. It is restricted to
`hku-gpu2`, freezes its exact tool/upstream revision and static dependencies,
and watches the registered timing tmux handle and terminal state. A missing
handle before a published result, any upstream failure, a nonfinite projection
or a failed audit terminates the watcher without invoking CUDA.

After timing passes, the watcher enters a separate
`waiting_for_explicit_stage1_release` state. It never creates or accepts a
release. It requires the exact Stage-1 protocol, `accepted=true`, the timing
audit SHA256, `stage=stage1`, reviewer and decision note. Only then does it
call the formal paired runner. It records the runner exit code and requires
the terminal paired state and independent audit before reporting completion.

Commit `3c7e29fc33cbb781090db6792e4fef42dec6dbcf` was checked out cleanly on
`hku-gpu2`. The complete focused suite passed 63/63 and watcher Bash syntax
passed. A negative run with a deliberately missing timing tmux handle exited
6 with state `stage1_timing_watcher_handle_missing`, before CUDA and without
creating final or partial training output. Evidence is stored under
`remote/hku_gpu2_formal_stage1_watcher_tooling_20261018/`.

The validated watcher is now live in tmux session
`baseline_2x5090_formal_stage1_3c7e29fc_20261018`. At
`2026-10-18T12:25:20+08:00`, its pane PID 1422177 and the upstream timing
session were live, its state was `waiting_for_audited_stage1_timing`, the
explicit release was absent, neither final nor partial training output
existed, and GPU 0/1 remained at 46/18 MiB and 0% utilization.

## 22. Post-Stage-1 Stage-2 ladder watcher (2026-10-18)

`watch_and_run_in_place_ttt_2x5090_stage2_ladder.sh` persistently bridges the
paired formal Stage-1 result to the registered 64K/128K Stage-2 resource
ladder. It is restricted to `hku-gpu2`, binds a clean exact tool/upstream
revision, frozen plans and independently audited exact resource inputs, and
monitors both the formal Stage-1 tmux handle and upstream watcher state.

The watcher accepts only terminal state `completed_100m_stage1` with the exact
paired audit protocol, a clean pass and both arm records. Missing handles,
upstream failure, invalid audit or timeout stops before the ladder runner. A
successful launch still requires all four 64K/128K arm-specific boundaries,
terminal ladder state and its independent audit before reporting completion.

Commit `1b04d8cf55821056e1dfb3a724d3ded1a87ebffc` was checked out cleanly on
`hku-gpu2`; the complete focused suite passed 63/63 and Bash syntax passed. A
negative run with an absent formal Stage-1 tmux handle exited 6 in state
`formal_stage1_watcher_handle_missing`, before CUDA and without ladder output.
The validated watcher is now live in tmux session
`baseline_2x5090_stage2_ladder_1b04d8cf_20261018`, pane PID 1424039, state
`waiting_for_audited_formal_stage1`. Evidence is stored under
`remote/hku_gpu2_stage2_ladder_watcher_tooling_20261018/`.

## 23. Stage-2 timing watcher and GPU-host restriction (2026-10-18)

All remaining baseline GPU training, resource measurement and evaluation are
restricted to `hku-gpu2`. No other server is an eligible fallback. CPU-only
preparation may run there while the registered prerequisites are incomplete.

`watch_and_run_in_place_ttt_2x5090_stage2_timing.sh` persistently connects the
four-boundary 64K/128K Stage-2 resource ladder to the registered paired 128K
three-step timing run. It freezes the exact tool revision, runner, independent
auditor and plans, monitors the Stage-2 ladder tmux handle and watcher state,
and accepts only the exact ladder audit protocol with both arm-specific start
models and all four independently audited boundaries.

A missing upstream handle, terminal upstream failure, timeout, incomplete
boundary set or failed audit stops before invoking the timing runner. After a
run, the watcher requires terminal state `completed_stage2_timing`, an exact
passing audit, both arms and finite positive formal wall-time projections. The
timing checkpoints remain discarded, and this watcher neither creates nor
accepts the explicit release required by formal Stage 2.

Commit `fbc02a6e6ad88834b2bb2533906264ddd670d2f3` was checked out cleanly on
`hku-gpu2`. The focused suite passed 63/63 and Bash syntax passed. A negative
run with a deliberately missing Stage-2 ladder tmux handle exited 6 in state
`stage2_ladder_watcher_handle_missing`, created no timing output and left GPU
0/1 at 46/18 MiB with zero utilization. The validated watcher is live in tmux
session `baseline_2x5090_stage2_timing_fbc02a6e_20261018`; at
`2026-10-18T12:45:04+08:00`, pane PID 1425558 was live, its state was
`waiting_for_audited_stage2_ladder`, and the corpus partial directory was 57
GB. Evidence is stored under
`remote/hku_gpu2_stage2_timing_watcher_tooling_20261018/`.

## 24. Formal Stage-2 release watcher (2026-10-18)

`watch_and_run_in_place_ttt_2x5090_100m_stage2.sh` closes the persistent
training chain after Stage-2 timing. It is restricted to `hku-gpu2`, freezes
the exact formal runner, independent auditor, release helper and plans, and
monitors both the timing watcher tmux handle and its terminal state.

The watcher requires the exact paired Stage-2 timing audit protocol, clean
pass, both arm projections, a positive finite sequential projection and exact
sum consistency. It then waits in a separate state for an explicit Stage-2
release that binds the timing-audit SHA256 and contains a reviewer and decision
note. It never creates or accepts that release automatically. Missing handles,
upstream failure, invalid release or timeout stops before formal CUDA work.

After release, the existing formal runner still revalidates the formal
Stage-1 starts, Stage-2 ladder, full corpus, disk, host and idle GPUs. The
watcher reports completion only for terminal state `completed_100m_stage2`
with the exact passing paired audit and both arms.

Commit `73950ecd1d0a8c2163863dce48de272e5db7d8d8` was checked out cleanly on
`hku-gpu2`. The focused suite passed 63/63 and Bash syntax passed. A negative
run with a deliberately missing Stage-2 timing tmux handle exited 6 in state
`stage2_timing_watcher_handle_missing`, created no formal output and left GPU
0/1 at 46/18 MiB with zero utilization. The validated watcher is live in tmux
session `baseline_2x5090_formal_stage2_73950ecd_20261018`; at
`2026-10-18T12:51:11+08:00`, pane PID 1426724 was live, state was
`waiting_for_audited_stage2_timing`, the explicit release was absent, and the
corpus partial directory was 58 GB. Evidence is stored under
`remote/hku_gpu2_formal_stage2_watcher_tooling_20261018/`.

## 25. Persistent paired RULER evaluation chain (2026-10-18)

The formal RULER matrix runner now writes terminal state
`completed_ruler_matrix` only after its summary and independent matrix audit
both succeed. Two opt-in watchers connect the paired formal Stage-2 result to
the registered evaluation while keeping all GPU execution on `hku-gpu2`.

The resource watcher accepts only an exact, clean paired formal Stage-2 audit
with both arms. It then runs the paired one-sample-per-task 4K/32K/64K/128K
resource ladder and requires eight audited boundaries, discarded scores and
the terminal ladder state. The matrix watcher waits on that exact ladder and
then runs both arms over 4K/8K/16K/32K/64K/128K with 100 examples per task.
It requires the terminal matrix state, all six recomputed context scores and
the independent matrix audit before reporting completion.

Both watchers bind clean exact revisions, original Qwen3-4B tokenizer, frozen
plans, runners, summarizer and auditors. Missing upstream handles, terminal
failure, invalid audit or timeout stops before the next GPU stage. The 256K
extrapolation remains separately sealed and cannot alter the registered
4K–128K baseline result.

Commit `5603910cdadf617b7073d689967dfa9c26a73449` was checked out cleanly on
`hku-gpu2`. The focused suite passed 63/63; the matrix runner and both watchers
passed Bash syntax checks. The registered OpenCompass environment imported
successfully with PyTorch 2.8.0+cu128 and Transformers 4.57.3. Negative missing
handle runs exited 6 in states `formal_stage2_watcher_handle_missing` and
`ruler_resource_watcher_handle_missing`, created no GPU outputs and left GPU
0/1 at 46/18 MiB with zero utilization.

The validated resource watcher is live in tmux session
`baseline_2x5090_ruler_resource_5603910c_20261018`, pane PID 1428296, state
`waiting_for_audited_formal_stage2`. The validated matrix watcher is live in
session `baseline_2x5090_ruler_matrix_5603910c_20261018`, pane PID 1428336,
state `waiting_for_audited_ruler_resource_ladder`. The observation time was
`2026-10-18T12:58:00+08:00`; the corpus partial directory was 58 GB. Evidence
is stored under `remote/hku_gpu2_ruler_watchers_tooling_20261018/`.

## 26. Machine-audited paper-reference comparison (2026-10-18)

Before any formal RULER score existed, the paired matrix summarizer was
extended to embed the seven Table 1 Qwen3-4B reference values for matched
baseline and In-Place TTT. For each registered 4K--128K context it now reports
the public no-TTT result minus the paper matched baseline, the public TTT
result minus the paper TTT value, and the difference between the public and
paper TTT gains.

The independent matrix auditor recomputes every comparison from raw task
scores and rejects any changed paper value, missing context, arithmetic drift
or claim-boundary change. The summary explicitly records
`public_data_matched_replication_not_paper_exact`; the 256K paper values are
retained for reference while the local 256K result remains behind its separate
extrapolation resource gate.

Commit `439fcea142c4d3bb51413d75480d5231d0c62cab` was checked out cleanly on
`hku-gpu2`; the six RULER tests and complete 63-test focused suite passed. The
two prior RULER watchers were confirmed live, still in their initial waiting
states and without final/partial GPU outputs before being replaced. The
paper-aware resource watcher is now live in session
`baseline_2x5090_ruler_resource_439fcea1_20261018`, pane PID 1430095, and the
matrix watcher in `baseline_2x5090_ruler_matrix_439fcea1_20261018`, pane PID
1430136. At `2026-10-18T13:04:10+08:00`, their states were respectively
`waiting_for_audited_formal_stage2` and
`waiting_for_audited_ruler_resource_ladder`; GPU 0/1 remained at 46/18 MiB and
zero utilization. Evidence is stored under
`remote/hku_gpu2_ruler_paper_comparison_20261018/`.

## 27. Separate RULER-256K extrapolation resource gate (2026-10-18)

`IN_PLACE_TTT_BASELINE_2X5090_RULER_256K_PLAN.md` is frozen before any 256K
run or score. It preserves the paper's 256K point as a separate YaRN
extrapolation because the public-data Stage-2 checkpoints are registered to
131,072 positions. A failure cannot invalidate or alter the primary
4K--128K matrix.

The opt-in paired resource runner starts only after the independently audited
primary matrix. It evaluates one example for each of the 13 tasks, with no-TTT
on GPU 0 and TTT on GPU 1, and discards scores. Its independent auditor binds
the final Stage-2 checkpoints, primary matrix, plans, exact tools and both
child audits; it also recomputes paired input hashes and requires at least 5%
GPU headroom. The persistent watcher monitors the paper-aware matrix handle
and launches this resource decision only after the exact matrix audit passes.

A passing resource gate releases preparation of a separately frozen formal
100-example 256K runner; it does not launch formal scoring automatically. OOM,
nonzero exit or any audit mismatch stops without shortening inputs, changing
tasks, quantizing, or relabelling a subset as the paper metric.

Commit `b33b93ace74240b2bb2332368647c4adda31b7dc` was checked out cleanly on
`hku-gpu2`. The dedicated test passed 1/1, the full focused suite passed 64/64,
both Bash scripts parsed and the Python auditor compiled. A negative run with
a deliberately missing matrix handle exited 6 in state
`ruler_matrix_watcher_handle_missing`, created no resource output and left GPU
0/1 at 46/18 MiB with zero utilization.

The validated watcher is live in tmux session
`baseline_2x5090_ruler_256k_resource_b33b93ac_20261018`; at
`2026-10-18T13:15:39+08:00`, pane PID 1431970 was live in state
`waiting_for_audited_ruler_matrix`, the GPUs remained idle, and the corpus
partial directory was 60 GB. Evidence is stored under
`remote/hku_gpu2_ruler_256k_resource_20261018/`.

## 28. Formal RULER-256K release and scoring closure (2026-10-18)

`IN_PLACE_TTT_BASELINE_2X5090_RULER_256K_FORMAL_PLAN.md` was frozen before
the separate resource decision and before any formal 256K score. It registers
13 tasks with 100 examples per task, seed 42, deterministic decoding and
byte-identical paired inputs at 262,144 tokens. This remains a public-data
YaRN extrapolation outside the registered 131,072-position training range and
is excluded from the primary 4K--128K mean.

The opt-in release helper accepts only the exact, independently passing
resource audit and binds its SHA256. Without `--accept` it exits without
writing a release; an accepted release also requires a reviewer and decision
note. The persistent formal watcher never creates or accepts this release. It
waits first for the live resource watcher and then in a separate
`waiting_for_explicit_ruler_256k_release` state.

After release, the formal runner evaluates no-TTT on GPU 0 and native TTT on
GPU 1, both exclusively on `hku-gpu2`. The summarizer reports per-task and
macro accuracy, paired TTT gain and differences from the paper's 41.7/43.9
references. The independent auditor reconstructs all 2,600 predictions,
checks paired input hashes, checkpoint and dependency bindings, release and
plan hashes, arithmetic, terminal states and the explicit
`public_data_yarn_extrapolation_not_paper_exact` claim boundary.

The three dedicated formal/resource tests and the complete focused suite pass
3/3 and 66/66 respectively; both Bash entry points parse, the Python tools
compile, and `git diff --check` passes. No formal release has been accepted and
no 256K formal score has been produced at this implementation checkpoint.

Commit `a0375e93a530591ca3e36d6d4c8498c4772d8784` was pushed and checked out
cleanly on `hku-gpu2`; the remote focused suite again passed 66/66. A negative
preflight with a deliberately missing resource-watcher handle exited 6 in
state `ruler_256k_resource_watcher_handle_missing`, before CUDA and without
formal or partial output.

The validated formal watcher is live in tmux session
`baseline_2x5090_ruler_256k_formal_a0375e93_20261018`. At
`2026-10-18T13:28:55+08:00`, pane PID 1435575 was live, its state was
`waiting_for_audited_ruler_256k_resource`, the explicit release and formal
output were absent, and GPU 0/1 remained idle at 46/18 MiB. Evidence is stored
under `remote/hku_gpu2_ruler_256k_formal_tooling_20261018/`.

## 29. Full-corpus streaming-audit fallback watcher (2026-10-19)

The active full-corpus wrapper froze original auditor SHA256
`9c5dbadba76c4ebbba5248952455b3bfe4860b9b2b79bdbb555034dcc76c7e31`.
That version retains all historical continuity entries and may exhaust host
memory at formal scale after the builder has atomically published the corpus.
The previously frozen scalability addendum registered streaming auditor SHA256
`5ec753a29996f6abc386d7e16529218bbaaf6c1b1ff271638db730b9c41f8b87`,
but the active wrapper had no automatic fallback.

`IN_PLACE_TTT_BASELINE_CORPUS_AUDIT_FALLBACK_WATCHER_ADDENDUM.md` now freezes
that missing state transition before launch. The opt-in CPU watcher accepts a
valid existing original audit, otherwise waits. It can run the streaming
auditor only after atomic full-corpus publication, exact manifest/dependency
validation, explicit original-wrapper `FAILED` state, zero live builder or
auditor processes, and complete absence of canonical audit and partial files.
It rechecks the decision after five seconds and never replaces a failed or
malformed existing audit. Thus a genuine corpus failure remains terminal.

The fallback writes canonical `independent_audit.json` only when no canonical
artifact exists, then independently verifies full profile, all-true checks,
manifest binding and registered auditor identity. Existing Stage-1 timing and
formal-training gates therefore require the same canonical passing contract.
Nine tests cover release after post-publication original failure,
active-auditor waiting, valid existing-audit acceptance, failed-audit refusal,
pre-publication failure refusal, completed-without-audit refusal and exact
active-command classification for direct, `/usr/bin/time`-wrapped and
dependency-only auditor paths.

Commit `990da145d814aded46a560f64947bc17e7e89fb6` is pushed and checked out
cleanly on `hku-gpu2`. The complete focused suite passes 75/75. A ten-second
negative preflight exited 9 in state
`timed_out_waiting_for_original_audit`, created no canonical audit and left
GPU 0/1 at 46/18 MiB. The persistent watcher is live in tmux session
`baseline_public_corpus_fallback_990da145_20261019`, with pane PID 1605090 and
Python PID 1605091. At `2026-10-19T01:08:27+08:00` it remained in `wait`
because the full corpus was not yet atomically published; the builder PID
1397987 was live at 60,421 Stage-2 rows and both GPUs were idle. Evidence is
archived under
`remote/hku_gpu2_public_corpus_streaming_fallback_20261019/`.

## 30. Unified completion matrix (2026-10-19)

[`IN_PLACE_TTT_BASELINE_COMPLETION_MATRIX.md`](../../IN_PLACE_TTT_BASELINE_COMPLETION_MATRIX.md)
now indexes the authoritative evidence required to close every remaining
baseline stage. It distinguishes a live watcher, smoke, resource boundary,
formal paired training result, release decision and independently audited
metric matrix. It also records the remote roots for the full corpus, both
training stages and RULER without changing any frozen plan or artifact hash.

At this checkpoint, released-code Level A, the public source inventory and the
continuous packing pilot are complete. The full corpus is still building;
formal training, primary RULER scores and the baseline conclusion remain
incomplete. Update-admission work remains sealed. The matrix is a status and
acceptance index only: it creates no release and launches no process.

The matrix does not redefine 100M tokens as the paper-scale baseline. The
98,566,144-token pair is the first resource-constrained end-to-end milestone.
Its measured training and RULER result must be reported together with the
32K/128K wall-time projection and explicit feasibility boundary. A 1B
expansion needs a new pre-outcome addendum, and a full Level-B scale claim
still requires the registered approximately 35B-token curriculum.

## 31. Stage-2 75% sparse provenance boundary (2026-10-19)

Before the registered boundary existed, a one-shot CPU-only watcher fixed
Stage-2 indices 85,824--85,829 and script SHA256
`3eab1785aae80f393057fcc29bc0fd6ecdf57c95ec43fb6df93c2b8bc73ec389`.
It ran on `hku-gpu2`, emitted no source text or token IDs, waited for at least
85,830 complete provenance rows, atomically wrote its result, and exited. The
builder remained the original PID 1397987; no GPU process was started.

The audit passed with no errors. The six rows have exact consecutive stage
indices, strict FineWeb-Edu/LongCrawl64 alternation, paired source indices
42,912--42,914, 131,071 content tokens, 131,072 training tokens, 131,071 raw
source tokens consumed, zero synthetic EOS tokens, present spans and valid
64-hex training-token hashes. A separate read-only recomputation from the live
provenance file independently passed all ten semantic checks.

The atomic audit is stored at
`/home/ctj/in_place_ttt_public_corpus_full_run_20261018/stage2_sparse_75pct_audit.json`
with SHA256
`7ae916550d98d60dbd355fc599ed43e1cd383dec8526ade6c3efe52ff3139695`.
This is an intermediate continuity check only. It does not replace full-corpus
atomic publication or the registered canonical independent audit, and it does
not release activation, training, RULER or update-admission work.

## 32. Frozen RULER configuration revalidation (2026-10-19)

The frozen OpenCompass environment on `hku-gpu2` loaded the actual 2x5090
configuration without running inference. At 4K it expanded to exactly the 13
official tasks: two QA, eight NIAH, variable tracking, frequent-word extraction
and common-word extraction. Every task had 100 examples, seed 42 and a 4,096
maximum sequence length, and the official `ruler_4k` summary group was selected.

The same load-time check passed at 4K, 8K, 16K, 32K, 64K, 128K and 256K. At
each length all 13 task suffixes, sample counts, seeds, maximum lengths and
summary groups matched the registered context. The environment reported
OpenCompass 0.5.4, Transformers 4.57.3, Datasets 2.21.0 and Torch 2.8.0+cu128.
All ten registered generator/scorer module hashes and all three registered
RULER source-file hashes matched the frozen constants.

The released `eval_config/ruler_128k.py` and the resource-constrained config
both import the same five OpenCompass generator groups and the same official
summary groups. The 2x5090 path changes process scheduling, explicitly fixes
the seed and tokenizer policy, and binds the evaluated checkpoint; it does not
change the task set or metric. This revalidation produced no prediction or
score. The 4K--128K matrix remains sealed behind audited paired Stage 2, and
256K remains a separately released extrapolation.

## 33. Qwen3-4B formal-training configuration audit (2026-10-19)

The independent `in_place_ttt_2x5090_config_audit_v1` auditor passed all four
formal configurations with no errors. It bound exact per-file hashes, the
execution-addendum hash, world size two and 98,566,144 total tokens. Within
each stage, no-TTT and native-TTT are identical after normalizing runtime paths
and `ttt_mode`; the latter is the only algorithmic arm difference.

A direct field comparison with the released `configs/pretrain/qwen3_longct.yaml`
found every shared optimizer, scheduler, sharding, batch, mixed-precision,
gradient-checkpointing, data-loader and TTT writer field equal. The registered
adaptations are the 32K-to-128K curriculum, 27+5 optimizer steps for the first
resource-constrained tier, seed 42, disabled WandB, final-only retention and
activation offload required by two 32-GB devices.

The released generic config lists TTT layers `[0,6,12,18,24,30,36]`. The
downloaded Qwen3-4B checkpoint has 36 decoder layers and the training model
constructs them with zero-based indices `range(36)`, so index 36 does not
exist. The Qwen3-4B configs therefore use `[0,6,12,18,24,30]`, matching the
repository's Qwen3 configuration default. This removes an invalid generic
8B-oriented layer entry; it does not remove a real 4B writer layer.

## 34. Training-step and global-batch runtime semantics (2026-10-19)

The pinned VeOmni implementation was inspected and instantiated in the remote
training environment. Its `IterativeDataset` inherits `IterableDataset` and
does not implement `__len__`; it streams the registered JSONL, applies the
released buffer-10,000 shuffle and splits examples by data-parallel rank.
Because the formal path uses `rmpad=false`, `compute_train_steps` selects the
explicit `max_steps` when the dataset length is absent. The entry point passes
that result to both the native dataloader and cosine scheduler, then performs
one optimizer and one scheduler step per outer training iteration.

With `WORLD_SIZE=2`, `ulysses_parallel_size=1`, global batch 64 and microbatch
one, the actual `TrainingArguments` derived DP size two, 32 gradient-accumulation
microbatches and a per-rank dataloader batch of 32. Runtime instantiation gave:

| Stage | train steps | tokens/step | exact tokens |
|---|---:|---:|---:|
| 32K | 27 | 2,097,152 | 56,623,104 |
| 128K | 5 | 8,388,608 | 41,943,040 |

Both exact totals equal the frozen configuration. Here explicit `max_steps` is
required: unlike the released dynamic-token path, a non-packed streaming
dataset has neither a finite dataset length nor an enabled token-based step
calculation. Omitting it would make this pinned VeOmni path unable to determine
the formal step count; it would not automatically infer the registered tier.

## 35. Native plaintext dataloader closure (2026-10-19)

The corpus builder writes one `content_split` string whose frozen tokenizer
round trip is exactly `max_seq_len - 1` tokens. Its provenance hashes the
training sequence obtained by appending one EOS. The pinned official training
entry selects VeOmni `process_pretrain_example` for `data_type=plaintext`; that
function encodes with `add_special_tokens=false`, appends one EOS, splits at
`max_seq_len`, and copies the resulting input IDs into labels.

A CPU-only probe on `hku-gpu2` exercised that actual tokenizer builder and
transform rather than reimplementing them. It checked the first FineWeb-Edu
and LongCrawl64 rows in both the 32K and 128K pilot stages. All four rows
produced one chunk of the exact registered length, a terminal EOS, all-one
attention mask, labels equal to inputs and a training-token SHA256 identical
to provenance. Only lengths, booleans and hashes were emitted; source text and
token IDs were not recorded. `CUDA_VISIBLE_DEVICES` was empty.

The evidence binds upstream commit
`be2324829b0e91c8fd10a74d4b43714fde6676e1`, the exact official training
entry, installed VeOmni transform and tokenizer builder, the pilot manifest and
three tokenizer assets. It is stored under
`remote/hku_gpu2_dataloader_closure_20261019/`. This closes the pilot
builder-to-native-dataloader representation boundary: there is no duplicate
EOS, hidden BOS or second truncation on these registered rows. The full corpus
still requires atomic publication and its canonical independent audit before
any GPU gate can run.

## 36. Paired data-parallel sample-order replay (2026-10-19)

The no-TTT and native-TTT configs share seed 42, one corpus file, the same
10,000-example streaming shuffle, DP size two and identical native dataloader
settings. The pinned Datasets code derives its shuffle generator from the
dataset seed and epoch, then splits the shuffled iterable by DP rank. Worker
assignment only shards those prepared data sources; it does not use model
initialization RNG to select examples.

This was exercised twice in separate CPU-only launches on `hku-gpu2` with two
Gloo ranks and the exact Stage-1 batch structure: 32K, global batch 64,
microbatch one, `rmpad=false`, seed 42 and configured `num_workers=2`. Each
rank returned 32 one-example microbatches. In both launches, the ordered 32
training-token hashes on rank 0 matched exactly, as did the ordered 32 hashes
on rank 1. The two rank streams were disjoint; all 64 hashes were unique and
all resolved to rows in the audited pilot provenance. The two structured run
artifacts were byte-identical with SHA256
`1030083c664521b56decdb8a9415af1c13c3f002c8544db2c34dbd80c590e0db`.

Because the corpus is supplied as one JSONL, Datasets exposes one shard and
automatically stops one of the two configured workers in each rank. Thus the
actual loader has one effective worker per rank. This does not change paired
ordering, but any throughput claim must use the measured timing gate rather
than assuming two active workers per rank.

The comparison passed every registered probe condition and has SHA256
`a8a93b7e8924b73ed24c4a40280db259be4dfc83d952d67a27a934a207c0c5e0`.
The script, both runs and comparison are archived under
`remote/hku_gpu2_dataloader_pairing_20261019/`. No GPU was visible to the
process. This confirms the deterministic pairing mechanism and one complete
pilot global batch; the formal audits must still bind the published full
corpus and completed checkpoints.

## 37. Checkpoint consumer and RULER model-class isolation (2026-10-19)

The formal checkpoint chain has two runtime consumers. Stage 1 writes DCP at
step 27 and converts it with pinned VeOmni `ckpt_to_state_dict` and
`save_model_weights`; the Stage-2 runner selects each arm-specific HF path
from the passing Stage-1 audit, and the training entry loads that path before
the 128K stage. Stage 2 repeats the DCP-to-HF conversion at step five. The
RULER resource runner then selects each arm-specific Stage-2 HF path from its
paired audit, records every model-file hash and delegates loading to
OpenCompass `HuggingFaceBaseModel`, whose pinned implementation calls
`AutoModelForCausalLM.from_pretrained` before inference. Thus Stage-1 loading
is exercised by Stage 2, while Stage-2 loading is exercised by the mandatory
RULER resource gate before formal scoring.

The development/tool checkout also contains later CBF modifications under its
own `inference_model`, so module shadowing would invalidate a baseline result.
The frozen single-context runner exports the clean upstream checkout before
the tool checkout in `PYTHONPATH`; its registered entry imports
`inference_model` before invoking OpenCompass. A CPU-only runtime audit in the
exact evaluation environment resolved `Qwen3Config` and
`Qwen3ForCausalLM` to
`/home/ctj/in_place_ttt_official_be232482_resource_clean/inference_model` at
upstream commit `be2324829b0e91c8fd10a74d4b43714fde6676e1`. The resolved
modeling file has SHA256
`98ec94bcf1780635d7817b993a0bf4fa1f1a7e3a35415dce8a46c345a52a2858`,
contains native `TTT_update`, and has no `cbf_ttt.runtime` import. The frozen
tool copy did contain that import, so the audit tested the relevant ambiguity.

The structured audit passed with SHA256
`28445763f9e8261fbac7f32c3880e76b6afc696bdd58f3c2babed1e8d8313781`
and is archived under `remote/hku_gpu2_ruler_import_resolution_20261019/`.
This proves model-class isolation and the registered loading path without
using a GPU. It does not assert that future checkpoint tensors are valid:
formal training must still emit the complete file manifests, and the RULER
resource runs must successfully load those exact manifests and generate all
paired resource predictions before the matrix is released.

## 38. Pre-registered Stage-2 90% sparse boundary (2026-10-19)

Before the target rows existed, a new CPU-only watcher fixed Stage-2 indices
102,990--102,995 and script SHA256
`2b2c78cfffcb3933300e1080a0cdb3f82a8912937ecc36575f4bd207536141ac`.
The target begins just beyond 90% of the registered 114,432-row Stage-2
artifact. The checks match the earlier 75% boundary: six exact consecutive
indices, FineWeb-Edu/LongCrawl64 alternation, paired source indices, exact
content/training/raw-consumed token counts, zero synthetic EOS, nonempty spans
and valid training-token hashes.

The polling implementation now reads only the final four provenance lines and
uses the last complete JSON record as its progress counter. This removes the
earlier repeated full-file `wc` scan while preserving the fixed-boundary
decision. It scans to the six target records only once after they exist. If
the original builder exits before the boundary or the three-hour timeout is
reached, the watcher writes a failing atomic result.

At `2026-10-19T04:34:03+08:00`, builder PID 1397987 and watcher PID 1666185
were live, 90,575 rows were complete, `CUDA_VISIBLE_DEVICES` was empty, and
GPU 0/1 were 46/18 MiB at zero utilization. The launch metadata has SHA256
`25bdb2188c27f925abccc60b8bd607e810241119dd10b30e543e1f1d8754010f`
and is archived under `remote/hku_gpu2_stage2_sparse90_watcher_20261019/`.
The future boundary result remains pending and cannot replace atomic corpus
publication or the canonical independent full audit.

## 39. Registered 100M selected-stream composition (2026-10-19)

The full public corpus is constructed with exact 1:1 FineWeb-Edu/LongCrawl64
token balance, but the resource-constrained run consumes only the deterministic
prefix selected after the native buffer-10,000 streaming shuffle and DP2
split. A CPU-only replay on `hku-gpu2` used the exact seed 42, per-rank batch
32 and registered 27+5 optimizer steps. It repeated both rank schedules from
scratch and checked exact equality, cross-rank disjointness, unique stage
indices and training hashes, valid source labels, exact counts and sufficient
provenance prefix coverage. Every check passed; no model was loaded and
`CUDA_VISIBLE_DEVICES` was empty.

Stage 1 selected 827 FineWeb-Edu and 901 LongCrawl64 records, or 27,099,136
and 29,523,968 tokens. Stage 2 selected 161 and 159 records, or 21,102,592 and
20,840,448 tokens. Combined, the registered tier selects 48,201,728 FineWeb-Edu
tokens (48.9029%) and 50,364,416 LongCrawl64 tokens (51.0971%) out of
98,566,144. The 2.1941% absolute source-fraction difference is deterministic
finite-sample variation. Consequently the 100M tier is approximately 1:1; only
the complete registered 35B corpus is exactly token-balanced.

The structured result has SHA256
`7134c35d98fd24f4781a32aa6ea3713b2f73033218a2dc762a3f04ee4e6fd7f0`
and the frozen audit script has SHA256
`834874e9ab013f6c8e29d09459aba1adffb8775449f975b9fffab9a7083641d4`.
Both are archived under `remote/hku_gpu2_100m_selected_stream_20261019/`.
This closes the finite-tier composition boundary but does not replace the
canonical full-corpus audit or release training.

## 40. Active execution-chain integrity snapshot (2026-10-19)

A read-only CPU audit on `hku-gpu2` bound the complete waiting execution chain
before any activation-offload GPU run. The clean official checkout resolved to
upstream commit `be2324829b0e91c8fd10a74d4b43714fde6676e1`. Eleven detached
tool checkouts for activation, both Stage-1 gates, both Stage-2 gates, formal
training, primary RULER, separate 256K evaluation and the corpus-audit fallback
all matched their registered commits and had empty Git status.

All thirteen registered watcher PIDs were live, including the Stage-2 90%
boundary watcher. For every process, `/proc/<pid>/cmdline` contained the exact
expected frozen script, whose SHA256 was recorded. The host was exactly
`user-Super-Server`; devices 0 and 1 were both 32,607-MiB RTX 5090s. The
Stage-1, Stage-2 and formal 256K release JSON files were all absent, confirming
that no timing review or evaluation boundary had been opened early.

All 55 checks passed with `CUDA_VISIBLE_DEVICES` empty. The result has SHA256
`6962c87fa39cbed91523bc551d8b7aa1e9da4602e8f5ae1b7178424b0b8e6315`
and the auditor has SHA256
`2f8fdc682ebf874e0af25e61ef0e0987e727c329202ef2c9c157ea995db3f831`;
both are archived under `remote/hku_gpu2_active_chain_integrity_20261019/`.
This proves waiting-chain immutability at the recorded time. It is not evidence
that any future child run, checkpoint or metric has completed.

## 41. Qwen3-4B-Base identity correction (2026-10-19)

An independent pre-training identity check found that the checkpoint staged at
`/home/ctj/models/Qwen3-4B` is the official post-trained `Qwen/Qwen3-4B`, not
the `Qwen/Qwen3-4B-Base` required by the parent plan and paper. Its local
configuration has 40,960 maximum positions and EOS 151645; tokenizer metadata
contains a 4,168-character chat template and its 239-byte generation config
enables sampling. These properties and the local config/tokenizer/generation
file sizes match the official post-trained repository. The required Base
configuration instead has 32,768 maximum positions and EOS 151643.

This is a formal baseline identity failure. The earlier engineering smoke
results cannot become paper-baseline evidence, and the active checkpoint-bound
activation/Stage/RULER chain is superseded. At discovery time, the activation
watcher remained in `waiting_8k_no_ttt`, no activation child directory existed,
all formal releases were absent and both GPUs were idle. Thus no formal GPU
score or checkpoint needs reinterpretation.

[`IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_IDENTITY_CORRECTION_ADDENDUM.md`](../../IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_IDENTITY_CORRECTION_ADDENDUM.md)
freezes containment, exact Base download/identity requirements, a tokenizer
compatibility decision for the still-building corpus, and replacement-chain
rules before any corrected GPU run. The corpus may be reused only if ordinary
token encoding and every packed-row length are proven identical; its EOS-bound
training hashes, tokenizer manifest and old audit are invalid for Base in all
cases and must be regenerated. Gate exploration remains sealed.

The exact Base revision
`906bfd4b4dc7f14ee4320094d8b41684abff8539` was subsequently downloaded to
`/home/ctj/models/Qwen3-4B-Base` and passed all nine identity checks. The three
model shard SHA256 values match the official LFS objects. Its config SHA256 is
`304b2545a258d35620f1d4bf46940c0471d9baa00715ff8e77f84c2fca5057c1`
and tokenizer SHA256 is
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
The compact identity record has SHA256
`5e5edb5bef03218c91a1bea0d4f8a80a4814a120734b433a2e6fb765a0a154d4`.

Compatibility probes confirmed identical vocabulary/merges and identical
ordinary-text and packed-edge encodings. They did not justify payload reuse:
the post-trained tokenizer adds `<think>`, `</think>`, `<tool_response>` and
`</tool_response>`, and reserved literal occurrences were found in both old
packed stages. Those strings can change tokenization and packing boundaries.
The old 126-GiB partial corpus, its source-composition replay and all
checkpoint-bound child chains are therefore preserved but superseded. All 14
registered old tmux sessions are absent, no formal release exists and no old
GPU child had started.

A new Base-bound source inventory was generated from the same 140 FineWeb-Edu
files and 256 LongCrawl64 shards. Its manifest SHA256 is
`70cd90464a77ed81ca2d33bc7b394d91c2ab4d100441545c93e22bc02ce7d48a`;
its independent audit SHA256 is
`a113cbc1f40684547f5c2459d049fb910f9b4aa8a3f4624ccd7302905d6f26ec`
and all seven checks pass. The corrected continuous pilot contains 144 records
and 6,291,456 training tokens, with zero synthetic EOS tokens. Its independent
audit passes all ten checks; manifest and audit SHA256 values are
`73fdb39dab7732416effd44ff9b21477d7762a5fc0d37d0d76f93a41cdc4c567`
and `32c8fa6928718e1f9b5c22611878146c193b02f4d34f9b58948e949cd75cf7b1`.
The pilot used about 1,017 MiB peak host RSS and 38.87 seconds. A later
convenience print exited nonzero because it treated the manifest's stage list
as a mapping; the builder and independent auditor themselves both exited zero,
and direct schema-aware verification confirms the recorded counts and checks.

At `2026-10-19T04:59:10+08:00`, a new full build entered `RUNNING_BUILD` in
tmux session `public_corpus_qwen3_4b_base_full_20261019`. It targets the new
immutable root
`/home/ctj/in_place_ttt_public_corpus_qwen3_4b_base_906bfd4b_20261019`,
binds clean tool commit `3e09b18727ae6fcda17684da44abf53c59754465`, the exact Base assets and the
new inventory, and will run the canonical streaming audit after atomic
publication. It launches with `CUDA_VISIBLE_DEVICES` empty. The recorded
snapshot shows GPU 0/1 at 46/18 MiB and 0% utilization. All future GPU work is
restricted to `hku-gpu2`; other servers are not eligible fallbacks. The full
corpus is not complete until the final directory exists without `.partial` and
all independent audit checks pass.

Compact correction evidence is archived under
`remote/hku_gpu2_qwen3_4b_base_identity_correction_20261019/`. The corrected
Base Level-A smoke, selected-stream audit, activation gate, training and RULER
matrix all remain unexecuted. No update-admission experiment is released.

## 42. Corrected Base Level-A replacement chain (2026-10-19)

Two pre-execution addenda now freeze the corrected resource-input and Level-A
closure. The resource-input builder/auditor retain the old pilot hash/profile
as their default for compatibility, while new opt-in arguments accept an exact
independently audited source-manifest hash and require either `pilot` or
`full`. The corrected invocation requires `full` and binds the new Base corpus
manifest, canonical audit and tokenizer assets.

`scripts/watch_and_run_in_place_ttt_qwen3_4b_base_level_a.sh` is restricted to
hostname `user-Super-Server`. It waits for the corrected full corpus wrapper to
report `COMPLETE`, verifies atomic publication and every audit check, derives
and independently audits the 8K--128K sparse inputs, and only then permits GPU
execution. The registered GPU order is no-TTT then TTT, using the same Base
checkpoint and 8K two-record stream. Each arm must pass the existing
activation-offload resource audit before upstream DCP-to-HF conversion and a
13-task, one-example-per-task RULER-4K smoke. A new terminal auditor re-hashes
the model identity, corpus, resource inputs, both child audits, converted
checkpoints and smoke summaries. Its claim is restricted to Level-A execution
closure.

The new and existing In-Place-TTT focused test set passes 78/78 locally and in
the clean remote worktree; Bash syntax and Python compilation also pass. A
CPU-only Base-pilot exercise of the generalized resource-input path completed
in 3.80 seconds and its independent audit in 4.67 seconds, at about 813/810
MiB peak RSS. All seven checks pass; manifest and audit SHA256 values are
`01835ff805be13720d27fb100c3a8ddd5a9be24fcb4703cc0cdcaa2f90fff22e`
and `df5453781fca587965a05d04610006a25def7c3f5b4f9312c6129059fbcb3ada`.
This validates code routing only; formal inputs will come from the full corpus.

The frozen watcher now runs in tmux
`baseline_qwen3_4b_base_level_a_6a3bb2bf_20261019` from clean commit
`6a3bb2bf54c803d4bc9a9fb789e5f11c7f1a6b65`. Its output root is
`/home/ctj/in_place_ttt_qwen3_4b_base_level_a_6a3bb2bf_20261019`. The
authoritative launch snapshot records `WAITING_BASE_CORPUS`, a live watcher
PID, a live Base corpus builder, zero training children and GPU0/1 at 46/18
MiB with 0% utilization. No GPU work is released while the corrected full
corpus remains `RUNNING_BUILD`. Evidence is archived under
`remote/hku_gpu2_qwen3_4b_base_level_a_watcher_20261019/`. Formal 100M
training, the RULER matrix and gate exploration remain sealed.

## 43. Corrected Base Stage-1 resource ladder registration (2026-10-19)

`IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_STAGE1_RESOURCE_PLAN.md` freezes the next
resource-only dependency boundary before any 16K/32K score exists. It accepts
only the corrected Level-A root after its wrapper reports `COMPLETE` and its
terminal audit revalidates Base identity, full corpus, resource inputs, matched
8K arms, conversion and RULER smokes. The four registered children run in the
fixed order 16K no-TTT, 16K TTT, 32K no-TTT and 32K TTT, all on `hku-gpu2`.

The new runner reuses the upstream training entry, existing activation-offload
resource runner, frozen Stage-1 configs and per-child auditor. A separate
corrected ladder auditor imports the existing child validator, recomputes the
Level-A prerequisite, checks the Base-derived `full` resource-input manifest,
verifies four start timestamps and re-audits every child resource trace and
checkpoint. The watcher cannot launch a child while Level A is waiting or
failed. Any missing handle, terminal failure or audit failure stops it.

The focused In-Place-TTT suite now passes 81/81 locally; both new Bash scripts
pass syntax validation and the auditor compiles. This registration does not
release the 32K timing gate, formal 100M training, Stage 2, RULER matrix or
update-admission exploration.

Commit `97c4d786e0c6c9d38de665312a92d41b26d63ab1` also passes 81/81 in a clean
detached `hku-gpu2` worktree. Its watcher is live in tmux
`baseline_qwen3_4b_base_stage1_ladder_97c4d786_20261019`, with watcher root
`/home/ctj/in_place_ttt_qwen3_4b_base_stage1_ladder_97c4d786_20261019.watch`.
The initial state is `waiting_for_corrected_level_a`; the future result root
does not exist, Level A remains `WAITING_BASE_CORPUS`, and no 16K/32K child was
created. GPU0/1 remained at 46/18 MiB and 0% utilization. Compact evidence is
archived under
`remote/hku_gpu2_qwen3_4b_base_stage1_ladder_watcher_20261019/`.

## 44. Corrected Base Stage-1 timing gate registration (2026-10-19)

`IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_STAGE1_TIMING_PLAN.md` freezes the
three-step 32K timing boundary before any corrected timing result exists. It
requires the full Base-tokenizer corpus and corrected 16K/32K ladder to pass,
then runs no-TTT before native TTT from the same exact Base checkpoint and
audited stream. Each arm uses two RTX 5090s on `hku-gpu2`, global batch 64 and
exactly three optimizer steps; step 1 is warm-up and steps 2--3 define the
measured projection. No other GPU host is eligible.

The new watcher remains CPU-only while its predecessor is pending. It checks
the corrected ladder watcher's terminal state and independent audit before it
can invoke the existing released-loop timing runner. A new wrapper auditor
then re-hashes the Base identity, full corpus/tokenizer binding, corrected
ladder, frozen plans, existing runner and underlying three-step audit. The
wrapper writes `formal_release_created=false`; a passing timing result still
requires a separate human-reviewed release before formal Stage-1 training.

Three new tests cover the frozen order and release boundary, Bash syntax and
the wrapper's missing-input failure path. The complete focused suite passes
84/84 locally, the auditor compiles, and `git diff --check` passes. This
registration creates no GPU child and does not release formal training,
Stage 2, RULER or update-admission work.

The first remote launch attempt at commit `ca317132` exposed a fail-closed
watcher bug before output creation: it required the future full-corpus audit
while that audit was still being built. No watcher result root or CUDA child
was created. Commit `226a1198a462b2d132ccacb896a863d093873924` moves that
check to the point after the corrected ladder releases its inputs and records
their hashes there. The repaired clean remote worktree passes 84/84 tests.

The repaired watcher is live in tmux
`baseline_qwen3_4b_base_stage1_timing_226a1198_20261019`, with watch root
`/home/ctj/in_place_ttt_qwen3_4b_base_stage1_timing_226a1198_20261019.watch`.
Its authoritative state is `waiting_for_corrected_stage1_ladder`; the timing
result root is absent, while the corpus, Level A and resource ladder remain at
`RUNNING_BUILD`, `WAITING_BASE_CORPUS` and
`waiting_for_corrected_level_a`. The pane is live and GPU0/1 remain at 46/18
MiB with 0% utilization. Launch evidence is archived under
`remote/hku_gpu2_qwen3_4b_base_stage1_timing_watcher_20261019/`.

## 45. Corrected Base formal Stage-1 registration (2026-10-19)

`IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_100M_STAGE1_PLAN.md` freezes the matched
27-step 32K formal Stage-1 boundary. Both arms start from exact
Qwen3-4B-Base revision `906bfd4b...`, consume the same Base-tokenizer stream,
and run no-TTT before native TTT on `hku-gpu2` only. Each arm processes
56,623,104 tokens at global batch 64. Stage 2, RULER and update-admission work
remain sealed.

The corrected release tool first validates the existing timing audit and the
Base wrapper audit. Without explicit `--accept`, reviewer and decision note it
writes no release. An accepted record binds both audit hashes and the exact
Base revision while retaining the protocol consumed by the existing formal
runner. The watcher waits for the corrected timing terminal state, then waits
within its registered budget for this explicit release; it never creates or
infers acceptance.

Formal execution reuses the existing 27-step runner and paired auditor. A new
terminal wrapper re-hashes Base identity/tokenizer, complete corpus, both
timing audits, explicit release, plans, configs, runner and the paired formal
audit/checkpoint manifests. Four new tests cover the plan boundary, watcher
syntax/order, two-audit release behavior and missing-input failure closure.
The full focused suite passes 88/88 locally. No release or formal GPU child
exists at registration time.

Commit `997748173594ad7f26732a65fabf2327012f6a5b` passes the same 88/88
suite and static checks in a clean detached `hku-gpu2` worktree. Its CPU-only
watcher is live in tmux
`baseline_qwen3_4b_base_100m_stage1_99774817_20261019`, with watch root
`/home/ctj/in_place_ttt_qwen3_4b_base_100m_stage1_99774817_20261019.watch`.
The initial state is `waiting_for_corrected_stage1_timing`; the explicit
release and formal result root are absent. The timing watcher remains at
`waiting_for_corrected_stage1_ladder`, the corpus remains `RUNNING_BUILD`, and
GPU0/1 remain at 46/18 MiB with 0% utilization. Evidence is archived under
`remote/hku_gpu2_qwen3_4b_base_100m_stage1_watcher_20261019/`.

## 46. Corrected Base Stage-2 resource ladder registration (2026-10-19)

`IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_STAGE2_RESOURCE_PLAN.md` freezes the
post-Stage-1 64K/128K resource boundary. Each arm must start from its own
current corrected formal Stage-1 HF checkpoint. The fixed order is 64K no-TTT,
64K native TTT, 128K no-TTT and 128K native TTT, with one global-batch-2
optimizer step per child. GPU execution remains restricted to `hku-gpu2`.

The watcher defers the future full-corpus sparse inputs until corrected formal
Stage 1 has passed, then revalidates both underlying and corrected Stage-1
audits before invoking the existing Stage-2 ladder runner. A new terminal
wrapper re-hashes the Base identity, full-profile resource inputs, arm-specific
start-model manifests, four child audits, plans, configs and existing
runner/auditor. It produces resource evidence only; 128K timing, formal Stage
2, RULER and update admission remain sealed.

Three tests cover the frozen order/scope, Bash syntax and deferred-input
behavior, and missing-input failure closure. The complete focused suite passes
91/91 locally, the auditor compiles, and `git diff --check` passes. No Stage-2
resource child exists at registration time.

Commit `1098b08d2098a9d4bd6f600474cd3d2d2f3cc7a6` passes 91/91 and the
same static checks in a clean detached `hku-gpu2` worktree. Its watcher is
live in tmux `baseline_qwen3_4b_base_stage2_ladder_1098b08d_20261019`, with
watch root
`/home/ctj/in_place_ttt_qwen3_4b_base_stage2_ladder_1098b08d_20261019.watch`.
The initial state is `waiting_for_corrected_formal_stage1`; the Stage-2 result
root is absent, formal Stage 1 is waiting for corrected timing, and the corpus
is still building. GPU0/1 remain at 46/18 MiB and 0% utilization. Launch
evidence is archived under
`remote/hku_gpu2_qwen3_4b_base_stage2_ladder_watcher_20261019/`.

## 47. Paper-specified Qwen3 inference update clipping (2026-10-19)

The paper states that each Qwen3-4B inference fast-weight update delta is
rescaled to Frobenius norm `tau=1e-5` when it exceeds that boundary. Direct
inspection of clean upstream commit `be232482...` found that the released
Qwen3 inference path adds `dw` without clipping and exposes no corresponding
configuration field. No corrected-Base formal RULER score existed when this
gap was found.

`IN_PLACE_TTT_BASELINE_PAPER_INFERENCE_CLIP_ADDENDUM.md` is now frozen before
scoring. Released-code Level A remains an unmodified execution check. The
paper-facing public-data Level-B 2x5090 path instead builds an immutable
output-local overlay from all seven hash-verified clean upstream
`inference_model` files. It changes only Qwen3 configuration and modeling,
adds opt-in `ttt_update_clip_norm=None`, and clips each native update matrix
immediately before its existing addition. The clean upstream checkout,
checkpoints, writer, training and baseline defaults remain unchanged.

The registered OpenCompass 2x5090 config passes `1e-5` through
`model_kwargs`. Before every resource or formal run, a CPU-only validator
requires the imported model to resolve inside the overlay, verifies an
above-boundary update, zero-update behavior, invalid-value rejection,
serialization and the real `from_pretrained` override, and confirms CUDA was
not initialized. Single-run, resource-ladder and matrix audits independently
rehash the source/generated files and require the same frozen addendum hash
from every child.

On `hku-gpu2`, the official evaluation environment measured an unclipped
fixture norm of `24.207931518554688` and a clipped norm of
`1.0000000656873453e-5`; all CPU checks passed. The focused overlay/audit suite
passes 9/9 and the complete `tests/test_in_place_ttt*.py` suite passes 94/94
with all GPUs hidden. No RULER score or GPU job was started. The legacy 80-GB
runner is outside the user-authorized host path; all future GPU execution
remains restricted to `hku-gpu2`.

The immutable manifest, numerical validation JSON, clean-worktree test log and
static-check log are archived under
`remote/hku_gpu2_paper_inference_clip_20261019/` with a separate evidence
manifest.

## 48. Corrected Base Stage-2 timing registration (2026-10-19)

`IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_STAGE2_TIMING_PLAN.md` freezes the 128K
timing boundary at global batch 64 and exactly three optimizer steps per arm.
No-TTT runs before native TTT. Step one is warm-up; the median of steps two
and three projects each registered five-step formal Stage-2 arm and their
sequential total. All GPU execution remains restricted to `hku-gpu2`.

The CPU-only watcher waits for the corrected Stage-2 ladder terminal state.
It then verifies the underlying four-boundary audit, corrected Base wrapper
audit and corrected formal Stage-1 binding before releasing the complete
128K corpus to the existing timing runner. Future corpus checks occur only
after the predecessor releases them, so a pending chain neither touches CUDA
nor fails because a future audit is absent.

The new terminal wrapper re-hashes Base identity, formal Stage-1 evidence,
both Stage-2 ladder audits, complete corpus, plans, configs and the existing
runner/auditor. It requires three finite progress records per arm and a
positive five-step projection, and records `formal_release_created=false`.
Three tests cover plan/order/release scope, Bash syntax with deferred inputs,
and missing-input failure closure. Registration starts no GPU child and does
not release formal Stage 2, RULER or update-admission work.

Commit `80c414c7232dfb33abe2b4f5294bf8b8fd49a042` is pushed and passes 98
focused tests in a clean detached `hku-gpu2` worktree with all GPUs hidden;
five environment-specific tests are skipped. Python compilation, Bash syntax,
diff and clean-worktree checks also pass. The CPU-only watcher is live in tmux
`baseline_qwen3_4b_base_stage2_timing_80c414c7_20261019`, with watch root
`/home/ctj/in_place_ttt_qwen3_4b_base_stage2_timing_80c414c7_20261019.watch`.
Its initial state is `waiting_for_corrected_stage2_ladder`, the timing result
root is absent, and GPU0/1 remain at 46/18 MiB with 0% utilization. Evidence
is archived under
`remote/hku_gpu2_qwen3_4b_base_stage2_timing_watcher_20261019/`.

## 49. Corrected Base formal Stage-2 registration (2026-10-19)

`IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_100M_STAGE2_PLAN.md` freezes the matched
five-step 128K formal Stage-2 boundary. Each arm starts from its own audited
formal Stage-1 HF checkpoint with a fresh optimizer/scheduler. No-TTT runs
before native TTT; each arm consumes 41,943,040 tokens at global batch 64.
All GPU execution remains restricted to `hku-gpu2`.

The corrected release tool validates the underlying Stage-2 timing audit and
the Base wrapper audit. Without explicit `--accept`, reviewer and decision
note it writes no release. An accepted record binds both audit hashes and the
exact Base revision while retaining the protocol consumed by the existing
formal runner. The CPU-only watcher waits for the corrected timing terminal
state, then for this explicit release; it never creates or infers acceptance.

Formal execution reuses the existing five-step Stage-2 runner and paired
auditor. A new terminal wrapper re-hashes Base identity/tokenizer, complete
corpus, corrected Stage-1, both resource/timing audit layers, release, plans,
configs, runner and the paired final checkpoint audit. Four new tests cover
the plan boundary, watcher syntax/order, two-audit release behavior and
missing-input failure closure. The complete focused suite passes 102/102
locally. Registration creates no release or GPU child and keeps RULER and
update-admission work sealed.

Commit `a21afd2799a33a42469676c8b6cb5769371c0544` is pushed and passes 102
focused tests in a clean detached `hku-gpu2` worktree with all GPUs hidden;
five environment-specific tests are skipped, and all static checks pass. The
CPU-only watcher is live in tmux
`baseline_qwen3_4b_base_100m_stage2_a21afd27_20261019`, with watch root
`/home/ctj/in_place_ttt_qwen3_4b_base_100m_stage2_a21afd27_20261019.watch`.
Its initial state is `waiting_for_corrected_stage2_timing`; both the formal
result root and explicit release are absent. GPU0/1 remain at 46/18 MiB and
0% utilization. Evidence is archived under
`remote/hku_gpu2_qwen3_4b_base_100m_stage2_watcher_20261019/`.

## 50. Corrected Base paper-facing RULER chain registration (2026-10-19)

Review of the existing 2x5090 RULER path confirmed that it already preserves
the registered 13 tasks, tokenizer, deterministic decoding, 4K--128K matrix
and paper-specified per-update Frobenius `1e-5` clip. Its entry gate, however,
accepted only the underlying formal Stage-2 audit and did not require the
corrected Base wrapper terminal state.

`IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_RULER_PLAN.md` now freezes this corrected
boundary. The paired resource ladder uses 4K/32K/64K/128K with one example
per task and discards every score. The formal matrix uses
4K/8K/16K/32K/64K/128K with 100 examples for each of all 13 tasks. GPU work
remains restricted to `hku-gpu2`; 256K remains a separate extrapolation gate.

Two Base-specific CPU watchers require `completed_corrected_formal_stage2`
and then `completed_corrected_ruler_resource` before reusing the existing
OpenCompass runners. Two terminal wrappers bind Base identity/tokenizer,
corrected prerequisites, paper inference addendum, plans, raw matrix summary
and underlying resource/matrix audits. No task, input, decoding or metric is
changed. Four new tests cover the frozen scope, watcher order/Bash syntax and
missing-input failure closure. The complete focused suite passes 106/106
locally. No RULER GPU child or score exists at registration time; only the
passing corrected matrix audit can release update-admission work.

Commit `c89c6f8c91e2b6bb435142e6fc87a221018be585` is pushed and passes 106
focused tests in a clean detached `hku-gpu2` worktree with all GPUs hidden;
five environment-specific tests are skipped and all static checks pass. The
CPU-only resource and matrix watchers are live in tmux
`baseline_qwen3_4b_base_ruler_resource_c89c6f8c_20261019` and
`baseline_qwen3_4b_base_ruler_matrix_c89c6f8c_20261019`. Their initial
states are `waiting_for_corrected_formal_stage2` and
`waiting_for_corrected_ruler_resource`; both result roots are absent, and
GPU0/1 remain at 46/18 MiB with 0% utilization. Evidence is archived under
`remote/hku_gpu2_qwen3_4b_base_ruler_resource_watcher_20261019/` and
`remote/hku_gpu2_qwen3_4b_base_ruler_matrix_watcher_20261019/`.

## 51. Corrected Base 100M selected-stream audit registration (2026-10-19)

The former selected-stream artifact is non-authoritative because it consumed
the superseded tokenizer corpus and iterated the Hugging Face stream directly.
It did not exercise the `StatefulDataLoader(num_workers=2)` path used by the
formal configs.

`IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_SELECTED_STREAM_ADDENDUM.md` freezes a
replacement before the corrected corpus publishes and before either formal
release. The replay uses seed 42, shuffle buffer 10,000, two DP ranks, 32
records per rank, 27 Stage-1 steps and five Stage-2 steps. It builds a compact
15,000-row prefix from each provenance file containing only stage index,
source and training-token SHA-256, then invokes the installed datasets node
split and torchdata StatefulDataLoader. A separate process independently
rebuilds both prefixes and schedules and matches all ordered digests, counts
and rank-disjointness checks.

Each formal stage is one JSONL shard. Although the frozen config requests two
workers, datasets 2.21.0 reports one shard and stops one excess worker. The
artifact therefore records both requested workers 2 and effective workers 1.
This is the actual formal input behavior rather than a resource assumption.

Both formal release creators now require this passing audit and bind the
applicable stage's ordered index/hash digests and corrected corpus manifest.
The selected-stream result does not release training: an explicit human
timing review, `--accept`, reviewer and decision note remain mandatory.

On `hku-gpu2` with both GPUs hidden, the synthetic implementation replay
covered 1,728 32K records plus 320 128K records, or 98,566,144 total selected
tokens. The main replay passed in about 9.9 seconds with about 857 MiB peak
RSS; the independent reconstruction passed in about 4.8 seconds with about
707 MiB peak RSS. The 12 focused tests pass. A broad 928-test run had 921
passes, five environment skips and two unrelated existing utility-record
fixture errors.

Commit `2b8644a0ed0dcf2e8d94a9b51360e8428db695de` is pushed and deployed as a
clean detached checkout. The CPU-only watcher is live in tmux
`baseline_qwen3_4b_base_selected_stream_2b8644a0_20261019`, output root
`/home/ctj/in_place_ttt_qwen3_4b_base_selected_stream_2b8644a0_20261019`,
state `waiting_for_corrected_base_corpus`. At registration GPU0/1 used 46/18
MiB at 0% utilization. Evidence is archived under
`remote/hku_gpu2_qwen3_4b_base_selected_stream_20261019/`.

## 52. Corrected formal-release review packet registration (2026-10-19)

The formal timing gates intentionally stop before writing an accepted release.
To make the eventual human decision concrete and auditable,
`IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_RELEASE_REVIEW_ADDENDUM.md` freezes a
CPU-only review stage for both formal training stages.

The review builder reuses the exact validation functions from each formal
release creator. It combines the underlying and corrected timing audits,
measured arm and sequential projections, Base revision, corrected
selected-stream audit, corpus-manifest hash, applicable ordered stream
digests, and tool/plan hashes. Its immutable protocol requires
`ready_for_human_review=true` and `formal_release_created=false`.

The generic watcher waits for both the applicable corrected timing result and
the corrected selected-stream result while checking both tmux handles. It
fails closed on any terminal prerequisite error, stale binding, missing live
handle, pre-existing target release, or output reuse. It runs with all GPUs
hidden. There is no `--accept` call and it cannot write the formal release or
launch training. A human must still inspect the packet and separately provide
acceptance, reviewer identity and a decision note.

Commit `d6d47cb041aa4630a10f5def7ae652b5544d2984` passes all 15 related tests
locally and in a clean detached `hku-gpu2` checkout. The Stage-1 and Stage-2
watchers are live in tmux
`baseline_qwen3_4b_base_stage1_release_review_d6d47cb0_20261019` and
`baseline_qwen3_4b_base_stage2_release_review_d6d47cb0_20261019`, currently
waiting for their timing and selected-stream prerequisites. GPU0/1 used 46/18
MiB at 0% utilization at registration. Evidence is archived under
`remote/hku_gpu2_qwen3_4b_base_release_review_watchers_20261019/`.

## 53. Review-bound formal-release transition (2026-10-19)

`IN_PLACE_TTT_BASELINE_QWEN3_4B_BASE_RELEASE_TRANSITION_ADDENDUM.md` was frozen
before either formal release existed. It defines the sole supported path from
a non-authorizing review packet to an accepted release. The target path must
match the packet, every timing and selected-stream input must retain its exact
SHA-256, the applicable release validator source must be unchanged, and the
entire embedded review is independently recomputed before acceptance.

`create_in_place_ttt_qwen3_4b_base_release_from_review.py` then invokes the
existing Stage-1 or Stage-2 release creator in a temporary directory and
atomically writes a previously absent target. It additionally binds the review
packet and transition plan hashes. The command still requires explicit
`--accept`, a nonempty reviewer and a nonempty decision note. No watcher calls
this command, and neither a passing packet nor elapsed time supplies human
acceptance.

Two new tests prove that an omitted acceptance flag creates no release and
that changing a selected-stream audit after review is rejected. A static test
also excludes the transition tool from the review watcher. The combined 17
focused tests pass locally and in the clean detached `hku-gpu2` checkout at
commit `ccf17a70bc41b899b976d69be03730abfdebb36a`, with
`CUDA_VISIBLE_DEVICES=` on the remote run.

At the 06:40 Asia/Shanghai snapshot, the CPU corpus builder remained live and
had written 61,296 of 610,368 Stage-1 records. GPU0/1 used only 46/18 MiB with
0% utilization; no review packet, accepted release or new GPU child existed.
All GPU execution remains restricted to `hku-gpu2`. Test and chain snapshots
are archived under
`remote/hku_gpu2_qwen3_4b_base_release_transition_20261019/`.
