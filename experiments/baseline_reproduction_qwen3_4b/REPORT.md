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
