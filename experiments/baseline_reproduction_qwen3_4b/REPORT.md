# In-Place TTT Qwen3-4B baseline reproduction report

Date: 2026-10-16  
Frozen plan: [`IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md`](../../IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md)  
Plan SHA256: `fbce7e0e52c495467c5b6383242a7350cf16042d61af8795309b9d6eda021fd3`  
Official source commit: `be2324829b0e91c8fd10a74d4b43714fde6676e1`

## 1. Conclusion

The released-code evaluation pipeline is operational, but the paper baseline
has **not** been reproduced yet.

- A clean official checkout and isolated official-version environment load on
  `hku-gpu2`.
- The official Qwen3-4B training path reaches model loading, the full forward
  pass, backward pass, and `optimizer.step()` for both matched arms.
- Both 8K resource probes OOM while AdamW initializes optimizer state. The
  current shared 2xRTX 5090 host therefore fails the frozen training resource
  gate before a single completed optimizer step.
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
  the run still peaks at 32,086/32,103 MiB. A truly exclusive 32-GB rerun
  remains unmeasured because 724/644 MiB of unrelated CARLA use was still
  present.
- The unmodified Qwen3-4B OpenCompass RULER-4K pipeline completes all 13 tasks.
  Its one-example-per-task smoke score is `76.9231`; this validates execution
  and score extraction only and is not a precision estimate comparable to the
  paper's 100-example result.
- The paper's exact continual-pretraining corpus and several stage settings
  are not public. FineWeb-Edu plus LongCrawl64 remains a declared public-data
  substitute and cannot be called the original paper dataset.

Consequently, Level A is partially complete, Level B has not started formal
training, and Level C remains blocked by private data/configuration. Update
admission and forgetting-gate exploration remains paused.

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
- `scripts/summarize_in_place_ttt_baseline_resource.py`;
- `eval_config/baseline_reproduction/qwen3_4b_base_4k_smoke.py`;
- `scripts/run_in_place_ttt_ruler_smoke.sh`;
- `scripts/summarize_in_place_ttt_ruler_smoke.py`;
- two focused summary tests.

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

### Alternate host check

At `2026-10-16 23:13:47 HKT`, `hku-a100` exposed one
`NVIDIA A100-SXM4-80GB` (81,920 MiB). A `VLLM::EngineCore` process occupied
75,124 MiB, leaving 6,007 MiB, so the host was not available for this run.
Host RAM was 125 GiB total and 82 GiB available, while its 2-GiB swap was
fully used. If the A100 becomes exclusive, its 80-GiB capacity warrants a new
single-GPU resource pilot; it is not a drop-in execution of the frozen
two-GPU FSDP2 protocol.

Snapshot: [`remote/hku_a100_resource_snapshot.json`](remote/hku_a100_resource_snapshot.json).

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

## 6. Verification

- CUDA bf16 tensor creation and FlashAttention forward passed on RTX 5090.
- Official source checkout commit and clean status were checked before every
  run.
- Plan, config, data, model-config, log, GPU trace, and result hashes are stored
  in the compact summaries.
- Focused resource and RULER summarizer tests: `2 passed`.
- RULER task set is checked against an explicit 13-task allowlist.
- Official group score is independently recomputed from task result JSONs.

## 7. Commands

Environment setup:

```bash
bash scripts/setup_in_place_ttt_official_env.sh
bash scripts/setup_in_place_ttt_official_eval_env.sh
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

RULER smoke:

```bash
UPSTREAM_REPO=/home/ctj/in_place_ttt_official_be232482 \
PYTHON_BIN=/home/ctj/miniconda3/envs/in_place_ttt_official_eval/bin/python \
MODEL=/home/ctj/models/Qwen3-4B OUTPUT_ROOT=/new/output/ruler_4k_smoke \
PLAN=$PWD/IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md TOOL_REPO=$PWD \
bash scripts/run_in_place_ttt_ruler_smoke.sh
```

## 8. Next required experiment

1. Allocate an exclusive host with enough memory for the frozen recipe,
   preferably 4--8 GPUs with at least 80GB each, or rerun 8K on exclusive
   5090s to determine whether unrelated processes alone caused the boundary.
   `hku-a100` cannot currently serve this role because its only 80-GB GPU has
   75,124 MiB allocated to vLLM.
2. Repeat both 8K arms from new output roots. Only after both complete, measure
   16K and 32K; attempt 64K only if 32K retains at least 15% free memory.
3. On the selected host, construct a fully hashed FineWeb-Edu/LongCrawl64
   public substitute and train byte-identical no-TTT/TTT token streams.
4. Convert both final DCP checkpoints and execute the 100-example RULER matrix
   from 4K through 256K once.
5. Report it as a public-data matched replication. Resume support/update gates
   only after this baseline report is complete.

Exact paper reproduction additionally requires the authors' private mixture,
32K/128K stage recipe, and inference clipping details.

The current host has now exhausted all preregistered, unmodified released-code
paths: native FSDP2 is GPU-memory limited even after the allocator-only retry,
its offload flag is not implemented, and FSDP1 offload is host-memory limited.
Continuing training requires an external-state change: exclusive larger GPUs
or a host with materially more RAM for the explicitly non-paper offload path.
