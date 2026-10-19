# Qwen3-4B-Base two-RTX-5090 no-headroom capacity addendum

Status: **frozen before execution**

Date: 2026-10-19

Execution host: `hku-gpu2` only

## Purpose

The preceding clean serial Level-A retry showed that native TTT can complete
one 8K optimizer step but retains only approximately 2% GPU memory headroom.
At the user's direction, this separate diagnostic tests whether two RTX 5090
devices can sustain native-TTT training when the earlier 5% headroom threshold
is treated as an observation rather than a stopping rule.

This addendum does not amend, supersede, or retroactively pass the frozen
Level-A resource gate. Its output cannot release Stage 1, checkpoint conversion,
RULER scoring, or update-admission experiments.

## Frozen workload

- model: exact local `Qwen/Qwen3-4B-Base` revision registered by the baseline;
- upstream code: clean commit `be2324829b0e91c8fd10a74d4b43714fde6676e1`;
- hardware: both 32 GiB RTX 5090 devices on `hku-gpu2`;
- branch: native TTT with the released `ttt_conv`/`ttt_proj` writer unchanged;
- sequence length: 8,192 tokens;
- TTT chunk: 4,096 tokens;
- global/micro batch size: 2/1;
- optimizer: the existing fused AdamW path;
- duration: 10 consecutive optimizer steps from the Base checkpoint;
- data: the independently audited preliminary corpus Stage-1 stream;
- checkpoint: one DCP save after step 10; no intermediate save and no HF save;
- allocation: activation offload and `expandable_segments:True` as in the clean
  Level-A retry.

## Preflight and measurements

Execution requires both GPUs to report at most 128 MiB used, no registered
CUDA compute process, at least 48 GiB host memory available, a clean upstream
checkout, and at least 80 GB free disk. The runner records 0.5-second GPU and
host-memory traces, exact input/config/addendum hashes, the raw training exit
code, and checkpoint size.

## Decision rule

The diagnostic passes only if all ten steps finish, the process exits zero,
the final logged loss and gradient norm are finite, no OOM/nonfinite signature
appears, and a nonempty step-10 DCP exists. GPU free fraction and host-memory
minima are reported but have no pass/fail threshold in this diagnostic.

An OOM, killed process, missing step, nonfinite value, missing checkpoint, or
nonzero exit is a capacity failure. A pass means only that the exact 8K
workload can execute for ten steps on two 5090s without a reserved safety
margin; it is not an accuracy, convergence, stability-at-scale, or paper
reproduction result.
