# Preliminary Level-A resource stop

This directory archives compact, text-free evidence from the first corrected
Qwen3-4B-Base Level-A attempt using the independently audited
`preliminary_100m` corpus. The run used both RTX 5090 devices on `hku-gpu2`.

## What completed

- Corpus release and derived-input audits passed.
- The no-TTT arm completed one real AdamW optimizer step at 8K with global
  batch size 2.
- Loss was 2.0122 and gradient norm was 1.9313; both were finite.
- The full DCP checkpoint was written successfully and occupied
  49,846,590,933 bytes.
- The measured wall time was 48.81 seconds; the optimizer step took 6.15
  seconds.

## Frozen stop condition

The no-TTT child audit failed exactly the predeclared resource gates:

- GPU 0/1 peaks were 31,622/31,594 MiB of 32,607 MiB, leaving
  3.02%/3.11% rather than the required 5%.
- Minimum host available memory was 7,216.37 MiB rather than the required
  8,192 MiB.
- Peak swap use was 424.93 MiB and remained below its 3.5 GiB limit.
- There was no OOM, nonfinite loss or nonfinite gradient.

The trace separates the two pressures. The GPU peak occurs during the optimizer
step. Host use then rises during the 49.85 GB optimizer-bearing DCP save. The
TTT arm, conversion and RULER smoke were not started.

## Wrapper-state defect

The child correctly returned nonzero, and the tmux process exited, but the
parent status file remained `RUNNING_RESOURCE_NO_TTT`. Bash `errexit` did not
invoke the generic trap at the function boundary. The watcher now handles each
arm failure explicitly and writes `FAILED_${CURRENT_STAGE}` before exit. This
fix changes orchestration only; the model, writer, optimizer, configs and
existing result are untouched.

## Integrity

- Resource-input manifest SHA256:
  `5aead49455bdcdd2ebdc4082d3ff13ff8cc8b428f14563f291a15944c8cf78e6`
- Resource-input audit SHA256:
  `1751e09fcf6b7bd1789a5484b0f05d328024b9b24b65de0cdce797ff89bffcf5`
- no-TTT summary SHA256:
  `203fda9c08ee9e4e93e87f79af5385aa14e4e373f5fe42f130b23c53a8ee3b4e`
- no-TTT audit SHA256:
  `5299fc56fc5b0f466e04e9799c4f14a43fcc54e2307c089f33acd4a903f4cbf2`

This result is a resource stop, not evidence for or against TTT efficacy.
