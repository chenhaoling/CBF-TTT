# Corrected Base selected-stream registration evidence

This directory records the pre-outcome implementation smoke and live watcher
registration for commit `2b8644a0ed0dcf2e8d94a9b51360e8428db695de`.

- Host: `hku-gpu2` (`user-Super-Server`).
- GPU policy: every replay, audit and test ran with `CUDA_VISIBLE_DEVICES=`.
- Focused result: 12/12 tests passed.
- Synthetic native replay: passed for 1,728 Stage-1 records and 320 Stage-2
  records, totalling 98,566,144 selected training tokens.
- Independent reconstruction: passed and matched every ordered digest and
  source aggregate.
- Configured workers: 2. Effective workers: 1, because each formal stage is a
  single JSONL shard and the installed Hugging Face runtime stops the excess
  worker. This is recorded explicitly in the replay artifact.
- Installed runtime: Python 3.11.16, datasets 2.21.0, torchdata 0.11.0 and
  VeOmni 0.1.0.
- Live CPU watcher:
  `baseline_qwen3_4b_base_selected_stream_2b8644a0_20261019`.
- Live output root:
  `/home/ctj/in_place_ttt_qwen3_4b_base_selected_stream_2b8644a0_20261019`.
- Registration state: `waiting_for_corrected_base_corpus`.
- GPU snapshot at registration: GPU0 46 MiB, GPU1 18 MiB, both 0% utilization.

The synthetic artifact contains only stage indices, source labels and SHA-256
digests. It is an implementation smoke, not the authoritative corrected-corpus
result. The watcher will create that result only after the full Base-tokenized
corpus and canonical independent audit publish successfully. It never creates
a formal training release.

The broad repository suite observed 928 tests: 921 passed, five were skipped
for unavailable upstream fixtures, and two unrelated pre-existing utility
record fixture tests errored because their synthetic group lacked five stable
relations. The focused baseline suite is archived here and is authoritative
for this change.
