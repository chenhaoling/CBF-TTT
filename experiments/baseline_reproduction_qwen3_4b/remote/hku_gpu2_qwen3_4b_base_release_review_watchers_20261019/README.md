# Corrected Base formal-release review watcher evidence

This directory records the CPU-only registration of the Stage-1 and Stage-2
formal-release review watchers at commit
`d6d47cb041aa4630a10f5def7ae652b5544d2984`.

- Host: `hku-gpu2` (`user-Super-Server`).
- Focused tests: 15/15 passed with `CUDA_VISIBLE_DEVICES=`.
- Stage-1 tmux:
  `baseline_qwen3_4b_base_stage1_release_review_d6d47cb0_20261019`.
- Stage-1 output:
  `/home/ctj/in_place_ttt_qwen3_4b_base_stage1_release_review_d6d47cb0_20261019`.
- Stage-1 state: `waiting_for_stage1_timing_and_selected_stream`.
- Stage-2 tmux:
  `baseline_qwen3_4b_base_stage2_release_review_d6d47cb0_20261019`.
- Stage-2 output:
  `/home/ctj/in_place_ttt_qwen3_4b_base_stage2_release_review_d6d47cb0_20261019`.
- Stage-2 state: `waiting_for_stage2_timing_and_selected_stream`.
- Registration GPU snapshot: GPU0 46 MiB, GPU1 18 MiB, both 0% utilization.

The watchers can only write non-authorizing review packets. They contain no
`--accept` path and cannot write either target formal release. After both
prerequisite audits pass, each packet will bind the measured projection,
selected-stream digests, exact corpus manifest and tool hashes. A separate
human decision remains necessary before formal training can start.
