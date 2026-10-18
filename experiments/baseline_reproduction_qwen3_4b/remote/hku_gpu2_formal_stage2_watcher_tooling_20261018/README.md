# hku-gpu2 formal Stage-2 watcher verification

This directory records CPU-only verification and the live launch of
`watch_and_run_in_place_ttt_2x5090_100m_stage2.sh` on `hku-gpu2`.

- Exact tool commit: `73950ecd1d0a8c2163863dce48de272e5db7d8d8`.
- The isolated checkout was clean (`tool_status.txt`).
- The focused baseline suite passed 63/63 (`focused_tests.txt`).
- Bash syntax validation passed (`bash_syntax.txt`).
- A deliberately missing Stage-2 timing tmux handle exited 6 in state
  `stage2_timing_watcher_handle_missing`; no final or partial formal training
  output was created (`negative_exit_code.txt`, `negative_state.txt`).
- GPU snapshots before and after that negative run stayed at 46/18 MiB with
  zero utilization (`gpu_before.csv`, `gpu_after.csv`).
- The real watcher was launched in tmux session
  `baseline_2x5090_formal_stage2_73950ecd_20261018`. At
  `2026-10-18T12:51:11+08:00`, pane PID 1426724 was live, state was
  `waiting_for_audited_stage2_timing`, and the explicit release was absent
  (`live_handle.txt`, `live_state.txt`, `live_release.txt`).
- The live GPU snapshot remained 46/18 MiB at zero utilization
  (`live_gpu.csv`); the corpus partial directory was 58 GB
  (`corpus_partial_size.txt`).

The watcher and all future GPU work are restricted to `hku-gpu2`. Formal
Stage 2 cannot launch until the timing audit passes and a separate explicit
release is written.
