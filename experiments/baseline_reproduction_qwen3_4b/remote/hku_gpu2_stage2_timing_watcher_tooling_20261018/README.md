# hku-gpu2 Stage-2 timing watcher verification

This directory records the CPU-only verification and live launch of
`watch_and_run_in_place_ttt_2x5090_stage2_timing.sh` on `hku-gpu2`.

- Exact tool commit: `fbc02a6e6ad88834b2bb2533906264ddd670d2f3`.
- The isolated checkout was clean (`tool_status.txt`).
- The focused baseline suite passed 63/63 (`focused_tests.txt`).
- Bash syntax validation passed (`bash_syntax.txt`).
- A deliberately missing upstream Stage-2 ladder tmux handle exited 6 with
  state `stage2_ladder_watcher_handle_missing`; it created no final or partial
  timing output (`negative_exit_code.txt`, `negative_state.txt`).
- GPU snapshots before and after the negative test stayed at 46/18 MiB with
  zero utilization (`gpu_before.csv`, `gpu_after.csv`).
- The real watcher was launched in tmux session
  `baseline_2x5090_stage2_timing_fbc02a6e_20261018`. At
  `2026-10-18T12:45:04+08:00`, pane PID 1425558 was live and the watcher state
  was `waiting_for_audited_stage2_ladder` (`live_handle.txt`,
  `live_state.txt`).
- The live GPU snapshot was still 46/18 MiB at zero utilization
  (`live_gpu.csv`). The formal corpus partial directory was 57 GB
  (`corpus_partial_size.txt`).

The watcher is restricted to `hku-gpu2`. It does not create the explicit
post-timing release and therefore cannot launch formal Stage 2 by itself.
