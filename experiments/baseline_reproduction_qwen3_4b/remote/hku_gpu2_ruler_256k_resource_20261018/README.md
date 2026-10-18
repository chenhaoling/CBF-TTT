# hku-gpu2 RULER-256K resource-gate verification

This directory records CPU-only verification and live launch of the separate
RULER-256K extrapolation resource watcher on `hku-gpu2`.

- Exact tool commit: `b33b93ace74240b2bb2332368647c4adda31b7dc`.
- The isolated checkout was clean (`tool_status.txt`).
- The dedicated test passed 1/1 and the focused baseline suite passed 64/64
  (`specific_tests.txt`, `focused_tests.txt`).
- Both Bash scripts parsed and the Python auditor compiled
  (`syntax_compile.txt`).
- A deliberately missing formal matrix handle exited 6 in state
  `ruler_matrix_watcher_handle_missing`; no final or partial resource output
  was created (`negative_exit_code.txt`, `negative_state.txt`).
- GPU snapshots before and after the negative check stayed at 46/18 MiB with
  zero utilization (`gpu_before.csv`, `gpu_after.csv`).
- The real watcher is live in tmux session
  `baseline_2x5090_ruler_256k_resource_b33b93ac_20261018`. At
  `2026-10-18T13:15:39+08:00`, pane PID 1431970 was live in state
  `waiting_for_audited_ruler_matrix` (`live_handle.txt`, `live_state.txt`).
- The live GPU snapshot remained 46/18 MiB at zero utilization; the corpus
  partial directory had reached 60 GB (`live_gpu.csv`,
  `corpus_partial_size.txt`).

The watcher cannot launch until the complete paper-aware 4K--128K matrix and
its exact independent audit pass. A successful resource run still does not
automatically execute formal 256K scoring.
