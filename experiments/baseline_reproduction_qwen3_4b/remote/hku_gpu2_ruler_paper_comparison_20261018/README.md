# hku-gpu2 paper-reference RULER verification

This directory records remote verification of the machine-audited Table 1
comparison and replacement of the two still-waiting RULER watchers.

- Exact tool commit: `439fcea142c4d3bb51413d75480d5231d0c62cab`.
- The isolated checkout was clean (`tool_status.txt`).
- The six RULER tests and all 63 focused baseline tests passed
  (`ruler_tests.txt`, `focused_tests.txt`).
- Both old `5603910c` watchers were confirmed live and still in their initial
  waiting states, with no final or partial GPU output, before replacement
  (`superseded_*_handle.txt`).
- The replacement resource watcher is live in tmux session
  `baseline_2x5090_ruler_resource_439fcea1_20261018`, pane PID 1430095, state
  `waiting_for_audited_formal_stage2`.
- The replacement matrix watcher is live in tmux session
  `baseline_2x5090_ruler_matrix_439fcea1_20261018`, pane PID 1430136, state
  `waiting_for_audited_ruler_resource_ladder`.
- Replacement completed at `2026-10-18T13:04:10+08:00`. GPU 0/1 remained at
  46/18 MiB with zero utilization (`replacement_gpu.csv`).

No training or evaluation GPU process was interrupted. Only the two CPU-only
RULER waiters were superseded so the eventual formal matrix uses the frozen
paper-reference comparison and independent audit.
