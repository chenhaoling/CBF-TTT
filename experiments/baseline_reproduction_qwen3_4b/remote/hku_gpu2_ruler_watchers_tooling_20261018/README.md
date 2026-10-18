# hku-gpu2 persistent RULER watcher verification

This directory records CPU-only verification and live launch of the paired
RULER resource and formal matrix watchers on `hku-gpu2`.

- Exact tool commit: `5603910cdadf617b7073d689967dfa9c26a73449`.
- The isolated checkout was clean (`tool_status.txt`).
- The focused baseline suite passed 63/63 (`focused_tests.txt`).
- The matrix runner and both watcher scripts passed Bash syntax validation
  (`bash_syntax.txt`).
- The registered evaluation environment imported OpenCompass and reported
  PyTorch 2.8.0+cu128 and Transformers 4.57.3
  (`eval_environment.txt`).
- A missing formal Stage-2 handle made the resource watcher exit 6 in state
  `formal_stage2_watcher_handle_missing`; a missing resource-watcher handle
  made the matrix watcher exit 6 in state
  `ruler_resource_watcher_handle_missing`. Neither negative run created a GPU
  output (`resource_negative_*`, `matrix_negative_*`).
- GPU snapshots before and after verification remained 46/18 MiB with zero
  utilization (`gpu_before.csv`, `gpu_after.csv`).
- The real resource watcher was launched in tmux session
  `baseline_2x5090_ruler_resource_5603910c_20261018`; at
  `2026-10-18T12:58:00+08:00`, pane PID 1428296 was live in state
  `waiting_for_audited_formal_stage2`.
- The real matrix watcher was launched in tmux session
  `baseline_2x5090_ruler_matrix_5603910c_20261018`; pane PID 1428336 was live
  in state `waiting_for_audited_ruler_resource_ladder`.
- The live GPU snapshot was still 46/18 MiB at zero utilization and the
  public-corpus partial directory was 58 GB (`live_gpu.csv`,
  `corpus_partial_size.txt`).

All GPU work is restricted to `hku-gpu2`. The watchers can advance only after
the exact upstream audit and tmux handle gates pass.
