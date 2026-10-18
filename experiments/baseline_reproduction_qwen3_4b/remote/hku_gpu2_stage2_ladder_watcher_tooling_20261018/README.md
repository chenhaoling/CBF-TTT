# hku-gpu2 Stage-2 ladder watcher verification

This directory records CPU-only verification of clean commit
`1b04d8cf55821056e1dfb3a724d3ded1a87ebffc` on `hku-gpu2` on 2026-10-18.

- The complete focused baseline suite passed 63/63.
- The Stage-2 ladder watcher passed `bash -n`.
- A negative preflight with a deliberately absent formal Stage-1 tmux session
  exited 6 with state `formal_stage1_watcher_handle_missing`.
- The negative preflight created no ladder output and launched no CUDA process.
- The validated watcher was then launched as
  `baseline_2x5090_stage2_ladder_1b04d8cf_20261018`. Its pane PID 1424039 was
  live in state `waiting_for_audited_formal_stage1`; GPU 0/1 remained at
  46/18 MiB and 0% utilization.

`bash_syntax.txt`, `git_status.txt` and `negative_preflight.log` are empty
successful/no-diagnostic outputs. Frozen hashes, the missing-handle snapshot,
negative terminal state and live watcher snapshot are retained separately.
