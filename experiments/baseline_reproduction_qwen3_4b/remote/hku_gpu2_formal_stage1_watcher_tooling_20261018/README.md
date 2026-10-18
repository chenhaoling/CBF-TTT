# hku-gpu2 formal Stage-1 watcher verification

This directory records CPU-only verification of clean commit
`3c7e29fc33cbb781090db6792e4fef42dec6dbcf` on `hku-gpu2` on 2026-10-18.

- The complete focused In-Place-TTT baseline suite passed 63/63.
- The formal Stage-1 watcher passed `bash -n`.
- A negative preflight used a deliberately absent timing tmux session. It
  exited 6 with state `stage1_timing_watcher_handle_missing`.
- The negative preflight created neither final nor partial formal-training
  output and launched no CUDA process.
- GPU 0/1 remained at 46/18 MiB and 0% utilization. The corpus builder and
  repaired 8K watcher remained live.
- The validated watcher was then launched in tmux session
  `baseline_2x5090_formal_stage1_3c7e29fc_20261018`. At
  `2026-10-18T12:25:20+08:00`, pane PID 1422177 was live with state
  `waiting_for_audited_stage1_timing`; no release, final output or partial
  output existed, and both GPUs remained idle.

`bash_syntax.txt`, `git_status.txt` and `negative_preflight.log` are
intentionally empty successful/no-diagnostic outputs. The watcher state,
missing-handle snapshot and frozen dependency hashes are stored separately.
