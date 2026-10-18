# hku-gpu2 formal RULER-256K tooling validation

- Host: `user-Super-Server` (`hku-gpu2`)
- Tool commit: `a0375e93a530591ca3e36d6d4c8498c4772d8784`
- Clean worktree: `/home/ctj/cbf_ttt_tool_a0375e93`
- Validation: 66/66 focused tests, Bash syntax and Python compilation passed.
- Negative preflight: missing resource watcher handle exited 6 with
  `ruler_256k_resource_watcher_handle_missing`; no final or partial GPU output
  was created.
- Live watcher:
  `baseline_2x5090_ruler_256k_formal_a0375e93_20261018`, pane PID 1435575.
- Initial state: `waiting_for_audited_ruler_256k_resource`.
- Explicit release and formal output were absent at launch.
- GPU 0/1 were idle at 46/18 MiB. All GPU work remains restricted to
  `hku-gpu2`.

`remote_validation.txt` contains the remote clean-worktree test output.
`watcher_status.txt` records the successful launch observation and the
negative-preflight outcome.
