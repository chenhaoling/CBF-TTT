# Reviewed release transition evidence

This directory records the CPU-only validation of the formal-release
transition at commit `ccf17a70bc41b899b976d69be03730abfdebb36a` on
`hku-gpu2` (`user-Super-Server`).

- `remote_focused_tests.txt`: exact remote commit, hidden CUDA environment and
  the 17 passing review/selected-stream/formal-release tests.
- `remote_chain_snapshot.txt`: GPU snapshot, live corpus-builder process,
  partial Stage-1 record counts and registered tmux watcher handles at
  2026-10-19 06:40 Asia/Shanghai.

No review packet or accepted release was created during validation. No GPU job
was launched; all future GPU execution remains restricted to `hku-gpu2`.
