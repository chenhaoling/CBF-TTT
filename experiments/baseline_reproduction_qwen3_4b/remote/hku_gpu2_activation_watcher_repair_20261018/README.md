# hku-gpu2 activation-watcher snapshot repair

This directory records the 2026-10-18 diagnosis and repair of the live 8K
activation resource watcher on `hku-gpu2` (`user-Super-Server`). No GPU process
was started during the repair.

- `cp /proc/meminfo` created a mode-`0444` destination and returned 1 when the
  same destination was overwritten on the next poll.
- Writing through shell redirection created an owner-writable destination and
  returned 0 on the second write. The server umask produced mode `0664` in the
  isolated reproduction.
- The candidate script passed `bash -n`; its SHA256 matches the repository
  version recorded with this evidence.
- Commit `40bedb659a104d3177584f5afc528c9cc955d1b8` was checked out cleanly on
  `hku-gpu2`. The complete focused baseline suite passed 59/59 and the fixed
  script passed `bash -n`; its SHA256 remained
  `fa5bd27911c6dbf08473e08bd007403ca8624a6adad295df692b7817df2273c5`.
- The live snapshot was restored from `0444` to `0644`. Its timestamp then
  advanced on every scheduled poll while the watcher PID remained live.
- The watcher correctly continued waiting on the two public-corpus builder
  processes. GPU 0/1 remained at 46/18 MiB and 0% utilization.

`cp_second.stdout` is intentionally empty. The corresponding stderr file
contains the reproduced permission error. `fixed_bash_syntax.txt` and
`fixed_git_status.txt` are intentionally empty successful outputs.
