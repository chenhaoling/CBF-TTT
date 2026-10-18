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
- The live snapshot was restored from `0444` to `0644`. Its timestamp then
  advanced on every scheduled poll while the watcher PID remained live.
- The watcher correctly continued waiting on the two public-corpus builder
  processes. GPU 0/1 remained at 46/18 MiB and 0% utilization.

`cp_second.stdout` is intentionally empty. The corresponding stderr file
contains the reproduced permission error.
