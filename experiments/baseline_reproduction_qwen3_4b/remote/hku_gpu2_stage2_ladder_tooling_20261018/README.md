# hku-gpu2 Stage-2 resource-ladder tooling evidence

This directory records CPU-only verification of the arm-specific 64K/128K
Stage-2 resource ladder at clean detached commit
`4860ea6ed695caf78df6843c3b72d6206227ad47`.

The remote focused tests passed 2/2 and Bash syntax passed. A negative
preflight supplied a future formal Stage-1 root. It exited 1 while that audit
was absent, created no ladder output and left GPU0/GPU1 at 46/18 MiB and 0%.

At `2026-10-18T11:11:47+08:00`, the full-corpus builder PID 1397987 was still
live, the partial corpus was about 49 GiB, and all three upstream watcher
states remained waiting. No Stage-2 CUDA work was released.
