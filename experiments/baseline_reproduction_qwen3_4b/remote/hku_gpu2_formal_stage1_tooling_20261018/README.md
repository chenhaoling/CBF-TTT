# hku-gpu2 formal Stage-1 tooling evidence

This directory records CPU-only verification of the formal 2x5090 Stage-1
path at clean detached commit
`8b9a56b462f7e77440c374e2e870e9f6ef253459`.

The remote focused tests passed 2/2 and Bash syntax passed. A negative
preflight supplied the future corpus, timing and release paths while they were
still absent. It exited 1, created neither output nor partial directory and
left GPU0/GPU1 at 46/18 MiB and 0% utilization.

At `2026-10-18T11:06:28+08:00`, the public-corpus builder PID 1397987 and all
three upstream watcher states remained live/waiting. The host had
926,365,290,496 bytes free, above the plan's 160-GB pair gate. This is tooling
and guard evidence only; formal Stage-1 training was not released or started.
