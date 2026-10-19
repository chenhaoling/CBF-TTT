# Clean serial Level-A retry on `hku-gpu2`

This directory contains compact, text-free evidence from the clean serial
retry of the preliminary Qwen3-4B-Base Level-A chain. The run used tool commit
`c10d7b8c279d70f55d81e638696cfd361e3bbf72` and executed the two training arms
in the frozen order `no-TTT -> native TTT`. Checkpoint conversion and the
RULER-4K smoke remained fail-closed after the TTT resource audit failed.

Immediately before the no-TTT arm, GPU 0/1 used 46/18 MiB, both reported 0%
utilization, no CUDA compute process was registered, and the conflict scan was
empty. The TTT arm repeated the process/conflict preflight before launch.

| Arm | Training result | GPU peak (MiB) | Minimum GPU free | Minimum host available | Resource audit |
|---|---:|---:|---:|---:|---:|
| no-TTT | one finite AdamW step; DCP saved | 23,944 / 23,916 | 26.57% / 26.65% | 8,657.812 MiB | pass |
| native TTT | one finite AdamW step; DCP saved | 31,942 / 31,914 | 2.04% / 2.13% | 6,685.281 MiB | fail |

The TTT child exited with code zero, recorded loss `2.0122`, finite gradient
norm `531.5888`, and wrote a 50,319,528,929-byte DCP. It failed the predeclared
resource limits of at least 5% free memory on both GPUs and at least 8 GiB host
memory available. This is a resource-headroom failure rather than CUDA OOM,
nonfinite training, or interference from a concurrent CUDA process.

Raw training logs are intentionally excluded because they contain truncated
token-id displays. The archived summaries, scalar GPU/host traces, preflight
snapshots, manifests, dependency bindings, and independent audits are sufficient
to recompute the resource decision. `ARTIFACTS_SHA256.txt` binds every archived
file except itself.
