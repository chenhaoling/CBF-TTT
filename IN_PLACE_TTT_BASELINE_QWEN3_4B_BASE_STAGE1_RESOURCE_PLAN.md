# Corrected Qwen3-4B-Base Stage-1 resource plan

Status: **frozen before corrected 16K/32K GPU resource measurements**  
Freeze date: 2026-10-19 (Asia/Shanghai)

## Scope and prerequisite

This plan registers the corrected Base checkpoint's Stage-1 resource ladder.
It may start only after the corrected Level-A chain reports `COMPLETE` and its
terminal independent audit passes every registered check. The prerequisite
must bind the exact Base identity, full-corpus manifest/audit, Base-derived
resource-input manifest/audit, matched 8K no-TTT/TTT arms, conversion and both
RULER smokes.

All GPU work is restricted to `hku-gpu2` (hostname `user-Super-Server`). The
post-trained-Qwen chain and its resource inputs are forbidden prerequisites.

## Registered boundaries

Run four one-step measurements in this fixed order:

1. 16K no-TTT;
2. 16K native TTT;
3. 32K no-TTT;
4. 32K native TTT.

Every run starts from the same exact `Qwen/Qwen3-4B-Base` checkpoint and uses
the corresponding two-record input derived from the corrected full corpus.
The two arms at each length differ only in native `ttt_mode`. Reuse the frozen
BF16 FSDP2, activation-offload, gradient-checkpointing, AdamW `5e-6`, weight
decay `0.1`, global-batch-2 resource protocol and upstream writer settings.

Each child must have finite loss, gradient norm and optimizer-step duration; a
complete DCP checkpoint; at least 5% free GPU memory on both devices; at least
8 GiB available host memory throughout; swap use no greater than 3.5 GiB; and
a passing child independent audit. Any failure stops the ladder immediately.

## Acceptance and claim boundary

The final audit revalidates the Level-A prerequisite, resource-input hashes,
configuration matrix, fixed execution order, every raw resource trace and all
four child audits. It records measured throughput and memory without imposing
a directional efficacy condition.

Passing this ladder is resource evidence only. It does not release the 32K
timing run, formal 100M Stage-1 training, Stage 2, RULER matrix, 256K
extrapolation or update-admission experiments.
