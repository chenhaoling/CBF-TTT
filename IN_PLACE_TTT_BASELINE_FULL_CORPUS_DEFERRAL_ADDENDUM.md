# Full public-corpus builder deferral addendum

Status: **frozen before stopping or moving the live full-corpus builder**  
Freeze date: 2026-10-19 (Asia/Shanghai)

## Decision

The user selected a preliminary validation instead of waiting for the
approximately 35B-token public replacement corpus. The live `full` builder may
therefore be stopped only after the independently audited
`preliminary_100m` corpus is complete. This releases host resources for the
paired Qwen3-4B-Base Level-A and 100M training runs on `hku-gpu2`.

## Preservation procedure

The operation must run only on `hku-gpu2` and must verify all of the following:

- the exact registered Python builder process is alive and its command line
  binds the `full` profile and expected output root;
- the wrapper status is `RUNNING_BUILD`;
- the final full-corpus root does not exist and the `.partial` root does;
- the preliminary corpus and its independent audit both pass every check.

Then:

1. record the builder PID, command-line hash, elapsed time, partial artifact
   sizes, available host memory, swap, disk, record counts and latest complete
   provenance index without reading source plaintext;
2. send `SIGSTOP` to the Python builder and confirm it is stopped;
3. atomically rename the `.partial` directory to a unique `.deferred_*` path;
4. send `SIGKILL` to the stopped Python builder so its exception cleanup cannot
   delete the preserved directory;
5. allow `/usr/bin/time` and the original wrapper to terminate naturally and
   record their terminal status;
6. independently confirm the builder is absent, the original `.partial` and
   final roots are absent, the deferred root is present, and both GPUs remain
   idle.

No file in the deferred root may be deleted, truncated, edited or presented as
an atomically published corpus. The prefix is not resumable with the current
builder and cannot satisfy any `full` corpus or formal-scale gate. Its only
purpose is recovery/forensics if the paper-scale path is resumed under a new
plan.

## Claim boundary

This is a user-priority deferral, not a data-pipeline failure and not evidence
against paper-scale training. The original wrapper is expected to report a
nonzero builder termination because its child was deliberately stopped. The
preliminary corpus remains a distinct, atomically published and independently
audited artifact. Gate training and sealed test remain closed.
