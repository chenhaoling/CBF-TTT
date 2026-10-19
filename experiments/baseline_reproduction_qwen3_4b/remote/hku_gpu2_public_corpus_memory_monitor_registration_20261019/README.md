# hku-gpu2 public-corpus memory-monitor registration

This directory preserves the first failed registration and the corrected
successful registration of the read-only full public-corpus memory monitor.
Both attempts ran only on `hku-gpu2` with CUDA hidden. Neither attempt read
corpus text, changed the builder, sent a signal, released a downstream gate or
used a GPU.

## Preserved failed attempt

The monitor from clean tool commit `32aa45e09cce...` failed closed before its
first sample with `FAILED_MULTIPLE_BUILDERS`. Its argv matcher found both the
`/usr/bin/time` wrapper (PID 1673441) and the actual Python builder (PID
1673442). The output root remains preserved; the corpus builder continued
running.

## Corrected registration

The process-classification correction was frozen before retry in
`IN_PLACE_TTT_BASELINE_PUBLIC_CORPUS_MEMORY_MONITOR_PROCESS_CORRECTION_ADDENDUM.md`.
Clean tool commit `38d167c34641...` additionally requires the executable
basename to start with `python`, excluding the time wrapper without relying on
PID values or parent-child ordering. All four focused tests passed on
`hku-gpu2`.

The corrected monitor entered `MONITORING`, selected only builder PID 1673442
and wrote one initial scalar sample. At that sample:

- builder RSS: 10,511,416 KiB (10.02 GiB)
- builder anonymous RSS: 10,231,320 KiB
- builder swap: 0 KiB
- host available memory: 50,362,340 KiB (48.03 GiB)
- free filesystem space: 708,514,455,552 bytes
- Stage-1 data/provenance size: 24,809,233,735 / 1,423,814,927 bytes
- wrapper state: `RUNNING_BUILD`
- GPU 0/1 memory at registration: 46 MiB / 18 MiB, both at 0% utilization

The live monitor continues in tmux session
`baseline_public_corpus_memory_38d167c3_20261019` and samples every 300
seconds for at most 48 hours. Its live output is separate from this immutable
registration snapshot.

## Evidence

- `failure_*`: identity, fail-closed state and header-only sample file from the
  superseded attempt
- `corrected_*`: identity, initial state and first scalar sample from the
  corrected monitor
- `focused_tests.txt`: four remote focused tests
- `remote_identity.txt`: host, time, commit, wrapper state and GPU snapshot
- `SHA256SUMS`: hashes created and verified on `hku-gpu2`
- `ARCHIVE_SHA256.txt`: hashes of this complete local archive
- `.gitattributes`: preserves the remote CSV byte stream and its CRLF record
  terminators without text normalization

## Claim boundary

This proves only that the corrected monitor uniquely binds the active Python
builder and can record scalar resource samples without using CUDA. It is not a
corpus audit, does not predict whether the build will finish, does not release
Level A, and is not training or RULER evidence. All GPU execution remains
restricted to `hku-gpu2`; update-admission experiments remain sealed.
