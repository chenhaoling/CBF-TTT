# Stage-2 90% sparse provenance watcher

This directory records the pre-boundary script and launch metadata for a
CPU-only sparse provenance audit on `hku-gpu2`.

The fixed boundary is Stage-2 record indices 102,990 through 102,995. The
watcher was launched while only 90,575 records were complete, so the target
rows had not been observed when the script and checks were frozen. It runs
with an empty `CUDA_VISIBLE_DEVICES` and emits no source text or token IDs.

Unlike the earlier 75% watcher, the polling loop reads only the last four
provenance lines and selects the last complete JSON row. This avoids repeatedly
scanning the multi-gigabyte provenance file. Once the fixed boundary exists,
the script reads the six target rows and verifies exact indices, strict source
alternation, paired source indices, 131,071 content/raw tokens, 131,072
training tokens, zero synthetic EOS, spans and token-hash format.

Frozen artifacts:

- `audit_stage2_sparse_90.py`:
  `2b2c78cfffcb3933300e1080a0cdb3f82a8912937ecc36575f4bd207536141ac`
- `stage2_sparse_90pct_launch.json`:
  `25bdb2188c27f925abccc60b8bd607e810241119dd10b30e543e1f1d8754010f`

At launch, builder PID 1397987 and watcher PID 1666185 were live. GPU 0/1
were at 46/18 MiB with zero utilization. A later passing or failing result
must be copied separately; this launch record is not an audit result and does
not replace the full canonical audit.
