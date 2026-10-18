# Corrected Qwen3-4B-Base Table 1 conclusion plan

Status: frozen before any corrected formal RULER score exists.

Freeze date: 2026-10-19 (Asia/Shanghai)

## Entry gate

Generate the final baseline conclusion only after both corrected audits pass:
the primary 4K--128K matrix and the separate formal 256K extrapolation.  The
reporter is CPU-only, runs with CUDA hidden on `hku-gpu2`, and never launches
training, evaluation, a release transition, or a gate experiment.

## Registered outputs

Produce one machine-readable JSON and one deterministic Markdown report.  For
each of 4K, 8K, 16K, 32K, 64K, 128K, and 256K report:

- matched no-TTT, native In-Place TTT, and `TTT - no-TTT` macro accuracy;
- both arm scores and paired delta for all 13 official RULER tasks;
- 13 tasks, 100 examples per task and arm, paired-input status, failures, and
  source summary/audit hashes;
- differences from the paper's matched/TTT/gain Table 1 values;
- available one-example resource peak memory and total memory without treating
  discarded resource scores as efficacy evidence.

Report the registered six-context 4K--128K mean separately.  Keep 256K marked
as YaRN extrapolation and outside that mean.

## Uncertainty

Before observing scores, register a paired task bootstrap with seed 42 and
10,000 resamples.  The sampling unit is one of the 13 RULER tasks.  For each
context, resample task deltas with replacement and report the 2.5th and 97.5th
percentiles using sorted indices `floor(p * (N - 1))`.  For the primary mean,
resample the same task identities and average their deltas over the six primary
contexts.  These intervals describe task-set uncertainty only; they are not
example-level intervals and impose no directional acceptance threshold.

## Claim and gate boundary

The final artifact is a public-data matched replication using FineWeb-Edu and
LongCrawl64, not an exact reproduction of the private paper corpus or its
approximately 35B-token scale.  It may establish that the registered baseline
evidence is complete, but it must not automatically start update-admission
training or open any sealed gate test.  Gate exploration requires a separate
pre-outcome plan after this report is audited.
