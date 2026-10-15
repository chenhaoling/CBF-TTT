# V7 cross-source failure diagnostic

## Evidence boundary

The independently audited V7 cross-source controller improved every seed in
both development domains and added no unrelated-query errors, but it failed
the preregistered release gate.  Mean gain / safe fact-key-oracle gap closure
were 2.569 points / 59.62% on TempLAMA and 0.859 points / 44.59% on
LAMA/ParaRel.  The already opened TempLAMA test and original-source
confirmation remain closed.  LongMemEval-S and LoCoMo remain unopened.

This is a development-only diagnostic.  It cannot release a checkpoint or
authorize an external benchmark.  It may only select the class of the next
preregistered development method.

## Locked inputs

Use exactly the audited V7 inputs and its three locked checkpoints:

- TempLAMA train: 4,392 residual examples from 512 groups;
- LAMA/ParaRel train: 4,684 residual examples from 512 groups;
- TempLAMA development: 2,204 candidates from 256 groups;
- LAMA/ParaRel development: 588 candidates from 64 groups; and
- V7 seeds 26101651, 26101652 and 26101653.

Verify every upstream audit and V7 checkpoint hash before analysis.  No test,
confirmation, LongMemEval-S or LoCoMo file may be loaded.

## Fixed diagnostic questions

### 1. Ranking and calibration

For each locked V7 checkpoint and development domain, report benefit and risk
prevalence, AUROC, average precision, Brier score and ten-bin expected
calibration error.  Also report these metrics by lifecycle mechanism whenever
both classes are present.  These measurements use record-level labels only to
diagnose an already failed development model.

### 2. Policy-selection ceiling

At each checkpoint's locked epoch, recompute the original V7 grid.  Keep the
V6 fact-key projection and zero-collateral constraint.  In addition to the
locked dual-domain selection, select the best configuration independently in
each development domain by gain, deletion precision, lower deletion fraction
and the existing deterministic simplicity tie breaks.  This is an explanatory
ceiling and cannot replace the locked V7 policy.

### 3. Source-transfer matrix

Train the unchanged 66-feature V5 MLP separately on TempLAMA only and
LAMA/ParaRel only, using the same three seeds, optimizer, epochs and config
grid.  Evaluate each model on both development domains and report the best
zero-collateral domain-specific configuration.  These diagnostic models are
not saved or released.  Combined-training values come from the locked V7
checkpoints.

### 4. Covariate shift

Normalize the combined training features, compute each source centroid and
use the difference in squared centroid distance as a fixed source score.
Report development-domain source AUROC and the ten largest absolute
standardized mean differences.  No source identifier enters a forgetting
controller.

## Fixed interpretation rule

Choose exactly one next-method class:

1. **selection/calibration limited** if the locked joint checkpoint's
   domain-specific zero-collateral ceiling reaches both the 1.5-point gain and
   50% closure gates on LAMA;
2. **negative-transfer limited** otherwise, if the LAMA-only diagnostic model
   reaches both LAMA gates and exceeds the joint domain-specific LAMA gain by
   at least 0.5 point; or
3. **representation/ranking limited** otherwise.

The first outcome preregisters a cross-fitted calibration/selection V8.  The
second preregisters a domain-robust or gradient-balanced V8.  The third
preregisters a richer pairwise fact-compatibility representation before any
new controller fit.  No threshold may be weakened based on this diagnostic.

## Required outputs and audit

Write a JSON summary containing the locked hashes, all metrics, the transfer
matrix, covariate-shift results, the deterministic diagnosis and explicit
held-out seals.  An independent auditor must reload inputs and checkpoints,
recompute the complete summary and verify exact agreement before the
diagnosis is used.
