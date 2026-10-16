# V11 frozen relation-transfer failure diagnostic

## Scope

V11 passed every integrity and policy-effect condition but failed its registered
release because relation-group DRO reduced benefit average precision by 3.71
points relative to equal-source ERM.  Its three policy gains were
3.244/3.185/3.170 points, both clustered confidence intervals were positive,
collateral errors were zero and removal of FewRel reduced gain by 3.194 points.
Internal/external FewRel tests remain sealed.

This diagnostic uses only the nine saved checkpoints and the already opened
eight-relation development rows.  It does not fit a model, select a checkpoint,
change a threshold or open another source.

## Frozen analyses

1. independently reconstruct selected actions for all three arms and seeds;
2. compute source-group and relation-cluster intervals for every arm;
3. report benefit/risk AP, policy gain, deletion precision and coverage;
4. compare ERM and DRO actions boundary by boundary, including net future-query
   correctness and collateral changes; and
5. report seed-averaged per-relation and per-mechanism policy gains.

If ERM and DRO both have positive clustered intervals while the no-FewRel arm
has negligible gain, classify the evidence as
`constructed_corpus_effect_supported_robust_loss_not_supported`.  The next
study must keep ordinary ERM frozen and use a new natural source; it may not
open the V11 test partitions or tune again on these development relations.
