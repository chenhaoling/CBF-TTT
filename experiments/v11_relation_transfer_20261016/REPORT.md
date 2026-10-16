# V11 unseen-relation transfer report

## Decision

All corpus, label, feature, checkpoint and result integrity audits passed.  The
registered V11 development release failed one auxiliary condition, so the
eight FewRel internal-test and sixteen external-test relations remain sealed.

The frozen diagnosis is conclusive: natural counterfactual FewRel training
data caused the unseen-relation policy gain, while relation-group DRO did not
improve ordinary equal-source ERM.  The next study must use the frozen ERM
method on a new natural source.

## Data and resources

- 8 wholly unseen target relations;
- 112 source groups, 224 twin episodes and 672 sparse boundaries;
- 3,360 future queries and 2,016 residual candidate examples;
- 4,032 unique material rows/entity pairs with zero V10 donor reuse;
- retain accuracy 81.027%, safe-oracle accuracy 91.429%, headroom 10.402 pp;
- positive safe-benefit support 13.294%, risk support 16.716%;
- label critical path 214.76 s, peak reserved memory 8.02 GiB;
- feature critical path 26.56 s, peak reserved memory 7.87 GiB; and
- nine-controller training plus bootstrap 189.32 s.

## Policy result

| Arm | Mean gain | Relation-bootstrap 95% CI | Benefit AP |
|---|---:|---:|---:|
| Equal-source ERM | +3.239 pp | [2.406, 4.137] pp | 0.6827 |
| Relation-group DRO | +3.199 pp | [2.302, 4.132] pp | 0.6456 |
| DRO without FewRel | +0.005 pp | [-0.025, 0.035] pp | 0.1987 |

Every ERM and DRO seed improved over lifecycle physics.  DRO's source-group
bootstrap interval was [2.242, 4.201] pp and every selected policy added zero
unrelated-query errors.  It closed 82.38% of the safe-oracle gap.  Its gain was
positive on every one of the eight target relations and nonnegative on every
lifecycle mechanism.

The registered release failed because DRO benefit AP was 3.714 points below
ERM rather than at least 3 points above it.  This rejects the robust-loss
hypothesis even though the downstream policies are similarly effective.

## Causal interpretation

Removing FewRel reduced DRO gain by 3.194 points and made both clustered
intervals cross zero.  Natural counterfactual corpus coverage is therefore the
supported mechanism.  Relation-group DRO changed only 22--29 of 672 boundaries
per seed relative to ERM, with net future-query changes of -5/+4/-7 and no
collateral changes.  It adds complexity without a policy benefit.

## Claim boundary

The result supports selective forgetting over the audited addressable Qwen
residual memory with current writes fixed at `g=1`.  It does not establish
content binding by the original dense `ttt_proj`/`ttt_conv` writer, whose
independent favorable single-record test failed.  It is development evidence;
the failed registered release prevents using the reserved FewRel tests.
