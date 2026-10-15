# Qwen utility record controller V6: semantic fact-key safety projection

## Freeze point and motivation

This protocol is frozen after the independently audited V5 development result
failed only its zero-collateral gate and before scoring any V6 action.  V5
already delivers a large, robust utility gain over explicit lifecycle physics:
the three-seed mean is 3.381 points, the target-group 95% interval is
`[2.552, 4.290]` points, Oracle-gap closure is 73.6%, and all nine relations
are nonnegative.  Its three seeds nevertheless introduce 6/14/21 unrelated
answer errors.  TempLAMA test remains sealed.

The preregistered hypothesis is that subject-only locality conflates distinct
facts about the same entity.  Earlier model-free support analysis localized all
single-action collateral in the `same_subject_different_relation` mechanism,
whose V5 mean gain is only 0.345 points, while overwrite and stale-repetition
gains are 3.664 and 11.157 points.  A memory record is therefore eligible for
learned residual deletion only when its immutable semantic key, defined at
construction time by `(normalized subject, relation id)`, exactly matches the
incoming record's semantic key.

## Frozen V6 policy

V6 reuses the three audited V5 checkpoints and selected configurations without
retraining, threshold changes or additional learned inputs.  At each boundary:

1. apply the unchanged explicit lifecycle physics action;
2. run the unchanged V5 benefit/risk scorer over exact-subject candidates;
3. retain V5's at-most-one residual deletion only if candidate and incoming
   records have exactly the same immutable semantic key;
4. otherwise fall back to the lifecycle-physics action.

The safety projection reads record metadata available when the memory entry is
created.  It does not inspect query answers, counterfactual utilities or
benchmark labels at inference time.  The physical interpretation is that one
fact may supersede older values of the same fact, while an update about one
relation cannot erase another relation merely because the entity is shared.

## Development analysis

Score the frozen projection once on the complete 256-group TempLAMA validation
development set.  Report the same deterministic baselines, V5 subject-only
controllers, V6 fact-key controllers, majority policy, safe single-action
fact-key Oracle, per-query-role/mechanism/relation results, target-group and
relation-cluster uncertainty, background sensitivity, and the exact source of
every V5 and V6 collateral error.  An independent exact-rerun audit must verify
the semantic-key reconstruction, checkpoint hashes and all held-out seals.

The development release gates are unchanged:

1. safe single-action fact-key Oracle headroom is at least three points;
2. every seed gains at least one point and the mean gain is at least two points;
3. the pooled target-group 95% lower bound is above zero;
4. every seed introduces zero unrelated or anchor errors;
5. mean fact-key Oracle-gap closure is at least 40%;
6. overwrite and stale-repetition mean gains are nonnegative;
7. macro-relation gain is positive, at least eight relations are nonnegative,
   and the worst relation loses at most two points;
8. every upstream, policy and result audit passes.

## Confirmation boundary

This V6 architecture was chosen using TempLAMA validation, so another
validation pass is development evidence only.  If and only if all gates pass,
construct the previously unopened TempLAMA test corpus with the same source
inspection, decontamination, relation allocation, episode grammar, sparse
boundaries, binary action grid and independent data/label/feature audits.
Evaluate the unchanged V5 checkpoints, thresholds and fact-key projection once
on test.  No test-dependent repair, threshold search or architecture choice is
allowed.

## Claim boundary

A later test pass would support a learned residual forgetting controller with
a deterministic semantic fact-key safety layer for session memory.  V6 does
not train the current-input write gate, modify Qwen weights, infer dense
`ttt_conv`/`ttt_proj` content bindings, or claim that subject identity alone is
a safe deletion key.
