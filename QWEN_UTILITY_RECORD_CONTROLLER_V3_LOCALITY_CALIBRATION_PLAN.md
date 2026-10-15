# Qwen utility record controller V3: train-only locality calibration

## Freeze point and V2 evidence

This protocol is frozen after the independently audited V2 TempLAMA validation
evaluation failed its prospective release gate, and before extracting any
controller feature or prediction from the TempLAMA train resource pilot.  The
official TempLAMA test and original LAMA/ParaRel confirmation remain sealed.

V2 established a real but insufficient zero-shot effect.  Against explicit
lifecycle physics, its three frozen V1 controllers gained 1.543, 2.201 and
1.686 accuracy points; every subject-group interval had a positive lower bound
and the pooled interval was `[1.126, 2.539]` points.  The mean gain was 1.810
points and mean Oracle-gap closure was 35.2%.  It failed because the registered
requirements were 2 points and 40%, the seeds introduced 83/28/87 unrelated
future-query errors, semantic overwrite and two relations mildly regressed,
and only seven of nine relations were nonnegative.  The test was not opened.

The diagnostic pattern is consistent with a cross-source safety-calibration
failure: stale repetition gained 9.3--11.2 points and mixed lifecycle gained
3.9--4.8 points, while deletion-matched random controls were worse and caused
approximately 283--333 collateral errors.  V3 tests whether a train-only
semantic-locality projection can retain the useful learned deletions while
blocking unrelated ones.  V3 cannot reinterpret V2 as a pass.

## Fixed data boundary

Calibration uses only the already audited 72-group TempLAMA train resource
pilot: eight real temporal trajectories per relation, 144 twin episodes and
432 boundaries.  Its existing ternary labels contain every binary endpoint,
so the safe binary teacher is reconstructed without running a new label grid.
The same frozen 66-dimensional V1 feature extractor is applied.  No validation
or test record may enter feature extraction, conformal calibration or method
selection.

The V2 validation corpus remains the development evaluation for V3.  Because
the V3 family was motivated by observed V2 validation failures, this is an
iterative development result rather than independent confirmation.  Only the
still-sealed TempLAMA test can support a generalization claim.

## Frozen controller and candidate projections

The three V1 checkpoints and normalization statistics remain unchanged.  For
each seed, enumerate the original registered utility thresholds
`{0.4,0.5,0.6}`, risk penalties `{0,0.5,1,2}` and train-pilot conformal levels
`{0.01,0.03,0.05}`.  A conformal risk threshold is computed only from positive
single-delete risk targets in the train pilot.

Explicit reset, referenced revoke and exact-statement duplicate behavior remain
the physical base.  A neural deletion that is not physically forced must also
satisfy exactly one candidate locality rule from this frozen set:

1. `unrestricted`;
2. `exact_subject` (`exact_subject = 1`);
3. `subject_value_conflict` (`subject_value_conflict = 1`);
4. `conflict_pair_positive` (subject-value conflict and frozen-Qwen yes/no pair
   margin at least zero);
5. `conflict_cosine_098`, `conflict_cosine_099`, or
   `conflict_cosine_0995` (subject-value conflict and masked-relation cosine at
   least the named fixed threshold);
6. `conflict_pair_cosine_098`, `conflict_pair_cosine_099`, or
   `conflict_pair_cosine_0995` (both the positive pair margin and corresponding
   cosine condition).

These rules use only inference-time record and incoming-update features.
Relation ID, mechanism, answer, teacher, timestamp and future-query outcome are
forbidden inputs.

## Train-only selection

For each seed, evaluate the complete Cartesian grid on all train-pilot groups.
A configuration is eligible only when it adds zero unrelated or anchor errors
against retain-all on every pilot boundary.  Select the eligible configuration
by, in order:

1. highest primary accuracy;
2. highest deletion precision;
3. lowest deletion fraction;
4. the locality-rule order above;
5. higher utility threshold;
6. higher risk penalty;
7. smaller conformal level;
8. lexical configuration name.

No validation result can change this selection.  The selector, all candidate
metrics and exact selected actions require independent deterministic audit.

## Validation evaluation and unchanged release standard

Apply each selected seed configuration once to all 256 V2 validation groups.
Report the same deterministic baselines, safe binary Oracle, 20 deletion-
matched random draws, query roles, mechanisms, nine relations, group and
relation-cluster intervals, two-way target/background bootstrap and the nine
background-independent groups.

V3 opens TempLAMA test only if the original V2 development gates all hold:

1. safe binary Oracle headroom over the strongest zero-collateral deterministic
   baseline is at least three points;
2. every seed gains at least one point and mean gain is at least two points;
3. pooled subject-group 95% gain lower bound is above zero;
4. every seed adds zero unrelated or anchor errors;
5. mean Oracle-gap closure is at least 40%;
6. temporal overwrite and stale repetition mean gains are nonnegative;
7. macro-relation gain is positive, at least eight of nine relations are
   nonnegative, and the worst relation loses no more than two points;
8. all source, data, label, feature, selection, result and seal audits pass.

Failure keeps TempLAMA test sealed.  The next permissible branch would train a
domain risk/utility head on a prospectively constructed larger TempLAMA train
corpus; it cannot search another validation-only rule grid.

## Claim boundary

A V3 test pass would establish that small train-source calibration transfers a
locality-constrained forgetting controller to unseen TempLAMA subjects.  It
would not recover V2's failed zero-shot claim and would not establish content
binding in the original dense `ttt_conv`/`ttt_proj` writer.
