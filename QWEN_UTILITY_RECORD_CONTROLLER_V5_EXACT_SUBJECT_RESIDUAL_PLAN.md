# Qwen utility record controller V5: exact-subject residual forgetting

## Freeze point and motivation

This protocol is frozen after the independently audited V4 evaluation failed
and after a model-free post-hoc sweep of the ten already registered locality
rules, but before training or scoring a V5 model.  TempLAMA test remains sealed.

V4 improved over explicit lifecycle physics by 0.799 points with a positive
group-bootstrap interval, but its `conflict_cosine_099` candidate set has only
1.341 points of safe residual headroom on the evaluation half.  A structured
action ranker cannot meet the two-point release target inside that support.
The residual sweep found no boundary in which an accuracy gain required two or
more extra deletions while every single extra deletion was unhelpful.

The `exact_subject` scope is the registered expansion with the best useful
safety/coverage tradeoff.  It contains 4,392/1,124/1,080 optional records on
formal train/calibration/evaluation; single-action collateral rates are
4.58%/4.45%/4.91%, while safe residual headroom is
5.57/4.58/4.61 points.  V5 therefore learns one residual deletion beyond the
unchanged explicit lifecycle physics action rather than relearning the whole
forgetting action.

## Training examples

For every formal TempLAMA train boundary:

1. construct the frozen `explicit_lifecycle_physics` action;
2. retain only records that physics keeps and whose subject matches the
   incoming record exactly;
3. for each retained candidate, evaluate the already collected counterfactual
   action obtained by deleting only that record in addition to the physics
   deletions;
4. set the benefit target to one only if the action preserves the retain-all
   unrelated-query barrier and increases the number of correct future answers;
5. set the risk target to one if it loses any retain-all unrelated answer;
6. record correct-count gain and answer-NLL change as diagnostic margins.

NLL-only improvements are reported but are not positive benefit labels.  This
keeps the primary target aligned with accuracy and avoids rewarding unnecessary
deletions.  Validation and test rows are forbidden training inputs.

## Model and inner selection

Use the unchanged 66 inference-safe features for the optional record and its
candidate context.  Train a `66 -> 128 -> 64` MLP with separate benefit and
risk sigmoid heads, AdamW (`1e-3`, weight decay `1e-4`), batch size 256,
dropout 0.1 and gradient norm 1.0.  Use seeds `26101651`, `26101652` and
`26101653`, candidate epochs 25/50/100 and inverse-frequency weights fit only
on the inner-train groups.

For each seed, hash-split complete target groups within each relation into 80%
inner train and 20% inner validation.  Keep the selected inner model rather
than refitting it on all groups, so its thresholds remain calibrated on unseen
train groups.  On inner validation, enumerate utility thresholds
`{0.4,0.5,0.6,0.7,0.8}`, risk penalties `{0,0.5,1,2}` and conformal risk levels
`{0.01,0.03,0.05}`.  Start from lifecycle physics, choose at most the highest
scoring eligible residual deletion per boundary, and fall back to physics.
Only configurations with zero added unrelated errors are eligible.  Select by
future-query accuracy, residual deletion precision, lower residual deletion
fraction, earlier epoch and the registered numeric tie order.

## Development evaluation

After all three train-only models and configurations pass an independent
reconstruction audit, score V5 once on the full 256-group TempLAMA validation
development set.  This set has already informed the V5 architecture through
the failed V2--V4 results and residual support diagnostic, so it is development
evidence rather than claim-bearing confirmation.

Report retain-all, global clear, last-1/last-2, explicit lifecycle physics,
V2--V4, the exact-subject safe residual Oracle, three V5 seeds, their majority
policy, and 20 deletion-matched random controls.  Include query-role,
mechanism, relation, group-bootstrap and relation-cluster results.

TempLAMA test may be opened only if the independently audited V5 development
result satisfies all of the following:

1. exact-subject safe residual headroom over lifecycle physics is at least
   three points;
2. every seed gains at least one point and mean gain is at least two points;
3. the pooled target-group 95% lower bound is above zero;
4. every seed adds zero unrelated or anchor errors;
5. mean exact-subject residual-Oracle gap closure is at least 40%;
6. overwrite and stale-repetition mean gains are nonnegative;
7. macro-relation gain is positive, at least eight relations are nonnegative,
   and the worst relation loses at most two points;
8. every upstream, training, policy and result audit passes.

On failure, test remains sealed and the residual MLP branch stops.  On pass,
the unchanged three train-only checkpoints and policies are evaluated once on
official TempLAMA test.

## Claim boundary

A later test pass would support a learned exact-subject residual forgetting
policy layered on deterministic lifecycle memory management.  V5 does not
train the current-input write gate, modify Qwen weights, claim dense
`ttt_conv`/`ttt_proj` content binding, or use benchmark test labels for model
selection.
