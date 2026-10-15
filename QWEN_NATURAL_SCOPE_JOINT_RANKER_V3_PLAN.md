# Preregistered Qwen natural joint key/defer ranker V3

## Motivation and claim

The audited V2 calibration improved oracle-lifecycle candidate accuracy from
87.05% to 91.54% while retaining 96.59% no-match specificity.  It stopped
because correct-top recall was 89.03% and the gap to oracle selection was 6.77
points.  V2's remaining errors are concentrated in rephrased auxiliary facts:
their frozen cosine top accuracy is about 92.5% and their V2 retrieve rate is
about 72.2%.

V3 tests whether one jointly trained compatibility score can select the correct
record and defer when no record matches.  It replaces neither the writer nor
the forgetting policy and is an enabling reader repair.  Natural development,
counterfactual test and all external benchmarks remain sealed until the V3
calibration gate passes.

## Frozen inputs and examples

Use exactly the V2 calibration sessions, five session folds, oracle lifecycle
snapshots, natural queries and deterministic in-fold, subject/target-disjoint
no-match controls.  Re-extract the same frozen Qwen pair feature for every
memory key.  Concatenate each key's pair feature with the same four frozen
ensemble-geometry features used by V1/V2.  Pad to eight keys with a mask.

For natural queries, the multiclass target is the active record index.  For
no-match controls, the target is a ninth defer class.  Donor texts never cross
folds.  Qwen3-4B, contextual transforms, value scale, candidates, prompts,
memory states and V1 scores remain fixed.

Give each of six strata equal total loss weight:

1. primary exact;
2. primary rephrase;
3. auxiliary exact;
4. auxiliary rephrase;
5. no-match exact;
6. no-match rephrase.

## Registered model and search

Use one shared linear compatibility head over each key feature and one learned
scalar defer logit.  Initialize the compatibility head from the original
seed-7301 decomposed gate and initialize defer to `logit(0.4)`.  Optimize
weighted multiclass cross entropy plus squared distance from this initialization
with deterministic full-batch AdamW.

Search only:

- anchor weight: `0.001, 0.01, 0.1, 1.0`;
- learning rate: `0.001, 0.005`;
- epochs: `25, 50, 100`;
- inference defer offset: `-1.0, -0.75, ..., 1.0`.

Emit predictions only out of fold.  Among configurations with no-match
specificity at least 92% and exact natural selection at least 98%, maximize
rephrased natural selection, then overall natural selection, no-match
specificity and six-stratum macro accuracy.  Break remaining ties by larger
anchor, lower learning rate, fewer epochs and defer offset closest to zero.
Train one final all-calibration checkpoint after selection.

## Exact end-to-end scoring

For every OOF natural decision:

- correct-key selection reuses the already scored `oracle_oracle` residual;
- defer reuses the already scored `zero` residual;
- wrong-key selection is explicitly rescored with Qwen and the selected
  record's natural value.

No wrong-key outcome may be imputed.  Use the same 12 candidates, fresh KV,
scale 8, fixed-correct-length final generation and 20,000 subject-group
bootstrap draws as V1/V2.

## Calibration release gate

All checks must pass:

- OOF natural record-selection accuracy at least 93%;
- OOF exact selection at least 98%;
- OOF rephrase selection at least 88%;
- OOF auxiliary-rephrase selection at least 85%;
- OOF no-match specificity at least 92%;
- six-stratum macro accuracy at least 92%;
- exact candidate accuracy at least 93%;
- gap to `oracle_oracle` at most 5.5 points;
- final candidate accuracy at least 91%;
- final generation exact match at least 74%;
- candidate-accuracy gain over audited V2 has a subject-group bootstrap 95%
  lower bound above zero.

Passing freezes the final V3 checkpoint and releases one natural-development
run.  Development requires candidate accuracy at least 90%, oracle gap at most
seven points, no-match specificity at least 90%, and a positive confidence
bound over the frozen V1 reader.  Failure keeps development sealed and is
localized to compatibility ranking rather than value realization or memory
lifecycle.  Release thresholds may not be relaxed after scores are observed.
