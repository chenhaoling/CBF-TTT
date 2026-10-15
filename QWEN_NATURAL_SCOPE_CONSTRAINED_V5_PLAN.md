# Preregistered constrained selection for natural scope gate V5

## Reason for the protocol correction

V4's residual gate improved OOF natural-memory candidate accuracy to 94.58%,
rephrase accuracy to 90.55% and final generation exact match to 77.62%, with a
3.02-point gain over V2 and bootstrap 95% CI `[2.07, 4.03]`.  It nevertheless
stopped because its registered hyperparameter selector maximized correct-top
recall after constraining two specificity metrics but did not constrain the
balanced-accuracy metric that was later used as a release gate.  The selected
threshold reached 91.899% balanced accuracy against the frozen 92% gate.

The already written V4 search contains eleven configurations satisfying all
four decision gates.  V5 corrects only this mismatch between model selection
and release criteria.  It does not add a model, hyperparameter, threshold,
feature, example or score.  This plan is frozen before exact Qwen outcome
scoring for the newly selected configuration.  Natural development and all
test/external data remain sealed.

## Immutable artifacts and selection

Require the audited V2 feature SHA256
`1d3f89ee7db51208b1d1934d7229a6b1c5d3e302cc2ebc6a67b8c6c09535a140`
and the immutable V4 search.  Admit only V4 records satisfying all of:

- correct-top recall at least 92%;
- natural wrong-top specificity at least 85%;
- no-match specificity at least 92%;
- three-stratum balanced accuracy at least 92%.

Among admitted records, apply the original V4 preference order: maximize
correct-top recall, then no-match specificity, wrong-top specificity and
balanced accuracy; break ties by smaller hidden width, larger penalty, lower
learning rate, fewer epochs and threshold closest to 0.5.

Recompute the selected five-fold OOF logits from the V2 feature artifact and
verify exact equality with its immutable V4 search record.  Train the final
all-calibration residual checkpoint with that same configuration.  No other
V4 record may be scored with Qwen.

## Exact scoring and unchanged release gate

Use the V4 score-reuse rules: correct-top retrieval reuses `oracle_oracle`,
defer reuses `zero`, and every retrieved wrong top receives an explicit fresh-
KV Qwen score at scale 8.  Use 20,000 subject-group bootstrap draws against the
audited V2 OOF rows.

All V4 release checks remain unchanged:

- the four decision constraints above;
- candidate accuracy at least 93%;
- gap to `oracle_oracle` at most 5.5 points;
- final candidate accuracy at least 91%;
- final generation exact at least 74%;
- gain over V2 with bootstrap 95% lower bound above zero.

Passing freezes the V5 checkpoint and releases exactly one natural-development
evaluation.  It is calibration-selected architecture evidence, not an
independent held-out result.  Development must retain at least 90% no-match
specificity, 90% candidate accuracy, an oracle gap no greater than seven
points, and a positive confidence bound over V1 `oracle_auto`.  No threshold
or gate may be changed after V5 outcome scores are observed.
