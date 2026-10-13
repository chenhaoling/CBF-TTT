# Episode-barrier forgetting V6 preregistration

## Motivation

V5 reached 99.81% current retention, 99.70% non-target retention, and 99.86%
hard-retain accuracy on strict nested OOF evaluation, but stale deletion and
pair state were 83.99/83.90%, roughly one point below release. Threshold tuning
is closed. V6 changes the training objective so the one update in an episode
must outrank the most dangerous retain event.

## Frozen components

Keep V3 scope/value factorization, three-ranker candidate ensemble, top-4
shortlist, 48-dimensional gate features, five subject folds, rotating
train/calibrate/validate assignments, fixed 50 epochs, threshold grid,
calibration rule, median-margin deployment ensemble, OOF prerequisite, and
development gate unchanged from V5. LongMemEval-S and LoCoMo remain sealed.

## Barrier objective

For every training episode, let `z+` be the gate logit at its sole local update
and `z-` the maximum logit over all retain events. Train the same rank-16 gate
with:

`softplus(0.5 - (z+ - z-)) + 0.25 * BCE(logits, labels)`.

The first term is an episode-level safety barrier: a valid controller must put
the update at least 0.5 logit above the worst retain event. The BCE term keeps
absolute scores identifiable for cross-fold calibration. No loss weight,
margin, epoch, feature, threshold, or fold is selected after seeing OOF.

## Gates and stop rule

The unchanged OOF prerequisite requires current ≥97%, non-target ≥95%,
hard-retain ≥95%, stale deletion ≥85%, and pair state ≥85%. Development is
loaded only after a complete OOF pass, then the unchanged ten V4 checks control
LoCoMo release.

If V6 fails OOF, training on this constructed corpus stops. The paper result is
then a verified factorized candidate mechanism plus an explicit safety/recall
frontier, and the next data collection must use an independent natural
long-session source rather than further ZsRE threshold or loss iteration.
