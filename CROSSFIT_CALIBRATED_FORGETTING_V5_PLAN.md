# Cross-fit calibrated forgetting V5 preregistration

## Motivation

V4 solved candidate retrieval and cumulative safety but missed the mean
deletion gate because single fit/tune calibration did not transfer through
refitting. One seed already reached 91.48% strict pair state with all safety and
length metrics above 90%, while another reached 70.04%. V5 changes calibration
and deployment only. The V3 resolver, scope/value representation, feature
family, and memory actions remain frozen.

## Nested cross-fit protocol

Assign the 267 calibration subjects deterministically to five folds. For outer
fold `i`:

- fold `i` is validation and never trains or calibrates that model;
- fold `(i+1) mod 5` calibrates its threshold;
- the remaining three folds train the gate.

Train rank-16 gates for exactly 50 epochs with seeds 32101/32102/32103. No
epoch selection or refit is allowed. Each model chooses from the unchanged V4
threshold grid using complete causal calibration replay. A feasible threshold
requires current retention ≥98%, non-target retention ≥97%, and hard-retain
accuracy ≥97%; among feasible thresholds it maximizes strict pair state,
update accuracy, and non-target retention. If none is feasible, choose the
highest threshold, yielding the safest defined action.

Pool the five untouched validation folds for an OOF estimate. V5 may read the
subject-disjoint development set only if pooled OOF reaches current retention
≥97%, non-target retention ≥95%, hard-retain ≥95%, stale deletion ≥85%, and
strict pair state ≥85%.

## Deployment ensemble

The candidate is selected by averaging all three frozen V3 ranker scores inside
the unchanged scope top-4 shortlist. Gate features exclude absolute ranker
logits and margins, removing their train-versus-test calibration shift. For
each event, the 15 calibrated fold/seed models produce signed logit margins
relative to their own calibration thresholds. Delete the shared candidate only
if the median margin is nonnegative. Models are not refit after calibration.

## Development gate

If the OOF prerequisite releases development, the same V4 ten checks apply:
shortlist recall ≥99%, candidate top-1 ≥95%, current retention ≥97%, non-target
retention ≥95%, hard-retain ≥95%, stale deletion ≥90%, pair state ≥90%, every
length pair state ≥85%, clustered pair-state lower-bound gain over V1 >25
points, and gain over action-count-matched random >10 points.

A complete pass freezes V3+V5 for a one-time LoCoMo protocol. Failure ends gate
calibration on this constructed corpus. LongMemEval-S remains locked and LoCoMo
remains unopened until that pass.
