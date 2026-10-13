# Factorized causal forgetting V4 preregistration

## Released evidence and claim

Factorized scope-key candidate V3 passed all six release checks at 97.07%
top-1 and 98.38% MRR. V4 asks the remaining causal question: when a new
session arrives, can a high-specificity gate delete the selected stale session
while preserving current and unrelated session memory?

This stage is development only. LongMemEval-S outcomes remain locked and
LoCoMo remains unopened. V3 scope transforms, candidate-ranker epochs, and
candidate checkpoints are frozen.

## Candidate cascade

For every incoming session, use frozen scope-only maximum cosine to shortlist
four active sessions, then apply the corresponding frozen V3 listwise ranker
inside that shortlist. Unit order, candidate age, session ID, subject, chain,
target, event kind, and split are unavailable. Positive-event target recall@4
is reported and must be at least 99%.

## Conflict evidence and deletion gate

The rank-16 tanh gate receives only observable evidence:

- the selected candidate's frozen 33-dimensional V3 scope feature;
- candidate-ranker top score and top1/top2 margin;
- shortlist scope maximum and its active-memory margin;
- whether both selected unit payloads exist and whether they are exactly equal;
- three frozen value-projection cosines, their mean/std, and lexical payload
  overlap.

Thus deletion is learned as a conjunction of scope match and payload conflict.
Payload never participates in candidate ranking. Candidate age and the fixed
quarter/three-quarter event schedule remain excluded.

Each seed trains on calibration-fit oracle states, selects epoch and threshold
by complete causal replay on calibration-tune, and refits on fit+tune after all
choices freeze. Thresholds are
`{0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.925,0.95,0.975,0.99,0.995}`.
Tune selection first requires current retention at least 98%, non-target
retention at least 97%, hard-retain accuracy at least 97%, and update action
accuracy at least 85%; it then maximizes strict pair state, update accuracy,
non-target retention, earlier epoch, and threshold proximity to 0.9.

## Controls

- retain all;
- scope maximum plus a tune-selected scalar threshold;
- exact deletion-count-matched random timing and target;
- archived V1 learned controller;
- frozen factorized candidate plus learned conflict gate;
- oracle local deletion.

## Development gate

All checks are required:

- shortlist target recall@4 at least 99%;
- candidate top-1 at least 95%;
- current retention at least 97%;
- non-target retention at least 95%;
- hard-retain action accuracy at least 95%;
- stale deletion at least 90%;
- strict stale/current pair state at least 90%;
- every length strict pair state at least 85%;
- primary-seed strict pair-state gain over V1 has a subject-cluster bootstrap
  lower bound above 25 points;
- primary-seed strict pair-state gain over action-count-matched random has a
  subject-cluster bootstrap lower bound above 10 points.

Passing freezes the complete resolver/gate for a one-time LoCoMo
preregistration. Failure keeps LoCoMo sealed and is decomposed into shortlist,
ranking, false-delete, and missed-update errors; no V4 threshold is changed
after development.
