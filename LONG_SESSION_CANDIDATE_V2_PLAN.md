# Hierarchical long-session candidate resolver V2 preregistration

## Motivation and scope

Sparse-gate V1 failed because maximum-cosine session retrieval reached only
33.99% positive-event top-1. The local-deletion oracle remained perfect at all
four session lengths. V2 therefore changes only the stale-session candidate
resolver and does not train or evaluate a deletion gate. Threshold tuning is
not an allowed response to V1 because a gate cannot delete the correct session
when the candidate is wrong.

The V1 development outcomes are already open and are treated as method
development evidence. V2 architecture, features, epochs, selection rule, and
release gate are frozen in this document before V2 is evaluated. LongMemEval-S
remains locked; LoCoMo is not loaded or scored.

## Data and split

Reuse the audited `long_session_sparse_corpus_v1` without rebuilding labels:

- calibration-fit trains the resolver;
- calibration-tune selects epoch independently for each fixed seed;
- fit+tune refits at the selected epoch;
- the subject-disjoint ZsRE development trajectories are read once for V2.

Only the single positive update event in each episode contributes a listwise
candidate label. All active sessions at that event are candidates. Subject,
chain ID, session ID, split, target ID, event kind, and unit position are never
model features.

## Hierarchical evidence

For each active-session/new-session pair, form each frozen projection's 4-by-4
unit cosine matrix. The order-neutral feature vector contains:

- maximum, top-2 mean, top-4 mean, full mean, and standard deviation for each
  of the three projections;
- mean/std of the three maxima, cross-projection argmax agreement, mean-matrix
  top-1/top-2 margin, and fractions above 0.3/0.5/0.7/0.9;
- lexical and relation-token overlap for the best mean-cosine unit pair;
- observable unit counts.

The primary ranker is a rank-16 tanh MLP trained by one listwise softmax loss
per update event. It cannot use candidate age or unit position, preventing it
from exploiting the constructed quarter/three-quarter schedule or the current
anchor-first serialization.

## Fixed controls

- V1 mean maximum cosine;
- robust mean top-4 cosine;
- full unit-pair mean;
- projection-consensus maximum (`mean(max)-std(max)`);
- anchor-to-anchor cosine as a construction-only diagnostic, ineligible for
  selection or release;
- oracle target.

## Training and selection

Use seeds 29101/29102/29103, rank 16, AdamW with learning rate 0.01 and weight
decay 0.1, and checkpoints 10/25/50/100/200. Each seed selects the checkpoint
lexicographically by calibration-tune top-1, worst-length top-1, MRR, then the
earlier epoch. Refit on fit+tune for exactly that epoch. No development metric
changes a checkpoint, feature, or hyperparameter.

## Development release gate

V2 advances to a new sparse deletion-gate preregistration only if all checks
pass:

- three-seed mean top-1 at least 95%;
- minimum seed top-1 at least 93%;
- every length's mean top-1 at least 93%;
- mean MRR at least 97%;
- primary-seed top-1 gain over V1 max-cosine has a subject-cluster bootstrap
  95% lower bound above 20 points.

Failure stops this feature-level resolver and requires query-aware unit routing
or a learned token-level scope representation. Passing V2 does not itself
establish forgetting efficacy; it only permits a separately preregistered
causal deletion-gate experiment. LoCoMo remains sealed in either case.
