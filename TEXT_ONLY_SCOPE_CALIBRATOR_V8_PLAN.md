# Text-only scope calibrator V8 preregistration

## Motivation

V7 stopped scalar weight expansion at 93.63% top-1. Relation compatibility is
strong under oracle segmentation (97.83%), while the automatic entity heuristic
has complementary errors. V8 replaces the single scalar combination with a
small pairwise calibrator over fixed, text-only evidence.

## Features and model

For each current-input / memory-key pair, V8 uses only:

- V5 posterior relation compatibility and its log;
- V6 IDF entity score, weighted overlap, block mass, and matched-span length;
- token-length ratio;
- six existing lexical similarity statistics.

Subject strings, relation IDs, chain IDs, target values, memory age, candidate
position, and future queries are excluded. A rank-16 MLP predicts a residual on
the fixed V6 score `log(relation)+8*entity`.

## Evaluation

- CounterFact remains observed development-only.
- V5 relation and V6 span features are recomputed OOF.
- Five outer folds are deterministic by session.
- The next fold selects epochs from `{10,25,50}` after being excluded from
  training; the outer model is then retrained on all four non-outer folds.
- Three seeds are ensembled by mean logits.

## Gate

All checks are required:

- top-1 at least 95%;
- MRR at least 97%;
- at least ten points gain over V4 with positive bootstrap lower bound;
- at least one point gain over V7 with positive bootstrap lower bound.

A pass freezes the resolver and authorizes causal on-policy forgetting. A
failure ends calibration on the current fixed feature family and requires a
learned entity encoder. LongMemEval-S and LoCoMo remain sealed.

