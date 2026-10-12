# Entity-fusion scope V10 preregistration

## Motivation

V9's same-entity head reached 96.57% pairwise AUC, but rigidly multiplying its
probability with relation compatibility reduced candidate top-1 to 92.48%.
V8 remains the best automatic method at 94.01%. V10 tests exactly one remaining
hypothesis: entity probability is complementary when calibrated jointly with
V8 features rather than imposed as a product rule.

## Protocol

- Reuse the frozen, subject-disjoint V9 entity heads for each outer fold.
- Append entity probability, log probability, and entity logit to the thirteen
  V8 text-only features.
- Keep V8's fixed base score, rank-16 residual, five outer folds, next-fold
  epoch selection from `{10,25,50}`, and three-seed ensemble.
- Inference sees only query/key text. Subject/relation/chain identifiers,
  targets, age, position, and future queries remain excluded.

## Gate and stop rule

V10 passes only if top-1 ≥95%, MRR ≥97%, gain over V8 ≥1 point with positive
paired-bootstrap lower bound, and gain over V4 ≥12 points with positive lower
bound.

A pass freezes the automatic resolver and starts causal on-policy forgetting.
A failure permanently stops CounterFact score-fusion experiments and requires a
token-level entity span extractor. LongMemEval-S and LoCoMo remain sealed.

