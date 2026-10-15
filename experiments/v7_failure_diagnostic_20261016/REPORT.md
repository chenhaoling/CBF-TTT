# V7 cross-source failure diagnostic

The preregistered diagnostic and its full recomputation audit passed.  It
classified the V7 miss as `representation_ranking_limited`.

## Decisive evidence

- The locked joint controller's LAMA domain-specific ceiling remained at
  +0.859 points and 44.59% fact-key-oracle gap closure.
- A LAMA-only controller was worse: +0.495 points and 25.68% closure.  The
  result therefore does not support negative transfer from joint training.
- LAMA benefit AUROC ranged from 0.7881 to 0.8192 and average precision from
  0.2093 to 0.2933 at 6.29% prevalence.  Risk AUROC was 0.9728--0.9858, so the
  main weakness is benefit ranking rather than collateral-risk detection.
- A frozen centroid source score separated the two development domains with
  AUROC 0.8360.  The largest shifts were subject-masked and full-statement
  cosine features, supporting a representation-level domain mismatch.

All source-only transfer cells retained zero collateral.  TempLAMA-only,
LAMA-only and joint models used only the registered training/development
sources.  TempLAMA test, the original-source confirmation, LongMemEval-S and
LoCoMo remained sealed.

The deterministic next stage is a preregistered V8 pairwise fact-compatibility
representation.  Threshold relaxation and further single-source fitting are
excluded by this result.
