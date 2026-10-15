# Qwen natural joint key/defer ranker V3

V3 tested whether the frozen pair feature could replace cosine ranking and
retrieve/defer with one nine-class linear model.  The model used five-fold
session OOF predictions, six equally weighted natural/no-match strata and a
fully preregistered anchored search.

No configuration simultaneously reached the required 92% no-match specificity
and 98% exact selection, so the protocol stopped before any new Qwen outcome
score.  The best configuration under the no-match constraint achieved only
90.59% exact, 56.18% rephrase and 33.75% auxiliary-rephrase selection, with
92.57% no-match specificity.

This negative result shows that the pairwise linear feature is useful for
retrieve/defer calibration but cannot replace the already strong frozen cosine
ranker.  Subsequent work therefore preserves cosine rank one and calibrates
only whether that record should be read.  Development, test and external
benchmarks remained sealed.
