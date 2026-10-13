# Factorized causal forgetting V4

## Status

Completed and independently audited. Six of ten release checks passed. The
registered gate failed, so LoCoMo remained sealed.

## Result

The frozen candidate cascade retained the V3 gain: target recall@4 was 99.25%
and top-1 was 96.63%. The three-seed controller mean was:

| Metric | Result |
|---|---:|
| Action accuracy | 96.51% |
| Strict stale/current pair state | 82.74% |
| Stale deletion | 83.71% |
| Current retention | 98.85% |
| Non-target retention | 97.34% |
| Hard-retain action accuracy | 97.32% |
| Primary-update action accuracy | 80.93% |

Safety passed at every aggregate gate. Strict pair state by length was
84.52/81.52/82.52/82.40% for 16/32/48/64. The remaining failure was missed
updates rather than cumulative over-deletion.

Seeds were heterogeneous. Seed 31102 reached 91.48% pair state, 92.42% stale
deletion, 98.97% current retention, and 97.51% non-target retention; every
length exceeded 90%. Seed 31103 reached only 70.04% pair state. The primary
seed beat V1 by 28.37 points, but its clustered CI `[23.03,33.52]` missed the
registered 25-point lower-bound floor. It beat exact deletion-count-matched
random by 79.31 points with CI `[75.19,83.05]`.

## Calibration diagnosis

No fit-only seed had a tune threshold satisfying both the 85% update constraint
and the causal safety constraints. The registered fallback therefore selected
the best unsafe tune point. Refitting on fit+tune shifted score calibration and
produced safe development behavior, but with large seed variance. This makes
single-split threshold transfer the remaining bottleneck. The next method must
estimate deletion calibration out of fold and deploy calibrated fold models
without a score-changing refit.

## Reproducibility

- runtime: 383.54 seconds;
- peak GPU allocated/reserved: 8.026/8.988 GiB;
- 333,316 detailed evidence-cache entries and 6,326,407 cache hits;
- independent audit: all ten checks passed;
- summary SHA-256: `b796c9bf64c1ebd3f744df3c9b333478595ae3ba7fe41b6d7d48748ed2c9f613`.

The 54 MiB raw traces remain at
`/home/ctj/cbf_ttt_factorized_causal_forgetting_v4` on `hku-gpu2` and are
covered by archived hashes.
