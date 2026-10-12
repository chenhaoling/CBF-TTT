# Automatic scope boundary V7

## Objective

Audit whether V6 failed because every inner fold hit the maximum registered
entity/relation weight. V7 changed only the grid from values ending at 8 to
`{8,12,16,24,32,48,64}`. Features, folds, checkpoint, metrics, and gate were
unchanged.

## Result

| Method | Top-1 | MRR |
|---|---:|---:|
| V6 original grid | 93.25% | 96.37% |
| **V7 extended grid** | **93.63%** | **96.76%** |
| V5 oracle segmentation | 97.83% | 98.92% |

V7 improved over V4 by 10.70 points, CI `[7.90,13.63]`, but still failed the
95% top-1, 97% MRR, and within-three-points checks. The one-shot boundary audit
therefore failed. Scalar entity/relation weight expansion stops permanently.

The next method must learn a richer pairwise combination of fixed relation,
automatic span, and lexical evidence under nested OOF evaluation.

## Claim boundary

CounterFact remains observed development-only. LongMemEval-S and LoCoMo were
not scored. V7 provides calibration evidence only and does not support causal
forgetting or native fast-weight claims.

