# Natural forgetting Factor Gate V2

## Result

The registered low-dimensional factorized controller passed every calibration
release check and its independent audit passed all 10 integrity checks.  The
controller uses 13 inference-visible variables, excludes all 2564 raw reader
coordinates, and separately predicts the future benefit and collateral risk
of exact scope-top deletion.

| Policy | Candidate accuracy | Added unrelated errors |
|---|---:|---:|
| Retain all | 80.8691% | 0 |
| Always delete scope top | 81.5647% | 855 |
| Barrier-Q V1 | 80.9476% | 120 |
| Direct classifier V1 | 81.6634% | 418--492 |
| Factor Gate V2 | **82.0336%** | **2** |
| Benefit-only V2 | 82.3474% | 35 |
| Safe binary oracle | 83.1976% | 0 |
| Full safe oracle | 83.6438% | 0 |

Factor Gate V2 improved over retain-all by 1.1712 points with a paired
session-bootstrap 95% CI of [0.9949, 1.3538] points.  It improved over the
strongest fixed action by 0.4846 points with CI [0.1686, 0.8019].  It captured
41.97% of the full safe-oracle headroom.  All three seeds improved both
registered baselines; four of five outer folds were feasible.

The risk gate removed 33 of the 35 collateral errors made by benefit-only at a
cost of 0.3137 accuracy points.  This directly supports the factorization:
predicting benefit alone is more accurate, but its extra forgetting is unsafe.

## Frozen configuration

The released engineered model is linear (`hidden=0`) with learning rate
0.005, zero weight decay, benefit threshold 0.8 and risk threshold 0.1.  The
registered no-schedule ablation reached roughly 81.48--81.70%, below the
primary model, showing that memory size, event position and recency carry
useful decision information.

## Scope of evidence

These are nested grouped OOF results on 65 calibration sessions and 774 event
states.  Each label evaluates one intervention from a retain-all prefix.  The
result authorizes the separately preregistered on-policy V3 development
rollout; it does not itself prove that the controller remains effective after
its earlier decisions change later memory states.

## Integrity

- Terminal status: `completed_qwen_natural_forgetting_factor_gate_v2_development`
- Exit code: `0`
- Training time: 329.62 seconds on CPU
- Factor checkpoint SHA256:
  `3a0f2ff062d458640fd516b75521d72ba1af5d3ad06e7cffd127ebc410e8e691`
- Independent audit: 10/10 checks passed
- Development, test and external benchmarks were sealed during training.

Machine-readable results are in `result/summary.json`; full OOF decisions,
search tables, checkpoint, features and the independent audit are archived in
this directory.
