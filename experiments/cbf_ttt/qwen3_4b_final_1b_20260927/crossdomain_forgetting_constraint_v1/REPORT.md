# Constraint-calibrated V2 causal replay

## Change

No model weight changed. Fold thresholds were selected only from saved inner-OOF
grids by satisfying hard-retain ≥85% and update ≥75%, then maximizing action.
The resulting thresholds were 0.50, 0.45, 0.40, 0.45, and 0.40. One fold used
the registered minimum-deficit fallback.

## Result

| Metric | Constraint calibration | Original V2 | V1 |
|---|---:|---:|---:|
| Query accuracy | 48.64% | 46.79% | **49.66%** |
| Action accuracy | 84.29% | 78.26% | **84.93%** |
| Candidate top-1 | 79.62% | 79.62% | 78.47% |
| Initial retain | 96.69% | 93.38% | **98.73%** |
| Hard retain | 88.66% | 80.25% | **92.36%** |
| Update | 67.52% | 61.15% | 63.69% |

Constraint calibration recovered 1.83 query points over original V2 (95% CI
`[1.43,2.24]`) and 6.03 action points (CI `[4.76,7.30]`). Its oracle-state
action/update were 85.82/76.82%, but causal action/update were 84.29/67.52%.
The complete development gate still failed on overall action, update, and query
above fixed retain.

The 9.30-point oracle-to-causal update gap confirms that static calibration does
not solve exposure bias. A false deletion changes later memory membership and
cannot be represented by replaying an oracle candidate set. The next registered
experiment fits only the gate on policy-induced states and selects thresholds
through causal inner-fold rollout.

Runtime was 195.07 seconds on one RTX 5090 with 9.026/12.580 GiB peak
allocated/reserved memory. CounterFact remains observed development data;
LongMemEval-S and LoCoMo remained unscored.

