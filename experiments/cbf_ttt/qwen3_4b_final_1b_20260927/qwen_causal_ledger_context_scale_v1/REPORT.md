# Causal-ledger contextual residual-scale diagnostic

## Decision

The frozen diagnostic completed on all 1,008 failed-pilot queries and passed
its independent audit. Scale 32 is the smallest registered scale satisfying
both selection thresholds: 99.7024% free five-token exact match and 99.7024%
teacher-forced full-vocabulary sequence accuracy. It is eligible for a new
versioned content validation. It does not retroactively change the failed V1
pilot or authorize development construction by itself.

## Results

| Scale | Free exact | Teacher sequence exact | Free token accuracy | Answer NLL | Minimum family exact |
|---:|---:|---:|---:|---:|---:|
| 16 | 27.08% | 27.08% | 84.50% | 0.37882 | 20.14% |
| 20 | 36.11% | 36.11% | 86.57% | 0.22474 | 25.69% |
| 24 | 55.06% | 55.85% | 90.81% | 0.12012 | 45.83% |
| 32 | 99.70% | 99.70% | 99.94% | 0.02319 | 97.92% |
| 48 | 100.00% | 100.00% | 100.00% | 0.00040 | 100.00% |
| 64 | 100.00% | 100.00% | 100.00% | 0.000006 | 100.00% |

At scale 32 the five free-rollout position accuracies were
`[1.0, 0.9970, 1.0, 1.0, 1.0]`. The discrete transition between scales 24 and
32 agrees with the preceding geometry result: the unit-row target is already
the LM-head top token, but its scale-16 margin does not reliably overcome the
contextual preference for a no-space second token.

## Execution and integrity

- Fixed scales: 16/20/24/32/48/64.
- Rows: 1,008; no sampled or dropped queries.
- Two RTX 5090 critical path: 271.3971 seconds.
- Peak allocated/reserved memory: 7.6696/7.6934 GiB.
- Model and tokenizer are identical to the failed content pilot.
- No missing, unexpected or mismatched checkpoint keys.
- Development, confirmation and OOD remain ungenerated.

`execution_audit.json` passed terminal/exit, source coverage, shard hashes,
row-level token recomputation, independent aggregation/selection and split
sealing. The raw shard rows, resource manifests, logs, summary and audit are
included in this directory.

## Next stage

The successor must use a fresh resource namespace and freeze scale 32. It must
generate exactly five tokens and separately verify:

1. Correct/Wrong/Zero/Random content causality;
2. an explicit twin-aware counterfactual wherever the lifecycle leaves twin
   answers different;
3. the complete multi-record causal ledger, including absent/null routing;
4. dense similarity aggregation versus an addressable top-1 key lookup;
5. family, role and absence floors.

Only a passing independently audited successor may construct development data.

