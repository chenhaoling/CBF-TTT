# On-policy forgetting gate V3

## Method

V3 froze each V2 residual candidate ranker, rolled the constraint-calibrated
policy through training sessions, and refit only the forgetting gate on the
encountered memory states. Retain errors received twice the loss weight because
deletion is irreversible. A separate inner CounterFact fold selected each
threshold by full causal state evolution.

## Result

| Metric | V3 on-policy | V1 best | Constraint diagnostic |
|---|---:|---:|---:|
| Query accuracy | 49.15% | **49.66%** | 48.64% |
| Action accuracy | 82.34% | **84.93%** | 84.29% |
| Candidate top-1 | **79.62%** | 78.47% | **79.62%** |
| Initial retain | 98.34% | **98.73%** | 96.69% |
| Hard retain | 88.92% | **92.36%** | 88.66% |
| Update | 59.75% | **63.69%** | 67.52% |

V3 was 0.49 query points below V1, with paired 95% CI
`[-0.93,-0.08]`, and 2.59 action points below, CI `[-3.99,-1.23]`.
Candidate and hard-retain checks passed, while overall action, update, and query
above fixed retain failed.

All five inner causal sweeps used the registered minimum-deficit fallback; no
fold contained a threshold satisfying hard-retain ≥85% and update ≥75%
simultaneously. Each training rollout contained 1,416–1,445 available local
updates and 51–86 local events whose old chain had already been deleted. Runtime
was 344.82 seconds on one RTX 5090, with 9.026/12.551 GiB peak
allocated/reserved memory.

## Stop decision

The preregistered stop condition fired. The current explicit candidate-memory
formulation has a stable approximately 80% candidate ceiling, but neither
oracle-state, constrained, nor one-round on-policy gate fitting yields reliable
causal updating. More threshold tuning on observed CounterFact is not justified.

The next architecture change must improve the relation/scope representation,
then retain the causal rollout and asymmetric deletion evaluation introduced
here. CounterFact remains development-only. LongMemEval-S and LoCoMo have not
been scored and remain available for a later frozen final method.

