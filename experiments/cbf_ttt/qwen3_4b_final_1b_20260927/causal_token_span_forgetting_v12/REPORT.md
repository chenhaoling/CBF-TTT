# Causal token-span forgetting V12 report

## Result

V12 froze the passing V11 resolver and replayed every CounterFact trajectory
causally. It produced a statistically significant improvement over retaining
all history, but failed the registered action-quality gate.

| Metric | Result |
| --- | ---: |
| Static candidate top-1 | 97.07% |
| Causal query accuracy | 50.92% |
| Query gain over fixed retain | +1.46 points |
| Paired-bootstrap 95% CI | [0.99, 1.96] points |
| Overall action accuracy | 90.40% |
| Hard-retain accuracy | 93.50% |
| Primary-update accuracy | 80.00% |
| Oracle-local query gap | -0.95 points |
| Runtime | 441.11 s |
| Peak allocated / reserved GPU memory | 9.03 / 10.74 GiB |

All folds selected epoch 50. Thresholds were 0.60, 0.65, 0.65, 0.65, and
0.70. Every inner fold used the minimum-constraint-deficit fallback; none could
simultaneously achieve the registered 95% hard-retain and 90% update floors.

## Error audit

One deterministic action trajectory contains 2,355 decisions and 226 errors:

- 18 primary-initial false deletions;
- 51 hard-retain false deletions;
- 121 primary-update false retains;
- 36 primary-update wrong-candidate deletions.

Thus 121/157 update errors arise because a true conflict is rejected by the
binary gate, while only 36 arise after authorizing the wrong candidate. This is
consistent with V11's 97% static candidate accuracy and identifies the gate,
rather than the candidate resolver, as the current bottleneck. Sessions with no
action error reached 53.68% query accuracy; sessions with at least one error
reached 49.30%.

## Interpretation

The positive paired query interval establishes a causal memory benefit: the
frozen automatic scope resolver changes retained memory in a way that improves
later queries. The failed action floors prevent claiming a finished controller.
The linear oracle-state gate is conservative and does not cover its own
policy-induced states. V13 will keep V11 frozen, expose richer relation-posterior
geometry to a small nonlinear gate, and add one on-policy data-aggregation
round. It will include an oracle-state nonlinear ablation to separate the two
changes.
