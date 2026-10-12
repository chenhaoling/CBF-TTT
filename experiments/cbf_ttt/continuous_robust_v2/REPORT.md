# Robust Continuous Forgetting: Final Controlled Experiment

## 1. Objective

This experiment tests the paper hypothesis that a session-memory controller should choose a continuous retention/forgetting strength from the current memory state, and that broader training support can make this rule portable across memory sizes, update strengths, conflict ratios, and unseen fast-weight projections.

It follows a preregistered failed stress result: the original controller beat every hindsight-selected fixed alpha, but a 0/1 endpoint controller was 0.00117 NLL better for strong updates. The failed criterion is retained in `continuous_stress_v1`; this run changes only the training distribution, not the model architecture or features.

## 2. Data, model, and release protocol

The robust training set equally mixes base, long-memory, weak-update, strong-update, and extreme-overlap domains. Each contributes 64 source groups x 20 scenarios x three training projection seeds, totaling 19,200 labeled seed-scenario examples. The controller is the same sigmoid MLP regressor trained on five-point counterfactual alpha labels.

Development uses disjoint groups and unseen projection seeds 2901/2902/2903. All seven registered development checks must pass before test scenario generation. Final test uses new seeds 3901/3902/3903, 32 groups x 20 scenarios in each of five component domains and three unseen compound domains: long+weak, long+strong, and extreme-overlap+extreme-scale.

Controller inputs contain only update-boundary statistics. Future loss, answer identities, group IDs, and split IDs are excluded. Every reported policy is replayed through the same fast-weight equation before scoring.

## 3. Development gate

All seven checks passed. Robust training reduced NLL versus the original frozen controller by 0.01023 with source-group bootstrap 95% CI 0.00710–0.01388. The strong-update gap to endpoint control was 0.000539, within the registered 0.002 noninferiority margin. Macro development results were:

| Policy | NLL | Accuracy | Alpha MAE |
|---|---:|---:|---:|
| Robust continuous | 0.6001 | 91.07% | 0.0565 |
| Endpoint learned | 0.6604 | 89.00% | 0.1574 |
| Structure blind | 0.7027 | 85.30% | 0.2579 |
| Hindsight best fixed alpha=0.25 | 0.7449 | 81.12% | 0.2885 |
| Five-point grid oracle | 0.6015 | 90.89% | 0 |

## 4. Once-released final test

The robust controller achieved macro NLL 0.7591 and accuracy 87.78%. The hindsight best fixed alpha achieved 0.9139 and 76.80%; the controller improved NLL by 0.1548 (95% CI 0.1476–0.1620) and accuracy by 10.98 points (9.98–12.00). Endpoint control achieved NLL 0.8165 and accuracy 87.07%; continuous control improved NLL by 0.05737 (0.05048–0.06445) and accuracy by 0.72 points (0.25–1.17).

| Test domain | Continuous NLL / acc. | Best fixed NLL | Endpoint NLL | NLL gain vs fixed (95% CI) | NLL gain vs endpoint (95% CI) |
|---|---:|---:|---:|---:|---:|
| Base | 0.3534 / 96.03% | 0.4398 | 0.4170 | 0.0864 (0.0796–0.0933) | 0.0636 (0.0569–0.0698) |
| Long memory | 0.7505 / 88.82% | 0.8761 | 0.8071 | 0.1256 (0.1181–0.1332) | 0.0566 (0.0508–0.0627) |
| Weak update | 1.1611 / 79.37% | 1.2626 | 1.3172 | 0.1015 (0.0922–0.1111) | 0.1561 (0.1432–0.1694) |
| Strong update | 0.1540 / 97.39% | 0.1592 | 0.1533 | 0.00519 (0.00384–0.00663) | -0.00067 (-0.00096–-0.00037) |
| Extreme overlap | 0.5113 / 93.88% | 0.6557 | 0.5387 | 0.1444 (0.1358–0.1545) | 0.0274 (0.0230–0.0316) |
| Long + weak | 1.7805 / 68.70% | 1.9499 | 1.9077 | 0.1693 (0.1565–0.1834) | 0.1271 (0.1171–0.1374) |
| Long + strong | 0.4695 / 88.70% | 0.4891 | 0.4739 | 0.0196 (0.0151–0.0240) | 0.00432 (0.00235–0.00637) |
| Extreme + scale | 0.8923 / 89.37% | 1.0958 | 0.9168 | 0.2035 (0.1884–0.2203) | 0.0245 (0.0211–0.0281) |

The isolated strong-update domain remains the expected boundary: most decisions need no decay, so endpoint alpha=0 is 0.00067 NLL better. In the harder unseen long+strong composition, continuous control again significantly beats endpoint control. This supports a conditional mechanism claim: intermediate coefficients matter when overwrite and preservation compete, while the controller should approach an endpoint when the current write already dominates.

The continuous regressor has macro alpha MAE 0.0550. Its NLL is 0.0010 lower than the five-point grid oracle because continuous outputs interpolate between label grid points; the grid oracle is not a continuous optimum.

## 5. Integrity, resources, and reproducibility

- Nine focused server tests passed.
- Test files were absent until every development check passed.
- Independent audit rebuilt 19,200 train labels, five development domains, eight test domains, all policy rollouts, bootstrap intervals, and hashes; it passed.
- Training plus released evaluation took 147.08 seconds on CPU. The audit deliberately recomputed all artifacts and took longer.
- Large checkpoint and raw records remain in `/home/ctj/cbf_ttt_continuous_robust_v2/formal`.

```bash
FROZEN=/home/ctj/cbf_ttt_continuous_forgetting_v1/formal \
ROOT=/home/ctj/cbf_ttt_continuous_robust_v2 \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
TEST_PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python \
bash scripts/run_cbf_continuous_robust.sh
```

## 6. Claim supported and remaining risk

The result is suitable as controlled mechanistic evidence for state-dependent continuous session forgetting in additive fast-weight memory. It demonstrates interior optima, fixed-policy failure, endpoint-policy limits, dependence on conflict structure, projection generalization, distribution-shift robustness, and unseen composition generalization.

It does not establish end-to-end gains on Qwen natural-language benchmarks. Earlier Qwen experiments show that the current native TTT writer does not reliably bind random session facts in memory-only readout, so applying this controller there would not repair the underlying writer. The next model-level milestone is a readable TTT-compatible memory writer or an activation-derived proxy whose observable features replace the synthetic conflict counts.
