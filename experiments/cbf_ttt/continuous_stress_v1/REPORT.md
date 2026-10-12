# Frozen Continuous Controller Stress Test

## Goal and protocol

The IID continuous-forgetting checkpoint was frozen and evaluated without retraining on unseen projection seeds 2901/2902/2903. Four preregistered shifts each contained 32 source groups x 20 scenarios: longer memory, weaker updates, stronger updates, and extreme overlap. The primary fixed control was selected with hindsight separately on each suite. The full protocol is in `CONTINUOUS_STRESS_PLAN.md`.

## Results

The independent reconstruction audit passed. The controller beat the hindsight-best fixed alpha in all four suites with positive source-group 95% CI lower bounds, retained macro alpha MAE below 0.20, and beat the macro fixed control on every unseen projection seed. The stricter all-suite endpoint check failed.

| Suite | Learned NLL / acc. | Best fixed NLL / acc. | Learned minus endpoint NLL gain (95% CI) |
|---|---:|---:|---:|
| Long memory | 0.8028 / 85.21% | 0.8934 / 82.26% | 0.0252 (0.0170–0.0334) |
| Weak update | 1.1969 / 83.14% | 1.2908 / 62.33% | 0.1820 (0.1657–0.1987) |
| Strong update | 0.1456 / 97.49% | 0.1511 / 97.43% | -0.00117 (-0.00171–-0.00060) |
| Extreme overlap | 0.5316 / 94.29% | 0.6820 / 84.78% | 0.0243 (0.0209–0.0278) |

The macro NLL gain over hindsight-best fixed alpha was 0.1421 (0.1305–0.1542), with a 10.80-point accuracy gain. In the strong-update suite, 79.4% of labels were alpha=0, so an endpoint decision was usually sufficient and beat the frozen continuous regressor by 0.00117 NLL. This is retained as a failed preregistered criterion rather than hidden or relaxed.

## Interpretation and next step

The state-dependent rule generalizes to new random projections and remains better than any fixed decay under all tested shifts. Intermediate control adds clear value when overwrite and preservation compete; it is unnecessary when a very strong current update already resolves most conflicts. The next experiment keeps the architecture fixed and expands training support to all five domains, then uses new projection seeds and unseen compound shifts for the final evaluation.

## Artifacts and compatibility

`summary.json`, `manifest.json`, `execution_audit.json`, and `RUN_STATUS.json` contain complete aggregate results, frozen-checkpoint hashes, raw-record hashes, and the failed gate. Raw JSONL files remain at `/home/ctj/cbf_ttt_continuous_stress_v1/formal`. No baseline code, Qwen weights, pretraining data, or previous checkpoints were modified.
