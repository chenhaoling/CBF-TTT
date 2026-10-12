# Continuous-Alpha Fast-Weight Forgetting: Formal Report

## 1. Objective and interpretation

This experiment follows the forgetting-only paper direction. It asks whether a controller should choose an intermediate global session-memory decay coefficient when a current update must overwrite stale conflicting facts while preserving unrelated history. The controlled memory is additive and TTT-compatible: `M' = (1 - alpha) M + Delta`. It does not claim that the current Qwen writer has solved random text binding.

The implementation assumes that overlap, duplicate status, update scale, matrix norms/alignment, and key interference can be measured at the update boundary. Counterfactual future-query losses are used only to create offline training labels and are excluded from controller inputs.

## 2. Implementation

- `tasks/train_cbf_continuous_forgetting.py`: deterministic scenario construction, five-point counterfactual labeling, controller training, policy replay, group aggregation, development gate, and delayed test release.
- `scripts/run_cbf_continuous_forgetting.sh`: focused tests, label-distribution probe, formal run, and audit.
- `scripts/audit_cbf_continuous_forgetting.py`: independent reconstruction of labels, rollouts, summaries, hashes, ablations, gate, and conditional test result.
- `tests/test_cbf_continuous_forgetting.py`: split isolation, scenario invariants, observed-loss label, policy bounds, and test sealing smoke checks.
- `CONTINUOUS_FORGETTING_PLAN.md`: protocol and thresholds frozen before the server probe result was read.

All functionality is isolated behind new entry points. Baseline model, pretraining, TTT runtime, datasets, and configurations are unchanged.

## 3. Corpus and labels

The probe used 40 groups x 20 scenarios x 3 embedding seeds. The formal corpus used 128/32/32 train/dev/test groups, 20 scenarios per group, and seeds 1901/1902/1903. Each scenario held 6–12 old facts, changed a nonempty proper subset, optionally repeated old assignments and introduced new keys, and sampled current-write scale from 0.55–1.55.

The registered probe passed: 56.71% of 2,400 labels were interior. Counts for alpha 0/0.25/0.5/0.75/1 were 868/688/427/246/171, so all three interior points independently exceeded 5%. Formal training contained 7,680 labeled seed-scenario pairs with counts 2679/2186/1364/865/586.

## 4. Controllers and controls

The primary controller is a sigmoid MLP regressor trained by MSE against the grid-optimal alpha. Controls include a grid classifier, a structure-blind regressor, an endpoint-only 0/1 classifier, a physical conflict heuristic, the train-label mean, every fixed grid value, and the five-point counterfactual oracle. Policies are compared by replaying their selected coefficient and recomputing query NLL and top-1 accuracy.

## 5. Formal results

Seven server tests passed. All seven preregistered development checks passed, so test was generated and scored once. The independent audit passed and reproduced every record and aggregate.

| Split / policy | NLL | Accuracy | Alpha MAE |
|---|---:|---:|---:|
| Dev learned continuous | 0.3506 | 95.96% | 0.0622 |
| Dev best fixed alpha=0.25 | 0.4400 | 89.98% | 0.2591 |
| Dev endpoint learned | 0.4139 | 90.32% | 0.1854 |
| Test learned continuous | 0.3697 | 95.67% | 0.0640 |
| Test grid classifier | 0.3740 | 95.38% | 0.0358 |
| Test best fixed alpha=0.25 | 0.4536 | 90.05% | 0.2374 |
| Test endpoint learned | 0.4395 | 88.96% | 0.1965 |
| Test structure blind | 0.4539 | 93.00% | 0.2535 |
| Test physical heuristic | 0.4190 | 95.57% | 0.1990 |
| Test five-point grid oracle | 0.3717 | 95.72% | 0.0000 |

On development, learned continuous alpha improved NLL over the best fixed alpha by 0.0894 (source-group bootstrap 95% CI 0.0821–0.0974) and accuracy by 5.98 points. It improved NLL over the endpoint controller by 0.0634 (0.0579–0.0690) and accuracy by 5.64 points.

On the once-released test split, the NLL gain over the best fixed alpha was 0.0839 (0.0760–0.0920) and the accuracy gain was 5.62 points (4.60–6.72). The NLL gain over endpoint control was 0.0698 (0.0630–0.0765); the NLL gain over the structure-blind ablation was 0.0842 (0.0769–0.0916). Every embedding seed beat the best fixed alpha. The continuous prediction had 0.0020 lower NLL than the discrete grid oracle because it can interpolate between grid points; this does not make it an oracle for the continuous interval.

## 6. Commands

```bash
ROOT=/home/ctj/cbf_ttt_continuous_forgetting_v1 \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
TEST_PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python \
bash scripts/run_cbf_continuous_forgetting.sh
```

The complete run, including training, took 16.35 seconds before audit on the CPU host. The experiment does not require GPU memory.

## 7. Conclusion and limitations

The result supports three claims inside this controlled fast-weight setting: intermediate alpha values are common; a learned state-dependent coefficient beats every fixed coefficient; and allowing intermediate values materially beats an otherwise learned endpoint-only decision. The structure-blind failure shows that the gain depends on conflict/preservation state rather than only matrix scale.

The result is not yet direct evidence on natural-language Qwen TTT memory. Train and IID test share projection seeds, values are synthetic random vectors, and the controller observes clean semantic overlap statistics. The next registered experiment freezes the checkpoint, changes all projection seeds, and evaluates long memory, weak/strong updates, and extreme-overlap distributions. Later work must estimate comparable signals from real model activations and evaluate an actual TTT layer.

## 8. Artifact map and TODO

- `probe.json`: immutable label-distribution gate.
- `summary.json`: complete dev/test policy metrics and groups.
- `training.json`: label counts, curves, runtime, and checkpoint hash.
- `training_manifest.json`: frozen seeds, dimensions, policies, split sizes, and release rule.
- `execution_audit.json`: independent audit result and raw-record hashes.
- `test_comparisons.json`: deterministic source-group bootstrap comparisons derived from the archived test groups.
- Remote large artifacts: `/home/ctj/cbf_ttt_continuous_forgetting_v1/formal/{controllers.pt,dev_records.jsonl,test_records.jsonl}`.

Outstanding work: complete the frozen-controller stress test, repeat with correlated/nonlinear keys, then implement an activation-derived proxy in the real TTT layer. Risks include distribution shift, imperfect observable conflict features, grid-label discretization, and the gap between low-rank associative memory and Qwen in-place parameter updates.
