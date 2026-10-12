# Frozen Continuous-Forgetting Controller Stress Test

## Question

The in-distribution formal result uses new source groups but the same three random fast-weight projections and the same scenario ranges as training. This experiment tests whether the frozen controller learned a portable state rule rather than a projection-specific or range-specific shortcut.

## Frozen protocol

The checkpoint from `continuous_forgetting_v1` is loaded without retraining. Evaluation uses three unseen embedding seeds `2901/2902/2903` and four preregistered shifts, each with 32 source groups and 20 scenarios per group:

- `long_memory`: 16–28 old facts, up to eight new facts;
- `weak_update`: current-write scale 0.20–0.50, below training support;
- `strong_update`: current-write scale 1.60–2.20, above training support;
- `extreme_overlap`: conflict ratio at most 0.25 or at least 0.70, wider counts and scales.

Every frozen policy from the formal experiment is replayed. For each suite, the primary comparison is against the best fixed grid alpha selected with hindsight on that suite, making the fixed baseline stronger than a deployable development-selected constant. Other controls are the endpoint-only controller, structure-blind controller, physical heuristic, grid classifier, and grid oracle.

Success requires the learned continuous controller to beat the hindsight best fixed alpha and endpoint controller in every suite with a source-group bootstrap 95% lower bound above zero, macro alpha MAE at most 0.20, and lower macro NLL than the macro best fixed alpha for every unseen embedding seed. The thresholds and suites are frozen before any stress scenario is generated. `scripts/audit_cbf_continuous_stress.py` independently reconstructs every scenario, rollout, aggregation, bootstrap interval, hash, and check.
