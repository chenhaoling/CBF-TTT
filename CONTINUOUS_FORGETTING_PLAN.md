# Continuous-Alpha Fast-Weight Tradeoff Experiment

## Question

The low-rank experiment validates selective local/global forgetting, but alpha 0.5 was optimal in less than 0.3% of decisions. This experiment asks whether a continuous global coefficient is useful when the same update must both overwrite conflicting facts and preserve unrelated historical facts.

## Scenario

Each scenario starts with a low-rank outer-product memory for 6–12 facts. The current delta overwrites a variable subset, repeats some unchanged assignments, and introduces new facts. Evaluation queries every currently valid fact. Increasing alpha suppresses stale conflicts but also weakens unrelated history, creating a measurable tradeoff.

Counterfactual labels use the fixed grid `{0, 0.25, 0.5, 0.75, 1}` and minimize future-query NLL with the smaller-alpha tie rule. Controller features are available at the update boundary: memory/delta norms and alignment, conflict/overlap/duplicate/new ratios, fact counts, update scale, and key-interference statistics. Answers and future query outcomes are excluded.

Before formal training, a deterministic probe must show at least 30% interior labels and at least 5% support for two distinct interior grid points. Otherwise the scenario generator is revised before any development or test result is observed.

The formal experiment will use disjoint train/development/test scenario groups and three independent fast-weight embedding seeds. Test remains ungenerated and unread until the registered aggregate development gate passes.

## Frozen formal protocol

The formal corpus contains 128/32/32 train/development/test source groups with 20 scenarios per group and embedding seeds 1901/1902/1903. The test scenarios are generated only after the aggregate development gate passes. A pooled controller sees all three training projections, while source groups remain disjoint across splits.

The primary controller is a two-hidden-layer sigmoid regressor trained by MSE against the five-point counterfactual optimum. Controls are a five-way grid classifier, a regressor that only sees matrix norms/alignment and key-correlation statistics, an endpoint-only classifier trained against alpha 0 versus 1, a fixed physical heuristic, the training-label mean, every fixed grid value, and the counterfactual grid oracle. All policies are evaluated by replaying their selected alpha in the fast-weight equation and recomputing current-truth NLL and top-1 accuracy.

The registered development gate requires: alpha MAE at most 0.15; mean NLL regret to the grid oracle at most 0.03; NLL gain over the best fixed alpha at least 0.05 with source-group bootstrap lower bound above 0.02; NLL gain over the endpoint-only learned controller at least 0.02 with lower bound above 0.005; and lower NLL than the best fixed alpha for every embedding seed. No threshold is changed after development evaluation. A failed gate leaves test ungenerated and starts a separately recorded development iteration.

`scripts/run_cbf_continuous_forgetting.sh` executes focused tests, the registered probe, the formal run, and an independent reconstruction audit. The audit rebuilds train labels and development rollouts, verifies all hashes and controller ablations, and conditionally reconstructs test only when the stored development gate passed.
