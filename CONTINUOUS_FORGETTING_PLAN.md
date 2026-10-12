# Continuous-Alpha Fast-Weight Tradeoff Experiment

## Question

The low-rank experiment validates selective local/global forgetting, but alpha 0.5 was optimal in less than 0.3% of decisions. This experiment asks whether a continuous global coefficient is useful when the same update must both overwrite conflicting facts and preserve unrelated historical facts.

## Scenario

Each scenario starts with a low-rank outer-product memory for 6–12 facts. The current delta overwrites a variable subset, repeats some unchanged assignments, and introduces new facts. Evaluation queries every currently valid fact. Increasing alpha suppresses stale conflicts but also weakens unrelated history, creating a measurable tradeoff.

Counterfactual labels use the fixed grid `{0, 0.25, 0.5, 0.75, 1}` and minimize future-query NLL with the smaller-alpha tie rule. Controller features are available at the update boundary: memory/delta norms and alignment, conflict/overlap/duplicate/new ratios, fact counts, update scale, and key-interference statistics. Answers and future query outcomes are excluded.

Before formal training, a deterministic probe must show at least 30% interior labels and at least 5% support for two distinct interior grid points. Otherwise the scenario generator is revised before any development or test result is observed.

The formal experiment will use disjoint train/development/test scenario groups and three independent fast-weight embedding seeds. Test remains ungenerated and unread until the registered aggregate development gate passes.
