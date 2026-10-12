# Domain-Augmented Continuous Forgetting Experiment

## Motivation

The frozen IID controller beat the hindsight best fixed alpha in all four preregistered stress suites, but missed the all-suite endpoint criterion because endpoint alpha=0 was 0.00117 NLL better under strong updates. This result is retained as a failed stress gate. The next experiment tests whether wider training support repairs that extrapolation error without changing the controller architecture.

## Frozen protocol

The same sigmoid MLP and features are trained on an equal mixture of the base, long-memory, weak-update, strong-update, and extreme-overlap domains. Each domain contributes 64 groups x 20 scenarios under projection seeds 1901/1902/1903. Development uses disjoint groups and unseen projection seeds 2901/2902/2903. The original IID controller is evaluated on exactly the same development rows as a frozen baseline.

The robust development gate requires macro alpha MAE at most 0.12; source-group bootstrap lower NLL-gain bounds above 0.03 versus the macro hindsight-best fixed alpha and above 0.01 versus endpoint control; improvement over the original frozen controller with lower bound above 0.003; at most 0.002 NLL disadvantage to endpoint control in the strong-update domain; significant improvement over hindsight-best fixed alpha in every domain; and improvement over the macro fixed alpha for every development projection seed.

Only after all checks pass are final scenarios generated. Final evaluation uses projection seeds 3901/3902/3903, 32 new groups per domain, the five development domains, and three unseen compositions: long+weak, long+strong, and extreme-overlap+extreme-scale. The controller is not retrained after development. The independent audit rebuilds training labels, all released records, group bootstraps, hashes, and conditional release state.
