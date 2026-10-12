# Qwen-grounded low-rank memory bridge

## Purpose

The audited final-writer experiment showed a generic populated-memory effect without value binding. This bridge changes the memory representation before returning to forgetting-policy training.

For an entity key `k` and observed value token `v`, the session stores a rank-one component `v k^T`. `k` is the normalized mean Qwen input embedding of the entity string. `v` is the normalized Qwen LM-head direction of the observed value token. At query time the same entity key retrieves a residual `M k`, which is added to the frozen final query hidden state before the unchanged LM head.

This is a controlled, readable memory substrate. It does not claim that the native dense writer learned this representation. Its purpose is to connect the established low-rank forgetting result to the actual Qwen embedding/readout geometry.

## Isolation and selection

- dev groups 0--7 select one residual scale from `[0, 1, 2, 4, 8, 16]`;
- Q groups 8--15 confirm the fixed scale;
- V groups 16--23 remain sealed until Q passes;
- every group includes all eight counterfactual colors under the identical query and background identity.

Correct, opposite-value Wrong, Zero, and norm-matched Rademacher Random memories are scored with fresh attention KV. The Q/V gate requires at least 75% Correct candidate and full-vocabulary first-token accuracy in each domain; at least 50 points over Wrong and Zero candidate accuracy; and Wrong/Zero/Random minus Correct answer NLL at least 0.2 with source-group bootstrap 95% lower bound above zero. EOS first-token rate must be at most 5%.

Only Q passing unlocks V. A passing V authorizes the next multi-block experiment, where rank-one components accumulate and the previously validated local/global forgetting controller chooses which components to retain.

The bridge is opt-in and does not modify baseline model definitions, checkpoints, or default inference.
