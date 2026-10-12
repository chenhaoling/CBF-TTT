# Qwen-grounded multi-component forgetting

## Hypothesis

A state-dependent controller that chooses retain, global half-decay, referenced-component deletion, or global reset will outperform every fixed action and a learned global-only controller when additive low-rank memory components live in Qwen's embedding/readout geometry.

## Data and isolation

- generate a fresh event curriculum with seed `20261101`;
- 128/32/32 disjoint train/development/test worlds;
- 24 episodes per world across retain, revoke-last-1, revoke-last-2, reset, one to three decision boundaries, and twins;
- entity namespaces, values, and surface templates are split-disjoint;
- test is not loaded or scored until the aggregate development gate passes.

## Qwen-grounded memory

Entity keys start from the final Qwen3-4B input-embedding mean. Values start from the final Qwen3-4B LM-head token-direction mean. Shared prefix structure is removed by centering all entity vectors and all value vectors separately. Three fixed Gaussian projections map the centered vectors to 128 dimensions and are normalized.

Each visible assignment writes the outer product `value * key^T`. Memory is the weighted sum of batch components. Query retrieval and candidate ranking use the same projected Qwen value directions. The model checkpoint remains frozen; only the small forgetting controllers are trained.

## Policies

- fixed retain / half / referenced local deletion / global reset;
- deterministic random;
- history/reference-blind learned controller;
- learned global-only controller restricted to retain/half/reset;
- full learned controller;
- visible physics rule;
- counterfactual oracle.

Controllers use only online memory invariants and visible event features. Private answers and future blocks never enter features.

## Development gate

Across all three projection seeds:

1. oracle mean accuracy at least 90%;
2. full learned mean accuracy at least 88%;
3. every seed at least 85%;
4. learned remains within three points of oracle;
5. learned exceeds the stronger of the best fixed policy and global-only learned by at least eight points;
6. source-world bootstrap 95% lower bound on that gain exceeds five points.

Only a passing aggregate gate unlocks one test evaluation. The same policies, controllers, projections, thresholds, and temperature are frozen for test.

## Scope

A pass bridges the prior controlled low-rank forgetting result into Qwen-derived representation geometry. It does not yet prove native dense `ttt_conv/ttt_proj` memory, automatic natural-text key/value extraction, or open-vocabulary generation. The next paper-level step after a pass is natural ZsRE/RULER-QA/LongMemEval-S evaluation with an automatic extractor and capacity/latency accounting.
