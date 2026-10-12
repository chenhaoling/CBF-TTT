# ZsRE same-subject hard-retain forgetting plan

## Trigger

Natural-forgetting pilot v1 was nearly solved by a cosine threshold: cosine local reached 97.21% versus a 97.77% oracle, while direct learned-gate transfer reached 84.43%. Every v1 retain event introduced another subject, so the task did not require semantic update scope.

## Calibration-only repair

For each primary update chain, turn the first version's two ZsRE locality aliases into one independently stored auxiliary fact:

- alias one is the auxiliary key;
- alias two is its rephrase query;
- their source ground-truth sets must match exactly;
- the shared first ground truth is the auxiliary value.

All 26/31/52 calibration/development/test primary chains pass this consistency check. Each session packs at most four primary chains plus their four auxiliary facts, keeping the ideal active memory at eight records.

## Event sequence

1. Insert primary version 0: `retain`.
2. Insert its same-subject auxiliary locality fact: hard `retain`.
3. Insert later primary versions: `local`, deleting only the preceding primary record.

The auxiliary prompt may be lexically and relationally close to the primary prompt, but its source ground truth must remain intact. Policies receive no chain identity or action label.

## Policies and readout

Reuse the six v1 policies and three frozen gate seeds. Query every active primary and auxiliary fact with its current and all historical exact/rephrase forms; every query must return the newest value for its own chain. No `defer` query is counted in v2 because both main and locality facts are explicitly stored.

## Stage A2

Run calibration only, with the same runtime, peak-memory, action, stale-read and chain-bootstrap accounting. Development and test remain sealed. A2 must demonstrate that the hard-retain construction creates a nontrivial gap between cosine and oracle and that at least one learned or hybrid policy improves on cosine before any development gate is frozen.

Bootstrap units pair each primary chain with its auxiliary fact, so correlated same-subject queries are never treated as independent. Report primary-initial retain, hard-retain, and primary-update actions separately. The preregistered numerical gate requires a cosine-to-oracle accuracy gap of at least three points and a learned improvement over the best fixed baseline whose paired-unit bootstrap 95% interval is strictly above zero.

If the frozen learned gate still fails, the next admissible calibration-only step is to train a small action controller on update-time features using leave-one-session-out calibration. Do not inspect development or test to choose it.

## Claim boundary

This remains a deterministic counterfactual sequence of natural ZsRE edits. It tests structured-memory deletion, not native dense TTT fast weights or official ZsRE generation.
