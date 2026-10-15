# Frozen one-time development evaluation for natural scope V5

## Release evidence and question

The constrained V5 calibration passed all nine preregistered gates and its
independent audit passed all eight integrity checks.  The frozen residual gate
checkpoint SHA256 is
`3a0ba5a83ece4850dd3824fc2277dc2e0dfdb04902e361dbff764e1b630abadc`.
This protocol evaluates reader transfer once on the previously sealed,
subject-disjoint natural development split.  It does not train or select any
parameter and does not load ZsRE test or an external benchmark.

## Frozen evaluation

Replay the exact oracle lifecycle for all 64 development sessions and five
registered checkpoints.  Use the same Qwen3-4B checkpoint, contextual
transforms, cosine key ranker, 12 natural candidates, scale-8 value residual,
fresh KV and fixed-correct-length final generation as calibration.

Evaluate four reader arms on every exact and rephrased active query:

1. `v5_auto`: frozen cosine top plus the frozen V5 residual gate at threshold
   0.30;
2. `v1_auto`: the original decomposed gate at its frozen threshold 0.40;
3. `oracle_oracle`: direct active-record selection;
4. `zero`: no memory residual.

At every session checkpoint also construct the same two deterministic,
in-fold, subject/target-disjoint no-match controls used by V2--V5.  Score their
V5 and V1 defer decisions without injecting a value.  Candidate construction,
donor construction and feature extraction may not use development labels to
select a parameter.

Record candidate scores, answer NLL, selected chain/version, retrieve
probability, stale/wrong/defer outcomes, final generation, no-match decisions,
runtime and peak memory.  Use 20,000 deterministic subject-group bootstrap
draws for V5 versus V1 candidate accuracy.

## Frozen development gate

All checks must pass:

- V5 no-match specificity at least 90%;
- V5 candidate accuracy at least 90%;
- V5 gap to `oracle_oracle` no greater than seven points;
- V5 final candidate accuracy at least 88%;
- V5 final generation exact at least 70%;
- V5 rephrase candidate accuracy at least 85%;
- V5 candidate-accuracy gain over V1 has a subject-group bootstrap 95% lower
  bound above zero.

Passing freezes V5 as the infrastructure reader for the next forgetting-
controller experiment.  It does not establish a learned-forgetting advantage.
Failure returns to calibration-only reader work.  No test or external result is
released by either outcome.
