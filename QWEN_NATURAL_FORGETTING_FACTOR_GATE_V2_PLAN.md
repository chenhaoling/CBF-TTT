# Low-dimensional factorized forgetting gate V2

## Failure evidence and repair hypothesis

Barrier-Q V1 passed its independent audit but stopped before development.  Its
2564-dimensional per-action value/risk model reached 80.9476% mean candidate
accuracy and added 120 unrelated errors; retain-all was 80.8691% and the
strongest fixed action was 81.5647%.  All five inner searches were infeasible.
The gain-only ablation added more than 1,200 unrelated errors, while direct
classification reached about 81.66% but added 418--492 errors.

The formal labels show that the safe two-action oracle over `retain_all` and
`scope_top1_a0` reaches 83.1976% with zero added unrelated errors, retaining
83.9% of the full action oracle headroom.  A calibration-only OOF diagnostic
then found that 13 engineered state variables with a 16-unit hidden layer can
reach 82.56% with two added errors, whereas the raw 2564-dimensional feature
overfits.  V2 tests this dimensionality and factorization repair under a new
nested grouped protocol; the diagnostic result is not treated as confirmatory
evidence.

Natural development, test and external benchmarks remain sealed.

## Targets and actions

Restrict deployment actions to `retain_all` and exact deletion of the frozen
scope top, `scope_top1_a0`.

For every formal calibration state define:

- `benefit=1` when scope-top deletion produces more future candidate-correct
  queries than retain-all, or ties candidate correctness and has lower total
  answer NLL;
- `risk=1` when scope-top deletion creates at least one additional candidate
  error on an unrelated chain.

The factorized policy deletes only when predicted benefit is at least the
selected threshold and predicted risk is at most the selected threshold.
Retain is the fallback.  Report the benefit-only ablation, which ignores the
risk gate, and the exact safe two-action oracle.

## Feature sets

The primary `engineered` representation uses exactly the 13 inference-visible
features already sealed in the V1 artifact:

`has_memory`, `memory_size`, `event_position`, `top_recency`, `top_cosine`,
`cosine_margin`, `cosine_mean`, `cosine_std`, `cosine_entropy`,
`v5_probability`, `lexical_jaccard`, `lexical_containment`, and `length_ratio`.

It excludes every 2564-dimensional `reader_*` coordinate.  Normalization is
fit inside each training fold.

Train a registered `no_schedule` ablation after applying the same nested
selection, excluding `memory_size`, `event_position` and `top_recency`.  This
tests whether the controller depends on the constructed event schedule.  The
ablation may not replace the primary model or change the release decision.

## Nested grouped training

Reuse the V1 deterministic five outer session folds and four inner folds.
For each outer fold, train benefit and risk heads only on outer-training
sessions, create inner OOF probabilities, select configuration and thresholds,
then refit on all outer-training rows and predict the held-out sessions.  Use
three seeds `{9401, 9402, 9403}`.

Registered model grid:

- hidden width `{0, 16, 32}`;
- learning rate `{0.001, 0.005}`;
- weight decay `{0, 0.01}`;
- 300 full-batch AdamW epochs;
- inverse-frequency positive BCE weight clipped to `[1, 20]`, computed on the
  training fold separately for benefit and risk.

Registered threshold grid:

- benefit threshold `{0.3, 0.5, 0.7, 0.8, 0.9}`;
- risk threshold `{0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5}`.

For each feature set, keep inner configurations whose three-seed mean added
unrelated errors is no greater than the source-action control on the same rows
and whose accuracy exceeds retain-all by at least 0.5 points.  Select maximum
accuracy, then minimum errors, smaller hidden width, larger benefit threshold
and smaller risk threshold.  If no configuration is feasible, select minimum
errors then maximum accuracy and mark the fold infeasible.

Train final all-calibration checkpoints using the median outer selections.
Save every OOF probability and action, target, fold, search table and hash.

## Comparisons and calibration gate

Compare the primary factor gate with retain-all, always scope-top deletion,
V1 Barrier-Q, V1 direct classification, benefit-only V2, rate-matched random,
the non-deployable source-action rule, the safe two-action oracle and the full
safe oracle.  Use 20,000 session bootstrap draws for paired gains.

Authorize one-time online natural-development rollout only if all checks pass:

- three-seed mean accuracy exceeds retain-all by at least 0.75 points and its
  paired 95% bootstrap lower bound is above zero;
- mean accuracy exceeds always scope-top deletion by at least 0.25 points and
  its paired 95% lower bound is above zero;
- the policy captures at least 30% of full safe-oracle headroom;
- mean added unrelated errors are no greater than the source-action control;
- the risk gate adds fewer unrelated errors than benefit-only and loses no
  more than 0.5 accuracy points;
- primary accuracy exceeds V1 Barrier-Q and V1 direct classification means;
- at least four of five primary outer folds are feasible;
- all three seeds improve over retain-all and always scope-top deletion;
- the independent audit verifies feature exclusion, target reconstruction,
  nested group isolation, selections, OOF outputs, bootstraps, checkpoints and
  held-out seals.

Passing authorizes a separately preregistered *online sequential* development
rollout.  Failure keeps development sealed and triggers calibration-only
feature or on-policy-state diagnosis.
