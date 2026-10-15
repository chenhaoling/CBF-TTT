# On-policy sequential natural forgetting V3 preregistration

## Question and release condition

This experiment asks whether the frozen low-dimensional factorized controller
improves a session when each decision changes the memory seen by every later
decision.  The formal utility labels and Factor Gate V2 OOF scores are
one-intervention evaluations reconstructed from a retain-all prefix; they are
not evidence for an on-policy sequential benefit.

Natural development remains closed until Factor Gate V2 has a successful
terminal status, all registered calibration checks pass, and its independent
audit passes.  Before loading development, write a release manifest containing
the SHA256 hashes of this plan, the V2 summary, audit and checkpoint, the
natural-corpus manifest, contextual transforms, scope artifact and V5 reader.
The runner must refuse a failed or unaudited V2 artifact.  No V2 parameter,
threshold, feature, action or gate may be changed after development is loaded.

The 64-session natural development split has already been used once to verify
the frozen V5 infrastructure reader, but never to train or select the
forgetting controller.  This stage is therefore controller development rather
than final confirmation.  ZsRE test and all external benchmarks remain closed.

## Frozen controller

Use the V2 `engineered` ensemble with seeds 9401, 9402 and 9403.  At every
event, compute the same 13 inference-visible features from that policy's
current memory and the incoming prompt.  Exclude all `reader_*` coordinates.
Each seed produces benefit and risk probabilities using its all-calibration
checkpoint and normalization.  The primary ensemble averages probabilities
across seeds and applies the frozen V2 median thresholds:

```
delete scope top iff mean(P(benefit)) >= tau_b
                     and mean(P(risk)) <= tau_r
```

Otherwise retain the state.  Apply the decision immediately before appending
the incoming record.  Exact deletion means removing the current frozen-scorer
top-1 record; there is no global reset, delta-update gate, alpha interpolation,
oracle relation ID or access to future events.  Then continue from the
policy-induced state.  Also evaluate each of the three seed members with the
same thresholds as a robustness analysis; the ensemble is the sole primary.

## Registered arms

Replay every development session in its fixed event order with independent
state for:

1. `retain_all`: append every record and never delete;
2. `always_scope_top1`: delete the inferred top record before every non-empty
   write, then append;
3. `source_action_rule`: delete top only for source events marked `local`;
   this privileged non-deployable control cannot be the primary comparator;
4. `factor_ensemble`: the frozen three-seed factorized policy;
5. `factor_seed_9401`, `factor_seed_9402`, `factor_seed_9403`;
6. `benefit_only_ensemble`: omit the risk condition, using the same benefit
   models and threshold;
7. `no_schedule_ensemble`: use the separately trained registered V2 ablation.

All policies use the frozen V5 reader after their own state transition.  Score
every exact and rephrased active query at each registered session checkpoint.
At the final checkpoint, additionally run fixed-correct-length generation for
the primary ensemble, retain-all and always-scope arms.  Cache identical Qwen
outcomes without sharing memory state or decisions between arms.

## Outcomes and safety accounting

The primary outcome is micro candidate accuracy over all exact and rephrased
queries.  Report mean answer NLL, final candidate accuracy, final generation
token exact match, exact/rephrase accuracy, stale reads, wrong-chain reads,
defers, mean/maximum memory size, delete rate and action counts.

For every learned-policy query, record a paired retain-all outcome.  An added
error is a query that retain-all answers correctly and that policy answers
incorrectly.  A recovered error is the reverse.  At a checkpoint, a collateral
added error is an added error whose chain differs from the most recently
written chain.  Report total and collateral added errors rather than hiding
them inside mean accuracy.  The source-action arm supplies the registered
safety budget.

Use 20,000 deterministic session bootstrap draws.  Resample whole sessions,
never individual queries, and compute paired accuracy gains for the primary
ensemble versus retain-all and always-scope.  Use seeds 202610251 and
202610252 respectively.  Report intervals for exact, rephrase and final
subsets as secondary analyses without additional release decisions.

## Development gate

Sequential development passes only if every condition holds:

- primary candidate accuracy exceeds retain-all by at least 0.75 percentage
  points and its paired session-bootstrap 95% lower bound is above zero;
- primary candidate accuracy exceeds always-scope by at least 0.25 percentage
  points and its paired 95% lower bound is above zero;
- primary collateral added errors do not exceed the source-action rule;
- the risk gate produces fewer collateral added errors than benefit-only and
  loses no more than 0.5 accuracy points to benefit-only;
- final generation exact match is no lower than retain-all by more than 0.5
  percentage points;
- all three seed-member accuracies exceed retain-all;
- the primary delete rate is between 5% and 60%;
- an independent audit reconstructs all state transitions, features,
  probabilities, decisions, paired outcomes, aggregates, bootstraps, hashes
  and held-out seals.

Passing freezes the on-policy method and authorizes construction of a new
subject-disjoint confirmatory split plus external transfer.  Failure keeps
ZsRE test and external benchmarks sealed.  Failure diagnosis may use only the
already opened development traces and must be labeled exploratory; it cannot
change this V3 result or retroactively release confirmation.

## Interpretation limits

A pass supports learned selective forgetting in the frozen Qwen-derived
session-memory substrate.  It does not establish that native dense
`ttt_proj`/`ttt_conv` fast weights implement exact record deletion, and it does
not establish final benchmark generalization until a fresh confirmatory split
and external transfer pass their own preregistered evaluations.
