# V19 Independent Locality-Constrained Confirmation

## Status and purpose

This protocol is frozen before any V19 model scoring. V18 did not pass its
original unconstrained headroom gate because its oracle locality accuracy was
91.875%. A separately audited, explicitly post-hoc V18-LC diagnostic found
that the same frozen scores admit a 90.625% joint-success oracle at 97.8125%
locality under a 3% aggregate locality-error budget. V18-LC did not release
controller training. V19 is the independent confirmation required to decide
whether that post-hoc observation replicates.

## Registered hypothesis

On a new controlled-interference corpus, selective deletion under a frozen 3%
aggregate locality-error budget has positive and practically meaningful
causal headroom over an outer-fold-selected fixed memory policy.

## Frozen data and access order

- 320 episodes, 80 subjects, 20 template groups, and five outer folds.
- Each subject contributes the four registered mechanisms and all four session
  lengths: 8, 16, 32, and 64.
- Entities, opaque values, random seed, fact templates, and query templates are
  disjoint from the V18 pilot and V18 formal corpus.
- The final current-update session is never deletable.
- Available arms remain the five fixed policies, every causal single-session
  deletion, and every causal whole-key deletion.
- Corpus construction is released only by the passing independent V18-LC
  audit. The V19 corpus and its independent audit must be complete before model
  scoring begins.
- No external benchmark, future query outcome, or future answer is available
  to a deployable policy. Outcome labels are used only to measure oracle
  headroom.

## Registered comparisons

The fixed comparator is selected independently in each outer fold using only
the other four folds. The primary oracle is selected jointly over all 320 held
out episodes by exact dynamic programming with at most
`floor(0.03 * 320) = 9` locality failures. Its ranking is frozen as:

1. total joint success;
2. total update accuracy;
3. total retained-key accuracy;
4. total locality accuracy;
5. total generation joint success;
6. lower joint answer NLL;
7. fewer locality failures.

For diagnosis only, the same exact solver reports budgets 0%, 1%, 3%, 5%, and
10%. The 3% result is the sole primary endpoint. An unconstrained oracle is
reported as an upper bound and cannot determine progression.

## Frozen success gate

V19 confirms locality-safe causal headroom only if all checks pass:

1. subject-macro joint-success gain is at least 10 percentage points;
2. its subject-cluster bootstrap 95% lower bound is above zero;
3. template-macro joint-success gain is at least 10 percentage points;
4. its template-cluster bootstrap 95% lower bound is above zero;
5. constrained-oracle update accuracy is at least 90%;
6. constrained-oracle retained-key accuracy is at least 90%;
7. constrained-oracle locality accuracy is at least 97%;
8. every outer fold has non-negative joint-success gain;
9. all corpus, shard, model, tokenizer, coverage, and recomputation audits pass.

Bootstrap inference uses 20,000 deterministic cluster resamples. No threshold,
budget, ranking rule, or endpoint is changed after scoring.

## Stop/go rule

- Pass: release construction of causal write-time forgetting examples and a
  nested out-of-fold controller trained under the registered locality budget.
- Fail: stop the synthetic selective-forgetting main claim, report the V18-LC
  result as exploratory, and do not train a controller on V19 oracle labels.

Passing V19 establishes headroom, not a deployable learned policy and not a
real-benchmark result. Those require later model-level controller evaluation on
fresh held-out episodes and ZsRE/LongMemEval-S/LoCoMo/RULER tasks.
