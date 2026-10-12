# On-policy forgetting gate V3 plan

## Motivation

Constraint calibration improved V2 causal query/action from 46.79/78.26% to
48.64/84.29% and restored hard retain to 88.66%, but update remained 67.52% and
the development gate failed. The same thresholds reached 76.82% update on
oracle states. This 9.30-point gap is direct evidence of exposure bias: an
irreversible deletion changes later memory states, whereas every existing gate
was trained only on oracle histories.

## Method fixed before scoring

Keep each independently fitted V2 outer-fold candidate ranker frozen. For every
outer fold:

1. reserve one of the four outer-training CounterFact folds for causal threshold
   selection;
2. use the constraint-calibrated V2 policy to roll out on all observed ZsRE plus
   the other three CounterFact folds;
3. at each encountered nonempty memory state, label the old same-chain candidate
   when it is still present, otherwise label retain;
4. refit only the binary forgetting gate on these policy-induced states for 25
   epochs, weighting retain examples twice to reflect irreversible deletion
   cost;
5. replay the reserved inner fold causally over thresholds 0.05–0.95; choose a
   point satisfying hard-retain ≥85% and update ≥75% that maximizes overall
   action, with minimum total constraint deficit as fallback;
6. evaluate the untouched outer CounterFact fold once.

The ranker, Qwen features, current write gate (`g_t=1`), folds, thresholds, and
development checks remain unchanged. No outer-fold event, chain identity,
relation ID, or target string trains its gate. Save on-policy state counts,
fold-specific thresholds, checkpoints, and raw causal trajectories.

## Decision

Advance only if the original five development checks all pass. If candidate
top-1 remains above 75% but causal update still fails, the current explicit
candidate-memory formulation has reached its practical ceiling; stop further
CounterFact tuning and move the paper claim to risk-aware on-policy forgetting
with the limitation recorded. LongMemEval-S and LoCoMo remain unscored until a
development recipe passes.

