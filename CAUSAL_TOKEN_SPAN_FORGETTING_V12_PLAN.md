# Causal token-span forgetting V12 preregistration

## Motivation

V11 raised automatic local-update candidate top-1 from 94.65% to 97.20% and
passed its gate. V12 tests whether this offline localization improvement causes
better sequential memory, rather than merely improving a static ranking metric.
The current fact is always written (`g=1`); only session-memory retention is
controlled.

## Frozen components

- Freeze every V11 outer-fold token-span head, its selected 0.3 span threshold,
  and its `overlap8` relation score.
- Freeze the V5 relation classifiers and Qwen3-4B checkpoint.
- Keep the previously published structured-memory query readout unchanged so
  query-accuracy changes are caused by different memory contents.
- Keep the original five CounterFact subject-disjoint folds. CounterFact remains
  observed development-only.

## Retain/local gate

For each visible update and current memory, the frozen resolver returns the top
candidate plus relation compatibility, predicted-span overlap, score margin,
entropy, exact-span agreement, and candidate count. A three-seed linear gate is
trained to predict `local` versus `retain`. It receives no subject, relation,
chain, answer, target, age, dataset, or future-query identifier.

For each outer fold, the next fold is reserved for epoch and causal-threshold
selection. Gate heads are trained on the other three folds. Selection replays
the inner trajectories over thresholds 0.05--0.95 and prefers points satisfying
hard-retain at least 95% and primary-update at least 90%. Heads are then refit on
all four non-outer folds for the selected epoch; the threshold is unchanged.
The untouched outer fold is replayed once, so erroneous deletions change all
later states.

## Controls and metrics

Compare against the archived `fixed_retain`, `fixed_global`, `fixed_top`,
`cosine_local`, prior learned controller, and `oracle_local` trajectories.
Report candidate top-1, overall action accuracy, hard-retain and primary-update
accuracy, current exact/rephrase and historical query accuracy, stale reads,
mean memory size, runtime, and peak GPU memory. Paired bootstrap uses factual
subject groups.

## Gate

V12 passes only if:

- static outer-fold candidate top-1 is at least 96%;
- causal overall action accuracy is at least 95%;
- hard-retain accuracy is at least 95%;
- primary-update accuracy is at least 90%;
- query accuracy beats fixed retain with a positive paired-bootstrap 95% lower
  bound;
- query accuracy is within one point of the oracle-local policy.

A pass freezes the CounterFact causal recipe and authorizes external-domain
validation. A failure triggers a causal error audit; it does not reopen V8--V10
score fusion. LongMemEval-S and LoCoMo remain sealed until this gate passes.
