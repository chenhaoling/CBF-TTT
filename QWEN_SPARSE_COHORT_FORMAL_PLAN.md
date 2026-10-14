# Qwen sparse-reader cohort forgetting: formal confirmation

## Conditional release

This protocol is frozen before generating its corpus or training any controller.
It may run only if the independent audit of
`qwen_sparse_reader_diagnostic_v1` passes and selects `component_top1` under
the unchanged registered gate.  A failed or different diagnostic selection
blocks this experiment.

The reader diagnostic is model-selection evidence only.  This confirmation
uses a fresh corpus and a new development/test split.  No outcome from its test
split may select a feature, reader, coefficient, checkpoint or threshold.

## Hypothesis

Winner-take-all causal key routing removes unrelated rank-one interference but
does not decide which old write cohorts remain active.  Under that readable
memory, a state-dependent history/last-2/last-1 retention controller should
outperform both fixed retention and the explicit symbolic cohort rule.  The
key comparison is learned control versus physics, because the dense-reader
study already established a large gain over fixed policies while learned and
physics were tied.

## Frozen data and representation

- fresh event-curriculum seed: `202610152`;
- train/development/test source groups: `256/64/128`;
- 24 family/decision/twin episodes per group;
- unchanged causal replay, entity-disjointness, template-disjointness and twin
  audits;
- final 1B-token Qwen3-4B checkpoint;
- frozen `context_last_r0.01`, rank-256 transform and `unknown_raw=0.75`;
- projection seeds `2301/2302/2303`;
- extract train and development identifiers together with the registered
  batching path; do not load or encode test identifiers before the gate.

The reader is exactly `component_top1`: among active components, select the
largest entity-key cosine, breaking ties by newer write and then source fact
order, and return its weighted value contribution.  The reader receives no
answer, target role, event family, future text or oracle action.

## Action and supervision

Before every decision, partition old writes into `history`, `last2` and
`last1`.  Supervise the 27 retention triples in `{0, 0.5, 1}^3` by minimum
current-boundary query NLL.  Break NLL ties by minimum removed memory mass and
then retain-heavy grid order.  Always append the current write at unit weight.

Train the unchanged three-output sigmoid MLP with hidden widths 64/32, tanh,
AdamW, learning rate 0.01, weight decay `1e-4`, 300 epochs and controller seeds
`2611/2617/2621`.  Retain the 27-class controller and reference-blind regressor
as registered ablations.  Feature names and normalization are unchanged from
the dense-reader cohort study; no answer, family, split, entity/value identity,
future block or teacher state is a feature.

## Comparisons

Evaluate identical sessions under:

- retain all, global clear, global half, window-2 and window-3;
- deterministic random grid actions;
- reference-blind regression;
- 27-class grid classification;
- continuous learned cohort retention;
- explicit cohort physics;
- exact referenced-batch deletion;
- the causal 27-grid oracle.

Report query accuracy/NLL, current/historical/anchor accuracy, per-family
accuracy, retention MAE, grid agreement, retained mass, latency and peak GPU
memory.  Report all projection seeds and a macro average.  Compute paired
source-group bootstrap intervals with 20,000 draws.

## Development gate

Release one test extraction and one evaluation only if every check holds:

1. learned mean query accuracy at least 97%;
2. learned accuracy at least three points above cohort physics;
3. paired accuracy-gain 95% CI lower bound above 1.5 points;
4. paired NLL-gain 95% CI lower bound above zero;
5. learned beats physics on every projection seed;
6. learned is within two accuracy points of the oracle;
7. current and anchor accuracy at least 98%;
8. historical accuracy at least 93%;
9. minimum family accuracy at least 90%;
10. at least 10% of oracle training labels contain an interior coefficient;
11. every corpus, embedding, checkpoint, row, aggregate and sealing audit
    passes.

Retention MAE and exact grid agreement are reported diagnostics rather than
release conditions because multiple actions can have indistinguishable query
utility under winner-take-all reading.

## One-time test success

Test success requires learned accuracy at least 97%, a gain of at least three
points over physics, a paired accuracy-gain lower bound above 1.5 points, a
positive NLL-gain lower bound, no non-positive projection seed and an oracle
gap of at most two points.  Failure stops this controller architecture without
changing the reader, gates or test outcome.

Passing this controlled confirmation authorizes integration of the same
cohort retention interface into segmented native `ttt_proj`/`ttt_conv` fast
parameter deltas and evaluation on RULER/QA, LongMemEval-S and LoCoMo.  It does
not by itself claim utility of the existing native dense writer.

