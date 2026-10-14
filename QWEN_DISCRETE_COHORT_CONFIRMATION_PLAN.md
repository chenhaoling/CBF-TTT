# Discrete physical-state cohort controller: fresh confirmation

## Conditional status

This protocol is frozen after observing projection seed 2301, and before
observing seeds 2302/2303 or generating this experiment's corpus.  Seed 2301
showed that the 27-class controller reached 98.86% while the continuous
sigmoid controller reached 83.30% under `component_top1`.  The continuous
controller's retention MAE was small, but a small nonzero coefficient keeps a
component in the top-1 reader's active set.  The hypothesis is therefore that
exact physical states, especially exact zero, are required.

This experiment may run only if the complete predecessor audit passes and all
of the following predecessor development conditions hold:

1. test remained sealed;
2. the grid classifier averages at least 98.5%;
3. every projection seed reaches at least 98%;
4. grid accuracy exceeds cohort physics by at least five points;
5. the grid classifier is within one point of the oracle.

Failure of any prerequisite cancels this experiment without relaxing it.

## Fresh release

- corpus seed `202610153`;
- 256/64/128 train/development/test source groups;
- 24 family/decision/twin episodes per group;
- final 1B-token Qwen3-4B checkpoint;
- audited rank-256 `context_last_r0.01` geometry;
- audited causal `component_top1` reader;
- projection seeds `2301/2302/2303`;
- controller seeds `2611/2617/2621`.

Extract train and development identifiers together.  Test identifiers and
targets remain unloaded and unencoded until every development check passes.

## Primary method

The primary controller is the fixed 64/32 tanh MLP with a 27-way softmax over
`{0, 0.5, 1}^3`.  Train with cross entropy, AdamW, learning rate 0.01, weight
decay `1e-4` and 300 epochs.  Apply the selected grid action exactly, so zero
removes the corresponding cohort from the reader's active set.

Use the same 20 inference-safe state features as the predecessor.  Entity,
value, batch and family identity, answers, future text, teacher state and
oracle utility are forbidden at inference.

## Comparisons and ablations

Report:

- all fixed policies and deterministic random grid actions;
- reference-blind controller;
- continuous sigmoid retention;
- continuous retention snapped independently to the nearest grid point;
- primary 27-class discrete controller;
- explicit cohort physics and exact referenced-batch deletion;
- causal 27-grid oracle.

The snapped arm separates regression error from the active-set discontinuity.
All arms share the reader, sessions, writes, queries and candidate sets.
Report accuracy/NLL, role and family accuracy, action agreement, retention MAE,
retained mass, latency and peak GPU memory.  Use 20,000 paired source-group
bootstrap draws and report every projection seed.

## Development gate

The primary grid controller releases one test extraction only if all hold:

1. mean query accuracy at least 98.5%;
2. at least five accuracy points above cohort physics;
3. paired gain 95% CI lower bound above 3.5 points;
4. paired NLL-gain lower bound above zero;
5. every projection seed beats physics by at least four points;
6. oracle accuracy gap at most one point;
7. current and anchor accuracy at least 99%;
8. historical accuracy at least 96%;
9. minimum family accuracy at least 95%;
10. exact grid-action agreement at least 88%;
11. at least 20 points above the reference-blind controller;
12. at least 10% of oracle training labels contain an interior coefficient;
13. every corpus, reader dependency, checkpoint, row, aggregate and sealing
    audit passes.

## One-time test success

Test success requires at least 98.5% accuracy, at least five points over
physics with a paired lower bound above 3.5 points, positive NLL-gain lower
bound, every-seed gain of at least four points, and at most one point to the
oracle.  The test cannot tune the architecture or gates.

A passing result supports a paper claim that learned discrete cohort-state
control improves over fixed retention and explicit revoke/reset physics in a
frozen Qwen-grounded readable fast memory.  Native `ttt_proj`/`ttt_conv`
integration and RULER/QA, LongMemEval-S and LoCoMo remain separate required
stages.

