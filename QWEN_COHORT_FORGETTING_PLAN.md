# Qwen-grounded continuous cohort forgetting

## Frozen motivation

The controlled Qwen contextual-memory experiment established that a learned
state-dependent `retain/local/global` policy improves over every fixed policy,
but the explicit revoke/reset physics rule matched it.  Three independent
attempts to infer general relation identity from frozen representations
(CounterFact/MQuAKE, FewRel and DocRED) did not meet deletion-safety gates.
This experiment therefore stops semantic-pair tuning and tests the mechanism
suggested by the local/global forgetting analysis: independently retain old
history, the second-most-recent write and the most-recent write.

The experiment is frozen before generating its new corpus, extracting any new
Qwen representation or observing any new development/test outcome.

## Frozen representation dependency

Use the already audited Qwen3-4B contextual geometry without refitting it:

- geometry summary SHA256:
  `4fa6e0aba9fa5ab335ee9650b31547379a7525cd4c52db3b02d49aa49fbd272c`;
- training metadata SHA256:
  `01ecc4e9e5875485750b1f5a6121826eb254d9d865442a2c45ba79456ebb170c`;
- manifest SHA256:
  `d13cfa35397f65ed5231aace10d0338480482f08aba3f14f4e4f5b55c9347510`;
- transform SHA256:
  `524d213a82f158fddadab8973a28770a687138f528e269659430490affe74ae5`.

The fixed configuration is `context_last_r0.01`, rank 256, with
`unknown_raw=0.75`.  The Qwen3-4B checkpoint and tokenizer remain the final
1B-token artifacts used by the prior study.  No backbone, geometry or
readout parameter is fitted in this experiment.

## Fresh corpus and sealing

Generate a fresh event curriculum with seed `202610151`, 128 train groups, 32
development groups and 64 test groups.  Each group contains the complete
24-cell family/decision/twin matrix.  The generator must pass the existing
causal replay, entity-disjointness, template-disjointness and twin audits.

Train and development identifiers are extracted together and transformed with
the frozen contextual transform.  Test inputs, targets and identifiers are not
loaded or encoded until the unchanged development gate passes.  Previous
event-curriculum test outcomes cannot select a coefficient, threshold,
checkpoint or feature.

## Memory and action space

Maintain an additive list of write cohorts.  Before appending the current
write, partition existing cohorts into:

1. `history`: all cohorts older than the last two;
2. `last2`: the second-most-recent cohort, when present;
3. `last1`: the most-recent cohort, when present.

An action is a retention triple

\[
(r_h,r_2,r_1)\in\{0,0.5,1\}^3.
\]

Multiply every cohort in each group by its registered retention, then append
the current write at unit weight.  Candidate generation, queries and future
updates are identical across the 27 counterfactual branches.  Select the
minimum-NLL action; break ties by minimum removed memory mass, then by
retain-heavy lexicographic order `(1,1,1)` through `(0,0,0)`. This supervision
affects only the forgetting policy; the current write is always admitted.

## Controller and inference-safe features

Train a three-output sigmoid controller by MSE to the oracle retention triple.
Use a 64-unit and 32-unit tanh MLP, AdamW (`lr=0.01`, weight decay `1e-4`),
300 epochs and seeds `2611/2617/2621`.  Report a 27-way classifier with the
same hidden widths as a discrete-grid ablation.

Features may include memory/delta norms and alignment, cohort norm fractions,
cohort/delta alignments, memory length, block position, fact overlap,
duplication, and whether the already observed text explicitly references a
batch or reset.  Batch/entity/value IDs, family, split, answers, future blocks,
teacher events and oracle outcomes are forbidden inputs.  A reference-blind
ablation zeros reference existence, position and target-mass features.

## Comparisons

Evaluate identical sessions and queries under:

- retain all `(1,1,1)`;
- global clear `(0,0,0)`;
- global half `(0.5,0.5,0.5)`;
- window-2 `(0,0,1)`;
- window-3 `(0,1,1)`;
- deterministic uniform random grid actions;
- reference-blind learned regression;
- 27-way grid classifier;
- learned continuous cohort regression;
- the explicit cohort physics rule;
- exact referenced-batch deletion, as a stronger symbolic control;
- counterfactual 27-grid oracle.

Report query accuracy/NLL, current/historical/anchor accuracy, per-family
accuracy, retention MAE, exact grid-action agreement, retained memory mass,
latency and peak GPU memory.  Bootstrap paired gains over source groups with
20,000 draws.  Report all three projection seeds separately and their macro.

## Development gate and one-time test

The learned continuous controller passes only if all hold:

1. at least five accuracy points over the strongest fixed cohort policy;
2. paired group-bootstrap accuracy-gain lower bound above two points;
3. NLL-gain lower bound above zero;
4. every projection seed beats the strongest fixed policy;
5. current, historical and anchor accuracy each at least 85%;
6. minimum family accuracy at least 80%;
7. retention MAE at most 0.15;
8. at least 10% of oracle labels use an interior coefficient;
9. every corpus, feature, checkpoint, row and aggregate audit passes.

A pass releases one test extraction and one evaluation with frozen controllers.
Failure stops the cohort-continuous branch and leaves test unencoded.  Test
success requires a five-point gain with a positive group-bootstrap lower bound
and no negative projection seed.  The experiment does not claim native dense
TTT writer utility; it tests forgetting on the previously validated readable
Qwen-grounded additive memory.
