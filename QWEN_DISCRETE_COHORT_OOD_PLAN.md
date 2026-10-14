# Frozen discrete cohort controller: external lifecycle stress

## Conditional status

This protocol is frozen before the fresh discrete confirmation corpus has been
generated or scored.  It may run only if
`audit_qwen_discrete_cohort_confirmation_v1` passes, its one-time test is
scored, and the registered test gate passes.  A failed prerequisite cancels
this study without changing its suites or thresholds.

The question is whether exact cohort-state control learned a portable lifecycle
rule, rather than the eight-block schedule of the training curriculum.  The
confirmation checkpoint is loaded without retraining, calibration or threshold
selection.  Its joint 27-class controller remains the primary method.

## Fresh external release

- corpus seed `202610154`;
- 64 source groups per suite, with identifiers, values and templates disjoint
  from every earlier corpus;
- final 1B-token Qwen3-4B checkpoint;
- frozen audited `context_last_r0.01` geometry;
- frozen audited `component_top1` reader;
- projection seeds `2301/2302/2303`;
- no use of the confirmation train, development or test rows except their
  frozen controller checkpoint and independent audit hashes.

Generate all suites only after the dependency check succeeds:

1. `long_session`: 16 blocks and four to six decision boundaries, with valid
   assignments extending beyond the training horizon;
2. `distant_revoke`: 12 blocks and explicit revocations at lags three through
   six, forcing selective changes to the history cohort;
3. `mixed_lifecycle`: 12 blocks containing both session resets and later valid
   batch revocations, with protected post-reset anchors;
4. `duplicate_noop`: 12 blocks with repeated identical values, overlapping
   distractor keys and explicit revocations whose semantic effect may be empty.

Every decision asks one current, one historical and one protected-anchor
query.  Semantic answers are computed by replaying the event stream before any
model call.  Future blocks, private events, answers and oracle actions cannot
enter controller features.  The lifecycle text remains explicit because the
method controls application/session memory with observable record-management
events; this experiment does not claim latent natural-language contradiction
discovery.

## Frozen comparisons

Replay, without refitting:

- retain-all, global clear/half and last-1/last-2 windows;
- deterministic random grid actions;
- the confirmation reference-blind controller;
- continuous retention and its independently snapped output;
- the primary joint 27-class controller;
- explicit cohort physics and exact referenced-batch deletion;
- the causal 27-grid oracle.

All policies see identical writes, reader states, questions and candidate
sets.  Report accuracy, NLL, current/historical/anchor accuracy, every suite and
event family, action agreement, endpoint/interior action accuracy, retention
MAE, retained mass, wall time and peak allocated/reserved GPU memory.  Use
20,000 source-group bootstrap draws.  Source groups, not queries or twins, are
the sampling unit.

## Registered success gate

The frozen joint controller passes external stress only if all hold:

1. macro query accuracy is at least 97%;
2. every suite accuracy is at least 95%;
3. minimum projection-seed accuracy is at least 95%;
4. macro accuracy exceeds physics by at least three points, with paired 95%
   confidence lower bound above one point;
5. macro NLL gain over physics has a 95% confidence lower bound above zero;
6. no suite has negative mean accuracy gain versus physics;
7. oracle accuracy gap is at most two points;
8. current and anchor accuracy are at least 98%;
9. historical accuracy is at least 93%;
10. endpoint action agreement is at least 97% and interior agreement at least
    65%;
11. every corpus, dependency, row, aggregate, bootstrap and checkpoint audit
    passes.

There is no development tuning or second attempt on these suites.  Failure is
reported by suite and ends this architecture branch; it cannot be repaired by
dropping a failed suite or changing the action grid after scoring.

## Claim boundary and next release

A pass supports distribution robustness of learned exact cohort-state control
under explicit but unseen memory lifecycle schedules.  It does not establish a
native dense `ttt_proj`/`ttt_conv` writer, implicit semantic conflict discovery
or official LongMemEval-S/LoCoMo answer generation.  Those remain separate
required stages.  A failure localizes the schedule or length shift that breaks
the controller and prevents public-benchmark release from this branch.
