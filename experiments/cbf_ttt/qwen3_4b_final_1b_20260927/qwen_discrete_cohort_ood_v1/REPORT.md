# Frozen discrete cohort controller: audited lifecycle OOD result

## Conclusion

The frozen 27-class cohort controller did **not** generalize to the four
preregistered lifecycle shifts.  Its macro accuracy was **69.2612%**, compared
with **73.1289%** for explicit cohort physics and **90.9578%** for the
registered NLL grid oracle.  The controller-minus-physics accuracy difference
was **-3.1901 points**, with source-group bootstrap 95% CI
**[-4.2643, -2.0797]**.  Its NLL was also worse by 0.23732, CI
`[-0.27543,-0.19930]` when expressed as the registered physics-minus-controller
gain.  None of the eleven scientific release checks passed.

All 13 independent integrity checks passed.  The negative result is therefore
an audited model failure rather than a broken run, hash mismatch, data leak or
aggregate error.  Under the frozen protocol it terminates the exact fixed
cohort branch and blocks the planned residual-readout bridge.

## Protocol

- Fresh corpus seed: `202610154`.
- Four suites: long session, distant revocation, mixed lifecycle and
  duplicate/no-op.
- 64 source groups per suite and two one-anchor twins per group.
- Three frozen projection controllers: seeds 2301, 2302 and 2303.
- No controller retraining, calibration or threshold selection.
- All 13 confirmation policies were replayed on each suite and seed, producing
  the registered 156 raw result files.
- Source-group bootstrap used 20,000 paired draws.

## Macro result

| Policy | Accuracy | NLL | Historical | Anchor | Minimum seed |
|---|---:|---:|---:|---:|---:|
| Frozen 27-class controller | 69.2612% | 3.35431 | 44.8785% | 62.9051% | 67.1875% |
| Independently snapped continuous | 63.3632% | 3.33728 | 43.2292% | 46.8605% | 61.1400% |
| Continuous controller | 67.9012% | 3.22464 | 54.6875% | 49.0162% | 61.1690% |
| Cohort physics | 73.1289% | 3.07683 | 69.3866% | 50.0000% | 72.8588% |
| Retain all | 87.0853% | 3.36705 | 61.2558% | 100.0000% | 87.0660% |
| Exact referenced-record deletion | 89.7955% | 2.92170 | 69.3866% | 100.0000% | 89.5255% |
| Registered NLL grid oracle | 90.9578% | 2.61378 | 90.0608% | 82.8125% | 90.8420% |

Current-write accuracy remained 100% for every listed policy.  The learned
controller failure comes from deleting or retaining the wrong session records,
which damages historical and anchor queries.

## Suite localization

| Suite | Controller | Physics | Exact reference | Retain all | NLL oracle |
|---|---:|---:|---:|---:|---:|
| Long session | 70.6597% | 79.9479% | 91.0590% | 88.8889% | 80.7147% |
| Distant revoke | 77.7778% | 66.6667% | 100.0000% | 100.0000% | 100.0000% |
| Mixed lifecycle | 66.5799% | 75.6076% | 75.6076% | 66.6667% | 96.3542% |
| Duplicate/no-op | 61.3281% | 66.8837% | 91.8837% | 91.8837% | 91.8837% |

The exact-reference versus cohort-physics gap is 16.67 points in aggregate and
33.33 points on distant revocation.  This directly shows that a fixed
history/last-2/last-1 action cannot express deletion of an arbitrary old
record.  Duplicate/no-op additionally requires idempotence and record identity,
which the fixed cohort representation does not preserve.

The registered oracle minimizes answer NLL and does not lexicographically
maximize query accuracy.  This explains why exact-reference deletion has
higher accuracy than the oracle on long sessions while the oracle still has
lower NLL.  A successor must preregister an accuracy-first oracle so its upper
bound matches the primary endpoint.

## Action diagnostic

The frozen controller matched the oracle action on only **5.165%** of 6,912
decisions.  For endpoint oracle actions, action accuracy was 5.505% and query
accuracy 71.301%; for the 29.04% of decisions containing an interior
coefficient, action accuracy was 4.335% and query accuracy 64.275%.  Retention
MAE was 0.487/0.549 for endpoint/interior decisions.  This is a large
distribution and representation failure, not a marginal threshold miss.

## Registered gate

All checks failed: 97% macro accuracy, 95% per-suite accuracy, 95% minimum seed,
three-point physics gain, positive confidence intervals, nonnegative gain in
every suite, two-point oracle gap, current/anchor and historical floors, and
endpoint/interior action floors.  The recorded next stage is
`stop_exact_state_controller_branch`.

## Integrity and resources

- Runtime: 1,009.52 seconds.
- Embedding extraction: 24.28 seconds.
- Peak allocated/reserved memory: 7.9116/8.5781 GiB on RTX 5090.
- Runner and conditional chain exit codes: zero.
- Producer terminal: `completed_qwen_discrete_cohort_ood`.
- Audit terminal: `completed_qwen_discrete_cohort_ood_audit`.
- All source reconstruction, raw hashes, aggregates, bootstrap values,
  dependency bindings, action diagnostics and terminal transitions passed
  independent verification.

## Scientific consequence

The previous 98.68% fresh IID confirmation remains valid, but its claim is now
narrow: exact cohort-state learning works under the registered session
distribution and fails under variable lifecycle structure.  The next method
must replace fixed positional cohorts with dynamically selected, individually
addressable records and enforce no-op, locality, idempotence and reset
constraints.  This OOD corpus is now a disclosed diagnostic set and cannot be
used as the successor's held-out evidence.
