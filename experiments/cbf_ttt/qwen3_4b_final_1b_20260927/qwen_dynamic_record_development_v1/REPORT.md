# Dynamic record-level selective forgetting: development result

## Status

- Formal 512/64 train-development run completed with exit code 0.
- Three projection seeds and all ten registered policies completed.
- Independent audit passed all 13 integrity checks.
- The preregistered scientific gate failed.
- Matched test and external lifecycle OOD were never generated or scored.

## Main result

The shared record-equivariant, physically constrained controller reached
98.1481% development accuracy. It solved the fixed-position failure: the old
cohort controller reached only 72.5405%, cohort physics reached 81.4815%, and
retain-all reached 90.1042%. Candidate-selection recall was 100%, the minimum
projection-seed accuracy was 98.0903%, and no hard constraint was violated.

This did not establish a benefit from learning. Exact referenced-record
deletion and the visible dynamic lifecycle rule also reached 98.1481%, exactly
matching the accuracy-first oracle. The learned controller's paired accuracy
gain over the strongest non-oracle comparator was 0.0000 points with 95% CI
`[0.0000, 0.0000]`. Its NLL difference was -0.000756 with 95% CI
`[-0.002150, 0.000574]`.

| Policy | Accuracy | Minimum seed | NLL | Historical |
|---|---:|---:|---:|---:|
| retain all | 90.1042% | 90.1042% | 2.7801 | 84.3750% |
| old fixed-cohort controller | 72.5405% | 63.1944% | 2.8536 | 56.6840% |
| cohort physics | 81.4815% | 81.4236% | 2.5720 | 71.0069% |
| exact referenced-record deletion | **98.1481%** | 98.0903% | **2.4319** | **94.4444%** |
| unconstrained set controller | 98.0903% | 97.9167% | 2.4330 | 94.2708% |
| constrained set controller | **98.1481%** | 98.0903% | 2.4326 | **94.4444%** |
| accuracy-first oracle | **98.1481%** | 98.0903% | 2.4269 | **94.4444%** |

## Failure localization

The primary and oracle both reached 100% on retain, recent revoke, distant
revoke, exact duplicate, overwritten no-op and mixed two-event families. Both
reached only 86.8313% on reset, below the registered 90% family floor.

All reset errors were historical queries whose causal answer was `unknown`.
Current and anchor accuracy remained 100%. The failures occur at boundaries
where a reset leaves no active historical value for the queried key; the
`component_top1` reader is weaker at absence detection. Because the oracle is
subject to the same ceiling, changing the forgetting action cannot fix this
error.

The controller reproduced 98.55% of endpoint oracle actions but only 7.54% of
the 48 interior decisions across three seeds. This action mismatch did not
change query accuracy, showing that most interior labels resolve small NLL or
removed-mass ties after utility has saturated. They are not evidence of a
useful learned decision advantage.

## Registered gate

Passed checks:

- overall accuracy, minimum seed and oracle-gap bounds;
- current, anchor and historical accuracy bounds;
- candidate selection recall;
- endpoint action accuracy;
- zero hard-constraint violations.

Failed checks:

- every-family accuracy, due only to reset absence detection;
- three-point gain and positive paired confidence interval over the strongest
  non-oracle comparator;
- positive NLL-gain confidence interval;
- interior action accuracy.

The failed gate stops this learned dynamic-record branch. Opening the matched
test after this result would violate the frozen protocol.

## Resources and provenance

- Two RTX 5090 GPUs; critical path 391.24 seconds.
- Peak allocated/reserved memory 7.912/8.379 GiB.
- 30 raw development policy files, combined checkpoints, source manifests,
  logs, runner terminal files and the independent audit are included here.
- Producer commit on the experimental server: `79185504622108d77b361c31d83958c896f596ba`.
- Summary SHA256: `dbfe32076e143aedd438ad63874f6adda2b4bc2f0f48eadae5bfc0998f39b204`.
- Audit checkpoint SHA256:
  `000f09397dfa5037ad5fc482d1cf42296523f52c1f0094b655bb7bfd4e16c53a`.

## Supported conclusion and next experiment

This experiment supports dynamic, individually addressable lifecycle
transitions over fixed positional decay: exact record deletion improves by
25.61 points over the old fixed-cohort controller and matches the controlled
oracle. It does not support learning the explicit lifecycle decision, because
the deterministic record rule already saturates accuracy.

The next useful experiment should treat the addressable record ledger as the
method and validate it through the actual frozen Qwen LM head. It must first
pass correct/wrong/zero/random content controls, compare exact record
transitions against retain-all, windows and fixed cohorts on fresh data, and
separate memory absence from answer-token retrieval. Native dense
`ttt_proj`/`ttt_conv` content binding remains an independently established
negative result and cannot be claimed by this branch.
