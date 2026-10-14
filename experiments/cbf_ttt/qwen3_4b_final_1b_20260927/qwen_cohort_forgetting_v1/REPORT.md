# Qwen continuous cohort forgetting: formal development report

## Question

Can a forgetting-only controller independently retain old history, the
second-most-recent write and the most-recent write in the readable contextual
Qwen3-4B fast memory?  The current write is always admitted.  Each retention
coefficient is supervised on the fixed grid `{0, 0.5, 1}`.

The protocol and gates were frozen in `QWEN_COHORT_FORGETTING_PLAN.md` before
the fresh corpus was generated.  The final 1B-token Qwen3-4B checkpoint,
rank-256 `context_last_r0.01` geometry, reader and all gates remained frozen.

## Formal development result

The fresh corpus contains 128/32/64 train/development/test source groups and
24 episodes per group.  Three projection seeds were evaluated.

| Policy | Accuracy | NLL | Historical | Minimum family |
|---|---:|---:|---:|---:|
| Retain all | 66.41% | 1.1853 | 25.89% | 32.78% |
| Reference-blind controller | 60.87% | 1.2587 | 20.38% | 43.32% |
| 27-class grid controller | 88.04% | 0.4786 | 64.89% | 71.50% |
| Continuous cohort controller | **88.35%** | 0.4582 | 65.49% | 71.59% |
| Explicit cohort physics rule | 88.54% | **0.4296** | 66.36% | 71.76% |
| Exact referenced-batch deletion | 88.54% | **0.4296** | 66.36% | 71.76% |
| 27-grid oracle | 89.06% | 0.4356 | 67.90% | 71.82% |

The continuous controller improves over the strongest fixed policy by 21.94
accuracy points.  The paired 20,000-draw source-group bootstrap interval is
`[20.15, 23.80]` points.  Its NLL improvement is 0.7271, interval
`[0.6822, 0.7752]`.  Every projection seed is positive: learned accuracy is
88.997/87.912/88.151%, compared with 66.298/66.580/66.363% for retain-all.

The learned controller recovers 96.9% of the available accuracy improvement
between retain-all and the grid oracle.  Reference-blind control fails,
showing that visible revoke/reset and referenced-cohort state carry necessary
decision information.  Interior oracle coefficients occur in
31.58/32.71/34.80% of the three training-label sets.

## Registered gate

The experiment fails its complete development release gate despite the large
causal gain:

- accuracy gain, accuracy CI, NLL CI, every-seed and interior-label checks pass;
- minimum role accuracy fails because historical accuracy is 65.49%;
- minimum family accuracy is 71.59%, below 80%;
- continuous retention MAE is 0.1787, above 0.15.

The test split was not loaded, encoded or scored.  No gate, coefficient,
feature or checkpoint is changed after this outcome.

## Interpretation

The controller is close to the best action available to the registered
reader, while the oracle itself has only 67.90% historical accuracy.  The
absolute failure therefore localizes the dominant bottleneck to historical
readability of the dense accumulated rank-one memory.  The learned policy
does not beat the explicit physics rule: their accuracies are 88.35% and
88.54%.  This experiment supports state-dependent cohort forgetting relative
to fixed retention, but does not support a learned-policy advantage over
visible symbolic revoke/reset rules.

The next frozen diagnostic tests whether causal top-k key routing removes the
dense accumulation ceiling.  It is development-only and cannot retroactively
change this run's failed gate.

## Provenance

- server root: `/home/ctj/cbf_ttt_qwen_cohort_forgetting_20261015`;
- training code commit: `5ef9e7c1797e149473544c82c0750a02ae23d6e7`;
- runtime: 1,722.58 seconds;
- Qwen extraction: 11.65 seconds;
- peak CUDA allocated/reserved: 7.867/8.115 GiB;
- summary SHA256: `b8de681ae8ff62af0eb2a974724a2a6555d18c1b8c42bfafae39847e65591f67`;
- training SHA256: `33ad664dd628df8ee3b3c68d1dcb3eac6d1fd96a33d6a4a865e01c76ddcf8d47`;
- manifest SHA256: `8d3446d6959686e51b7ae1c5da4b1e3bc44d58ac2ee73fa0cba75fb4b24a732e`;
- controller SHA256: `bd96395c622f4747625e8dbfaf35da2dc5eb77b9c52342a92dad4a61c77c6d5b`;
- independent audit SHA256: `92949bdef771e25ad6509a3ee94e292ddacdcb37cdc116809fd7dc17abdd949f`.

All 13 independent source, checkpoint, row, aggregate, gate and test-sealing
checks pass.
