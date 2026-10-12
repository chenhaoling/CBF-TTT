# Frozen ZsRE forgetting controller: formal development result

## Purpose

This run evaluates the calibration-frozen forgetting controller on the subject-disjoint ZsRE development split. The controller checkpoint, action threshold (`0.4`), feature extractor, readout seeds, comparison policy, bootstrap seeds, and six release checks were fixed before development scoring. The test split was available to the runner only after every development check passed.

## Result

The controller improved over the frozen cosine-local policy on development:

| Metric | Controller | Cosine local | Paired gain | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| Query accuracy | 75.11% | 71.46% | +3.51 pp | [2.27, 4.82] pp |
| Action accuracy | 91.41% | 84.38% | +7.03 pp | [4.56, 9.77] pp |

The gains passed both paired statistical checks. The absolute release checks did not all pass:

| Frozen development check | Observed | Required | Passed |
|---|---:|---:|:---:|
| Overall action accuracy | 91.41% | 96% | no |
| Primary initial action | 99.61% | 98% | yes |
| Hard-retain action | 87.89% | 95% | no |
| Primary-update action | 86.72% | 92% | no |

Current-exact, current-rephrase, and historical query accuracies were 94.74%, 48.74%, and 98.05%. The main remaining error is therefore the boundary between retaining a related memory and replacing a contradicted memory, with rephrase readout as a second bottleneck.

The six-check development gate failed. The test set was neither loaded nor scored. The independent audit passed and verified the absence of test result files.

## Resources and provenance

- Code commit: `d7757ae`
- Protocol: `zsre_counterfactual_forgetting_controller_formal_v1`
- Development units: 256 subject groups
- Runtime: 41.69 seconds
- Peak CUDA memory: 8.233 GiB allocated, 11.297 GiB reserved
- Remote result root: `/home/ctj/cbf_ttt_zsre_counterfactual_forgetting_formal_v1`

The repository archives the compact `summary.json`, `training.json`, `training_manifest.json`, `execution_audit.json`, and terminal status. Raw per-session records remain on the server.

## Interpretation and next experiment

This result supports the narrow claim that a learned, state-dependent forgetting rule generalizes better than a cosine deletion heuristic. It does not meet the reliability threshold needed to release the held-out test, and it remains a constructed candidate-memory experiment rather than native dense TTT fast-weight evidence.

The next experiment will use the now-development-only records for error attribution and controller V2 selection. It will first test whether a development-specific threshold can reach the absolute floors without retraining. If it cannot, V2 will separate conflict detection from retention confidence and use relation-balanced grouped training. The write gate remains fixed at one; only old session-memory retention is learned. After freezing V2 on calibration plus development, the untouched test will be scored once. Native TTT fast-weight integration follows only after that release gate passes.
