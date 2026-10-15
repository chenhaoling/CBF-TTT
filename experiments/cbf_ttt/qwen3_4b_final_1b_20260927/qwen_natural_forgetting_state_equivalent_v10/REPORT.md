# Qwen3-4B state-equivalent safe-delete V10

## Corrected physical target

V10 treats deletion as safe when it removes a stale version from the same lifecycle chain or replaces a duplicate record with identical source-fact identity and target value. This adds 94 duplicate-equivalent positives to the 678 lifecycle positives. Source hash, target, chain and event metadata construct labels only; the inference features are exactly V7's ten no-schedule and five prompt entity-anchor features.

## Registered result

The five session-held-out outer models were retrained from the corrected labels using V7's unchanged nested grid.

| Metric | Result | Gate |
| --- | ---: | ---: |
| Precision | 99.01% | ≥98% |
| Recall | 77.46% | ≥50% |
| True positives | 598 | — |
| False positives | 6 | ≤4 |
| Delete rate | 27.06% | 5–50% |

Fold precision was 100%, 100%, 100%, 96.46% and 97.78%. All five inner selections were feasible, but every selected model disabled the prompt entity barrier. Four false deletes appeared in fold 3 and two in fold 4. The absolute false-delete gate therefore failed and on-policy replay/test were not authorized.

The independent audit passed all ten checks with zero probability/action mismatch and zero session leakage. Runtime was 113.43 seconds. Development, source test and external benchmarks remained sealed.

## Interpretation and decision

The state-equivalence correction is physically justified and reclassifies duplicate replacement correctly. It does not make the retrained controller safer: the enlarged positive class changes nested selection and creates six genuinely state-changing deletions. The unchanged frozen V7 actions look better under post-hoc relabeling, but that observation cannot replace V10's preregistered retraining result.

Per the registered stop rule, controller/threshold iteration on this calibration corpus ends. The source test remains unopened. The paper should keep V10 as a negative label-ablation and base its positive causal claim on the independently confirmed controlled ledger/cohort experiments plus the natural semantic oracle headroom, with deployment claims explicitly bounded.
