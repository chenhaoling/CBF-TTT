# MQuAKE-T Online Margin V14

## Goal

Test whether a deletion score should be interpreted relative to earlier scores
in the same session. V14 trains subject-equal episode-listwise gates and uses
the current normalized gate score minus the median of all prior action scores.
The policy is causal and has no position, length, subject, relation, or future
score input.

## Protocol

- Five subject-disjoint outer folds over MQuAKE-T train+tune.
- One separate calibration fold per outer fold.
- Three rank-16 gate seeds, 50 epochs, and fit-score median/IQR normalization.
- Exact causal replay over 20 registered margin thresholds.
- Frozen V12 anchor-16 candidate rankers and source artifacts.
- MQuAKE-CF, LongMemEval-S, LoCoMo, and V10 dev remain unscored.

## Audited result

The independent audit passed all eight checks. The 5,976 OOF episodes contain
239,040 actions.

| Metric | V14 |
| --- | ---: |
| Pair-state accuracy | 49.866% |
| Stale deletion | 50.000% |
| Current retention | 99.866% |
| Non-target retention | 99.850% |
| Hard-retain accuracy | 99.833% |
| Subject-macro pair | 51.490% |
| Subject safety LCB95 | 97.301% |
| Candidate top-1 | 98.929% |
| Gain over matched random | +48.042 points |
| Subject-bootstrap 95% CI | [37.468, 59.046] points |

Fold pair-state accuracy is 46.405%, 43.144%, 63.545%, 94.732%, and
1.342%. Selected thresholds are 3.0, 3.0, 3.0, 2.5, and 4.0. The strict
release gate fails on deletion, pair, subject-macro pair, minimum-fold pair,
and the 99% candidate top-1 floor.

Runtime was 7,458.63 seconds. Peak CUDA allocated/reserved memory was
7.867/8.115 GiB. The pair cache reached its 1,000,000-entry cap.

## Failure attribution

The registered relation diagnostic uses MQuAKE relation IDs after evaluation
only; it does not change V14 decisions. There are seven relations. Roughly 99%
of episodes have at least 500 fit episodes for their relation, while a
leave-fold-out same-relation predictor reaches only 46.12% binary accuracy.
Relation scarcity does not explain fold 4.

Fold 4 has update-margin median 3.011 against threshold 4.0. Fold 3 has
median 3.663 against threshold 2.5. This is a cross-subject calibration shift.

The frozen-trace normalizer screen compares seven causal transformations with
thresholds selected without the validation fold. MAD/IQR/mean z-scores lose
most update recall. Percentile achieves high recall but triggers about 4.8% of
non-executable actions. The original median margin is best: at a 0.25%
calibration FPR its pair proxy is 73.41% with 62.46% minimum-fold pair; at
0.5% FPR it is 88.28% with 81.80% minimum-fold pair. These are architecture
diagnostics on fixed V14 traces, not exact causal policy results.

## Conclusion

Session-relative scoring is useful but a threshold selected on one subject
fold is unstable. Relation conditioning and additional z-normalization are
rejected. V15 keeps the median margin and tests nested four-way subject
jackknife calibration with three-of-four consensus and exact causal replay.

## Artifacts

Raw OOF JSONL files remain on `hku-gpu2` and are locked by `training.json`.
This compact archive contains the summary, training metadata, manifest,
checkpoint, independent audit, relation diagnostic, normalizer screen, logs,
timestamps, and runner exit code.
