# Cross-domain forgetting reranker V1

## Purpose

Use observed ZsRE and CounterFact only as development data to test a decomposed
forgetting policy: a residual pairwise ranker first selects the old memory in the
same factual scope, then a separate safe-delete gate decides whether to forget
it. The current input is always written.

## Results

| Arm | Query | Action | Candidate top-1 | Hard retain | Update |
|---|---:|---:|---:|---:|---:|
| ZsRE-only transfer | 49.21% | 65.90% | 61.27% | 97.58% | 0.25% |
| Mixed ZsRE + CounterFact five-fold OOF | **49.66%** | **84.93%** | **78.47%** | **92.36%** | **63.69%** |
| Previous frozen ZsRE controller | 48.12% | 67.77% | — | 78.73% | 30.70% |
| Fixed retain | 49.44% | 66.67% | — | 100.00% | 0.00% |

The mixed OOF model improved over the previous controller by 1.15 query points,
with subject-paired 95% CI `[0.49,1.84]`, and 17.15 action points, CI
`[15.24,19.15]`. Candidate top-1 and hard retain passed their 75% and 85%
development floors, and query accuracy exceeded fixed retain. Overall action
missed 85% narrowly, and update accuracy missed the 75% floor; the complete gate
therefore failed.

ZsRE-only transfer selected a 0.95 deletion threshold and nearly never updated.
This confirms that cross-source conflict calibration does not transfer directly.
The mixed OOF ranker itself is useful, but the safe-delete gate was trained to
suppress deletion whenever its current candidate was wrong, producing high
retention and insufficient update recall.

## Resources and interpretation

The run encoded 7,254 unique texts and 86,874 pair prompts from 258 ZsRE and 197
CounterFact sessions. It processed 4,990 oracle-state decision groups. Runtime
was 307.87 seconds on one RTX 5090 with 9.026/12.551 GiB peak
allocated/reserved memory.

This is observed development evidence, not a final held-out result. It supports
the decomposition and rejects the V1 gate training objective. V2 keeps the
ranker family, uses all outer-training folds, learns semantic conflict with class
balancing, and selects thresholds from nested OOF predictions. LongMemEval-S and
LoCoMo remain unscored.

