# MQuAKE-T V15 Jackknife-Consensus Controller

## Status

- Protocol: `mquake_jackknife_consensus_v15`
- Terminal marker: `completed_mquake_jackknife_consensus_v15`
- Runner exit code: `0`
- Independent audit: passed all 8 checks
- Registered gate: failed
- External confirmation sets accessed: no

V15 used nested subject-jackknife calibration and a three-of-four consensus
decision. It was the final registered attempt in the listwise-score plus scalar
threshold family.

## Main results

| Metric | Result | Gate |
|---|---:|---:|
| OOF pair-state accuracy | 54.08% | >=90% |
| OOF stale deletion | 54.10% | >=90% |
| Current retention | 99.95% | >=99% |
| Non-target retention | 99.92% | >=97% |
| Hard-retain accuracy | 99.92% | >=97% |
| Subject-macro pair accuracy | 58.91% | >=85% |
| Subject-macro pair LCB95 | 53.53% | reported |
| Subject-macro safety | 98.34% | reported |
| Subject-macro safety LCB95 | 96.65% | >=95% |
| Candidate top-1 | 98.93% | >=99% |
| Candidate subject-macro top-1 | 97.34% | >=97% |
| Random-matched gain | +51.99 points | CI lower >0 |
| Random-matched 95% CI | `[45.30,57.92]` points | |

Fold pair-state accuracy was 69.48%, 57.02%, 46.82%, 61.37%, and 35.65%.
The minimum-fold requirement of 85% therefore failed substantially.

## Interpretation and decision

The controller preserved the current and non-target memories extremely well
and was clearly better than a matched random deletion policy. It nevertheless
failed to delete stale memory reliably and generalized poorly across subjects
and folds. These results do not show that forgetting is useless. They show that
the registered listwise feature representation plus scalar calibration family
does not provide a stable cross-subject decision rule.

Per the preregistered stop rule, V15 closed that controller family. V16 was not
launched. The next experiment is the V17 exact Qwen causal-headroom diagnostic;
a new controller is admissible only if the oracle shows at least five points of
subject-macro downstream accuracy headroom over the strongest fixed strategy.

The summary SHA-256 is
`a4b10fcdc010c5b769f7885b24b12b2276e4cc773c9c13ac0f403267aa993aa4`.
Raw OOF traces remain on the remote experiment root and are referenced by
hashes in the archived manifests; the local archive contains the compact
summary, training evidence and cross-fit checkpoint.
