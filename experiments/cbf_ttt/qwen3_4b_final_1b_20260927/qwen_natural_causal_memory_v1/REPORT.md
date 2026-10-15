# Qwen natural causal-memory composition V1

## Question

This preregistered calibration asks whether learned session forgetting,
automatic scope selection and the audited scale-8 natural-value residual work
together over sequential ZsRE updates.  It scores five lifecycle checkpoints
per full session, exact and rephrased queries, and eight matched policy arms.
Natural development, ZsRE test and all external benchmarks were release-gated.

## Audited calibration result

The recovery run completed all 5,144 expected rows from 65 sessions.  The final
audit passed all ten checks, including source hashes, split isolation, state
transition replay, tokenization, Qwen scores, independent aggregation,
bootstrap intervals, conditional release and sealed held-out data.

| Arm | All candidate | Final candidate | Final generation |
|---|---:|---:|---:|
| Oracle lifecycle + oracle reader | **98.31%** | **97.19%** | **83.33%** |
| Oracle lifecycle + automatic reader | 87.05% | 85.95% | 64.24% |
| Learned forgetting + automatic reader | 85.83% | 83.72% | 62.60% |
| Cosine forgetting + automatic reader | 85.28% | 85.08% | 63.47% |
| Retain all + automatic reader | 75.39% | 62.50% | 44.86% |
| Global reset + automatic reader | 41.27% | 37.60% | 17.15% |
| Oracle lifecycle + dense values | 42.83% | 42.44% | 31.40% |
| Zero memory | 44.25% | 40.02% | 7.95% |

Oracle lifecycle plus oracle reader passed its 95% overall/final gates, proving
that the value representation and lifecycle substrate are sufficient.  Oracle
forgetting beat retain all by 11.67 points with bootstrap 95% CI
`[10.92, 12.41]`.  Learned forgetting beat retain all by 10.44 points overall
and 21.22 points at the final checkpoint, establishing the usefulness of
selective session forgetting.

Learned forgetting did not establish superiority to cosine: the gain was 0.54
points with 95% CI `[-0.29, 1.45]`.  Its action accuracy was 96.38% versus
94.44% for cosine, showing that oracle-action imitation is not fully aligned
with downstream memory utility.

## Reader diagnosis

The score-free postmortem reconstructed the automatic reader's top record even
when its gate deferred:

- rank-one active-record accuracy: 96.09%;
- retrieve rate and selection accuracy: 74.30%;
- 1,121 rows deferred despite a correct rank-one record;
- exact top/retrieve accuracy: 100%/100%;
- rephrase top/retrieve accuracy: 92.19%/48.60%;
- auxiliary top/retrieve accuracy: 92.22%/50.12%.

Thus the dominant composition failure is gate transfer to rephrased auxiliary
facts.  A smaller residual ranker error remains on 201 rows.  This observation
registered the scope-transfer V2 experiment; it was not used to release
development.

## Execution and claim boundary

The two-GPU critical path was 955.11 seconds on two RTX 5090 cards.  Peak
allocated/reserved memory was 8.53/10.50 GiB.  Calibration failed the frozen
automatic-reader and learned-versus-cosine gates, so development was not
scored.  The archive contains all raw rows and states, corpus, resources,
summaries, failed and repaired audits, logs and the diagnosis.

The result supports selective forgetting over session memory and a viable
natural-value fast-memory substrate.  It does not yet support a learned-policy
advantage over cosine, automatic-scope sufficiency, native `ttt_conv` or
`ttt_proj` writing, or any held-out benchmark claim.
