# Hierarchical long-session candidate resolver V2

## Status

Completed and independently audited. The release gate failed, so no deletion
gate was trained and LoCoMo remained sealed.

## Result

| Resolver | Top-1 | MRR | Mean rank |
|---|---:|---:|---:|
| V1 mean maximum cosine | 33.99% | 47.18% | 7.07 |
| Top-4 mean | 19.85% | 36.48% | 8.03 |
| Full pair mean | 11.52% | 26.73% | 10.79 |
| Projection-consensus maximum | 33.05% | 46.69% | 7.10 |
| Anchor-only construction diagnostic | 79.21% | 86.21% | 2.04 |
| Learned hierarchical resolver | **67.38%** | **73.16%** | **4.86** |

The primary seed improved top-1 over V1 by 33.33 points, with subject-cluster
bootstrap 95% CI `[28.93,37.64]`. Results were stable across seeds
(67.32/67.51/67.32%). Learned top-1 by length was 70.54%, 66.29%, 66.17%, and
66.54% at lengths 16/32/48/64.

Only the registered gain check passed. Mean/minimum-seed/every-length top-1 and
MRR failed. The run therefore stopped before causal deletion.

## Interpretation

Order-neutral multi-instance learning recovers a large amount of candidate
signal, so V1's flat maximum was materially wrong. It does not solve the task.
Even the ineligible anchor-only diagnostic reaches only 79.21%, although it
knows which unit is the constructed anchor. The stored text joins a relation
prompt with a value that intentionally changes across the old/update pair.
Embedding this combined string contaminates the retrieval key with conflicting
payload. The next registered stage factorizes prompt scope from value payload;
unit position remains unavailable to the primary method.

## Reproducibility

- wall time: 38.73 seconds;
- peak GPU allocated/reserved: 8.090/10.023 GiB;
- selected epochs: 100/200/200;
- independent audit: all nine checks passed;
- summary SHA-256: `2c9825d3f0528e3f81a59213ae69818db937e590496bf496841c303dd6cc83fe`.

Hash-covered raw development records remain at
`/home/ctj/cbf_ttt_long_session_candidate_v2` on `hku-gpu2`.
