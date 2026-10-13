# MQuAKE Candidate Diagnostic V9 Report

V9 passed its independent audit and selected `anchor_4` on MQuAKE tune. Development,
MQuAKE-CF, LongMemEval-S, and LoCoMo were not loaded.

| Candidate policy | Target recall | Top-1 |
|---|---:|---:|
| all-unit max, width 4 | 34.52% | 34.52% |
| all-unit max, width 8 | 55.50% | 55.50% |
| all-unit max, width 16 | 92.09% | 92.09% |
| all-unit max, width 32 | 100.00% | 100.00% |
| explicit anchor, width 4 | 100.00% | 100.00% |
| learned full list | 100.00% | 100.00% |

Raw all-unit maximum cosine was 31.84% top-1 with mean target rank 7.25. Raw explicit
anchor cosine was 100% top-1 on every session length. This identifies the V8 failure
as max aggregation over natural multi-hop context units: a high incidental context
match hides the declared session scope. The four-item anchor shortlist recovers the
target at the lowest tested cost, and the frozen learned ranker preserves 100% top-1.

The next causal experiment freezes `anchor_4`, retrains only the deletion head on
train, calibrates only on tune, and conditionally reads dev. The scope anchor is an
explicit serializer field and does not use subject, relation, target, or age labels.
