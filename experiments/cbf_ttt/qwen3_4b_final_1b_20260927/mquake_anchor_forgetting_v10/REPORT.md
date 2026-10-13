# MQuAKE Anchor Forgetting V10 Report

## Result

V10 passed the six-check tune prerequisite and therefore evaluated the frozen method
once on entity-disjoint MQuAKE-T dev. The independent audit passed. Candidate retrieval
and selective deletion generalized strongly, but the registered safety gate failed.

| Metric | Tune | Dev | Dev requirement |
|---|---:|---:|---:|
| shortlist recall | 100.00% | 100.00% | 99% |
| candidate top-1 | 100.00% | 99.67% | 99% |
| stale deletion | 99.20% | 96.19% | 85% |
| current retention | 100.00% | 99.87% | 97% |
| strict pair state | 99.20% | 96.12% | 85% |
| non-target retention | 98.42% | 91.80% | 95% |
| hard-retain accuracy | 97.79% | 91.81% | 95% |

Every length reached at least 95.19% pair state. Exact deletion-count matched random
reached 11.63% pair state; V10 improved by 84.49 points with an 18-subject clustered
95% interval of `[81.71,88.18]` points. Current-memory safety, candidate accuracy,
deletion efficacy, every-length robustness, and the matched-random comparison passed.
Only non-target and hard-retain safety failed.

## Conclusion

The explicit scope anchor removes the context-max retrieval failure and supports highly
effective local forgetting on unseen temporal subjects. The remaining error is
over-deletion under cross-subject calibration shift: tune safety near 98% fell to about
92% on dev even though current retention stayed at 99.87%. This is evidence for the
method's central mechanism, but it does not satisfy the preregistered safe-forgetting
claim and does not release LoCoMo.

## Resources and provenance

- Runtime: 935.92 seconds.
- Peak GPU allocated/reserved: 7.867/8.115 GiB.
- Raw traces and checkpoint remain under
  `/home/ctj/cbf_ttt_mquake_anchor_forgetting_v10`.
- Compact summary, training metadata, audit, and timestamps are stored beside this
  report. MQuAKE-CF, LongMemEval-S, and LoCoMo were not used.
