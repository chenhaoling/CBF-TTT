# MQuAKE-T Adaptation V8 Report

## Result

V8 passed its independent audit and stopped at the tune prerequisite. Development was
not loaded or scored. MQuAKE-CF, LongMemEval-S, and LoCoMo remained unused.

| Tune metric | Result | Required |
|---|---:|---:|
| shortlist recall | 28.82% | 90% |
| candidate top-1 | 28.82% | 80% |
| current retention | 100.00% | 97% |
| non-target retention | 98.86% | 95% |
| hard-retain accuracy | 97.49% | 95% |
| strict pair state | 15.88% | 70% |

All three deletion heads found feasible safety thresholds (0.85, 0.85, and 0.75).
The ensemble therefore recovered high retention, but could delete the correct stale
session in only 15.88% of episodes. Shortlist recall equals candidate top-1, showing
that the learned ranker chooses the correct item whenever it survives the fixed
four-item shortlist; the primary bottleneck is shortlist construction.

## Resources

- Train: 4,484 episodes; tune: 1,492 episodes.
- Runtime: 892.01 seconds.
- Peak GPU allocated/reserved: 7.867/8.115 GiB.
- Compact results and the nine-check independent audit are stored beside this report.
  Hash-covered traces and checkpoints remain under
  `/home/ctj/cbf_ttt_mquake_temporal_adaptation_v8`.

## Next step

Do not tune the deletion head. On tune only, measure target ranks under the explicit
scope anchor, all-unit maximum, learned full-list ranker, and 4/8/16/32 candidate
shortlists. Freeze the cheapest resolver that recovers target recall before another
causal forgetting run. Dev and LoCoMo remain sealed.
