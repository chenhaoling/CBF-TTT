# LongMemEval-S zero-shot forgetting transfer V1

## Conclusion

The frozen short-fact forgetting controller does **not** transfer safely to
long natural sessions. It strongly suppresses stale evidence, but over-deletes
current and unrelated sessions. The preregistered transfer gate failed four of
five checks, so these 78 knowledge-update outcomes must not be used to tune and
rescore the same benchmark.

This negative result separates two questions. Local deletion is useful on this
task: the answer-ID oracle reaches perfect stale/current state and improves
retrieval. The failure is session-level scope calibration, not the physical
possibility of forgetting.

## One-time result

| Policy | Correct stale/current state | Current retained | Stale deleted | Non-answer retained | Current hit@5 | Stale hit@5 |
|---|---:|---:|---:|---:|---:|---:|
| Retain all | 0.00% | 100.00% | 0.00% | 100.00% | 60.26% | 61.54% |
| Window 8 | 50.00% | 52.56% | 97.44% | 16.44% | 52.56% | 2.56% |
| Always delete top | 8.97% | 8.97% | 100.00% | 2.00% | 8.97% | 0.00% |
| Random, matched deletion rate | 51.28% | 69.23% | 74.36% | 40.91% | 61.54% | 21.79% |
| **Frozen controller** | **44.87%** | **50.00%** | **91.03%** | **41.69%** | **50.00%** | **7.69%** |
| Oracle pair deletion | 100.00% | 100.00% | 100.00% | 100.00% | 66.67% | 0.00% |

The frozen controller reduced stale hit@5 relative to retain-all by **53.85
points**, question-bootstrap 95% CI **[41.03,66.67]**. This confirms that the
policy materially changes stale-memory exposure.

The same controller reduced current hit@5 by 10.26 points, CI
`[-24.36,3.85]`, and retained only half of the current answer sessions. Its
strict pair-state accuracy was 6.41 points below the rate-matched random arm,
CI `[-20.51,7.69]`. It therefore does not distinguish useful deletion from
general compression after the short-prompt to multi-turn-session shift.

The oracle is the important feasibility control: deleting only the stale
answer session increases current hit@5 from 60.26% to 66.67% while driving
stale hit@5 to zero. Selective forgetting can help LongMemEval retrieval if the
session scope resolver is corrected.

## Registered gate

| Check | Result |
|---|---|
| Pair state beats rate-matched random with positive lower CI | Failed |
| Current hit@5 non-inferior to retain-all within 2 points | Failed |
| Stale hit@5 reduction has positive lower CI | Passed |
| Current-session retention at least 95% | Failed |
| Non-answer retention at least 80% | Failed |

Overall gate: **failed**. Registered next state:
`stop_longmemeval_zero_shot_transfer`.

## Protocol and integrity

- Dataset: LongMemEval-S cleaned, fixed SHA-256
  `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
- Exactly 78 `knowledge-update` questions; two aligned answer sessions per
  question.
- Qwen3-4B final 1B-token checkpoint; frozen contextual transforms, controller
  weights, and ZsRE held-out threshold 0.675.
- Every dated, role-labelled session was split into non-overlapping 1024-token
  windows. All **10,101** windows were encoded; no session was truncated.
- Three empty-content placeholder turns were skipped before any scoring. The
  first launch stopped on these placeholders and produced no shard result; the
  formal retry used a fresh directory.
- Two RTX 5090 shards, 39 questions each. Per-shard times were 407.27 and
  400.93 seconds, executed concurrently.
- Peak allocated/reserved CUDA memory: 7.983/8.184 GiB.
- Independent local audit reloaded all 78 compact records, verified hashes and
  identities, recomputed every aggregate, 10,000-resample interval, gate, and
  claim lock: `audit_passed=true`.

## Claim boundary

This is a zero-shot mechanism-transfer result. It is **not** the official
LongMemEval answer-generation score. The answer text was not generated or
judged. It does not validate the failed native dense writer, and LoCoMo remains
unscored.

The valid conclusion is that the short-fact controller over-forgets under a
long-session distribution shift, while oracle local deletion shows a positive
task-level opportunity. Further development must use an independently
constructed long-session corpus. These 78 outcomes cannot select a new
threshold, feature, or checkpoint.

## Next experiment

Construct 16/32/48/64-session training and development episodes from
subject-disjoint ZsRE and observed CounterFact, with one stale/current update
pair and many hard-retain sessions. Train a session-level sparse deletion gate
over window-pair evidence and require current/non-answer retention, update
deletion, rate-matched-random superiority, and length robustness before any
new external benchmark is scored. LoCoMo remains the available independent
source for a later one-time test.
