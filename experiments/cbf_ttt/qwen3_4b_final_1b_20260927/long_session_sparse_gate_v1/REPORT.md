# Sparse long-session forgetting gate V1

## Status

Completed on the final 1B-token Qwen3-4B checkpoint and independently audited.
The registered development gate failed, so LoCoMo was not loaded or scored.
LongMemEval-S remained locked and was not used for training, selection, or
calibration.

## Protocol

- calibration-fit: 844 episodes / 33,760 events;
- calibration-tune: 224 episodes / 8,960 events;
- subject-disjoint development: 1,068 episodes / 42,720 events;
- lengths: 16, 32, 48, and 64 sessions, with 267 development subjects at each
  length;
- exactly one local replacement, two same-subject hard-retain events, and four
  fact units per session;
- frozen Qwen3-4B encoder and three frozen contextual projections;
- three rank-16 deletion-gate seeds, with epoch and threshold selected on tune;
- exact action-count-matched random, retain-all, cosine, fixed-top, and oracle
  controls.

## Main result

| Policy | Action accuracy | Pair state | Stale deletion | Current retention | Non-target retention |
|---|---:|---:|---:|---:|---:|
| Retain all | 97.50% | 0.00% | 0.00% | 100.00% | 100.00% |
| Fixed top | 2.50% | 0.00% | 100.00% | 0.00% | 3.57% |
| Cosine threshold | 59.80% | 48.13% | 61.89% | 78.37% | 64.60% |
| Action-count-matched random | 73.37% | 31.09% | 34.64% | 91.10% | 75.27% |
| Learned controller, three-seed mean | 55.20% | 55.18% | 73.75% | 77.81% | 58.43% |
| Oracle local deletion | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

The learned controller exceeded the registered primary-seed random control in
strict pair state by 27.25 points, with subject-cluster bootstrap 95% CI
`[22.66, 31.65]`. This gain is real but does not make the policy safe: it came
with substantial deletion of current and unrelated sessions.

The frozen maximum-cosine candidate selector retrieved the correct stale
session in only **33.99%** of positive development events. Controller action
accuracy on primary updates was 19.16% on average. Performance was unstable
across seeds: strict pair state was 58.33%, 59.93%, and 47.28%, while
non-target retention was 77.55%, 63.35%, and 34.40%.

Length did not create the only failure: mean strict pair state was 58.80%,
52.68%, 55.18%, and 54.06% at lengths 16/32/48/64. The central failure is the
candidate/session representation. Taking the largest cosine over 16 unit pairs
creates a multiple-comparison extreme; unrelated facts inside long sessions
frequently outrank the true old/update pair.

## Registered gate

Only the random-comparison check passed. Candidate top-1, current retention,
non-target retention, stale deletion, strict pair state, and every-length pair
state all failed their fixed thresholds. The next-stage marker is therefore
`stop_sparse_session_gate_v1`, and LoCoMo remains sealed.

## Supported conclusion

This experiment does not reject local forgetting. Oracle local deletion is
perfect under the same 1,068 episodes and all four lengths, matching the prior
LongMemEval oracle opportunity. It rejects the V1 assumption that a
maximum-cosine reduction over multi-fact sessions provides a reliable deletion
candidate. Threshold tuning cannot repair a wrong candidate. The next method
must learn or otherwise calibrate hierarchical unit-to-session scope before it
trains the sparse deletion gate.

## Runtime and audit

- wall time: 164.52 seconds;
- peak GPU allocated/reserved: 8.090/10.023 GiB;
- 41,652 development candidate rows;
- 170,845 detailed pair-cache entries and 2,136 episode score tables;
- independent audit: all 11 checks passed;
- summary SHA-256: `75f4c75f7fef175f3ba2d73c25939e8b1aac8fde247f633fcffa553a03be8584`.

Compact summaries, the 11.7 KiB controller checkpoint, hashes, and the audit
are archived in this directory. The 61 MiB raw causal traces remain at
`/home/ctj/cbf_ttt_long_session_sparse_gate_v1` on `hku-gpu2` and are covered
by hashes in `development/training.json`.
