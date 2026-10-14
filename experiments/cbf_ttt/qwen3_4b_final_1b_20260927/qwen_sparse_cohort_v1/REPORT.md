# Qwen top-1 cohort forgetting predecessor: audited report

## Conclusion

The registered continuous controller failed its development release gate under
the causal `component_top1` reader, so test remained unscored.  The experiment
nevertheless supplied the complete, independently audited prerequisite for the
separately frozen exact-discrete confirmation: the already included 27-class
controller reached **98.9258%** development accuracy versus **93.1351%** for
cohort physics, a **5.7907-point** descriptive paired gain with source-group
bootstrap 95% CI **[5.1468, 6.4019]**.  Its NLL gain was **0.08507**, CI
**[0.07239, 0.09808]**, and it stayed within **0.4376 points** of the
99.3634% grid oracle.

This is a development/prerequisite result rather than a new held-out-test
claim.  It establishes that the sparse reader exposes a stable exact-state
learning problem and justifies the fresh-corpus confirmation.  It does not
establish native dense `ttt_proj`/`ttt_conv` performance or natural benchmark
improvement.

## Development results

The fresh corpus used seed `202610152`, 256/64/128 train/development/test source
groups, 24 episodes per group and three projection seeds.  Across development
there are 9,216 decisions and 27,648 causal queries per projection seed.

| Policy | Accuracy | NLL | Historical | Min family | Action match | Retention MAE |
|---|---:|---:|---:|---:|---:|---:|
| Retain all | 66.6667% | 2.04643 | 25.0000% | 33.3333% | — | — |
| Reference blind | 63.8238% | 1.41901 | 25.0000% | 51.5914% | 6.4019% | 0.39224 |
| Continuous controller | 83.3152% | 0.80575 | 49.9457% | 66.6667% | 87.3698% | 0.04573 |
| Physics / exact reference | 93.1351% | 0.25222 | 79.5139% | 72.5405% | — | — |
| **27-class controller** | **98.9258%** | **0.16715** | **96.8207%** | **95.7031%** | **91.9596%** | **0.02519** |
| Grid oracle | 99.3634% | 0.16040 | 98.1337% | 97.4537% | 100.0000% | 0 |

The 27-class controller reached 98.8607%, 98.8607% and 99.0560% on projection
seeds 2301/2302/2303.  Current accuracy was 100%, anchor accuracy 99.9566%, and
the minimum seed accuracy was 98.8607%.  Its descriptive gain over retain-all
was 32.2591 points, CI `[31.9806,32.5304]`, and over the reference-blind arm was
35.1020 points, CI `[34.6933,35.5433]`.

## Registered decisions

The predecessor primary remained the continuous regressor.  It missed the
accuracy, physics-gain, confidence-interval, historical, family and oracle-gap
checks; therefore its development gate failed and test identifiers were not
loaded or encoded by the evaluator.  This failure was not relabelled as a
positive primary result.

After seed 2301 and before seeds 2302/2303 or the next corpus were observed,
the discrete confirmation protocol fixed four conditional prerequisites.  The
independent predecessor audit recomputed all of them:

| Conditional prerequisite | Result |
|---|---|
| Grid mean accuracy ≥ 98.5% | Pass: 98.9258% |
| Every seed ≥ 98% | Pass: minimum 98.8607% |
| Grid gain over physics ≥ 5 points | Pass: 5.7907 points |
| Oracle gap ≤ 1 point | Pass: 0.4376 points |

The fresh confirmation corpus was generated only after these conditions and
test sealing were verified.

## Exact-state mechanism diagnostic

The post-hoc development diagnostic is descriptive and is not read by either
release gate.  It hashes all three primary row files and separates actions by
whether the oracle contains an interior coefficient:

| Oracle action kind | Decisions | Action accuracy | Query accuracy | Retention MAE |
|---|---:|---:|---:|---:|
| Endpoint `{0,1}^3` | 6,962 | 99.2818% | 99.8324% | 0.00172 |
| Contains `0.5` | 2,254 | 69.3434% | 96.1254% | 0.09768 |

Interior actions account for 24.4575% of oracle decisions, while 24.7830% of
predictions contain an interior coefficient.  The controller has therefore
not reduced the task to binary keep/delete.  Most residual action error is
concentrated in the rarer interior states, while their downstream query
accuracy remains high.

## Integrity and resources

- Producer: 2026-10-15 03:51:46–04:42:19 +08:00.
- Total task time: 3,024.20 seconds; train/development embedding extraction:
  16.08 seconds.
- Peak allocated/reserved GPU memory: 7.8675/8.1152 GiB on RTX 5090.
- Producer code commit: `a6155cd71a9c3e941dc4a401d1cdf0fd7247a467`.
- All 16 independent audit checks pass: source reconstruction, data/geometry/
  reader dependencies, row hashes, aggregates, gates, checkpoint structure,
  terminal status and sealed-test state.
- Runner exit code is zero and terminal status is
  `completed_qwen_sparse_cohort_forgetting`.

Artifact SHA256 values:

| Artifact | SHA256 |
|---|---|
| summary | `f5dac98b5fcb7b5100547684dfc7a5ec73f37b5cc39c42efdb1a6d1ee0514a25` |
| training | `0b32928f15ce44c934bd55e0a14d18742c631393ab762a759c4efb0b561b2948` |
| manifest | `d1a8f3ec03844025f3a864db11809cbe7f524b2ad5dc0197395094db119ca392` |
| controller checkpoint | `5dbb133c53df71599304096e9f4ce9920f687e9ed739607f1e899af5f460f961` |
| independent audit | `bc99e2f696aa20e67cdeae51fe3a055f43928a6c2ea8e267faed3c20a4aa5b62` |
| action diagnostic | `6db91297a7a9bb0195a5bc799f34421870fcb13e1b9fdf68f3ab5900b87ad13c` |

## Reproduction

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_qwen_sparse_cohort_repro \
  bash scripts/run_qwen_sparse_cohort_forgetting.sh

/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
  -m scripts.audit_qwen_sparse_cohort_forgetting \
  --root /home/ctj/cbf_ttt_qwen_sparse_cohort_repro \
  --data-root /home/ctj/cbf_ttt_qwen_sparse_cohort_repro/data \
  --reader-audit /home/ctj/cbf_ttt_qwen_sparse_reader_20261015_retry2/execution_audit.json \
  --output /home/ctj/cbf_ttt_qwen_sparse_cohort_repro/execution_audit.json

/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
  -m scripts.summarize_qwen_sparse_action_diagnostic \
  --root /home/ctj/cbf_ttt_qwen_sparse_cohort_repro/result \
  --output /home/ctj/cbf_ttt_qwen_sparse_cohort_repro/action_diagnostic.json
```

The active next experiment is the fresh seed-`202610153` discrete confirmation.
Its test remains unavailable to the evaluator until all 12 registered
development checks pass.
