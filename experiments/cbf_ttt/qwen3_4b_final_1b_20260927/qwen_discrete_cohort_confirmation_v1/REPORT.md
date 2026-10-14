# Qwen exact-state cohort controller: audited confirmation

## Conclusion

The fresh, preregistered exact-state confirmation passed its development gate,
released the sealed test once, and passed every test gate.  Across three
projection seeds, the primary 27-class controller (`grid_classifier`) reached
**98.6762%** held-out accuracy versus **93.3955%** for the explicit cohort
physics rule.  The paired gain was **5.2807 points**, with source-group
bootstrap 95% CI **[4.8575, 5.7075]**.  Answer NLL improved by **0.07284**, with
95% CI **[0.06238, 0.08327]**.  The controller was only **0.6275 points** below
the 99.3037% causal grid oracle.

All 16 independent audit checks passed.  This result confirms that a learned
exact cohort-state decision can outperform a hand-written revoke/reset rule on
fresh controlled sessions.  It does not establish native dense
`ttt_proj`/`ttt_conv` writing, implicit extraction of lifecycle events from
natural text, or improvement on external language-model benchmarks.

## Data and protocol

- Fresh corpus seed: `202610153`.
- Source groups: 256 train, 64 development and 128 sealed test.
- Projection seeds: 2301, 2302 and 2303.
- Action space: all 27 values in `{0, 0.5, 1}^3` for history, last-2 and
  last-1 cohorts; the current write is always admitted.
- Reader: the previously audited causal `component_top1` reader.
- Primary: a 27-class controller trained independently for each projection.
- Comparators: continuous and independently snapped controllers, reference
  blind, retain/clear/half/window/random policies, explicit physics,
  exact-reference deletion and the causal grid oracle.
- Test identifiers and embeddings were unavailable until all 12 development
  checks passed.

The training labels contain an interior action in 23.71%, 23.81% and 23.88% of
examples for seeds 2301/2302/2303.  The experiment therefore includes a real
three-state decision problem rather than only binary keep/delete labels.

## Held-out results

| Policy | Accuracy | NLL | Historical | Minimum family | Action match | Retention MAE |
|---|---:|---:|---:|---:|---:|---:|
| Retain all | 66.6667% | 2.05709 | 25.0000% | 33.3333% | — | — |
| Reference blind | 63.6574% | 1.45170 | 25.0000% | 51.5119% | — | — |
| Continuous controller | 83.2881% | 0.80531 | 49.8752% | 66.6667% | 86.8978% | 0.04631 |
| Independently snapped | 98.5026% | 0.17609 | 95.5187% | 94.0104% | 88.7967% | 0.02856 |
| Physics / exact reference | 93.3955% | 0.24798 | 80.1866% | 73.5822% | — | — |
| **27-class controller** | **98.6762%** | **0.17513** | **96.2023%** | **95.0810%** | **90.6250%** | **0.02947** |
| Grid oracle | 99.3037% | 0.15939 | 97.9112% | 97.2150% | 100.0000% | 0 |

For the primary controller, current accuracy was 100%, anchor accuracy was
99.8264%, and minimum seed accuracy was 98.3236%.  Its family accuracies were
95.0810% reset, 99.7685% retain, 99.8553% revoke-last-1 and 100% revoke-last-2.
The result is therefore not driven by sacrificing current information or the
session anchor.

The continuous output is a registered negative control: although its
retention MAE is small, nonzero residual coefficients keep components active
under top-1 routing, and its accuracy remains 83.29%.  Independently snapping
the same output recovers 98.50%, while direct 27-class training reaches 98.68%.
This supports exact memory-state selection as the relevant mechanism for this
reader.

## Registered gates

Every held-out gate passed:

| Check | Audited result |
|---|---|
| Mean accuracy at least 98.5% | 98.6762% |
| Gain over physics at least 5 points | 5.2807 points |
| Gain 95% CI lower bound above 3.5 points | 4.8575 points |
| NLL-gain 95% CI lower bound above zero | 0.06238 |
| Every seed gains at least 4 points over physics | Pass |
| Oracle gap at most 1 point | 0.6275 points |

The development result was 98.9041% for the primary versus 93.2726% for
physics, a 5.6315-point gain with CI `[5.0239, 6.2102]`; all development
release checks passed before test access.

## Integrity and resources

- Producer runtime: 4,270.44 seconds.
- Train/development embedding extraction: 15.95 seconds.
- Peak allocated/reserved GPU memory: 7.8675/8.1211 GiB on RTX 5090.
- Runner exit code: zero.
- Producer terminal: `completed_qwen_discrete_cohort_confirmation`.
- Independent audit terminal:
  `completed_qwen_discrete_cohort_confirmation_audit`.
- All raw rows, reconstructed aggregates, bootstrap intervals, release gates,
  dependency hashes, checkpoint structure and sealed-test transitions passed
  independent verification.

Key artifact SHA256 values:

| Artifact | SHA256 |
|---|---|
| summary | `3649c879447260ada86af4a03d234ec983317bc96e6a8dd6083131a997cff659` |
| training | `ae1bf74e82509958d29e80eca6ead77b1075c11c890a4fd7cbb95044efd76275` |
| manifest | `b9f6f799cc08dd6f2370340bc6448a79b05374371995b373938627a785839c8d` |
| controller checkpoint | `15b42d85e3a9a5f648865f367b11113f8d0bc247376de1a3493f2df3517a6c65` |
| independent audit | `3fb154f7bc7392c417f2b880a046fc2a0579b45e97968c3ba707e1ccffada92d` |

## Reproduction

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_qwen_discrete_cohort_confirmation_repro \
  bash scripts/run_qwen_discrete_cohort_confirmation.sh

/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
  -m scripts.audit_qwen_discrete_cohort_confirmation \
  --root /home/ctj/cbf_ttt_qwen_discrete_cohort_confirmation_repro \
  --data-root /home/ctj/cbf_ttt_qwen_discrete_cohort_confirmation_repro/data \
  --reader-audit /home/ctj/cbf_ttt_qwen_sparse_reader_20261015_retry2/execution_audit.json \
  --predecessor-audit /home/ctj/cbf_ttt_qwen_sparse_cohort_20261015/execution_audit.json \
  --output /home/ctj/cbf_ttt_qwen_discrete_cohort_confirmation_repro/execution_audit.json
```

The independently released next experiment is the frozen-controller lifecycle
OOD evaluation on fresh seed `202610154`.  It retrains no controller and is
required before the actual-Qwen residual-readout bridge may run.
