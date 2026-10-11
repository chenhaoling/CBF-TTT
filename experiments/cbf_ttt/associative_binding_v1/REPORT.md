# All-Fact Associative Binding Pilot

## 1. Goal

The preceding candidate-ranking experiment showed that the original dynamic writer did not contain a usable entity-to-value binding hidden behind free-generation errors. This pilot tested whether direct supervision against a hard within-context permutation could create that binding.

The main intervention cyclically permuted the four queried values while preserving the natural background, entity strings, value multiset, context length, and all nonqueried tokens. A successful model therefore had to use the entity-to-value correspondence rather than detect a different domain, format, or value distribution.

## 2. Fixed design

| Item | Value |
|---|---:|
| Model | Qwen3-4B |
| Train / development worlds | 32 / 8 |
| Sealed test worlds | 32, unscored |
| Questions per context | 4 |
| Training rounds | 4 |
| Question exposures | 1,024 per arm |
| Optimizer updates | 256 per arm |
| Fixed checkpoints | 0 / 512 / 1,024 exposures |
| Learning rate | `1e-7` |
| Optimizer / compute | FP32 AdamW / BF16 forward |
| Context writes per update | 2 in both arms |
| Order seed / writer seed | 811 / 301 |

The matched arms were:

- `anchor_twin`: digit-weighted positive loss plus the previous single changing-anchor twin contrast;
- `all_fact_permutation`: the same positive loss plus the mean of four Correct-versus-permuted digit-NLL contrasts.

The fixed final development `global16` ranking gate required Correct top-1 at least 0.25, at least 0.10 top-1 separation from both permutation and Wrong, positive bootstrap lower bounds for both NLL gains, and Full-KV complete-code accuracy at least 0.80. The midpoint was diagnostic only.

## 3. Execution and integrity

The experiment completed on two RTX 5090 GPUs in about 21 minutes.

- Server tests: 11/11 passed.
- Token preflight: all 80 selected contexts produced length-preserving four-value derangements with no fixed points.
- Both arms completed all 256 updates and all three fixed evaluations.
- Independent audit: passed.
- Writer initialization was tensor-identical between arms.
- Frozen backbone hashes remained unchanged.
- Optimizer state and writer parameters were FP32.
- No NaN, OOM, or `||delta W||/||W|| > 1` safety stop occurred.
- Test remained unscored.

## 4. Primary fixed-endpoint result

### Development `global16`

| Arm | Memory | Top-1 | True digit NLL |
|---|---|---:|---:|
| anchor twin | Correct | 0.0625 | 2.7626 |
| anchor twin | permuted | 0.0625 | 2.7827 |
| anchor twin | twin | 0.0625 | 2.7687 |
| anchor twin | Wrong | 0.0938 | 2.6725 |
| anchor twin | Empty | 0.0625 | 2.7180 |
| all-fact permutation | Correct | **0.0469** | 2.5645 |
| all-fact permutation | permuted | 0.0625 | 2.5663 |
| all-fact permutation | twin | 0.0469 | 2.5647 |
| all-fact permutation | Wrong | 0.0469 | 2.5569 |
| all-fact permutation | Empty | 0.0625 | 2.7180 |

For the primary all-fact arm:

| Control | NLL gain over Correct | 95% source-world bootstrap CI | Correct top-1 advantage |
|---|---:|---:|---:|
| permuted | 0.0017 | [-0.0048, 0.0082] | -0.0156 |
| twin | 0.0002 | [-0.0029, 0.0027] | 0.0000 |
| Wrong | -0.0076 | [-0.1799, 0.1549] | 0.0000 |
| Empty | 0.1535 | [-0.0080, 0.3195] | -0.0156 |

Correct top-1 was below the 1/16 chance value of 0.0625 and below the permuted and Empty controls. Every content-specific gate check failed. Full-KV code accuracy was 0.9219, so prompt readability passed.

Complete-code generation accuracy was zero for Correct, permuted, twin, Wrong, and Empty memory in both arms.

## 5. Three-point trajectory

Development Correct-memory `global16` top-1:

| Exposure | anchor twin | all-fact permutation |
|---:|---:|---:|
| 0 | 0.0625 | 0.0625 |
| 512 | 0.0313 | 0.0625 |
| 1,024 | 0.0625 | 0.0469 |

At 512 exposures, the all-fact arm showed a small but statistically separated permutation NLL effect: permutation-minus-Correct was 0.0125 with 95% CI [0.0079, 0.0183]. Correct and permuted top-1 were nevertheless both 0.0625, and Correct did not beat Wrong. At the fixed endpoint the NLL effect shrank to 0.0017 and its interval crossed zero.

This transient midpoint effect was not selected. It suggests that the negative construction is meaningful, while showing that the current optimization does not turn it into stable candidate retrieval.

## 6. Training-set diagnostic

At the endpoint, train-probe Correct top-1 was 0.1719 for the all-fact arm, but permuted top-1 was also 0.1719. The anchor baseline similarly obtained 0.2031 for both Correct and permutation. Higher ranking on seen worlds therefore reflects learned value or world regularities rather than the required entity correspondence.

The all-fact objective components were:

| Window | Balanced loss | Permutation contrast |
|---|---:|---:|
| first 32 updates | 5.7064 | 0.7989 |
| middle 32 updates | 2.0600 | 0.7985 |
| last 32 updates | 2.0514 | 0.8072 |

The positive loss fell strongly, while the contrast did not decrease. With its fixed 0.5 coefficient, the writer continued optimizing generic answer likelihood and did not optimize the associative constraint. This explains why Correct-versus-Empty NLL can improve without Correct-versus-permutation ranking.

## 7. Resources

| Arm | Mean update | Peak allocated | Peak reserved | Max relative delta |
|---|---:|---:|---:|---:|
| anchor twin | 0.7457 s | 14.1099 GiB | 14.7773 GiB | 0.5603 |
| all-fact permutation | 0.8805 s | 16.0142 GiB | 16.4375 GiB | 0.5597 |

The all-fact arm was about 18% slower per update because it scored four negative queries instead of one. Context-write count remained matched.

## 8. Conclusion

The fixed pilot gate **failed**. The experiment establishes three useful points:

1. The within-context permutation is a valid hard negative: it briefly produced a development NLL separation that the original objective lacked.
2. Adding this negative as a weak auxiliary term is insufficient. Its training loss remained near the no-separation value throughout optimization.
3. The writer still learns generic populated-memory and answer-format effects rather than stable entity-conditioned value retrieval.

The result does not justify scaling the same objective, running the sealed test set, or returning to forgetting-controller training. It also does not reject the broader forgetting-only result, which assumes a memory substrate that already contains useful content.

## 9. Implementation and artifacts

- `tasks/train_cbf_associative_binding.py`: deterministic token-level derangement, matched training, all-fact objective, six memory controls, exact generation, and candidate ranking.
- `scripts/audit_cbf_associative_binding.py`: independent schedule, permutation, initialization, checkpoint, aggregation, gate, and test-sealing audit.
- `scripts/run_cbf_associative_binding.sh`: separate verified test and training environments, two-GPU execution, summary, and audit.
- `tests/test_cbf_associative_binding.py`: schedule, derangement, gradient direction, and fixed-gate tests.
- `ASSOCIATIVE_BINDING_PLAN.md`: preregistered protocol and stopping rule.

This directory contains compact manifests, checkpoint summaries, combined summary, audit, status, and hashes. Raw evaluation rows, optimizer state, writer checkpoints, and per-update logs remain under `/home/ctj/cbf_ttt_associative_binding_v1` on `hku-gpu2`.

The ignored transport archive SHA-256 is `6338a205302541e0d8f68b48166b863fc85c4b344b82725ecdd18b650f1a3a8a`.

## 10. Reproduction

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_associative_binding_v1 \
DATA=/home/ctj/cbf_ttt_dynamic_binding_v1/data \
MODEL=/home/ctj/models/Qwen3-4B \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
TEST_PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python \
bash scripts/run_cbf_associative_binding.sh
```

## 11. Recommended next experiment

The next experiment should align the primary training objective with retrieval rather than add another low-weight memory comparison:

1. For every query, score its correct value and the other three same-context values under the same Correct memory.
2. Optimize a four-way listwise cross-entropy or InfoNCE loss, making entity-conditioned candidate selection the primary objective.
3. Match the contrast gradient norm to the positive language-model loss, or remove the positive loss in an ablation, so the retrieval term cannot remain flat.
4. Retain Correct/permuted/Wrong/Empty controls and the fixed 32/8 pilot.
5. Require a train-probe Correct-versus-permutation gap as a basic optimization check before interpreting development results.
6. Only after candidate retrieval passes, run a fresh-seed exact-generation experiment and then resume forgetting-policy work.

An explicit associative retrieval head remains the next architectural option if the candidate-aligned objective cannot fit the training probe.

## 12. Compatibility and risks

- All new behavior is opt-in; baseline configs, datasets, optimizers, and entry points are unchanged.
- The pilot has only eight development worlds and one seed; small NLL intervals should be treated as localization evidence.
- Candidate ranking is easier than free generation and cannot replace the later generation gate.
- The cyclic permutation covers four queried facts while leaving the fifth unqueried sensor value fixed.
- Both arms match context writes but the all-fact arm performs more query readouts.

