# Physics-Informed Structured Forgetting Result

## Main result

The formal experiment completed at commit `4596028`. Eleven server tests, deterministic corpus construction, controller training, conditional development gate, one-time test release, and independent recomputation audit all passed.

The corpus contains 96 disjoint worlds: 64 train, 16 development, and 16 test. It has 2,304 episodes and 13,824 queries across retain, revoke-last-one, revoke-last-two, and reset families. Identifiers and wording templates are disjoint across splits. The controller was trained on 3,072 decision examples and has no language-model or GPU dependency.

| Policy | Dev query acc. | Dev action acc. | Test query acc. | Test action acc. | Test retained batches |
|---|---:|---:|---:|---:|---:|
| Fixed retain | 66.67% | 34.38% | 66.67% | 30.73% | 8.00 |
| Fixed last 1 | 68.14% | 25.00% | 68.66% | 25.00% | 6.00 |
| Fixed last 2 | **75.00%** | 25.00% | **75.00%** | 25.00% | 6.00 |
| Fixed global | 50.00% | 12.50% | 50.00% | 12.50% | 2.08 |
| Deterministic random | 60.81% | 22.27% | 62.50% | 26.30% | 4.96 |
| History-blind learned | 89.93% | 66.93% | 90.28% | 67.97% | 5.66 |
| **Physics-informed learned** | **99.91%** | **99.74%** | **99.74%** | **99.22%** | 5.66 |
| Visible physical rule | 100.00% | 87.50% | 100.00% | 87.50% | 5.52 |
| Counterfactual oracle | 100.00% | 100.00% | 100.00% | 100.00% | 5.63 |

On development, the learned controller exceeded the best fixed action (`fixed_last2`) by 24.91 percentage points; paired source-world bootstrap 95% CI `[24.74, 25.00]`. All four preregistered checks passed, which authorized the single test evaluation. On test, the gain over `fixed_last2` was 24.74 points, 95% CI `[24.48, 25.00]`.

The history-blind controller reached 90.28% test query accuracy. Restoring batch-recency state improved it by 9.46 points, 95% CI `[8.94, 9.98]`. This directly supports the local-memory branch: identifying whether a referenced batch is last one or last two contains useful information that a global or history-blind policy cannot recover.

## Family and role analysis

Learned test accuracy by event family:

| Family | Queries | Accuracy |
|---|---:|---:|
| Retain | 576 | 100.00% |
| Revoke last 1 | 576 | 100.00% |
| Revoke last 2 | 576 | 100.00% |
| Reset | 576 | 98.96% |

Learned test accuracy by query role was 100.00% for current facts, 100.00% for protected anchors, and 99.22% for historical facts. The history-blind ablation fell to 72.92% and 89.24% on the two revoke families, while retaining 100% on ordinary retain events. This localizes the benefit to history-sensitive forgetting rather than general memorization.

The linear controller trained in 0.96 seconds. Its training accuracy rose from 28.32% to 99.41%; the history-blind version saturated at 74.41%.

## What this establishes

This is positive preliminary evidence for the revised method:

- a single fixed forgetting coefficient cannot handle retain, local cancellation, and session reset together;
- state-dependent local (`last1`, `last2`) and global forgetting decisions can be learned from physical memory invariants;
- the learned rule generalizes to disjoint entities, values, and surface templates;
- recency features account for a measurable part of the gain.

The result validates the forgetting controller on a readable structured memory substrate. It does not yet establish an end-to-end gain in the original dense TTT fast-weight implementation. The dense writer remains unable to expose reliable entity-value content, so direct distillation into its current representation would confound policy quality with memory readability.

## Label ambiguity and limits

There are 1,114 tied counterfactual decisions among 3,072 training examples. Ties are resolved by the registered minimal-forgetting order. This explains why the visible physical rule has perfect query accuracy but 87.5% action agreement: a different tied action can answer every query correctly. Query accuracy and paired fixed-policy gains are therefore the main effectiveness outcomes; action accuracy measures agreement with the chosen conservative tie rule.

The corpus has explicit structural event syntax and four controlled families. Harder follow-up data should add irrelevant/noisy updates, duplicates, paraphrased references, delayed cancellations beyond two steps, imperfect retrieval, and capacity limits. Multiple corpus seeds are also required before a full paper claim.

## Preflight failure

The first launch at commit `4028948` built and audited the complete corpus and trained the controller, then stopped during fixed-policy evaluation. A policy could previously delete a batch that a later visible event referenced; the parser incorrectly treated that absent old reference as another new batch. No development summary or test score was produced. The fix identifies the last visible batch reference adjacent to facts as the new write and preserves earlier references independently of policy state. The failed directory is retained at `/home/ctj/cbf_ttt_structured_forgetting_v1_preflight_4028948`.

## Reproduction and compatibility

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_structured_forgetting_v1 \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
TEST_PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python \
bash scripts/run_cbf_structured_forgetting.sh
```

Small summaries, manifests, training curves, and audit files are archived here. Generated JSONL rollouts, the controller checkpoint, and full corpus remain on the server. The experiment is fully opt in and does not modify baseline model code, pretraining, or default TTT behavior.
