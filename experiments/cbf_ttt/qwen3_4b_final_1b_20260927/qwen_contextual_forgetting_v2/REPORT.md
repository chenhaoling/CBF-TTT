# Contextual Qwen forgetting controller v2

## Objective

Test the unchanged four-action forgetting controller on the frozen contextual Qwen memory selected by the preceding readability gate. The original 32-group test split was loaded exactly once, only after all development checks passed.

## Frozen method

- Final Qwen3-4B checkpoint after 1B-token continual pretraining.
- Final-layer hidden state at the last identifier token.
- 256-dimensional projection and covariance whitening with regularizer 0.01.
- Train-selected unknown score 0.75.
- Three representation seeds: 2301, 2302 and 2303.
- Actions: retain, half decay, referenced local deletion and global reset.
- Controller architecture, 300 epochs, features, fixed policies, ablations and gates are unchanged from v1.

## Development gate

| Policy | Accuracy | Minimum seed | NLL |
|---|---:|---:|---:|
| Fixed retain | 66.19% | 66.15% | 1.1838 |
| Fixed half | 46.11% | 45.88% | 1.4559 |
| **Fixed local** | **79.62%** | 79.25% | 0.8779 |
| Fixed global | 44.29% | 44.27% | 2.4616 |
| Random | 55.80% | 55.21% | 1.6573 |
| Reference-blind | 79.62% | 79.25% | 0.8779 |
| Global-only learned | 77.21% | 77.19% | 0.6871 |
| **Full learned** | **90.65%** | **90.30%** | **0.3810** |
| Physics | 90.66% | 90.30% | 0.3807 |
| Greedy oracle | 90.66% | 90.30% | 0.3803 |

Full learned exceeded fixed local by 11.03 points. The source-group bootstrap 95% interval was `[10.13, 12.00]` points. NLL improved by 0.4969 with interval `[0.4754, 0.5182]`. Oracle readability, learned accuracy, minimum-seed accuracy, near-oracle distance, control margin and confidence-bound margin all passed.

## One-time test result

| Policy | Accuracy | Minimum seed | NLL |
|---|---:|---:|---:|
| Fixed retain | 66.55% | 65.84% | 1.1901 |
| Fixed half | 44.81% | 44.40% | 1.5195 |
| **Fixed local** | **79.01%** | 78.60% | 0.8979 |
| Fixed global | 43.30% | 43.14% | 2.5436 |
| Random | 55.68% | 55.36% | 1.6513 |
| Reference-blind | 79.01% | 78.60% | 0.8979 |
| Global-only learned | 76.50% | 76.35% | 0.7264 |
| **Full learned** | **89.05%** | **88.80%** | **0.4321** |
| Physics | 89.05% | 88.80% | 0.4322 |
| Greedy oracle | 89.05% | 88.80% | 0.4309 |

Full learned improved over the strongest fixed policy by **10.04 accuracy points**, with source-group 95% CI **[9.35, 10.82]**, and improved NLL by **0.4657**, CI **[0.4437, 0.4908]**. It improved over global-only learned by 12.54 points, CI `[11.14, 13.86]`. Reference-blind exactly matched fixed local; the full controller's gain therefore depends on history/reference features rather than controller capacity alone.

Test accuracy by event family for full learned was:

- retain: 98.32%;
- revoke-last1: 93.23%;
- revoke-last2: 91.44%;
- reset: 73.21%.

Full learned selected the greedy oracle action on 97.79% of decisions. Physics reached the same query accuracy and nearly the same action agreement (97.70%). Learned and physics accuracy differences were exactly zero; learned NLL was lower by only 0.00010. The greedy oracle had 0.00122 lower NLL than learned.

## Interpretation

This result validates the central controlled mechanism in actual frozen-Qwen representation geometry: state-dependent local/global forgetting outperforms every fixed retention coefficient and a global-only learned controller under multi-component interference. The improvement reproduces across three projections and a source-disjoint one-time test.

The result does not establish that learning is superior to an explicit symbolic rule. The event text exposes revoke/reset structure, and the hand-built physics policy matches learned accuracy. A paper-level next experiment must replace perfect symbolic reference cues with automatic activation/text-derived evidence and test transfer to natural sequential updates such as ZsRE.

## Runtime and audit

- Formal run: 2026-10-12 16:39:13 to 16:48:48 Asia/Shanghai.
- Total task time: 570.04 seconds.
- Peak CUDA allocated/reserved: 7.868 / 8.277 GiB.
- Independent audit: passed all source, transform, embedding bridge, raw-record, aggregate, gate and conditional-test checks.
- Formal code commit: `30c70ce`.

## Artifacts

- Compact result and audit files are archived here.
- Controller checkpoint and 60 raw dev/test JSONL files remain at `/home/ctj/cbf_ttt_qwen_contextual_forgetting_v2` on `hku-gpu2`.

## Claim boundary

- Supported: adaptive local/global forgetting beats fixed and global-only policies in a controlled Qwen-derived low-rank memory.
- Not supported: superiority over a rule with direct access to explicit revoke/reset syntax.
- Not supported: native dense `ttt_proj/ttt_conv` writing, open-vocabulary generation, or official ZsRE/RULER/LongMemEval/LoCoMo improvement.
