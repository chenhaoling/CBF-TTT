# Within Update Counterfactual Pairing Result

## Result

The paired experiment completed at commit `3b758a5`; 22 server tests and the independent audit passed. Both arms used opposite assignment pairs in each update, covered all four assignments once per context, and completed 128 updates / 1,024 question exposures. Test remained sealed.

| Arm | First→last retrieval loss | First→last swap loss | First→last accuracy | Mean update | Peak allocated / reserved |
|---|---:|---:|---:|---:|---:|
| Paired listwise | 1.5900→1.3955 | 0.79804→0.79889 | 25.00→24.22% | 0.993 s | 24.385 / 24.752 GiB |
| Paired swap | 1.5887→1.3920 | 0.79807→0.79864 | 25.00→28.13% | 0.968 s | 24.385 / 24.752 GiB |

The swap loss did not decline from the no-separation value `softplus(0.2)≈0.7981`. The small final accuracy difference was not accompanied by causal memory separation.

Final development metrics:

| Arm / memory | Context4 top 1 | Global16 top 1 | Digit NLL |
|---|---:|---:|---:|
| Listwise / Correct | 24.22% | 7.81% | 16.9681 |
| Listwise / Counterfactual | 23.44% | 7.81% | 16.9655 |
| Listwise / Wrong | 25.78% | 2.34% | 16.9701 |
| Listwise / Empty | 26.56% | 5.47% | 2.7108 |
| Swap / Correct | 21.09% | 6.25% | 15.8077 |
| Swap / Counterfactual | 21.88% | 6.25% | 15.8035 |
| Swap / Wrong | 27.34% | 4.69% | 15.7943 |
| Swap / Empty | 26.56% | 5.47% | 2.7108 |

For the primary swap arm, Counterfactual minus Correct true NLL was `-0.00539`, bootstrap 95% CI `[-0.00842, -0.00204]`: the significant direction was wrong. Correct minus Counterfactual top 1 was also negative. Wrong and Empty controls did not support correct-memory content. Full-KV code readability remained 95.31% under the established LM metric. The registered gate failed.

Removing the LM term exposed a second failure: populated memory drove true digit NLL above 15 while Empty remained 2.71. The four candidate objective can preserve relative ordering while badly distorting absolute language likelihood. This makes the current LM-head readout unsuitable for a forgetting-policy prerequisite even when paired contrast is used.

## Decision

The result rules out cross-step gradient noise as the sole cause. Further weight tuning of listwise, permutation, or swap losses on the same dense delta plus frozen LM head is not justified. The next pilot adds a small explicit candidate retrieval head and compares a frozen-writer head-only shortcut control with joint head and writer training on the same balanced assignments. A pass will still require fresh permutations and a generation bridge.

## Artifacts and compatibility

`summary.json`, `execution_audit.json`, `RUN_STATUS.json`, and `git_commit.txt` are archived here. Raw steps, evaluations, checkpoints, optimizer states, and logs remain at `/home/ctj/cbf_ttt_paired_binding_v1`. Implementation is in `tasks/train_cbf_paired_binding.py`, the runner and audit are under `scripts/`, and tests are in `tests/test_cbf_paired_binding.py`.

All code is opt in. Baseline model, pretraining, datasets, and default TTT behavior are unchanged. The experiment uses one writer seed and eight development worlds.
