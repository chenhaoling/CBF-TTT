# ZsRE contrastive key alignment v1

## Objective

Test whether a calibration-only residual transform can repair the sole failure of the frozen ZsRE contextual readout: rejecting same-subject locality prompts. The backbone, values, associative memory, subject-disjoint split, and development release gate are inherited unchanged from v1.

## Registered arms

- `identity`: bypass the new head and reproduce the v1 readout byte for byte.
- `positive_only`: symmetric prompt/rephrase InfoNCE.
- `relation_aware`: the same InfoNCE plus same-subject locality hard negatives; this is the registered primary arm.

Each representation seed uses an independent zero-initialized rank-16 residual head. Heads train for 300 full-batch epochs on the 267 calibration subjects. Calibration independently selects each arm's unknown threshold from `{0.1, 0.2, 0.35, 0.5, 0.75}`. Development is never used for checkpoint or threshold selection.

## Calibration behavior

Both trained arms reached 100% prompt-to-rephrase top-1 on calibration for every representation seed. Calibration selected unknown thresholds 0.35/0.75/0.75 for identity/positive-only/relation-aware. The trained arms' high thresholds already indicate a substantially changed score scale.

For the primary relation-aware arm, the mean calibration prompt-to-rephrase versus hardest-locality cosine margin grew from about 0.526 at epoch 1 to 1.512 at epoch 300. The locality objective therefore optimized as intended on calibration.

## Development result

| Arm | Overall | Minimum seed | Original rephrase | Updated rephrase | Locality | Mean NLL |
|---|---:|---:|---:|---:|---:|---:|
| Identity | **93.80%** | **93.49%** | **98.75%** | **98.63%** | 88.26% | 0.8884 |
| Positive only | 92.34% | 91.85% | 70.66% | 68.29% | **99.97%** | **0.5144** |
| Relation aware | 91.87% | 91.53% | 69.66% | 69.29% | 99.09% | 0.5662 |

Relation-aware alignment repaired locality by +10.83 points relative to identity, but original and updated rephrase accuracy fell below the unchanged 70% gates. Exact accuracy remained 100.00% original and 99.63% updated. The development gate failed and the 534-subject test split remained sealed.

## Interpretation

The experiment establishes a real calibration-to-development tradeoff rather than a failure to optimize. A single global residual key map can create a strong locality rejection margin, but it overfits calibration paraphrases and does not preserve paraphrase invariance for unseen subjects. Positive-only training shows nearly the same failure, so increasing the locality-loss weight is not the main explanation.

Inspection of the source data also shows that ZsRE locality prompts can be lexically and relationally very close to the edited question. Their supplied ground truth represents information that should remain unchanged; in this memory abstraction `unknown` means “defer to the frozen model” rather than an unknown factual answer. The next method must resolve whether a query falls within an update's scope, using pairwise query–memory evidence instead of forcing all prompts through one global geometry.

## Runtime and audit

- Formal run: 2026-10-12 17:09:51 to 17:10:16 Asia/Shanghai.
- Task time: 21.94 seconds.
- Peak CUDA allocated/reserved: 8.027/8.811 GiB.
- Formal code commit: `f2c6ed9c1d71948def763faaadb11a5ce139cbc4`.
- Server tests: 6/6 passed.
- Independent source, checkpoint, raw-record, aggregation, gate, and conditional-test audit: passed.
- Identity raw development files exactly reproduce the v1 source files.

## Artifacts

Compact results and the independent audit are archived in this directory. The alignment checkpoint and nine raw development JSONL files remain under `/home/ctj/cbf_ttt_zsre_contrastive_keys_v1` on `hku-gpu2`.

## Claim boundary

- Supported: relation-aware calibration can almost eliminate locality activation in this constructed candidate-memory protocol.
- Supported: the learned global key transform fails to retain subject-disjoint paraphrase generalization under the fixed gate.
- Not supported: test performance, official ZsRE editing performance, open-vocabulary generation, or native dense TTT writer improvement.
