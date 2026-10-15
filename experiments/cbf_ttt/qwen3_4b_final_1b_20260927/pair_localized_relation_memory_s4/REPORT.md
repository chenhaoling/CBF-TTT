# Qwen pair-localized relation memory S4: audited stop report

## Question and registered decision

S3 obtained 71.79% relation retrieval recall@16 but only 51.65% direct-pair
ROC-AUC. S4 tested whether the failure came from averaging a complete dialogue
rather than encoding the turns that express the queried entity-pair relation.
The protocol first evaluates a gold-trigger-centered representation as an
upper bound. A direct-pair ROC-AUC below 0.75 stops the branch before training
the deployable localizer or opening DialogRE development/test.

## Data and isolation

The audited corpus contains 12,203 training mentions with 49,226 candidate
windows and 872 calibration mentions with 12,918 candidate windows. DialogRE
provides 1,567/452 trigger-supervised train/calibration mentions. Nineteen
training and 24 calibration examples use the registered full-dialogue fallback
because an entity span cannot be localized deterministically.

The independent data audit passed all ten checks. DialogRE development and test
parsed-row counts are both zero. The data manifest SHA256 is
`53f6506bcb01c4b6c8cf914ae6d0f61c3dfd73420be455f42874e7755a322826`.

## Frozen-Qwen extraction

The final 1B-token Qwen3-4B checkpoint was used with native TTT disabled. All
vectors are 2,560-dimensional, finite, FP32 after extraction and unit
normalized. Maximum tokenizer length is 630, below the frozen 1,024-token
limit, so no row was truncated.

| Dataset | Rows | Two-shard time | Peak reserved memory |
|---|---:|---:|---:|
| Train candidate windows | 49,226 | 158.38 / 156.12 s | 9.17 / 9.60 GiB |
| Calibration candidate windows | 12,918 | 41.98 / 41.23 s | 8.52 / 8.85 GiB |
| Train controls | 38,176 | 147.51 / 147.26 s | 10.25 / 10.38 GiB |
| Calibration controls | 3,068 | 19.97 / 20.39 s | 9.71 / 9.73 GiB |

The independent embedding audit passed every completeness, provenance,
normalization, finite-value, resource and sealing check for all four datasets.

## Oracle and controls

The gold-trigger oracle was trained only on the registered trigger-eligible
training rows, initialized from the selected S3 projection, and selected by
calibration recall@16, then direct-pair ROC-AUC, then earlier epoch.

| Representation | Recall@4 | Recall@8 | Recall@16 | Direct-pair ROC-AUC |
|---|---:|---:|---:|---:|
| Full dialogue | 42.43% | 58.37% | **71.79%** | 51.65% |
| Nearest entity radius 0 | 35.67% | 53.67% | 69.04% | 52.52% |
| Nearest entity radius 1 | 32.91% | 50.57% | 70.30% | 52.00% |
| Nearest entity radius 2 | 34.63% | 51.72% | 69.84% | 52.10% |
| Random window | 29.24% | 44.15% | 63.76% | 51.56% |
| Gold trigger, selected epoch 10 | **50.66%** | **59.07%** | 71.68% | **55.80%** |

Epochs 25 and 50 reached ROC-AUC 54.91% and 58.16%, but lower recall@16
(62.39% and 57.52%), so they were not selected under the frozen rule. The
selected weight SHA256 is
`6dcef22955942caed2da860a0680c8354e1be8bce887db0353ce76737e241bcf`.

## Registered gate and conclusion

The gold-trigger oracle ROC-AUC is **0.5580**, well below the registered 0.75
minimum. Its recall@16 is also slightly below the unchanged full-dialogue
control. The independent oracle auditor reconstructed all controls, selection,
weights, resources and sealing state and passed all nine checks.

Terminal status is `stopped_localized_representation_failure`. The deployable
learned localizer, S4 risk fusion and DialogRE development/test were not run.
The result rules out missing trigger localization as the main explanation for
S3. Generic relation-equivalence supervision remains too weak for safe
forgetting with this frozen Qwen representation.

The next experiment therefore uses downstream counterfactual future utility as
the supervision target for individually addressable records, with the current
write fixed to `g=1`. It does not continue relation-head tuning on these rows.

## Reproduction

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927

/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
  -m scripts.audit_pair_localized_relation_memory_s4_data \
  --root /home/ctj/cbf_ttt_pair_localized_relation_s4_data_20261016

bash scripts/run_pair_localized_relation_memory_s4_embeddings.sh

/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
  -m tasks.evaluate_pair_localized_relation_memory_s4_oracle \
  --data-root /home/ctj/cbf_ttt_pair_localized_relation_s4_data_20261016 \
  --embedding-root /home/ctj/cbf_ttt_pair_localized_relation_s4_embeddings_20261016 \
  --s3-projection-root /home/ctj/cbf_ttt_retrieval_evidence_s3_projection_20261015_retry1 \
  --output-root /home/ctj/cbf_ttt_pair_localized_relation_s4_oracle_20261016
```

The compact local archive contains the data manifest, tokenizer lengths,
embedding audit, oracle selection, run log and independent oracle audit. The
large embedding shards remain on `hku-gpu2` under the paths above.
