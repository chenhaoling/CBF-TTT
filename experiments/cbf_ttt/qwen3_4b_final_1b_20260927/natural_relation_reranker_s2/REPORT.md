# Natural relation retriever/reranker S2

## Objective

S2 tests whether a Qwen3-4B relation comparator can safely identify the
historical record whose semantic slot matches a new fact.  It was introduced
after S1 exposed a template-copy shortcut.  S2 therefore compares two
different natural mentions after replacing their subject and value spans with
placeholders.  An external session ledger supplies entity and value equality;
the model predicts relation equality only.

The physical operation is fixed:

```text
duplicate iff same_entity and same_relation and same_value
update    iff same_entity and same_relation and not same_value
retain    otherwise
```

The native In-Place TTT `ttt_proj`/`ttt_conv` writer is not changed by this
experiment and remains a separate downstream ablation.

## Data and construction

| Split | Source | Relation pairs | Ledger scenarios | Relations |
|---|---|---:|---:|---:|
| weight training | FewRel train | 19,200 | 48,000 | 48 |
| calibration | SemEval train | 10,722 | 26,805 | 17 |
| one-shot development | FewRel 2.0 PubMed validation | 2,000 | 5,000 | 10 |

Every anchor has one distinct same-relation mention and one
different-relation mention from the same 0.05 masked-trigram similarity bin.
Each pair expands into the five registered ledger cases.  Exact masked-text
positive coverage is zero.  A calibration-fitted trigram rule obtains zero
relation and deletion recall on PubMed at 100% precision.  The independent
data audit passed all checks over 111,727 rows.

Authoritative compact artifacts:

- `data_manifest.json`, SHA256
  `28b67effc911da68eec9fd74bfd964f9f9a9193b28b19755ed73cc473212584f`;
- `data_audit.json`, SHA256
  `cf74771c3111442a459f7461fd082724509dae2faf68c1621615e628ec370083`.

SemEval test was never parsed.

## Model and training

The backbone is the final 1B-token Qwen3-4B checkpoint.  Training freezes the
backbone and updates rank-16 LoRA adapters on attention `q/k/v/o` projections.
The input contains only two masked natural mentions.  The signed NLL
difference between the canonical `same_relation=true` and `false` suffixes is
the deployed score.

The formal two-GPU run completed 300 steps in 402.82 seconds.  Checkpoints at
0.25, 0.50, and 1.00 epoch each contain 288 LoRA tensors and no backbone
tensors.  Peak allocated memory was 9.36/9.52 GiB and peak reserved memory was
18.62/17.90 GiB.  The independent training audit passed.  Its manifest SHA256
is `9aea5279fc946ea9a6d344dd75285855081ee93e829b775f4d20bfa42c24a69c`.

## Calibration

Calibration selected the 0.50-epoch adapter and score threshold
`0.46428704261779785`.  At the registered 99% deletion-precision constraint,
it reached 99.07% deletion precision but only 2.98% recall.  The raw score has
78.77% ROC-AUC and 70.34% zero-threshold accuracy, showing that relation signal
was learned while the high-precision tail remained poorly separated.

Frozen-Qwen retrieval on calibration reached 65.36/77.75/87.82% recall at
4/8/16.  The registered selector retained `k=16`.  The selection and its
independent audit both kept PubMed model outputs and SemEval test sealed.

## One-shot PubMed development result

The selected checkpoint, threshold, and retrieval depth were applied once and
unchanged to PubMed.

| Metric | Result |
|---|---:|
| retrieval recall@4/8/16 | 79.3 / 89.8 / **96.8%** |
| same-relation precision | 87.5% |
| same-relation recall | **8.4%** |
| same-relation specificity | 98.8% |
| relation macro balanced accuracy | 53.6% |
| delete precision | **93.33%** |
| delete recall | **8.4%** |
| false-positive deletes | 12 |
| hard different-relation specificity | 98.8% |

Trigram and frozen-Qwen cosine baselines made no positive deletion at their
calibrated safe thresholds.  The selected model improved recall by 8.4 points,
but six of the ten preregistered gates failed: false-positive count, deletion
precision, deletion recall, ten-point safe-baseline gain, relation macro
balanced accuracy, and same-relation recall.  The independent development
audit reconstructed all 2,000 relation pairs, metrics, hashes, and gates and
passed.  Result SHA256 is
`a2a84e0aa8ee727beeefd2e65271e2d45ffedf2e4ef6eeac8be7b7a26520e328`;
audit SHA256 is
`90b6b784ac827073f343ceea19650d1dbb31af502669382c72ceaf1dc36544d4`.

Terminal status is `stopped_by_development`; SemEval test remains unopened.

## Interpretation

S2 removes the S1 template shortcut and establishes two useful facts:

1. frozen Qwen retrieval transfers well enough to place a matching relation in
   the top 16 on 96.8% of PubMed queries;
2. an independently scored pairwise cross-encoder does not preserve the
   calibration safety/coverage trade-off across domains.

The current bottleneck is therefore relation-score aggregation and risk
calibration rather than candidate generation.  These results do not establish
that learned forgetting improves a downstream session task.  PubMed is now an
observed diagnostic source and may not be reused for architecture or threshold
selection.

## Reproduction

The original remote data, checkpoints, score traces, and logs remain under:

```text
/home/ctj/cbf_ttt_natural_relation_reranker_s2_data_20261015
/home/ctj/cbf_ttt_natural_relation_reranker_s2_train_20261015
/home/ctj/cbf_ttt_natural_relation_reranker_s2_calibration_20261015
/home/ctj/cbf_ttt_natural_relation_reranker_s2_development_20261015
```

The complete registered pipeline is:

```bash
bash scripts/run_natural_relation_reranker_s2_development.sh
```

All new entry points are opt-in and leave baseline training and evaluation
configuration unchanged.

## Completed, stopped, and next work

Completed: shortcut-safe construction, LoRA training, frozen retrieval,
calibration, one-shot development, resource recording, and independent audits.

Stopped by protocol: three-seed confirmation, SemEval test, and causal session
replay.

Next: use a newly registered natural supervision source and aggregate multiple
retrieved supports before any deletion decision.  The next stage must use a
fresh one-shot development split and may not repair S2 by tuning on PubMed.

