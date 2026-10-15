# Retrieval-evidence forgetting S3 report

## Objective

S3 tested whether multiple natural relation mentions, a supervised contrastive
support bank, and calibrated multi-evidence fusion could supply a reliable
semantic-equivalence signal for session-memory forgetting. The current-write
gate was fixed to `g=1`; S3 did not train a write controller or modify the
baseline model.

The registered protocol is in
`QWEN_RETRIEVAL_EVIDENCE_FORGETTING_S3_PLAN.md`. All training and calibration
decisions used FewRel train, DialogRE train, and the registered unsupervised
PubMed rows. DialogRE dev/test, PubMed validation, SemEval test, and LoCoMo
remained sealed.

## Audited data

- 12,203 training mentions, 24,234 balanced relation pairs, and 60,585
  physical-action scenarios.
- 872 held-out DialogRE-train calibration mentions, 1,724 balanced pairs, and
  4,310 physical-action scenarios.
- 2,494 deduplicated PubMed rows with two entity-preserving self-supervised
  views.
- 79 train relation classes and 31 calibration relation classes.
- Character trigram and BM25 recall at 95% specificity were 5.22% and 6.50%;
  exact positive surface coverage was zero.
- The independent data audit passed over 41,527 rows. Manifest SHA256 is
  `0ed003c7e153ced83291e5654bd51217295b474970912c57315906bf70885078`.

## Representation and resource profile

The final 1B-token Qwen3-4B checkpoint was loaded with native TTT disabled.
All 14 `ttt_proj`/`ttt_conv` tensors were excluded from the frozen embedding
path. No mention exceeded the registered 1,024-token limit.

The projection resource pilot passed: two optimizer steps took 0.50 seconds,
1,000 support queries took 0.058 seconds, and peak reserved memory was 0.174
GiB after the cached Qwen embeddings had been loaded. The complete 48-arm
projection grid took 126.43 seconds and peaked at 0.162 GiB in the head-only
stage. Upstream Qwen extraction peaked at 9.79 GiB reserved per RTX 5090 and
the frozen S2 pair scorer peaked at 23.44 GiB.

## Projection result

The frozen selector chose PubMed self-supervision, dimension 512, temperature
0.05, margin 0.20, and epoch 25.

| Metric | Result |
|---|---:|
| retrieval recall@4 | 42.43% |
| retrieval recall@8 | 58.37% |
| retrieval recall@16 | 71.79% |
| MRR@16 | 30.19% |
| direct-pair ROC-AUC | 51.65% |
| positive mean cosine | 0.8931 |
| negative mean cosine | 0.8885 |

The no-PubMed arm reached 71.44% recall@16 and 51.74% AUC. Self-supervised
pretraining therefore changed neither failure mode materially.

## Fusion result

The registered selector chose the monotone full-evidence arm and threshold
`1.9021731615066528`.

| Metric | Result |
|---|---:|
| same-relation precision | 100.00% |
| same-relation recall | 0.812% |
| same-relation specificity | 100.00% |
| relation macro balanced accuracy | 50.35% |
| relation ROC-AUC | 53.02% |
| physical deletion precision | 100.00% |
| physical deletion recall | 0.812% |
| false-positive deletes | 0 |

The best diagnostic ablation was the unconstrained MLP over raw representations:
59.85% ROC-AUC and 4.06% deletion recall at 100% calibration precision. S2
direct scoring obtained 54.72% AUC and 1.51% safe recall. Projected cosine,
support posterior, no-S2, no-support, no-projection, no-conformal, and opposite
PubMed-pretraining arms all remained far below the registered release gates.

## Independent audit and decision

The projection auditor reconstructed the 48-arm grid, tie-breaking rule,
pretraining ablation, selected checkpoint, hashes, sealing flags, and resource
limits. All ten checks passed.

The fusion auditor reconstructed all 1,724 pair scores, empirical and
dialogue-conformal thresholds, pair confusion matrices, five physical actions,
macro balanced accuracy, arm selection, provenance, and hashes. All nine
checks passed. The raw-score SHA256 is
`8381c561c44ce60e30b549a026c681d6a42173b3a5262e96f72274ed1f2c2ed7`.
Both audits report zero DialogRE dev/test rows parsed.

S3 is stopped before DialogRE dev. Calibration recall@16 is 23.21 percentage
points below its 95% target, direct-pair discrimination is near chance, and
safe deletion recall is 0.812%. Further threshold or fusion-head search cannot
recover relation evidence absent from the full-dialogue embedding. The
registered next cause is **retrieval coverage / entity-pair evidence
localization**.

## S4 source finding

A train-only, aggregate inspection was completed before registering S4:

- 4,699 positive DialogRE annotations and 4,992 relation mentions;
- 2,233 annotations (47.52%) contain a nonblank trigger that can be found in
  the dialogue;
- 1,815 train-fold and 418 calibration-fold annotations have locatable
  triggers;
- the nearest entity interval with radius 0/1/2 covers any annotated trigger
  in only 43.89%/54.37%/65.34% of eligible annotations;
- 27 trigger-bearing relations occur in both train and calibration folds.

Consequently, S4 combines explicit trigger supervision with relation-level
multiple-instance learning over every positive annotation. Its frozen plan is
`QWEN_PAIR_LOCALIZED_RELATION_MEMORY_S4_PLAN.md`.

## Artifact inventory

- `data_manifest.json` and `data_audit.json`: constructed corpus and independent
  pre-model audit.
- `projection_resource_pilot.json`, `projection_selection.json`, and
  `projection_audit.json`: resource gate, complete projection grid, and audit.
- `fusion_selection.json`, `calibration_scores.jsonl`, and `fusion_audit.json`:
  complete fusion selection, raw per-pair scores, and independent reconstruction.
- `s4_source_inspection.json`: train-only trigger and deterministic-window
  feasibility statistics.

## Compatibility and limitations

No baseline configuration or native writer path was changed. All additions
are standalone task, audit, and artifact files. The result is a calibrated
negative finding on held-out DialogRE-train data, not a claim about official
DialogRE dev/test performance. The chief limitation is that only 47.52% of
positive train annotations provide explicit localized evidence; S4 must be
evaluated on all relation examples and cannot report trigger-bearing examples
as the primary task.
