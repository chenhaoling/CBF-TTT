# Qwen pair-localized relation memory S4

## Frozen motivation and hypothesis

The independently audited S3 calibration result obtained 71.79% relation
retrieval recall@16, 51.65% direct-pair ROC-AUC, and 0.812% physical deletion
recall at 100% precision. Its full-dialogue mean-pooled representation did not
isolate evidence for a queried entity pair. DialogRE dev and test remain
unparsed.

S4 tests one causal change: construct relation memory from pair-conditioned
local evidence windows. The hypothesis is that attention over relation-bearing
turns will improve both support retrieval and pair equivalence while preserving
the same ledger rule and risk constraint. S4 does not learn the current-write
gate; `g=1` remains fixed. It is an enabling stage for the paper's alpha-only
session-memory forgetting experiment.

## Registered source inspection

Before model or pair construction, a train-only aggregate inspection was
performed with source SHA256
`94b5799642107f9afecc5c46655d6d9612f41969893deff291b15a66298c08a0`.
Among 4,699 positive annotations, 2,233 (47.52%) have at least one nonblank
trigger found in the dialogue. The fixed folds contain 1,815 train and 418
calibration trigger-bearing annotations. There are 27 trigger-bearing
relations shared across the two folds, two train-only relations, and three
calibration-only relations.

The shortest entity interval covers any annotated trigger for 43.89% of the
trigger-bearing examples. Adding one and two surrounding turns reaches 54.37%
and 65.34%. These values register the deterministic localization baselines and
show that explicit trigger labels alone cannot cover the primary relation task.

## Data and sealing

S4 reuses the exact S3 FewRel train rows, DialogRE-train dialogue folds,
relation caps, masked arguments, pair balance, physical scenarios, and source
hashes. The S3 support split remains authoritative. No example may move between
train and calibration.

For each DialogRE source annotation, the builder emits:

1. every dialogue turn as a candidate centered window with radius one;
2. a deterministic nearest-entity interval at radii zero, one, and two;
3. the full masked dialogue for the unchanged S3 control;
4. trigger-bearing turn indices in an audit/label field only;
5. the relation and physical-action rows already registered by S3.

If either argument has no exact turn match, the deterministic nearest-entity
control falls back to the full dialogue and records the fallback. Learned
candidate attention still receives every turn and requires no such fallback.

Argument strings are replaced with `[ARG1]` and `[ARG2]` before model input.
The relation name, trigger string, source split, dialogue ID, entity string,
and gold trigger indices are prohibited from all inference-visible features.
FewRel mentions are single-window records and remain part of relation-head
training. PubMed unsupervised views initialize the same frozen selected S3
projection arm; PubMed validation is not loaded.

DialogRE dev may be opened once only after all calibration artifacts and an
independent audit are frozen. DialogRE test, SemEval test, LongMemEval-S, and
LoCoMo remain sealed until the development gate passes.

## Pre-model audit

An independent builder auditor must verify:

- all source/output hashes and zero dev/test rows parsed;
- exact reuse of the S3 dialogue-disjoint train/calibration assignment;
- reconstruction of every candidate window from raw train dialogue;
- exact argument masking with no entity or relation text in model fields;
- gold trigger fields absent from the serialized inference input;
- candidate completeness and turn-index bounds;
- relation caps, pair balance, scenario completeness, and source-instance
  disjointness inherited from S3;
- an exact independent rerun of the registered raw-source entity-window
  coverage, plus separately reported coverage on the capped S3 subset;
- no S2/S3 model score in constructed data.

Any failure permits only a construction repair and a new output root. It may
not change the model grid or release thresholds.

## Frozen representation extraction

Use the final 1B-token Qwen3-4B checkpoint with native TTT disabled. Each
candidate is formatted as an argument-role header followed by its masked turn
window. Mean-pool final non-padding hidden states and L2 normalize. Extract
full-dialogue embeddings through the unchanged S3 path and candidate-window
embeddings through a new shard-safe cache. Any truncation is a hard error; the
maximum length and batch size are frozen after tokenizer-only measurement.

The Qwen backbone remains frozen in S4. This isolates the information-source
change and avoids using a larger language model. `ttt_proj` and `ttt_conv`
remain excluded from the primary representation and are retained as a later
baseline ablation.

## Pair localizer and relation memory

For one annotation with candidate vectors `h_i`, a pair-conditioned scoring
head assigns `a_i`. Training uses two signals:

- **explicit localization loss:** binary focal loss on the 1,815 train-fold
  annotations with a locatable trigger; a candidate is positive when its
  centered window contains an annotated trigger turn;
- **relation multiple-instance loss:** soft attention over all candidates
  creates a bag vector for every relation annotation, followed by the same
  supervised contrastive and matched-negative margin losses used by S3.

Empty-trigger annotations participate fully in relation multiple-instance
learning. They are excluded only from explicit localization loss. The head
may use candidate embeddings, masked argument-marker occurrence flags,
relative turn distance, and normalized position. It may not use entity
identity, trigger text, relation ID, or gold trigger indices.

To isolate localization, reuse the selected S3 projection settings: dimension
512, temperature 0.05, margin 0.20, and the PubMed-pretrained initialization.
The finite S4 grid is:

- deployed top-k in `{1, 3}`;
- trigger-loss weight in `{0, 0.5, 1.0}`;
- attention entropy weight in `{0, 0.01}`.

This gives 12 learned arms. All arms use AdamW at learning rate `1e-3`, weight
decay `1e-4`, gradient norm 1.0, and save epochs 10/25/50. Selection maximizes
all-example calibration recall@16, then direct-pair ROC-AUC, then trigger top-k
recall, smaller k, lower trigger weight, and earlier epoch. Physical deletion
thresholds do not select the localizer.

## Required controls and upper bound

The same calibration rows evaluate:

1. frozen S3 full-dialogue representation;
2. random length-matched turn windows;
3. nearest-entity windows at radii zero, one, and two;
4. learned multiple-instance-only localization;
5. learned explicit-trigger-only localization;
6. learned hybrid localization, the candidate deployable arm;
7. learned-local plus full-dialogue fusion;
8. gold-trigger-centered radius-one oracle, reported only on examples with
   locatable train annotations.

The oracle is never eligible for deployment or fusion selection. It estimates
whether local evidence can be represented by the frozen Qwen checkpoint.
Its projection starts from the selected S3 PubMed-pretrained weights, trains
with the same supervised contrastive and matched-negative objectives, and is
evaluated at epochs 10/25/50. The oracle checkpoint is selected by eligible
calibration recall@16, then direct-pair ROC-AUC, then earlier epoch. This rule
is frozen before extracting or scoring any oracle representation.

## Calibration metrics and stop gates

Localization metrics are reported on the 418 trigger-bearing calibration
annotations: trigger-turn recall@1/@3, all-trigger coverage, average selected
turn count, and dialogue-length compression. Relation and deletion metrics are
always reported on the complete S3 calibration set, including examples with
empty triggers.

S4 may proceed to fusion only if the learned hybrid arm satisfies all of:

- trigger-turn recall@3 at least 80% and at least 15 points above the
  registered radius-two deterministic baseline;
- all-example relation retrieval recall@16 at least 90%;
- all-example direct-pair ROC-AUC at least 70%;
- at least five points recall@16 improvement over full-dialogue S3;
- oracle direct-pair ROC-AUC at least 75% on its eligible subset.

If the oracle fails, S4 stops with `localized_representation_failure`. If the
oracle passes and learned localization fails, S4 stops with
`localizer_failure`. A failed gate cannot be repaired by enlarging the head
grid or reading DialogRE dev.

## Risk fusion and physical forgetting

The selected local representation replaces the S3 full-dialogue embedding in
the otherwise unchanged registered support and fusion pipeline. Add only
localizer confidence, top-1/top-2 attention margin, selected-window agreement,
and local/full representation agreement. Refit the monotone and MLP heads on
train and select on calibration by maximum deletion recall, then macro
balanced accuracy, then the monotone head.

The fixed action remains:

```text
duplicate iff E=1, R=1, V=1, and relation confidence is safe
update    iff E=1, R=1, V=0, and relation confidence is safe
retain    otherwise
```

The deployed threshold is again the maximum of the empirical 99%-precision
threshold and the dialogue-group 99% conformal negative threshold. Before
opening dev, calibration must reach at least 99% physical deletion precision,
30% deletion recall, 70% same-relation recall, 97% specificity, and 75% macro
balanced accuracy.

## One-shot DialogRE-dev gate

After weights, thresholds, raw predictions, and independent audit are frozen,
open official DialogRE dev once. Every condition must hold:

- relation retrieval recall@16 at least 95%;
- same-relation recall at least 70% and specificity at least 97%;
- relation macro balanced accuracy at least 80%;
- physical deletion precision at least 99%, recall at least 40%, and at most
  four false-positive deletes;
- deletion recall at least 20 points over S2 direct transfer and ten points
  over every safe non-adapted baseline;
- localization/support evidence adds at least five recall points over the same
  head using full-dialogue evidence at the identical precision constraint;
- dialogue-bootstrap 95% lower confidence bound of the recall gain over the
  best safe baseline is positive.

Failure leaves test and long-session benchmarks sealed. Passing releases the
fixed seeds `35117/35119/35121`; every seed must retain at least 99% deletion
precision and 35% recall.

## Conditional alpha-only experiment

Only a confirmed S4 pass releases counterfactual session labeling. Keep `g=1`
and evaluate `alpha in {0, 0.5, 1}` on small natural sessions first, using
sparse boundary sampling. Record wall time and peak memory for every label
before deciding the formal scale. Train the alpha controller on
inference-visible local relation evidence, recency, repetition, surprisal,
memory/update norm, and alignment; future questions and answers are forbidden.

Compare retain-all, global reset, last-1, last-2, recency, S2, S3, S4 discrete
deletion, original `ttt_proj`/`ttt_conv`, and oracle alpha. After calibration
and three-seed confirmation, evaluate ZsRE update/locality, MMLU retention,
RULER and RULER/QA retrieval, LongMemEval-S, and LoCoMo. These benchmark test
sets do not select controller settings.

## Resource pilot

Before formal S4 training, run two steps and 1,000 local-support queries on two
RTX 5090 GPUs. Record per-GPU time, throughput, peak allocated/reserved memory,
candidate-cache size, and projected runtime. Formal execution requires finite
losses, both GPUs participating, less than 28 GiB peak reserved memory per GPU,
and less than eight projected hours per seed.
