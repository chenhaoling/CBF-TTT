# DocRED relation-description semantic slot space

## Frozen motivation

The source-disjoint FewRel controller reached 0.7680 development AUROC, and
its zero-false-deletion operating point retained only 0.19% positive recall.
Random and lexical-hard negatives had nearly identical specificity.  This
rejects threshold repair and further head tuning over the current frozen pair
features.  FewRel development tuning stops; internal and external FewRel test
features remain nonexistent.

This protocol is frozen before building a DocRED corpus, extracting a DocRED
feature, training a relation-space projection, or accessing a FewRel test
feature.  It tests a new representation hypothesis: align masked statements
to natural-language relation descriptions during training, then compare only
statement vectors at inference.

## Immutable sources and exclusions

Use the official `thunlp/docred` files:

- `train_annotated.json.gz`: SHA256
  `0d01cd07cabd7f9077db6ea8628832cf60281b4228ec1ba54647f836a3b17d02`;
- `dev.json.gz`: SHA256
  `6ae4d7f5b0b9d2cbe74b9634ed43b35b7cb5b7c0dc3a16226dbe343139a4ae05`;
- `rel_info.json.gz`: SHA256
  `7ef27efff537ba89ae66f6f4e60e4908d4df3860a4c2819ea94fa5ed696bdc70`.

The frozen FewRel manifest SHA256 is
`6f1dc9c9bacc934c5e44693b59a841220e18772ef0250eb64a5afb76aee77a1d`.
Exclude every relation assigned to FewRel `internal_test` or `external_test`
from all DocRED construction, description supervision and checkpoint
selection.  This exclusion is based only on public relation IDs and occurs
before a FewRel test feature or score exists.

## Relation and document partitions

Retain a non-target DocRED relation only if it has at least 50 annotated
training labels and 20 development labels with nonempty evidence.  This gives
67 eligible relations.  Sort them by SHA256 of
`2351:relation:<relation_id>` and assign the first 51 to
`representation_train`, the next eight to `relation_development`, and the
last eight to `relation_source_test`.

Use only `train_annotated` documents for representation training and only
official development documents for the two held-out relation partitions.
If one official development document contains eligible labels from both
held-out partitions, exclude all such labels from that document.  Require at
least 15 retained evidence instances per held-out relation.  No document,
relation or normalized masked surface may cross a partition.

## Masked evidence instances

For each annotated relation with nonempty evidence:

1. select the registered evidence sentences;
2. add the earliest head or tail mention sentence when the evidence does not
   contain that entity;
3. keep at most four sentences in document order;
4. replace all selected head mentions by `[SUBJECT]` and all selected tail
   mentions by `[VALUE]`;
5. join sentences with `[SEP]` and reject missing/overlapping markers.

Deduplicate normalized surfaces within each relation.  Retain at most 512
training and 128 held-out instances per relation by deterministic hash.  For
each retained anchor, construct two same-relation/different-document
positives, one lexical different-relation negative and one independent random
negative.  Deduplicate unordered pairs and balance each partition exactly.

Only descriptions for the 51 training relations are released to the training
artifact.  Held-out descriptions are audit metadata and cannot be encoded or
used by the model.

## Frozen Qwen representations

Use the final 1B-token Qwen3-4B checkpoint, native TTT disabled, BF16, fresh KV
and no gradient.  Encode each training/development statement with:

```
Identify the semantic relation between [SUBJECT] and [VALUE].
Evidence: <masked evidence>
Relation representation:
```

Encode each training relation description with:

```
Relation description: <official DocRED description>
Relation representation:
```

Use the normalized final-token state.  Relation IDs and descriptions are
never inference inputs.  Do not encode `relation_source_test` or either
FewRel test partition before their gate opens.

## Relation-description projection

Train rank-64 bias-free statement and description projections on the 51
training relations.  Use relation-balanced symmetric InfoNCE at temperature
0.07 plus a within-relation statement contrastive term and a 0.2
hard-negative margin.  Fixed seeds are `23531/23537/23543`; checkpoints are
25/50/100 epochs.  Select epoch and deletion thresholds only on the eight
held-out `relation_development` relations.

At inference, discard the description tower.  Map each masked statement
through the statement projection and compare two normalized projected vectors.
Baselines are raw statement cosine, character trigram similarity and the
frozen Qwen pair cross margin.  Report pair and relation-macro balanced
accuracy/AUROC, hard-negative specificity, five-percent false-deletion
operating point, latency and memory.

## Gates and one-time releases

Development must satisfy all of:

1. ensemble AUROC at least 0.85 and balanced accuracy at least 0.80;
2. every seed AUROC at least 0.82;
3. at least 80% delete recall with 95% retain specificity for the full binary
   claim, or at least 25% recall and 90% precision with a relation-cluster
   bootstrap false-deletion upper bound at most 5% for the selective claim;
4. at least a five-point balanced-accuracy gain over raw statement cosine;
5. no held-out relation below 0.60 AUROC;
6. all data, feature, checkpoint and independent reconstruction audits pass.

A pass permits one frozen `relation_source_test` extraction and evaluation
with unchanged checkpoints and thresholds.  A source-test pass under the same
thresholds then permits one extraction/evaluation of the still sealed FewRel
`internal_test` pairs.  FewRel external test, CounterFact/MQuAKE outcomes,
ZsRE, LoCoMo, LongMemEval-S, RULER/RULER-QA and MMLU remain sealed.

Failure at either gate stops this representation branch.  It cannot be rescued
by opening the next partition, changing relation eligibility, relaxing a
threshold, or training on FewRel development.
