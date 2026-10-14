# Source-disjoint selective forgetting on FewRel

## Motivation and frozen decision

The CounterFact/MQuAKE symmetric pair cross-controller reached 83.56% OOF
balanced accuracy and 0.8796 AUROC, but lexical-hard-negative specificity was
72.89%.  Its registered release gate failed, so no further controller or
threshold selection is allowed on those sources and their 320 model outcomes
remain sealed.  This protocol moves development to a new human-annotated
relation source and changes the deployable claim from exhaustive binary
deletion to conservative selective deletion.

This document is frozen before constructing FewRel pairs, extracting Qwen
features, or observing any FewRel controller score.

## Immutable public source

Use the official THU FewRel repository at commit
`278a2315d2138810a379cd8d5718914dc56e2582`:

- `data/train_wiki.json`: SHA256
  `8e9e58a05d559be773cbe18a8d4a16ba17e6d397663f9f05a056f894a94b92c1`,
  64 relations and 44,800 instances;
- `data/val_wiki.json`: SHA256
  `76bb44c2485b6faed358a3ce040eadafcc8a8835245cd7e6afc1bef672d4a0bf`,
  16 relations and 11,200 instances;
- `data/pid2name.json`: SHA256
  `eeedd25512a8e4c2bc06b78009836d27de81807bcba6ed929debd5cd713c07ea`.

No relation description or Wikidata identifier is model input.  Relation IDs
are supervision and audit labels only.

## Frozen source partitions

Sort the 64 training relation IDs by SHA256 of
`2311:relation:<relation_id>`.  Assign the first 48 to `train`, the next eight
to `development`, and the final eight to `internal_test`.  The official 16
validation relations form `external_test`.  No relation crosses partitions.

Remove every instance whose head entity occurs in more than one partition.
The head is the fact subject and is replaced by `[SUBJECT]`, so this is a
strict subject-disjoint safeguard.  Tail entities may repeat because every
tail mention is replaced by `[VALUE]`.  Remove surfaces shared across
partitions and surfaces associated with multiple relations inside one
partition.  Deduplicate within each relation, then retain at most 300
instances per relation by the fixed seed.  Publish every exclusion and require
at least 250 retained instances for every relation.

`internal_test` and `external_test` pair labels are mechanically constructed
but their model features and scores stay unopened until the corresponding
gate permits access.

## Pair construction

Replace all annotated head spans by `[SUBJECT]` and all tail spans by
`[VALUE]`; do not use entity names or IDs in model text.  For every retained
anchor, construct two positives from different subjects with the same relation
and two negatives:

1. a same-subject/different-relation negative when available, otherwise the
   most lexically similar different-relation surface among a deterministic
   trigram candidate pool;
2. an independently hashed random different-relation negative.

Deduplicate unordered surface pairs.  Record pair kind, relation and entity
metadata only in the audit record.  Each partition must be class balanced to
within two percentage points and must contain positive, lexical or
same-subject hard-negative, and random-negative examples.

## Frozen representation and controller

Reuse the already audited symmetric Qwen3-4B cross prompt and final-token
representation.  Encode both statement orders with the final 1B-token
checkpoint, native TTT disabled, BF16, fresh KV and no gradient.  Extraction
is label-free.  The controller input is:

- the 2,560-dimensional symmetric cross representation;
- the frozen one-token `yes - no` logit margin;
- character-trigram similarity.

Train the existing shared 64-unit utility/risk head only on `train`.  Select
epoch and thresholds only on `development`, with three seeds
`23031/23037/23043`.  Neither test partition may influence architecture,
optimizer, epoch, loss weight, calibration, checkpoint or threshold.

Report character similarity, zero-shot margin, the frozen
CounterFact/MQuAKE controller, a newly trained full binary controller, and a
calibrated selective controller.  Use relation-macro metrics and bootstrap
relations, rather than treating pairs as independent experimental units.

## Selective decision and gates

The semantic label is `delete-compatible`: the two masked statements express
the same relation slot.  The selective policy has three outputs:

- `delete` above the high-confidence threshold;
- `retain` below the low-confidence threshold;
- `defer` between them.  In causal memory evaluation, `defer` conservatively
  retains the historical component.

Two claims are kept separate:

### Full binary gate

On development, require at least 80% positive recall, 95% negative
specificity, 0.85 AUROC and 0.80 balanced accuracy for every seed.  Passing
releases one evaluation on `internal_test`.

### Selective deletion gate

This is a narrower safety claim, not a replacement for a failed full binary
gate.  Choose the largest-development-coverage threshold whose relation-
cluster bootstrap 95% upper bound on false deletion is at most 5%.  Also
require deletion precision at least 90%, positive recall at least 25%, and at
least 20% total non-deferred coverage for every seed.  Passing releases one
evaluation on `internal_test`; all criteria are then applied unchanged.

For either claim, `internal_test` additionally requires every relation to have
non-negative balanced-accuracy gain over the strongest non-oracle baseline
and a relation-cluster 95% lower bound above zero for the aggregate gain.

## Causal memory stage

Only an audited internal-test pass may construct FewRel memory episodes.  Use
real relation sentences for retained facts and deterministic within-relation
tail permutations for controlled conflicting updates.  Compare no forgetting,
global reset, last-1/last-2, lexical similarity, frozen old controller, new
selective controller and relation-ID oracle on joint update-and-retention
success, update, retention, false deletion, latency and memory.

The official `external_test` relations remain unopened until the causal
development gate passes.  CounterFact/MQuAKE outcomes, ZsRE, LoCoMo,
LongMemEval-S, RULER/RULER-QA and MMLU cannot be used for this stage's model or
threshold selection.

## Stopping rule

If neither full binary nor selective deletion passes on `internal_test`, stop
feature/head tuning on FewRel and report the representation limitation.  Do
not open `external_test` and do not relax the registered thresholds.  A pass
allows exactly one frozen external relation evaluation followed by the causal
memory study.
