# Qwen retrieval-evidence forgetting S3

## Frozen motivation

S2 achieved 96.8% same-relation retrieval recall@16 on one-shot PubMed, but
the independently calibrated pairwise score obtained only 8.4% relation recall
and 93.33% deletion precision.  Candidate discovery is sufficient for a new
stage; pairwise score transfer and safe-deletion calibration are not.

S3 therefore changes the learned object before reading any new source.  It
uses a support bank and aggregates evidence from multiple natural relation
mentions.  It does not tune S2 on PubMed.  PubMed validation, CounterFact,
MQuAKE, ZsRE, and every earlier controller trace are development-only evidence
and cannot select S3.  SemEval test, DialogRE test, and LoCoMo remain sealed.

S3 remains an enabling semantic-scope experiment.  The paper contribution is
session-memory forgetting.  Only a passed S3 release gate may construct new
counterfactual forgetting labels or train an `alpha` controller.

## Data registration and sealing

### Weight training

1. Retain the 48 registered FewRel training relations and all S2 source hashes.
2. Add official DialogRE v2 English train from
   `https://github.com/nlpdata/dialogre`, path
   `data_v2/en/data/train.json`.
3. Use the already downloaded but unparsed official FewRel 2.0
   `pubmed_unsupervised.json` only for a registered self-supervised arm.  Its
   SHA256 is
   `fa3555941364024c2ac1b1857396515ecc65a515bc742e9da0502568fad5dd1f`
   and its byte size is 1,093,301.

The DialogRE repository commit and train/dev/test file hashes must be recorded
before parsing any JSON.  DialogRE test may be hashed but never parsed during
S3.  If the unsupervised PubMed schema does not expose two entity spans or a
stable row identifier, the builder must record the arm as unavailable and
stop that arm; it may not silently synthesize labels from the observed PubMed
validation set.

### Calibration and fresh development

DialogRE train is divided by dialogue ID into five deterministic groups with
seed `35117`.  Four groups fit weights and the fifth selects checkpoint,
fusion regularization, and a safety threshold.  Group assignments are fixed
before model execution and no dialogue, entity pair, or exact masked text may
cross groups.

Official DialogRE v2 English dev is a one-shot development split.  It may be
opened only after the complete training/calibration selection and independent
audit are frozen.  Official DialogRE test and LoCoMo remain sealed even if dev
passes; dev first releases three fixed training seeds.

DialogRE is multi-label.  Each `(dialogue, argument pair, relation)` tuple is a
positive mention.  Pair construction excludes the no-relation label.  A
same-relation pair must come from different dialogues.  A negative pair is
matched to the positive by 0.05 masked character-trigram bins and argument
type compatibility when types are present.  Argument strings are replaced by
`[ARG1]` and `[ARG2]` before any model or lexical baseline sees the text.

## Pre-model integrity and shortcut gate

An independent auditor must verify:

- source repository commit and all file hashes;
- zero DialogRE test rows parsed;
- dialogue-, argument-pair-, and exact-text-disjoint folds;
- distinct natural texts for every positive pair;
- relation labels and multi-label expansion against the raw train source;
- equal positive/negative counts within occupied similarity bins;
- no model output or old controller score in constructed data;
- exact masked-surface positive coverage is zero;
- trigram and BM25 each have at most 20% relation recall at 95% specificity on
  the held-out DialogRE-train calibration group.

An integrity failure may repair construction and rerun the audit.  A shortcut
failure stops S3 before GPU training and requires a new construction plan.

## Model

### Frozen representation and support bank

Use the final 1B-token Qwen3-4B checkpoint with native TTT disabled.  Mean-pool
the final non-padding hidden states of every masked mention and L2 normalize.
The support bank contains only the current outer training groups.  For each
query mention retrieve the top 16 supports, excluding its dialogue and paired
mention.  Report recall@4/8/16 and MRR, but retrieval alone never authorizes
deletion.

### Supervised contrastive projection

Train a two-layer residual projection head on frozen Qwen embeddings.  Each
batch contains multiple relations and at least two dialogues per relation.
Use supervised contrastive loss with temperature grid `{0.05, 0.10}`, plus a
margin loss on the registered trigram-matched negatives.  Candidate dimensions
are `{256, 512}` and margin values `{0.10, 0.20}`.  This finite grid is selected
on the held-out DialogRE-train group only.

The PubMed-unsupervised arm, if schema-eligible, applies two deterministic
entity-preserving views of the same row and InfoNCE before supervised fitting.
The paired ablation skips this stage.  PubMed validation labels and S2 scores
are never loaded.

### Multi-evidence fusion and risk

For a candidate pair, compute inference-visible features only:

- projected direct cosine and raw frozen-Qwen cosine;
- S2 cross-encoder score as a frozen prior;
- reciprocal top-16 support overlap;
- dot product and Jensen-Shannon distance of neighbor relation posteriors;
- top-1/top-2 margins, entropy, support agreement, and score variance for both
  mentions;
- best, median, and trimmed-mean cross-support similarity;
- argument-type compatibility when the source itself supplies types.

Fit a monotone logistic fusion head and an unconstrained two-layer MLP as a
registered pair.  Training uses balanced relation labels.  Selection first
maximizes recall subject to 99% physical deletion precision on calibration;
ties use relation macro balanced accuracy, then the smaller head.  A conformal
lower confidence bound over dialogue groups supplies the deployed safe-delete
score.  No entity name, relation ID, gold trigger, or split identifier is a
model feature.

## Physical forgetting interface

The session ledger supplies normalized entity and value equality only.  The
semantic module supplies `same_relation` and its conservative confidence.
The fixed action is:

```text
duplicate iff E=1, R=1, V=1, and relation confidence is safe
update    iff E=1, R=1, V=0, and relation confidence is safe
retain    otherwise
```

For the later forgetting-policy stage, current-write gate `g=1` remains fixed.
S3 may release only evidence for choosing historical-memory forgetting
`alpha`; it may not introduce a learned current-write gate.

## Development metrics and baselines

Report frozen Qwen, lexical trigram, BM25, S2 direct pair score, projected
cosine, support aggregation, both fusion heads, and oracle relation.  Required
ablations remove S2 prior, support evidence, contrastive projection,
conformal risk, and PubMed self-supervision.

Primary measurements:

- relation precision, recall, specificity, F1, ROC-AUC, and macro balanced
  accuracy;
- retrieval recall@4/8/16 and MRR;
- physical delete precision, recall, false-positive count, and coverage;
- per-relation and per-dialogue bootstrap confidence intervals;
- latency, throughput, support-bank size, and peak allocated/reserved memory;
- every raw prediction and deterministic source/pair hash.

## One-shot DialogRE-dev release gate

Every condition must hold:

- retrieval recall@16 at least 95%;
- same-relation recall at least 70% and specificity at least 97%;
- relation macro balanced accuracy at least 80%;
- physical deletion precision at least 99%, deletion recall at least 40%, and
  no more than four false-positive deletes;
- deletion recall improves at least 20 points over S2 direct transfer and at
  least ten points over every safe non-adapted baseline;
- support aggregation improves deletion recall at least five points over the
  same projection without support evidence at the same precision constraint;
- the dialogue-bootstrap 95% lower confidence bound of the recall gain over
  the best safe baseline is positive.

If dev fails, DialogRE test and LoCoMo stay sealed.  Thresholds may not be
changed from dev errors.  The failure analysis may select only one of three
predeclared next causes: retrieval coverage, semantic evidence, or risk
calibration.  A further experiment must change the corresponding information
source, not repeat the same head search.

If dev passes, run fixed seeds `35117/35119/35121`.  Every seed must retain at
least 99% deletion precision and 35% recall.  Only then may the alpha-only
counterfactual session experiment be registered.

## Conditional alpha-only session experiment

This stage is not executed unless S3 passes.  Fix `g=1` and label natural
session boundaries with the counterfactual grid `alpha in {0, 0.5, 1}`.  The
controller may use S3 evidence, memory/update norm and alignment, pre-update
surprisal, repetition statistics, and recency.  It may not use future queries
or answers.  Compare learned alpha with retain-all, global reset, last-1,
last-2, recency, S2, S3 discrete deletion, and oracle under identical streams.

The primary causal outcomes are stale-fact removal, current-fact retention,
unrelated-memory retention, downstream answer accuracy, and counterfactual
utility.  The original `ttt_proj`/`ttt_conv` writer is a named ablation because
its earlier Correct-vs-Wrong/Empty content-readability gates failed; it is not
evidence for the primary readable-memory substrate.

## Resource pilot

Before formal training run two GPU steps and 1,000 support queries on two RTX
5090 GPUs.  Record wall time, examples/s, support-query latency, allocated and
reserved peak memory on each device.  Formal execution is allowed only when
all values are finite, each GPU participates, peak reserved memory is below
28 GiB, and the projected formal runtime is below eight hours per seed.

