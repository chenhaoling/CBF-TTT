# Post-V20 Semantic Controller and Natural Transfer

## Freeze point and motivation

This protocol is frozen while V19 Qwen scoring is still running and before V19
or V20 outcomes are available. It is released only if the audited V20 semantic
oracle gate passes.

V20 is a headroom experiment. Its 20 hand-fixed alias families recur across
folds, so V20 must not be used to claim that a learned controller generalizes
to unseen semantic relations. V20 examples, latent slot IDs and oracle actions
are excluded from controller training and model selection.

## Development sources and separation

Use only already observed development sources:

- CounterFact relation-labelled records for semantic relation supervision;
- MQuAKE-T train for natural temporal-update supervision;
- MQuAKE-T tune for threshold and constraint selection;
- ZsRE calibration for value/readout calibration only.

Group every source record by relation ID, subject and source row. Outer folds
must be relation-disjoint and subject-disjoint. A relation family, subject,
source row or paraphrase pair may occur in exactly one outer fold. The existing
ZsRE held-out test, CounterFact outcome traces, MQuAKE dev, LongMemEval-S
outcomes and all LoCoMo data are forbidden for fitting or selection.

Before training, publish an inventory containing source hashes, relation and
subject counts, fold assignments, overlap matrices and excluded records.

### Source-inventory addendum frozen before V19 outcomes

The inventory implementation and its allocation rule were frozen when the V19
confirmation had written 148/320 rows and before any V19 aggregate, arm outcome
or causal-rule result was read.  Relations are assigned deterministically to
five folds.  When one normalized subject spans relations assigned to different
folds, keep only the records in the subject's most frequent relation fold and
publish every removed row hash and reason.  ZsRE is read only from the already
materialized calibration file; its dev and test files are not opened.

The audited public-source inventory exposes an unavoidable MQuAKE-T imbalance:
among the 1,494 allowed train+tune rows, P6 contributes 995, P35 contributes
481, and the other five relations contribute 18 in total.  A five-fold split
cannot be both relation-disjoint and source-balanced.  A count-only pilot found
that exact equal-relation weighting reduced effective sample size to 239--1,534
from roughly 13,500 rows and assigned a maximum row weight of 560; this rule was
rejected before model outcomes were loaded.  The fixed response uses every
outer-training row once per epoch and weights a row in relation-source group of
size `n` by `n^-0.5`, normalized to mean one.  This square-root temperature
keeps effective sample size above 80% and maximum weight below 32 in every
outer split while reducing dominant-group influence.  The deterministic inner
fold is `(outer + 1) mod 5`; neither outer nor inner rows receive a training
weight.  Report micro, subject-macro, relation-macro and source-stratified
results.  Rows or relations are not copied across folds.  The gate continues
to use the registered subject- and
relation-cluster intervals and every-fold condition; source imbalance cannot be
hidden by a micro-only result.

## Frozen semantic representation

Mask the subject span and value span before encoding a relation surface. Use
the frozen final Qwen3-4B checkpoint to extract mean-pooled relation-token
states. The learned semantic resolver is a rank-16 symmetric projection trained
with supervised contrastive loss:

- positives: different surfaces of the same relation family;
- hard negatives: same-subject different-relation surfaces;
- cross-source negatives: lexically similar surfaces with different relation
  IDs.

The Qwen backbone, tokenizer and slow weights remain frozen. Relation IDs are
training targets only and are unavailable at inference. Subject text, answer
values, row IDs, fold IDs and future queries cannot enter controller features.

## Frozen forgetting decision

At a new write, compare its masked relation representation with every active
historical component. For each component, combine:

- projected semantic cosine and margin to the next candidate;
- value agreement/conflict;
- relative age and repetition count;
- current and historical representation entropy;
- component access count when available.

The current update is always retained. A two-head 64-unit MLP predicts stale
utility and locality risk. Use three seeds, nested five-fold relation/subject
OOF, epochs 25/50/100, utility thresholds 0.4/0.5/0.6 and risk penalties
0/0.5/1/2. Loss weights and constraint selection match the frozen V19
fallback. Combined retained states must be rescored by Qwen; action labels
alone are insufficient.

## Frozen comparisons and development gate

Report retain-all, global reset, last-1/last-2, exact string deletion, raw Qwen
cosine, projected semantic cosine, learned utility-risk control and a causal
oracle. Primary endpoints are joint update-and-retention success and locality;
secondary endpoints include top-1/MRR, NLL, free generation, deletion rate,
latency and peak memory.

Natural transfer is released only if the controller satisfies all:

1. joint gain over the strongest non-oracle baseline at least five points;
2. subject- and relation-cluster 95% interval lower bounds above zero;
3. no negative outer fold or seed;
4. update and retained-fact accuracy at least 90%;
5. locality and unrelated-session retention at least 97%;
6. at least 40% of the positive baseline-to-oracle gap closed;
7. at least two points over projected-cosine deletion with a positive
   relation-cluster lower bound;
8. every corpus, split, feature, checkpoint, prompt, resource and outcome audit
   passes.

Failure stops semantic-controller iteration on these observed development
sources. It cannot be rescued with LoCoMo, LongMemEval-S, MMLU or RULER.

## One-time natural confirmation

After a passing development audit, freeze all weights and thresholds. Score:

- LoCoMo as the untouched long conversational-memory confirmation;
- RULER and RULER/QA for long-context retention and distractor robustness;
- MMLU as a slow-knowledge non-regression check.

LongMemEval-S is already observed and remains diagnostic only; it cannot tune
or release this stage. Report LoCoMo session- and conversation-cluster
intervals, exact deletion-count matched random controls, answer generation,
current/stale evidence exposure, non-target retention, throughput and peak
memory. No benchmark may be rescored after a failed registered gate.

## Claim boundary

A passing chain supports a learned, causal semantic-forgetting mechanism on a
readable contextual/explicit memory substrate. It does not establish readable
content binding in the original dense `ttt_proj/ttt_conv` writer. That negative
native-writer result remains unchanged.
