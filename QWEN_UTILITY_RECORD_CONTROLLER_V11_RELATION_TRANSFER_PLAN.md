# V11 unseen-relation benefit-transfer development

## Frozen question

V10 added 672 naturalized FewRel training groups and produced a real but small
causal improvement on LAMA (+0.165 point versus its fixed removal ablation).
The frozen diagnostic showed that the remaining failure is benefit transfer:
FewRel reduced LAMA benefit average precision by 5.70 points, while the risk
head remained strong and FewRel was far from both target-domain centroids.
Further source weighting or threshold search on the already opened TempLAMA
and LAMA development sets is prohibited.

V11 asks whether relation-group robust benefit learning can transfer to wholly
unseen relations.  The current-write decision remains fixed at `g=1`; only
session-memory forgetting is learned.  The addressable Qwen residual memory is
the tested substrate.  The unchanged native `ttt_proj`/`ttt_conv` writer is a
negative baseline because its favorable single-record binding gate failed.

## Immutable FewRel partitions

Reuse the audited FewRel repository at commit
`278a2315d2138810a379cd8d5718914dc56e2582` and seed 2311 partition:

- 48 train relations: already used by V10 and available for fitting;
- 8 development relations: V11 model selection only;
- 8 internal-test relations: remain ungenerated until the development gate;
- 16 `val_wiki.json` external-test relations: remain ungenerated until the
  development gate and are scored once.

No internal/external test row, label, feature or relation ID may be loaded by
the V11 development pipeline.  Exact source rows, statements and masked
templates overlapping the existing TempLAMA/LAMA development episodes are
removed exactly as in V10.  Relations make the V10 train and V11 development
source rows disjoint by construction.

## Development corpus and counterfactual labels

For every one of the eight development relations, construct fourteen groups:
two repetitions of each of the seven registered lifecycle mechanisms.  Each
group has two counterfactual twins, three sparse boundaries, five future
queries per boundary, at most three candidate records and the unchanged
`alpha in {0, 0.5, 1}` action grid.  Natural FewRel templates are instantiated
with traceable subject/value substitutions exactly as in V10.  The target is:

- 112 source groups;
- 224 episodes;
- 672 labels; and
- 3,360 future queries.

The final 1B-token Qwen3-4B checkpoint and tokenizer are unchanged.  Labeling
records latency, answer coverage, cache statistics and peak CUDA memory.  The
corpus and labels must pass independent exact-reconstruction audits before
feature extraction.  Formal labels are usable only if answer coverage is at
least 95%, safe-oracle headroom is at least 1.5 points, positive safe-benefit
support is at least 5%, unrelated oracle errors are zero and peak reserved
memory is at most 28 GiB.

## Locked features and candidate training arms

Extract the unchanged 66 inference-safe V7 features.  V11 compares exactly
three 128/64 two-head controllers using the same seeds, optimizer, checkpoint
epochs and deployment grid:

1. `erm_equal`: the V10 equal-source loss over TempLAMA, LAMA and 48 FewRel
   train relations;
2. `relation_group_dro`: benefit loss is optimized with exponentiated
   relation-group weights, while risk loss remains the equal-source weighted
   loss;
3. `relation_group_dro_no_fewrel`: the same robust rule without FewRel, to
   identify whether unseen-relation support comes from the constructed corpus.

Group-DRO uses one group per `(source, relation)`, step size 0.05, weights
initialized uniformly and clipped to `[0.25, 4]` times their uniform value.
Within every optimizer step, each source contributes equally.  No source flag,
relation ID, counterfactual outcome or benchmark label enters the deployed
feature vector.  The exact-subject locality rule and V6 fact-key safety
projection remain mandatory.

Model/config selection uses only the eight new FewRel development relations.
TempLAMA/LAMA development results may be reported descriptively after the
choice but cannot change it.

## Development gate and sealed release

The branch advances only if an independent audit verifies all of the following:

1. all three selected robust seeds improve over lifecycle physics;
2. mean gain is at least 1.5 accuracy points;
3. source-group and relation-cluster bootstrap 95% lower bounds exceed zero;
4. every seed adds zero unrelated-query errors;
5. mean safe-oracle gap closure is at least 50%;
6. semantic-overwrite, stale-repetition and reset/revoke gains are nonnegative;
7. mean benefit average precision exceeds `erm_equal` by at least 0.03;
8. removing FewRel reduces mean gain by at least 0.2 point; and
9. every source, reconstruction, label, feature, checkpoint and result audit
   passes while both test partitions remain sealed.

A pass freezes one robust configuration and authorizes separately registered,
one-time 8-relation internal and 16-relation external tests.  A failure closes
this representation/loss branch without another search on these eight
development relations.
