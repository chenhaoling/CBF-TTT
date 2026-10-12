# Relation-supervised scope V5

## Objective

Test whether relation supervision resolves V4's same-subject/different-relation
failure. CounterFact relation IDs are class targets only; inference receives
text and the supplied subject span, not relation IDs. Every outer and inner fold
excludes all raw rows belonging to the evaluated subjects.

## Method

The frozen final Qwen3-4B checkpoint encodes 43,853 unique relation windows from
65,757 source examples. Each window replaces the supplied subject with
`<ENTITY>` and keeps ten tokens on either side. Three 34-way linear classifiers
train for ten epochs. An inner fold selects among posterior dot product, logit
cosine, query-class probability, and frozen-embedding cosine. All five folds
selected posterior dot product.

## CounterFact OOF result

| Metric | V4 | V5 |
|---|---:|---:|
| Candidate top-1 | 82.93% | **97.83%** |
| MRR | 90.91% | **98.92%** |
| Query relation class | — | 87.64% |

V5 improved candidate top-1 by **14.90 points**, with paired-bootstrap 95% CI
**[12.23,17.58]**. Only 17/785 candidate decisions were wrong. In contrast,
97/785 query relation classes were wrong; 82 of these still selected the
correct candidate. Only two candidate errors occurred when the query relation
class itself was correct.

This separation shows that pairwise compatibility is substantially easier and
more relevant than recovering the exact Wikidata relation identifier.

## Gate and next decision

Top-1, MRR, gain, and positive-CI checks passed. The preregistered auxiliary
query-relation threshold was 90%, while the result was 87.64%; the complete V5
gate therefore failed. Causal forgetting remains stopped.

V6 removes the supplied subject span at inference. It identifies a candidate
specific shared entity span using training-corpus IDF and combines that entity
evidence with V5's posterior relation compatibility. This directly tests
whether the 97.83% result survives automatic, text-only scope resolution.

## Resources and claim boundary

- Runtime: 103.86 seconds on one RTX 5090.
- Peak CUDA allocated/reserved: 8.693/10.604 GiB.
- Formal code commit: `51bbabc`.
- LongMemEval-S and LoCoMo remained unscored.

The result supports relation-supervised pairwise candidate compatibility under
oracle subject segmentation. It does not establish automatic segmentation,
unseen-ontology transfer, native fast-weight deletion, causal query gains, or
sealed benchmark performance.

