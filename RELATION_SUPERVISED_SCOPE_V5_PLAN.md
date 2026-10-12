# Relation-supervised scope V5 preregistration

## Fixed evidence before V5

Scoped Delta Ranker V4 improved CounterFact candidate top-1 from 79.62% to
82.93%, with a paired-bootstrap 95% confidence interval of +0.64 to +5.86
points. It still failed the registered 88% top-1 and 93% MRR gates. Of its 134
errors, 95 selected a different relation for the correct subject; all 95 had a
different CounterFact relation label. This motivates direct relation
supervision rather than another deletion-gate iteration.

## Protocol

- The final Qwen3-4B 1B-token checkpoint stays frozen.
- CounterFact relation IDs supervise a linear classifier over the concatenated
  three frozen 256-dimensional contextual projections.
- Relation IDs are targets only. Inference receives text and the supplied
  subject span, never a relation ID, chain ID, target value, candidate age, or
  future query.
- Text is cropped to a symmetric 10-token window around the subject and the
  subject is replaced by `<ENTITY>`.
- Each outer CounterFact session fold excludes all raw CounterFact rows whose
  subject occurs in that fold. The inner scoring-rule fold is also excluded
  while selecting among four fixed compatibility functions.
- Three initialization seeds, ten full-batch epochs, and class-balanced cross
  entropy are fixed before scoring.
- Candidate search is constrained to matching supplied subjects. This remains
  an oracle entity-segmentation diagnostic and is not a final deployable claim.

## Fixed compatibility arms

1. dot product of query/candidate relation posteriors;
2. cosine of relation logits;
3. candidate probability for the query's most likely relation;
4. cosine of the frozen relation-window embeddings.

The inner fold selects one arm by candidate top-1, MRR, and then the fixed order
above. No outer-fold metric changes the selected arm.

## Gate

V5 passes only if outer OOF predictions satisfy all of:

- candidate top-1 at least 90%;
- MRR at least 95%;
- query relation classification at least 90%;
- at least five points top-1 gain over V4;
- paired-bootstrap lower confidence bound above zero.

A pass authorizes automatic entity extraction and open-scope distillation. It
does not directly authorize sealed benchmarks or causal forgetting training.

