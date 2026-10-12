# Token-span relation scope V11 preregistration

## Evidence and hypothesis

V10 reached 94.65% top-1 and failed its registered gate. Of its 42 errors, 41
selected the correct subject but the wrong factual relation. Oracle subject
segmentation in V5 resolves 32 of those 42 errors and reaches 97.83% top-1.
V11 tests whether a frozen-Qwen token classifier can recover the subject span
from text and use that span to form the de-identified relation representation
required by V5.

The token extractor is therefore a relation preprocessing module. V11 does not
continue V8/V9/V10 score fusion.

## Supervision and leakage controls

- Build binary token labels from exact CounterFact subject character spans in
  the requested-rewrite prompt, paraphrases, and constructed scoped-delta text.
- Cache a deterministic low-dimensional projection of frozen Qwen3-4B token
  hidden states. The language model remains frozen.
- Use the existing five subject-disjoint outer folds. Every raw or constructed
  example containing an outer-fold subject is excluded from that fold's token
  classifier training set.
- Select the token-head epoch and span threshold on the next held-out inner fold;
  exclude both outer- and inner-fold subjects from this selection model.
- At inference, the resolver receives text only. Subject strings, subject IDs,
  relation IDs, chain IDs, targets, candidate ages, and future inputs are
  unavailable.
- CounterFact is observed development-only. LongMemEval-S and LoCoMo stay sealed.

## Model and scoring

For each token hidden state `h_i`, a small MLP predicts whether the token belongs
to the factual subject. Consecutive above-threshold tokens form candidate spans;
the highest-scoring run is selected, with the highest-scoring single token as a
deterministic fallback. The selected character span is replaced by `<ENTITY>`
and a ten-token relation window is passed through the frozen, fold-specific V5
relation classifier.

Two registered scoring arms are selected on the inner fold:

1. `hard_equal`: restrict relation ranking to candidates whose predicted subject
   token sequence exactly matches the query prediction when such a candidate
   exists;
2. `overlap8`: add eight times the predicted-span token Jaccard similarity to
   log relation compatibility, reusing V7's fixed boundary value.

V11 reports span exact match, token F1, relation-only candidate ranking, and the
selected primary ranking. The oracle V5 result remains a diagnostic upper bound.

## Gate

V11 passes only if all conditions hold on five-fold OOF predictions:

- primary top-1 accuracy is at least 96%;
- MRR is at least 98%;
- top-1 gain over V10 is at least one point;
- the paired-bootstrap 95% CI versus V10 has a positive lower bound;
- top-1 is within two points of oracle-segmented V5;
- no inference-time subject or relation metadata is used.

A pass freezes the resolver and starts causal on-policy forgetting-policy
experiments. A failure triggers an error audit before any new architecture is
registered. Existing-score fusion remains permanently stopped in either case.
