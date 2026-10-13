# MQuAKE Candidate Diagnostic V9 Plan

## Objective

Localize V8's 28.82% shortlist failure without training another deletion head or
reading development. The diagnostic uses the frozen V8 train-fitted candidate heads
and MQuAKE tune only.

## Arms

- full-list learned factorized ranker;
- all-unit maximum-cosine shortlist at widths 4, 8, 16, and 32, reranked by the
  frozen learned ranker;
- explicit scope-anchor cosine shortlist at the same widths, reranked identically;
- raw all-unit maximum and raw scope-anchor ranks as mechanism controls.

The scope anchor is the explicit first field emitted by the V7 session serializer. It
is a declared data field, not an inferred subject ID, relation ID, age, or target label.

## Gate

A resolver is eligible for another causal run only if tune target recall is at least
90% and top-1 is at least 80%. Among eligible arms, select the highest top-1, then the
smallest shortlist, preferring anchor over all-unit maximum at a tie. If no arm is
eligible, stop the current session serializer and redesign the memory key. MQuAKE dev,
MQuAKE-CF, LongMemEval-S, and LoCoMo remain unscored.
