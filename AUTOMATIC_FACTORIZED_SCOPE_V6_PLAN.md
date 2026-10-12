# Automatic factorized scope V6 preregistration

## Motivation

V5 reached 97.83% candidate top-1 but used a supplied subject span to crop
relation text and hard-filter candidates. Its exact 34-way relation
classification was only 87.64%; nevertheless 82 of the 97 relation-class errors
still selected the correct candidate. V6 therefore preserves pairwise posterior
compatibility and replaces oracle subject segmentation with a text-only entity
rule.

## Method

For each query/candidate pair, V6:

1. tokenizes both texts without subject metadata;
2. locates the shared contiguous token span with maximum inverse-document-
   frequency mass;
3. computes normalized weighted-overlap entity evidence;
4. replaces the selected shared span by `<ENTITY>` independently in both texts;
5. evaluates relation compatibility with the frozen, subject-disjoint V5
   classifier ensemble;
6. combines log relation compatibility and entity evidence.

IDF is fit separately for each fold from raw CounterFact text after excluding
all subjects in that fold. Candidate subject strings, relation IDs, chain IDs,
target values, age, and future queries are unavailable to the resolver.

The combination weight is selected from `{0, 0.5, 1, 2, 4, 8}` on the next
held-out fold using that fold's own V5 OOF classifier and IDF table. The selected
weight is then applied unchanged to the outer fold. CounterFact remains observed
development-only.

## Gate

V6 passes only if:

- text-only candidate top-1 is at least 95%;
- MRR is at least 97%;
- top-1 is no more than three points below oracle-segmented V5;
- top-1 improves over V4 by at least ten points;
- the paired-bootstrap confidence interval versus V4 has a positive lower
  bound.

A pass authorizes causal on-policy forgetting with the automatic resolver. A
failure returns to entity/scope representation work. LongMemEval-S and LoCoMo
remain sealed in either case.

