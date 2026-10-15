# Independent fact-key barrier V8 preregistration

## Motivation and separation from ZsRE calibration

V7's prompt entity view increased OOF true deletes from 551 to 609 and reduced
false deletes from six to five, but missed the registered maximum of four by
one. The five rows represent three events: same relation with different
entities, same entity with different relations, and a local update whose reader
top is another relation of the same entity. A physical delete rule therefore
needs a complete fact key, consisting of both entity and relation compatibility.

V8 trains and validates that key matcher entirely on the previously downloaded
CounterFact source. No ZsRE calibration row is used for its model, feature,
hyperparameter or threshold selection. ZsRE V7 is loaded only after an
independent CounterFact validation gate passes. Source test and external
long-memory benchmarks remain sealed.

## Independent CounterFact pair corpus

Verify the registered CounterFact SHA-256. Assign every subject to one of five
deterministic folds before forming examples. Negative partners must come from
the same fold so no subject surface crosses a split. For each eligible source
row construct exactly:

1. a positive pair between its clean prompt and first paraphrase, sharing
   subject and relation;
2. a same-relation/different-subject negative selected deterministically from
   that fold; and
3. a same-subject/different-relation negative selected deterministically from
   that subject's rows.

Rows lacking either hard-negative type are excluded and counted. Folds 0--2
are training, fold 3 is tune, and fold 4 is untouched validation. Subject sets
must be disjoint. Pair labels use CounterFact subject and relation IDs only
during corpus construction; these fields never enter inference features.

## Text-only inference features and selection

Reuse V7's deterministic entity-anchor extraction. Mask all extracted anchor
tokens with `<entity>` to form a relation view. The eleven registered features
are the five entity-anchor features, whole-prompt lexical Jaccard,
containment and length ratio, plus the same three lexical features on the
masked relation views.

Search the same linear/hidden-16/hidden-32 heads, learning rates
0.001/0.005, weight decay 0/0.01 and thresholds
0.50/0.70/0.80/0.90/0.95/0.98/0.99 with three seeds and 300 epochs. Selection
uses fold 3 only. A tune candidate is feasible with precision at least 99%,
recall at least 90%, and at least 98% specificity on each negative type.
Among feasible candidates prefer recall, precision, worst negative specificity,
simpler heads and higher threshold. Fit the selected configuration on folds
0--3 and evaluate fold 4 exactly once.

Independent validation passes only with the same 99/90/98/98 floors, at least
100 examples of each class/type, exact subject separation and an independent
audit. If it fails, do not load V7 or ZsRE calibration.

## Frozen zero-parameter transfer

After validation passes, freeze the three final heads, normalization and
threshold. For every audited V7 OOF row, reconstruct only its incoming and
reader-top prompt strings, compute the eleven features and fact-key
probability, and execute deletion only when both the original V7 OOF action
and the frozen fact-key barrier pass. No transfer threshold, feature or model
is selected on ZsRE.

The transferred offline gate is unchanged: precision at least 98%, recall at
least 50%, at most four false positives, at least 95% precision in each active
outer fold, 5--50% delete rate, exact V7 fold isolation, independent audit and
all held-out sources sealed. Passing authorizes an on-policy calibration
outcome replay, not test.

If the frozen transfer fails, stop controller development on the current ZsRE
calibration set. If it passes, the subsequent on-policy replay must still gain
at least five candidate-accuracy points over retain-all with positive paired
CI lower bound, at most four collateral errors, nonnegative fold gains and
preserved generation before the one-time source test can open.
