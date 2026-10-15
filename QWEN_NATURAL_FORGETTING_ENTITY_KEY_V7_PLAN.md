# Entity-key lifecycle safe-delete V7 preregistration

## Motivation and held-out boundary

V5 and its single registered DAgger iteration V6 both produced exactly six
false-positive deletes. V6 added only 19 distinct states and left all three
underlying failure events unchanged. In each event, the incoming and retrieved
prompts used the same relation template for different subjects. The existing
semantic and bag-of-token similarities therefore identify the relation but do
not identify the fact key.

V7 adds one inference-visible entity-key view and a physical barrier candidate.
It uses only the 2,232 audited V6 calibration states and the prompt strings
already present in those memory records. Dataset `subject`, source action,
chain ID, version, answer, future outcome, development, source test and
external benchmarks are forbidden model inputs. Test remains sealed.

## Deterministic prompt entity anchors

Tokenize each incoming and inferred-top prompt with the Unicode word pattern
`[\w]+(?:['’\-][\w]+)*`. An entity-anchor token contains a digit or begins
with an uppercase character, after removing the following fixed sentence and
query vocabulary (case-insensitive):

`a, an, and, are, at, by, did, do, does, for, from, how, in, is, name, of, on,
or, the, to, was, were, what, when, where, which, who, whom, whose, why`.

This rule uses only the two prompt strings at inference. It does not inspect a
gold subject field or learn corpus statistics. From the two anchor sets append
five features to the frozen ten-dimensional no-schedule view:

- both-anchor-sets-nonempty;
- anchor Jaccard overlap;
- anchor containment overlap;
- nonempty exact-set match; and
- anchor-count ratio.

The registered physical barrier is true only when both anchor sets are
nonempty and containment overlap is one. It encodes a necessary deletion
condition: the retrieved record must share an observable entity key with the
incoming write. Relation compatibility remains learned by the other features.

## Nested grouped candidate family

Repeat the V5/V6 five-fold grouped protocol on all V6 states. Use the same
linear/hidden-16/hidden-32 heads, learning rates 0.001/0.005, weight decay
0/0.01, thresholds 0.50/0.70/0.80/0.90/0.95/0.98/0.99, three seeds, 300
epochs, normalization and positive-class weighting. For every configuration
and threshold, register two action rules before observing V7 outcomes:

1. augmented learned probability only;
2. augmented learned probability AND the deterministic entity-key barrier.

Select solely from four-fold inner OOF predictions within each outer training
split. Inner feasibility remains precision at least 98%, recall at least 50%,
at most four false positives and at least 95% precision in every active inner
fold. Among feasible rows prefer higher recall, fewer false positives, higher
precision, the physical barrier, the simpler model and higher threshold.

## Release and next stage

Outer OOF release remains unchanged: precision at least 98%, recall at least
50%, no more than four false positives, at least 95% precision in each active
outer fold, 5--50% delete rate, exact session exclusion, independent audit and
all held-out sources sealed.

If V7 passes, freeze the five fold-specific rules and run the registered
on-policy calibration outcome replay. Compare retain-all, privileged source
action, V4 best, V6 and V7. It must gain at least five candidate-accuracy
points over retain-all with paired session-bootstrap 95% lower bound above
zero, add at most four collateral errors, be nonnegative in all folds and
preserve final generation within 0.5 points before test can open.

If V7 fails, keep test sealed and stop expanding the controller on this
calibration set. The next scientifically valid action is a new independently
generated training corpus with explicit same-relation/different-entity hard
negatives, followed by a fresh validation split.
