# Frozen-Qwen semantic fact-key barrier V9 preregistration

## Evidence and hypothesis

V8's subject-disjoint CounterFact validation obtained 100% specificity for
same-relation/different-subject negatives, but only 76.47% specificity for
same-subject/different-relation negatives and 49.30% recall for true paraphrase
pairs. The audited result localizes the failure to lexical relation matching.
Threshold search is closed.

V9 tests whether a frozen Qwen3-4B representation of an entity-masked relation
view can identify semantic relation equivalence while retaining V7's explicit
entity compatibility. It reuses the architecture and fixed epoch/threshold of
the previously successful V11 token-span extractor, but refits every span head
inside V9's subject split. V11 predictions and relation scores are not reused.

## Leakage controls

- Reuse V8's registered CounterFact source hash, eligible pairs and deterministic
  five-way subject assignment.
- Refit a three-seed token-span ensemble for each fold after excluding every
  subject assigned to that fold. The architecture, projection seed, epoch 15
  and threshold 0.3 are frozen from V11 and are not selected in V9.
- Pair-head folds 0--2 train, fold 3 selects a finite model/threshold grid and
  fold 4 is evaluated once.
- CounterFact subject strings supervise token spans and subject/relation IDs
  construct pair labels only. Inference receives prompt text only.
- ZsRE calibration is loaded only after the CounterFact validation and
  independent audit pass. ZsRE development, test and external benchmarks stay
  sealed.

## Frozen semantic representation

Load the final 1B-token Qwen3-4B checkpoint without updating it. For each prompt,
the cross-fitted span head selects one contiguous subject span. Replace it with
`<ENTITY>` and encode both the original prompt and relation view using the last
hidden state after a fixed `Fact key:` prefix. Project to 256 dimensions using
seed 29001 and L2 normalize.

The registered scalar feature arm contains:

1. five V7 prompt-anchor compatibility features;
2. five predicted-span token compatibility features;
3. three whole-prompt lexical features;
4. three masked-relation lexical features; and
5. cosine, mean absolute distance, RMS distance and maximum absolute distance
   for both full-prompt and masked-relation Qwen vectors.

The registered vector arm appends the elementwise absolute difference and
product of the two masked-relation vectors. These symmetric features cannot
encode pair order, gold subject, relation ID, chain ID, action or future outcome.

Search scalar/vector arms, linear/hidden-16/hidden-32 heads, learning rates
0.001/0.005, weight decay 0/0.01 and thresholds
0.50/0.70/0.80/0.90/0.95/0.98/0.99 with three seeds and 300 epochs. Selection
uses fold 3 only. Prefer feasible candidates, then recall, precision, worst
negative specificity, the scalar arm, simpler heads and the higher threshold.

## Registered gates and next action

Fold-4 validation passes only with precision at least 99%, recall at least 90%,
at least 98% specificity for each negative type, at least 100 examples of every
type, exact subject separation, a frozen Qwen checkpoint and a passing audit.

After a pass, freeze all semantic transforms, heads and threshold. Apply them to
the V7 out-of-fold ZsRE calibration states and delete only when both V7 and V9
approve. No feature, arm, head, threshold or span setting may be selected on
ZsRE. The transfer gate remains precision at least 98%, recall at least 50%, at
most four false deletes, at least 95% precision in every active fold and a
5--50% delete rate. Passing authorizes an on-policy calibration replay, not test.

If CounterFact validation fails, stop before ZsRE and audit semantic errors. If
frozen transfer fails, stop fact-key controller development on the current ZsRE
calibration set. Source test remains sealed in either case.
