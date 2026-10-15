# Qwen utility-supervised semantic record gate V1

## Freeze point and motivation

This protocol is frozen after the independently audited S4 gold-trigger upper
bound failed and before acquiring or parsing LAMA/T-REx or ParaRel. S4 showed
that pair-local evidence does not make generic relation equivalence reliable:
its selected gold-trigger representation reached only 0.5580 direct-pair
ROC-AUC. Continuing relation-head or localization tuning on FewRel, DialogRE,
CounterFact or MQuAKE is prohibited.

Natural semantic headroom nevertheless remains large. On the already audited
CounterFact/MQuAKE confirmation, relation-ID oracle deletion reached 92.19%
joint success versus 36.56% for exact-surface deletion. The new hypothesis is
that a controller should predict the downstream future utility and collateral
risk of each addressable memory record, rather than use relation equivalence
as its final training target.

The current write is always admitted (`g=1`). Only historical session memory
is controlled. The final 1B-token Qwen3-4B checkpoint and its slow weights stay
frozen.

## New public sources and contamination boundary

Acquire only the official sources:

- `facebookresearch/LAMA`, including the official T-REx data archive linked by
  that repository;
- `yanaiela/pararel`, including its relation-specific pattern JSON files.

The acquisition script records Git commits, file sizes and SHA256 hashes before
any builder parses an example. The following previously observed sources are
forbidden for training, selection and threshold calibration in V1:

- CounterFact and MQuAKE rows and outcomes;
- ZsRE calibration/development/test;
- FewRel and DocRED train/development rows;
- DialogRE train/development/test;
- LongMemEval-S, LoCoMo, RULER/RULER-QA and MMLU.

They remain prior evidence, diagnostics or conditional external evaluation.
No V1 decision may use their model outcomes.

## Eligibility and relation split

Use only relations present in both T-REx and ParaRel. A relation is eligible
when it has at least four distinct valid ParaRel patterns and at least 500
distinct T-REx `(subject, object)` facts after normalization. Reject malformed
patterns, missing `[X]/[Y]` slots, identical subject/object strings, blank
values and exact duplicate facts.

Sort eligible relation IDs by SHA256 of
`261016:utility-semantic-record:relation:<id>` and assign:

- first 60%: controller training;
- next 20%: development and threshold selection;
- final 20%: sealed relation confirmation.

All instances of a relation stay in one partition. Remove a normalized subject
that appears in more than one partition, then remove cross-partition rendered
surface duplicates. Publish every exclusion and require at least 20 eligible
relations in total, at least four development relations and at least four
sealed confirmation relations. Failure stops before model execution.

## Natural update episodes

Render facts with different ParaRel patterns while retaining source relation
and fact IDs only as audit metadata. Each source group has two counterfactual
twins that differ only in one protected anchor value. Episode lengths are
8/16/32/64 records and use seven balanced mechanisms:

1. same-subject, same-relation semantic overwrite with a different value;
2. same-subject, same-relation paraphrased duplicate/no-op;
3. same-subject, different-relation retention;
4. different-subject, same-relation retention;
5. stale-evidence repetition before a valid update;
6. explicit session reset or referenced-record revoke;
7. a mixed long session containing overwrite, duplicate and hard distractor.

Every boundary asks two surfaces of the updated fact, two retained-fact
queries and one protected-anchor query. Candidate values consist of the old
value, current value and four split-local distractor values selected without
model scores. Facts placed in memory are not copied into the query prompt.

Relation ID, future queries, answers, source row IDs, split ID and teacher
actions are prohibited from inference features.

## Sparse boundary and full counterfactual grid

Use exactly eight pilot source groups per mechanism. Each twin has three
registered sparse decision boundaries: the first conflict/no-op, the longest
distance conflict, and the final update. At each boundary, a deterministic
high-recall selector exposes at most three historical candidate records:

1. exact normalized-subject matches first;
2. remaining slots by frozen-Qwen masked-relation cosine;
3. ties by newer record and then source order.

Enumerate every valid coefficient tuple in `{0, 0.5, 1}^k` for the selected
historical records, plus registered retain-all and global-clear states. The
incoming record is appended at coefficient one after the intervention.
Equivalent states are scored once but every grid identity is retained.

For every boundary record label construction time, number of unique states,
number of future Qwen queries, cache hit/miss counts and CUDA peak
allocated/reserved memory. Report median, p90, p95, maximum and per-mechanism
statistics. These measurements choose only among 128/256/512 formal training
groups; they cannot change mechanisms, actions, model, prompts or gates.

## Frozen memory substrate and label objective

Use the already audited natural-value residual construction with fresh query KV
and its frozen scale. Each memory record remains individually addressable. The
native `ttt_proj`/`ttt_conv` writer is an ablation because its favorable
single-record content gate failed; V1 makes no native-writer claim.

Evaluate every action on all registered future queries in the episode. Choose
the safe oracle lexicographically:

1. forbid more errors on unrelated chains than retain-all;
2. maximize the number of correct future queries;
3. minimize total correct-answer NLL;
4. minimize retained memory mass;
5. prefer the retain-heavier action under an exact tie.

Store per-action future correctness, NLL, unrelated errors, memory mass and
utility margin. The controller never receives future outcomes at inference.

The full `0/0.5/1` grid is always labeled. Half-strength is considered a
distinct learnable state only if it is uniquely optimal by at least 0.01 mean
answer-NLL on at least 5% of training boundaries and changes at least one
future prediction relative to both endpoints. Otherwise the deployable primary
uses exact `{0,1}` states and `0.5` remains a registered soft-decay ablation.

## Pilot gates and formal-size rule

The resource pilot releases formal construction only if:

- all rows and outcomes are finite and independently reproducible;
- safe-oracle future accuracy improves over retain-all by at least five points;
- the source-group bootstrap 95% lower bound of that gain is above zero;
- the safe oracle adds zero unrelated-chain errors;
- every mechanism contains at least two distinct optimal actions;
- peak reserved memory is below 24 GiB per RTX 5090;
- projected two-GPU construction time is below six hours for at least 128
  training groups.

Choose the largest of 128/256/512 groups whose projected two-GPU critical path
is below six hours and whose projected output is below 50 GiB. Development and
sealed confirmation each use 64 groups per eligible relation partition and are
never reduced by the resource result.

## Controller

Train a shared record-equivariant utility/risk controller. Each candidate
record uses only:

- exact subject equality and normalized value agreement/conflict;
- frozen Qwen query/record cosine, top-one margin and four inference-safe
  pooled views;
- pair cross-encoding margin produced without relation labels;
- record age, repetition count, current memory size and access count;
- reader confidence, selectedness and record coefficient.

A shared two-layer 128/64 MLP produces keep/delete/half utility logits and an
unrelated-forgetting risk logit. A permutation-invariant pooled context is
concatenated to every record. Apply a physical projection that always retains
the current write, makes exact duplicates idempotent, honors explicit reset or
revoke targets, and conservatively retains a record when the predicted risk
exceeds the calibrated bound.

Train three seeds with relation-disjoint nested folds. Fixed epochs are
25/50/100; AdamW learning rate is `1e-3`, weight decay `1e-4`, batch size 256
labels and gradient norm one. Select only on development from utility
thresholds `{0.4,0.5,0.6}`, risk penalties `{0,0.5,1,2}` and conformal risk
levels `{0.01,0.03,0.05}`. No hidden-state or action grid is changed after a
development outcome.

## Comparisons and ablations

Report retain-all, global clear, last-1/last-2, exact string deletion, frozen
cosine deletion, nearest relation pair classifier, explicit lifecycle physics,
the utility-only controller, the risk-only controller, controller without
physical projection, soft continuous alpha, snapped binary alpha, the primary
record controller and the source relation-ID oracle.

Primary endpoints are joint update-and-retention accuracy and future query
accuracy. Secondary endpoints are answer NLL, update, retained-fact, anchor and
locality accuracy, false-deletion count, deletion precision/recall, memory
mass, per-mechanism results, action agreement, runtime and peak memory. Use
20,000 subject-group and relation-cluster bootstrap draws.

## Development and one-time confirmation gates

Development releases the sealed relation confirmation only if all hold:

1. oracle headroom over the strongest non-oracle deterministic baseline is at
   least ten accuracy points;
2. primary gain over every non-oracle learned or deterministic baseline is at
   least three points;
3. subject- and relation-cluster 95% gain lower bounds are above zero;
4. deletion precision is at least 99%;
5. update and retained-fact accuracy are each at least 90%;
6. anchor/locality and unrelated-chain retention are at least 97%;
7. every relation and mechanism has nonnegative mean gain;
8. every seed improves over retain-all and the strongest safe baseline;
9. at least 40% of the strongest-baseline-to-oracle gap is closed;
10. every construction, label, feature, checkpoint, outcome, resource and
    sealing audit passes.

The one-time relation confirmation uses unchanged weights and thresholds and
must pass the same absolute safety checks, at least a two-point primary gain,
positive subject- and relation-cluster lower bounds, no negative mechanism and
at least 30% oracle-gap closure. Failure terminates controller iteration on
LAMA/ParaRel.

## Conditional external evaluation and claim boundary

A passing confirmation freezes the system before one-time LoCoMo and
RULER/RULER-QA evaluation. MMLU measures slow-knowledge non-regression.
LongMemEval-S and ZsRE are disclosed as previously observed diagnostics and do
not tune or release V1. External evaluation must include deletion-matched
random controls, generated answers, session/conversation-cluster intervals,
latency and peak memory.

A pass supports learned utility-aware forgetting on a structured,
individually addressable test-time memory. It does not establish factual
content binding in the original dense `ttt_proj`/`ttt_conv` writer. If the
controller gate fails while the oracle headroom passes, the supported result
is that semantic action inference remains the bottleneck; no benchmark can be
opened to rescue the branch.
