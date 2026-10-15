# Qwen utility record controller V2: cross-source replication protocol

## Freeze point and evidence boundary

This protocol is frozen after the audited V1 development diagnostic and before
acquiring or parsing TempLAMA.  The V1 diagnostic result has SHA256
`5d8f7b543f00a4f0144296a6a95bb7da5086bbfe8a0e51d741d0a7b27e6f1f2a`.
On its untouched 32-group evaluation half, the three controllers improved over
the strongest safe deterministic baseline by 2.8125, 3.2292 and 3.1250 points;
all source-group bootstrap intervals had positive lower bounds and all added
unrelated-error counts were zero.  They closed 61.4%, 70.5% and 68.2% of the
baseline-to-full-oracle gap.

V1 nevertheless remains failed because its preregistered full-development
oracle-headroom gate required ten points and observed only 9.375 points.  Its
seven LAMA/ParaRel confirmation relations remain sealed.  V2 does not lower a
V1 threshold or reinterpret that result.  It registers an independent public
source replication before any confirmation can be opened.

## Claim and fixed method

The hypothesis is that a utility/risk controller trained on LAMA/ParaRel can
selectively forget historical, individually addressable session records under
an unseen factual source and unseen temporal relations while preserving
unrelated records.

The incoming write is always admitted (`g=1`).  The deployment action is
binary (`alpha` in `{0,1}`) over historical records.  The fixed method is the
V1 hybrid of explicit lifecycle projection and the learned shared-record
controller.  It uses the V1 66-dimensional inference-safe feature tensor and
the three already trained checkpoints locked by training-audit SHA256
`4260e090d6ebae309d0f0c0e39637de8691b9bea3fdb044ba40d244fa09e8d07`.

The deployment settings are copied verbatim from V1 development selection:

| Seed | Utility threshold | Risk penalty | Conformal level | Frozen risk threshold |
|---:|---:|---:|---:|---:|
| 2610161 | 0.6 | 2.0 | 0.05 | 0.004868066404014826 |
| 2610162 | 0.6 | 2.0 | 0.05 | 0.004372645635157824 |
| 2610163 | 0.6 | 2.0 | 0.01 | 1.1618091377840756e-8 |

No V2 source outcome can change a checkpoint, feature, normalization statistic,
threshold, risk penalty, projection rule or candidate selector.  Exact
duplicates remain idempotent, explicit reset clears historical records and an
explicit revoke removes only its referenced record.  The neural decision is
used for the remaining semantic overwrite, repetition and interference cases.

The memory substrate is the already audited addressable Qwen residual memory.
This protocol makes no content-binding claim about the original dense
`ttt_conv`/`ttt_proj` writer: its independent favorable single-record pilot
failed.  Native writer results remain a negative baseline and cannot be used
to tune V2.

## Independent public source

Acquire only the official TempLAMA implementation and data under
`google-research/language`, directory `language/templama`.  Record repository
commit, paths, sizes and SHA256 hashes before parsing.  TempLAMA was built from
temporally scoped Wikidata facts and contains nine relations whose objects
change over time.  It is disjoint from V1's T-REx/ParaRel construction at the
source and protocol level; exact rendered-surface and normalized subject-value
overlap with every V1 partition must also be removed and reported.

CounterFact, MQuAKE, ZsRE, FewRel, DocRED, DialogRE, LongMemEval-S, LoCoMo,
RULER/RULER-QA and MMLU remain prohibited for V2 fitting, threshold selection
and release.  The original LAMA/ParaRel confirmation rows also remain unread
until the cross-source replication gates below pass.

## Temporal episode construction

An eligible source trajectory has one subject-relation pair, at least two
chronologically ordered nonempty values, no ambiguous simultaneous value at a
selected transition and two distinct surface realizations.  Subjects are
disjoint across TempLAMA's official train, validation and test partitions.
When the official data violate this invariant, hash each normalized subject
into exactly one partition and publish all removals before model execution.

Construct twin session episodes of lengths 8, 16, 32 and 64 with three sparse
decision boundaries.  Real temporal transitions supply the old and new values.
Other eligible trajectories from the same locked partition supply unrelated
facts and hard same-relation/different-subject distractors.  Balance:

1. real temporal overwrite;
2. paraphrased duplicate/no-op;
3. same-subject different-relation retention when available;
4. different-subject same-relation retention;
5. stale old-value repetition before a real update;
6. explicit reset or referenced-record revoke;
7. a mixed long session.

Every boundary asks two surfaces of the current value, two retained-fact
queries and one protected-anchor query.  Past-year queries are excluded from
the primary latest-session endpoint because selective forgetting is not a
bitemporal archive.  They are reported separately as a tradeoff diagnostic.
Relation IDs, timestamps, answers, future queries, split names and teacher
actions are forbidden controller features.

The registered high-recall selector exposes at most three historical records:
exact normalized-subject matches first, then frozen-Qwen masked-relation cosine,
then newer record and source order.  Pilot labels enumerate `{0,0.5,1}^k` plus
retain-all and global clear.  If half strength is uniquely useful with at least
0.01 mean-answer-NLL advantage on at least 5% of pilot boundaries and changes a
prediction relative to both endpoints, stop and register a separate soft V2
before formal evaluation.  Otherwise formal replication labels use all binary
endpoints plus global clear; no outcome-dependent pruning is permitted.

## Stages and sizes

1. **Source and resource pilot:** official TempLAMA train only, exactly 8
   eligible subject groups per relation when available, capped at 72 groups.
   Measure per-label time, unique states, cache hits, peak allocated/reserved
   memory and safe-oracle headroom.  This stage cannot select controller values.
2. **Replication development:** exactly 256 official validation subject groups,
   balanced across eligible relations and lengths.  It is used only for the
   release decision below; the controller is already frozen.
3. **Cross-source confirmation:** exactly 256 official test subject groups,
   scored once with unchanged code and parameters only after development pass.
4. **Original-source confirmation:** the still-sealed LAMA/ParaRel confirmation
   relations are scored once only after cross-source confirmation passes.

If a partition contains fewer eligible groups, use all eligible groups only
when there are at least 128 and at least six relations.  Otherwise the protocol
stops before model execution.  Sampling is deterministic by SHA256 and twins
remain in one source-group cluster.

## Baselines, outcomes and uncertainty

Report retain-all, global clear, last-1, last-2, exact-string deletion,
exact-subject conflict, frozen cosine thresholds 0.98/0.99/0.995, explicit
lifecycle physics, each of the three frozen controllers, their majority action,
the safe binary oracle and the full pilot oracle.  Add deletion-matched random
controls using 20 independently seeded draws.

Primary accuracy pools the five registered current-session future queries.
Primary safety is the count of new errors on unrelated and anchor queries
relative to retain-all.  Also report answer NLL, update/retention/anchor
accuracy, deletion fraction and precision, teacher agreement, latency, peak
memory, per-mechanism and per-relation results, and the past-year tradeoff.
Use 20,000 subject-group bootstrap draws.  A pooled three-seed comparison uses
one controller seed sampled per subject group inside each draw; relation-cluster
intervals are reported separately when at least six relations are eligible.

## Prospective release gates

Replication development releases the official TempLAMA test only if all hold:

1. the safe binary oracle exceeds the strongest zero-collateral deterministic
   baseline by at least three accuracy points;
2. every controller seed exceeds that baseline by at least one point and their
   arithmetic mean gain is at least two points;
3. the pooled subject-group bootstrap 95% gain lower bound is above zero;
4. every seed adds zero unrelated or anchor errors;
5. mean oracle-gap closure is at least 40%;
6. temporal-overwrite and stale-repetition mean gains are nonnegative;
7. macro-relation gain is positive, at least eight ninths of eligible relations
   are nonnegative, and no relation loses more than two points;
8. all source, construction, label, feature, checkpoint, result, resource and
   seal audits pass.

Cross-source confirmation uses unchanged parameters and passes only with mean
gain at least 1.5 points, positive pooled 95% lower bound, every seed positive,
zero added unrelated/anchor errors, at least 30% oracle-gap closure and no
mechanism below minus two points.  Failure ends V2 without opening the original
confirmation.

The one-time original-source confirmation then requires mean gain at least two
points, positive subject- and relation-cluster lower bounds, every seed
positive, zero added collateral errors, at least 30% oracle-gap closure and no
negative mechanism.  A failure is reported and ends controller iteration on
this substrate.

## Conditional external evaluation and claim boundary

Only a pass on both confirmations freezes the method for LoCoMo,
LongMemEval-S and RULER/RULER-QA.  MMLU measures slow-knowledge non-regression;
ZsRE remains a disclosed previously observed diagnostic.  External evaluation
must include the original in-place TTT checkpoint, no-memory, retain-all,
explicit-physics, deletion-matched random and frozen-controller arms, with
conversation/session-cluster intervals, latency and memory accounting.

A complete pass supports cross-source utility-aware selective forgetting in an
addressable test-time memory attached to the final 1B-token Qwen3-4B model.  It
does not establish factual content binding in the original dense writer.  That
limitation must appear in the title, abstract, method diagram and conclusion
unless a separate preregistered native-writer replacement passes content
binding before any forgetting evaluation.
