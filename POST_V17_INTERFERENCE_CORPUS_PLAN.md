# Post-V17 Controlled Memory-Interference Experiment Plan

## Registration boundary

This plan was written after the frozen 40-row V17 resource pilot and before
reading any 272-row V17 formal outcome. V17 formal remains unchanged and keeps
its original primary-accuracy gate. The only evidence used here is the pilot's
completed fixed-arm table:

- `retain_all` reached 100% candidate accuracy on all four query kinds;
- `global_reset`/`last_1` reached 100% on both primary queries but only
  65.0% hard-retain and 52.5% unrelated accuracy;
- `no_memory` reached 57.5--77.5% depending on query kind;
- the selected primary-only fixed comparator therefore had a ceiling while
  discarding useful session history.

The new experiment addresses this measurement failure. It does not reinterpret
or replace the registered V17 result.

## Research question

Can selective forgetting improve **joint update and retention success** when
session memory contains genuinely interfering, previously unseen facts, and
can a causal policy approach that exact intervention headroom on unseen
subjects and templates?

## Controlled corpus

Build a source-disjoint controlled corpus with opaque entities and values that
cannot be recovered from Qwen's slow parameters. Each episode maintains
multiple key/value facts over 8, 16, 32 or 64 sessions. A session may update
one key while other keys remain valid. Values use seeded nonce strings with
token-length matching, and train/pilot/formal partitions use disjoint entity
prefixes, value pools and surface templates.

The corpus contains four balanced intervention mechanisms:

1. **clean update**: one stale and one current value for an updated key;
2. **stale repetition**: an obsolete value appears in 2--4 earlier sessions;
3. **irrelevant near duplicate**: lexically similar memory does not change the
   queried key;
4. **useful repetition**: repeated correct history should be retained.

Memory remains chronological, but the prompt does not tell the model that the
latest statement is always correct. The memory policy must present a coherent
state. Template wording is counterbalanced so that lexical overlap and recency
alone do not identify the deletion target.

## Queries and primary endpoint

Every query point contains:

- an update query using an old-surface paraphrase;
- the same update queried with a current-surface paraphrase;
- two retained-key queries whose facts come from earlier sessions;
- one unrelated/locality query.

Candidate accuracy, answer NLL and free-generation normalized exact match are
all recorded. The primary episode endpoint is **joint success**: both update
queries and both retained-key queries must be correct. Locality is a separate
safety endpoint. This endpoint prevents a global reset from winning by solving
only the current update.

## Interventions

Evaluate these fixed arms plus exact causal interventions:

- no memory, retain all, global reset, last 1, last 2;
- delete each single historical component;
- delete all components for one key;
- continuous counterfactual retention weights `0`, `0.5`, `1` only in the
  registered sparse boundary study.

The exact oracle maximizes joint success, then update accuracy, retention
accuracy, lower mean answer NLL, retain-all, and chronological deletion order.
It never deletes the current session or uses a future query, answer or loss as
an input to a deployed policy.

## Stage 1: 40-episode construction and resource pilot

Use 10 episodes at each length and balance the four mechanisms. Record every
label's latency and peak allocated/reserved memory. The corpus is admitted to
formal study only if:

- all rows reconstruct and score without leakage or context overflow;
- no-memory joint success is at most 70%;
- retain-all joint success is between 20% and 90%, avoiding floor and ceiling;
- global-reset retained-key accuracy is below 80%;
- the exact selective oracle reaches at least 90% update and 90% retained-key
  accuracy;
- oracle joint success exceeds the best fixed arm by at least 10 points.

These are corpus-validity checks, not final efficacy claims. If they fail,
change only the registered generator difficulty parameters on a new pilot;
never tune on the formal split.

## Stage 2: frozen formal causal-headroom study

Freeze at least 320 episodes from unseen entity prefixes, values and templates.
Select the strongest fixed arm inside each outer subject/template fold. Report
micro, subject macro, template macro, all folds and worst fold. Use 20,000-draw
subject/template-cluster bootstrap intervals.

The formal headroom gate requires:

- oracle-minus-fixed joint-success gain >=5 points;
- clustered 95% interval lower bound >0;
- oracle update and retained-key accuracy each >=90%;
- oracle locality accuracy >=97%;
- no outer fold has negative joint-success gain.

Failure closes controller training on this corpus. It does not justify another
threshold search.

### Formal split freeze after pilot admission

The audited 40-episode pilot passed all six construction gates before the
formal split was generated.  The formal split is therefore frozen as 320
episodes from 80 new subjects, with all four mechanisms represented once per
subject.  It contains 20 formal-only template groups and five outer folds.
Each subject belongs to exactly one template group and each subject/template
group belongs to exactly one fold; every fold has 16 subjects, four template
groups and 64 episodes.  Lengths 8, 16, 32 and 64 occur once per subject and
80 times overall.  Formal values start at the registered opaque-code offset
4096, so entity prefixes, all 2,560 answer values and every surface template
are disjoint from the pilot.

For each held-out fold, the strongest fixed arm is selected only on the other
four folds.  The exact oracle remains restricted to retain-all, one historical
component deletion, or one historical-key deletion.  The registered 20,000
draw intervals are computed separately over 80 subject clusters and 20
template clusters.  The formal gate requires both cluster-macro gains to be at
least five points and both 95% lower bounds to exceed zero, in addition to the
previous update, retention, locality and non-negative-fold requirements.
Free-generation joint success is a frozen secondary endpoint and cannot
replace a failed candidate-based primary gate.

## Stage 3: causal readout-utility policy

Only after a passed headroom gate, train a subject/template-disjoint policy to
predict the downstream loss change from deleting a component. Inputs may use
the current query/event representation, component readout, component
interactions and past-only scores. Future loss, answer labels, split identity,
absolute position and episode length are prohibited at inference.

Use at least three seeds and compare with all fixed arms, random matched
deletion, the V15 controller and the exact oracle. The policy must improve
joint success with a clustered CI lower bound above zero while preserving the
registered update, retention and locality floors.

## Stage 4: natural confirmation

Only a passed controlled policy can be evaluated once on MQuAKE-CF,
LongMemEval-S and LoCoMo. MMLU and RULER/RULER-QA then measure slow-knowledge
and long-context preservation. External data cannot select generator
difficulty, thresholds, features, epochs or architecture.

## Required artifacts

Every stage saves the generator configuration and seed, split identities,
source/template/value hashes, state matrix, raw model outputs, latency and GPU
memory per label, code/model/tokenizer hashes, summary, independent audit,
terminal status, report and artifact SHA-256 list.
