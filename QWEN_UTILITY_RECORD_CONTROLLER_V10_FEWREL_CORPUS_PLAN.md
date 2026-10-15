# V10 natural FewRel counterfactual training corpus

## Registered cause and objective

V8 showed that relation-posterior features improve LAMA benefit AP and policy
gain, but stopped at +1.189 points.  V9 raised within-boundary LAMA ranking
accuracy to 90.91% yet produced the identical +1.189-point policy gain and
reduced TempLAMA performance.  Its no-rank and base-rank ablations reached
only +1.155 and +1.137 points.  The learned pair-controller branch is closed.

The remaining registered cause is training coverage.  Across two existing
training sources, masking subject and value yields only 190 unambiguous
relation-slot texts over 25 relations.  V10 tests whether natural surface and
relation diversity, with unchanged downstream architecture, improves OOD
forgetting.  It constructs new counterfactual session episodes from official
FewRel **training** data and then retrains the locked V7 66-feature controller.

## Source boundary and leakage controls

Use the audited THU FewRel repository at commit
`278a2315d2138810a379cd8d5718914dc56e2582`.  The source root and file hashes
must match the existing FewRel audit.  Use only the 48 relations assigned to
the prior train partition from `train_wiki.json`.  `val_wiki.json`, FewRel 2.0
PubMed validation, SemEval test, TempLAMA test, the original-source
confirmation, LongMemEval-S and LoCoMo remain sealed.

Remove exact source-row, normalized-statement and masked-template overlaps
with all development records before group construction.  Development data may
supply exclusion hashes only; relation IDs, labels, outcomes and model scores
may not influence sampling.  Entity pairs and raw source rows are disjoint
across constructed groups.

## Natural episode construction

For every source row retain its natural sentence, marked head entity and tail
value.  Extract a natural relation template from that sentence.  A fact
statement instantiates a traceable FewRel template with the group's subject and
value; its exact query blanks the value in that instantiated statement, while
the rephrased query uses a different source-row template from the same relation.

This wording is a pre-model construction amendment.  The original wording
required every update statement to remain the verbatim source sentence, which
is incompatible with counterfactual overwrite because FewRel does not supply
multiple values for the same subject in a controlled lifecycle.  Entity
substitution is therefore necessary to form the intervention.  Every factual
and template donor row remains hash-traceable, and donor rows plus entity pairs
are globally disjoint across constructed groups.  This amendment was made
before capacity inspection, Qwen labeling, feature extraction or outcome
observation and does not change the registered mechanisms or gates.

Require each eligible relation to have at least 28 distinct source rows, 20
distinct subjects, 20 distinct values and seven distinct subject/value-masked
templates after exclusions.  Construct exactly 14 deterministic groups per
eligible train relation, two for each existing lifecycle mechanism:

1. semantic overwrite;
2. stale repetition;
3. semantic duplicate;
4. same subject, different relation;
5. different subject, same relation;
6. explicit reset/revoke; and
7. mixed lifecycle.

With all 48 registered relations eligible, the formal target is 672 groups.
Any capacity shortfall stops before model labeling and requires a documented
model-free amendment; relations or examples may not be fabricated.  Candidate
memories remain capped at three records so the existing `{0, 0.5, 1}` action
grid and global-clear action are unchanged.

## Pilot and formal labeling

Before the full run, deterministically select two relations and two groups per
mechanism (28 groups).  Label every registered sparse boundary on the final
1B-token Qwen3-4B checkpoint using two RTX 5090 shards.  Record per-label wall
time, peak allocated/reserved memory, answer coverage, action variance and
safe-oracle headroom.  The pilot passes only if:

- every label/action is finite and independently auditable;
- no shard exceeds 28 GiB reserved memory;
- at least 95% of registered answers are tokenizable and scored;
- at least 5% of residual candidates have positive safe benefit; and
- safe-oracle headroom is at least 1.5 points.

On a pass, label all formal groups with identical code, prompts, action grid,
model and tokenizer.  No pilot outcome may change the construction or
hyperparameters.  Persist per-label time and GPU peaks for scale accounting.

## Features and controller

Extract the unchanged V7 66 inference-safe features for formal FewRel labels.
Train the unchanged 128/64 benefit/risk MLP on TempLAMA, LAMA/ParaRel and
FewRel training data.  Every example is used; the loss is the equal mean of
the three per-domain weighted losses so corpus size cannot dominate.  Use the
same three seeds, optimizer, epochs, utility/risk grid, minimum-domain
conformal threshold, V6 fact-key projection and worst-domain selection rule.
No FewRel source flag enters the feature tensor.

The frozen comparison is V8, the strongest safe LAMA policy.  Also report the
unchanged V7 joint controller to isolate new-data benefit from V8's posterior
features.

## Development release gate

V10 passes only if an independent audit verifies:

1. every seed improves lifecycle physics on TempLAMA and LAMA development;
2. mean gain is at least 1.5 points in each domain;
3. TempLAMA mean gain is at least 3.5 points;
4. every seed/domain adds zero unrelated errors;
5. mean safe fact-key-oracle closure is at least 50% in each domain;
6. semantic-overwrite and stale-repetition gains are nonnegative;
7. LAMA gain exceeds locked V8 by at least 0.3 point;
8. FewRel removal reduces LAMA gain by at least 0.2 point in a fixed ablation;
9. every source, construction, label, feature, checkpoint and result audit
   passes; and
10. all held-out sources remain sealed.

A pass freezes the controller and authorizes a separately registered LoCoMo
natural-session evaluation.  A failure stops development on the current two
observed domains; it cannot justify another architecture or threshold search
there.
