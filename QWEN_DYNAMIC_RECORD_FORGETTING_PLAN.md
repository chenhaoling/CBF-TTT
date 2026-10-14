# Preregistered successor: dynamic record-level selective forgetting

## Motivation and evidence boundary

This protocol is frozen after the independently audited fixed-cohort lifecycle
OOD failure and before any successor corpus, label, feature, checkpoint or
metric exists. The failed OOD corpus seed `202610154` is permanently
diagnostic-only. It may be summarized to localize the failure but may not be
used for successor training, model selection, calibration, early stopping or
reported held-out evidence.

The preceding controller maps a session to one of 27 retention states for
three fixed positional cohorts: history, last-2 and last-1. It reached 98.68%
on a fresh matched test but only 69.26% on variable lifecycle OOD, with 5.16%
oracle-action agreement. Exact referenced-record deletion reached 89.80%,
showing that record identity and arbitrary history depth contain the missing
information. The successor tests whether forgetting should be an equivariant
state transition over individually addressable memory records rather than a
fixed positional decay vector.

The controlled reader remains the audited `component_top1` Qwen-derived memory
reader. Native `ttt_proj`/`ttt_conv`, actual LM-head residual readout and
natural-text event discovery remain separate claims. The previously frozen
residual-readout bridge stays locked because its lifecycle prerequisite failed.

## Method

At each decision boundary, construct the visible active-record set
`R_t = {r_1, ..., r_n}` from causal text only. Future blocks, teacher semantic
events, answers and oracle actions never enter a controller feature.

The method has two stages:

1. **Dynamic candidate selection.** Select at most three addressable old
   records using a deterministic union of the visibly referenced record, the
   most recent record sharing a key with the current write, and the strongest
   Qwen contextual key-conflict record. Ties use chronological order. Missing
   slots receive an explicit mask. Selection is permutation equivariant over
   record order except for declared age features.
2. **Constrained exact-state decision.** Predict a global session-clear bit and
   one state in `{0, 0.5, 1}` for each selected record. Nonselected records are
   retained unless the global clear bit is active. The resulting 54 actions
   are decoded jointly, then checked against hard lifecycle constraints.

Per-record features include visible reference identity, age and log age,
current/old key overlap, assignment duplication, overwrite status, record
size, Qwen contextual key similarities, value-direction conflict,
active-source status and a missing-slot indicator. Global features include
active record count, reset/no-op indicators, current-write size, duplicate
fraction and the same scale-free statistics used by the cohort controller. No
raw identifier or value string is used as a categorical feature.

The primary controller is a small shared record encoder with masked pooling
and a joint 54-class head. It is trained independently for projection seeds
2301/2302/2303. A plain MLP over the three selected slots and an unconstrained
set encoder are registered ablations.

## Physical constraints

The primary decoder enforces four causal rules:

- **Locality:** a record with no reference, overlapping key, duplicate relation
  or reset scope cannot be deleted.
- **Idempotence:** an exact duplicate/no-op cannot change states of unrelated
  records.
- **Monotone removal:** an explicitly revoked record cannot receive a larger
  retained state than before the event.
- **Session reset:** a visible reset activates global clear before the current
  write; the current write is never removed.

Train-time counterfactual twins vary one rule trigger while holding all other
visible content fixed. A consistency loss requires unchanged decisions for
unaffected records. The full method and every ablation use the same labels,
features and training budget.

## Fresh corpora and sealing

Use independent deterministic seeds and disjoint templates, identifiers and
values:

- development corpus seed `202610156`: 256 train and 64 development source
  groups;
- one-time matched test seed `202610157`: 128 sealed source groups;
- one-time external lifecycle OOD seed `202610158`: 64 source groups per suite.

Training/development families cover retain, recent revoke, distant revoke,
reset, exact duplicate, overwritten-record no-op and mixed two-event blocks at
session lengths 8, 12 and 16. Group twins differ in one initial anchor only.
The sealed matched test uses held-out surface templates and source groups.

The external OOD suites are fixed before materialization:

1. length extrapolation to 24 and 32 records;
2. two distant revocations in one block;
3. reset followed by a duplicate/no-op write;
4. alias reference, where a visible record alias is introduced earlier and
   later used for revocation;
5. mixed retain/revoke/reset composition with boundary schedules unseen in
   train/development.

No old OOD identifier, value, template or schedule may appear. Test and OOD
files are generated into separate roots and their identifiers are not loaded
by the trainer until the corresponding development gate passes.

## Counterfactual labels and resource pilot

First run exactly eight source groups per training family on the two RTX 5090
cards. For each boundary, enumerate the full `{0, 0.5, 1}^3` states for the
three selected slots under both global-clear values, subject only to hard
validity masks. This is the requested 0/0.5/1 grid with sparse boundary
sampling. Record for every label:

- wall time;
- prompt, record and candidate counts;
- token lengths;
- peak allocated and reserved GPU memory;
- valid-action count and oracle margin.

The oracle is lexicographic: maximize mean query accuracy first, minimize mean
answer NLL second, minimize removed mass third, then use registered action
order. This prevents the earlier NLL oracle from trading away the primary
accuracy endpoint. Ties use tolerances `1e-12` for accuracy and `1e-6` for NLL.

Choose the largest formal training scale among 128/256/512 groups that keeps
peak allocated memory below 24 GiB per GPU and estimated label construction
below six hours. The resource rule may change only sample size, never the
method, features, losses, splits, metrics or gates.

## Comparators and ablations

- retain all, global clear/half and fixed windows;
- deterministic random valid action;
- old frozen fixed-cohort controller;
- explicit cohort physics;
- exact referenced-record deletion;
- dynamic candidates with hand-written lifecycle rules;
- slot MLP without shared/equivariant encoding;
- set encoder without physical constraints;
- primary constrained set controller;
- lexicographic 54-action oracle.

Report full query accuracy and NLL, current/historical/anchor roles, every
family/suite, minimum projection seed, action agreement, slot-selection recall,
retention MAE, constraint violations, retained mass, latency and memory. Use
20,000 paired source-group bootstrap draws; twins and their queries are one
bootstrap unit.

## Registered release gates

Development releases the matched test only if all hold:

1. primary accuracy at least 95%;
2. every development family at least 90%;
3. minimum projection seed at least 93%;
4. gain over the strongest non-oracle comparator at least three points, with
   paired 95% lower bound above one point;
5. NLL gain over that comparator has 95% lower bound above zero;
6. oracle gap at most three points;
7. current and anchor accuracy at least 98%, historical at least 90%;
8. candidate-selection recall at least 98%;
9. endpoint/interior action accuracy at least 95%/65%;
10. zero hard-constraint violations and all integrity audits pass.

The one-time matched test uses the same gate. It releases external OOD only if
every check passes without retraining or calibration. External OOD passes only
if macro accuracy is at least 92%, every suite at least 85%, minimum seed at
least 90%, gain over the strongest non-oracle comparator is at least two points
with 95% lower bound above zero, oracle gap is at most five points, current and
anchor are at least 97%, historical is at least 85%, and all integrity checks
pass.

There is one attempt per sealed corpus. A failed gate stops before the next
split. Any successor after failure must use new seeds, templates and a newly
frozen protocol.

## Claim unlocked by a pass

A complete pass would support the controlled claim that record-equivariant,
physics-constrained selective forgetting generalizes across variable session
length and lifecycle composition better than fixed decay, fixed positional
cohorts and visible-rule baselines. It would still not establish native dense
TTT writing or natural benchmark gains. Actual Qwen LM-head readout and
LoCoMo/LongMemEval/RULER evaluation require separately sealed follow-up stages.
