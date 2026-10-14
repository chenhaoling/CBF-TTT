# Preregistered causal-ledger Qwen residual readout

## Freeze point and evidence boundary

This protocol is frozen after the independently audited dynamic record-level
development failure and before creating any corpus row, Qwen feature, residual
score or outcome for this successor. The failed development seed `202610156`
is diagnostic only. Its matched test and external OOD splits remain
ungenerated and may never be opened by this successor.

The predecessor established two facts. First, an individually addressable
record transition reaches 98.15% controlled accuracy and improves by 25.61
points over the frozen positional-cohort controller. Second, a learned
controller has zero accuracy gain over exact record semantics because the
visible lifecycle rule already matches the accuracy-first oracle. The remaining
reset errors are exclusively absent-value queries and survive the oracle.

This successor therefore treats the record ledger and its physical transition
operator as the method. It asks whether that state improves candidate
likelihood and generation through the frozen Qwen3-4B LM head. It does not
claim that the original dense `ttt_proj`/`ttt_conv` writer learned this
representation; its audited content-binding failure remains part of the claim
boundary.

## Method: causal fast-memory ledger

Each write creates an immutable record containing one or more key/value facts.
The ledger stores record identity, chronological order, active status and
Qwen-derived rank-one factors separately. At every event it applies four hard
state transitions:

1. **reset:** deactivate every older record before committing the current
   record;
2. **explicit revoke:** deactivate only the referenced record;
3. **shadowing and restoration:** the newest active assignment is visible for
   each key, while deactivating it restores the newest earlier active value;
4. **idempotence:** duplicate/no-op writes cannot change unrelated visible
   facts.

The current record is always committed and cannot be deleted by the same
event. There is no learned forgetting classifier, threshold or hidden action
label in the primary method.

For each fact, the normalized frozen-Qwen entity embedding is the key and five
position-specific normalized LM-head directions represent the opaque value.
At a query, every currently visible fact contributes its key-similarity-
weighted value directions. Their sum is added to fresh-KV Qwen hidden states
and scored by the unchanged LM head. An absent visible key routes a registered
five-token null value through the same readout; it does not choose a value from
the answer or future state. Native TTT adapter parameters are loaded but never
write memory.

## Fresh corpus and split isolation

Use new deterministic seeds, namespaces, templates and value pools:

- resource/content pilot: `202610159`;
- development and scale selection: `202610160`;
- one-time matched confirmation: `202610161`;
- one-time lifecycle OOD: `202610162`.

Pilot, development, confirmation and OOD identifiers, record IDs, templates
and opaque-value pools are pairwise disjoint. Every group has two twins that
differ in one initial anchor value. Each episode has three sparse decision
boundaries and covers one of seven families: retain, recent revoke, distant
revoke, reset, exact duplicate, overwritten revoke no-op, and mixed two-event.

Development lengths are 8/12/16. The matched confirmation uses held-out
surface templates at the same lengths. OOD is fixed before construction:

1. lengths 24 and 32;
2. two distant revocations in one boundary;
3. revoke of the newest assignment followed by restoration of an older value;
4. reset followed by a duplicate/no-op write;
5. mixed reset/revoke/overwrite schedules unseen in development.

The query and candidate table is derived from the causal ledger state. Every
answer is exactly five Qwen tokens. Candidates contain the causal answer, the
registered null value and four deterministic within-split distractors. Future
events and teacher actions never enter a policy state or model prompt.

## Resource and content pilot

Run exactly eight source groups per family on two RTX 5090 cards. Score scales
4/8/16 together. For every decision boundary, additionally enumerate the
registered global-clear plus full `{0, 0.5, 1}^3` grid over at most three
addressable old records. Record per-label wall time, valid action count,
prompt/answer tokens, active records, candidate count, Qwen-answer margin and
peak allocated/reserved GPU memory.

The pilot reports Correct, Wrong-twin, Zero and norm-matched sign-randomized
single-component controls in every family. It may choose only the largest
development size among 16/32/64 groups per family whose estimated two-GPU
runtime is below six hours and allocated peak is below 24 GiB. It cannot
change methods, scales, prompts, gates, candidates or splits.

The content gate requires, at primary scale 8:

- Correct candidate and greedy exact accuracy at least 95%/85%;
- Wrong and Zero candidate accuracy at most 10%/30%;
- Correct answer-NLL gain over Wrong, Zero and Random with source-group
  bootstrap 95% lower bounds above zero;
- every family Correct candidate accuracy at least 90%;
- scale 4 and 16 candidate accuracy no more than five points below scale 8 and
  EOS rate no more than five points above it.

Failure stops before development construction.

## Comparators and counterfactuals

All arms replay identical writes and queries:

- no memory and retain every written record;
- global reset at each decision boundary;
- last-one and last-two record windows;
- explicit reset/revoke only, without latest-active visibility;
- newest-same-key visibility without explicit lifecycle events;
- the full causal ledger;
- semantic state oracle reconstructed independently from the event stream;
- an actual-Qwen subset oracle over the same causally available record set,
  reported as diagnostic and never used to tune the ledger.

The semantic oracle is expected to equal the full ledger exactly; disagreement
is an implementation failure. The primary comparison is against the strongest
non-oracle fixed, window or single-rule arm, including explicit reset/revoke.

## Development and matched-confirmation gates

Development selects the primary scale from 4/8/16 by lexicographically
maximizing source-group macro candidate accuracy, greedy normalized exact
match, negative answer NLL and then lower scale. It releases the one-time
matched confirmation only if all hold:

1. ledger candidate accuracy at least 90% and greedy exact match at least 80%;
2. current and anchor accuracy at least 95%, historical at least 85%, and
   absent-value accuracy at least 90%;
3. every family at least 85%;
4. at least five-point accuracy gain over the strongest non-oracle comparator,
   with source-group bootstrap 95% lower bound above two points;
5. answer-NLL gain with 95% lower bound above zero;
6. no negative projection seed, family or length gain;
7. exact agreement with the independently replayed semantic state oracle;
8. content gate, prompt reconstruction, provenance, resource, state and result
   audits all pass.

The matched confirmation uses the frozen scale and identical gates without
retraining or calibration. There is one attempt. Failure leaves OOD
ungenerated.

## External OOD gate

After a passing independently audited matched confirmation, score the five OOD
suites exactly once. The full ledger must reach at least 85% macro candidate
accuracy, 75% in every suite, 80% greedy exact match, 90% current/anchor, 80%
historical, 85% absent-value accuracy, and at least a three-point gain over the
strongest non-oracle arm with a positive source-group 95% lower bound. State
oracle agreement and every integrity audit must remain exact.

## Claim unlocked by a complete pass

A complete chain supports the claim that causal, addressable record-state
transitions provide robust selective forgetting for a decomposable Qwen
residual fast memory, including reset, revocation, overwrite restoration and
absence. It would establish actual LM-head readout and generation under fresh
KV, rather than only component-space retrieval.

It would not establish natural-text event discovery, original dense-writer
content binding or gains on LoCoMo/RULER/MMLU. A natural explicit-lifecycle
transfer must be separately frozen and passed before those broader claims.
