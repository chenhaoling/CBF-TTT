# Qwen structured fact comparator S1

## Motivation and frozen decision

The state-equivalent V10 controller reached 99.01% deletion precision and
77.46% recall, but made six false deletions where the preregistered absolute
maximum was four.  Its error analysis and the preceding V8/V9 experiments
agree on the missing capability: lexical or frozen-vector features do not
reliably decide whether two natural statements address the same relation slot.
Further threshold or small-head tuning on CounterFact, ZsRE, FewRel
development, or DocRED development is closed by their respective stop rules.

S1 changes both the learning object and the development source.  It adapts the
final 1B-token Qwen3-4B checkpoint with LoRA to emit an auditable structured
comparison of two facts.  A deterministic state rule, rather than a learned
action head, maps that comparison to `duplicate`, `update`, or `retain`.
Only the original FewRel **training** relations provide weight updates.  All
selection and release decisions move to the previously unused SemEval-2010
Task 8 source.  This document is frozen before constructing S1 examples or
running a Qwen S1 forward pass.

S1 is a policy experiment over an explicit session-memory ledger.  It does not
alter the original dense `ttt_proj`/`ttt_conv` writer, whose content-binding
limitation is already separately documented.  A later causal integration is
conditional on this semantic gate.

## Immutable sources and sealing

### Weight-training source

Use the audited official THU FewRel repository at commit
`278a2315d2138810a379cd8d5718914dc56e2582`.  Reconstruct the relation
partition registered in `FEWREL_SELECTIVE_FORGETTING_PLAN.md` and use only its
48 `train` relations from `train_wiki.json`.  Do not read prior FewRel model
features or scores.  The eight old development relations and both old test
partitions are excluded from S1 output.

### Independent calibration and development source

Use the THUNLP OpenNRE SemEval release downloaded from the URLs in
`benchmark/download_semeval.sh`:

| File | Role | SHA256 | Rows |
|---|---|---|---:|
| `semeval_rel2id.json` | audit vocabulary | `c3153e7d20a127c82fbc9b7542fcdc8e48714bcf9165fafd611dadaed22c7de8` | n/a |
| `semeval_train.txt` | calibration only | `5a53be5e2dfbf865f77c287deec7b9e8a12be333ffb6157bfb5a692250a06220` | 6,507 |
| `semeval_val.txt` | one-shot development gate | `a91170e7c19354a895a58e582bf7fad43994467a0d27a8632319a70840b88e02` | 1,493 |
| `semeval_test.txt` | sealed external confirmation | `1f33efe3db4dc6ff99f52416331682f3ec6759de526786d3629e6b43f8020795` | 2,717 |

The builder verifies the test hash and line count but must never parse a test
row or write a derived test example.  S1 development failure leaves it sealed.
`Other` is heterogeneous and is excluded from positive/anchor construction;
it may not be treated as one semantic relation.

The first model-free build stopped before creating its output directory because
SemEval validation contains a single `Entity-Destination(e2,e1)` row, so the
registered different-subject positive cannot exist.  Before any S1 model
execution, the eligibility rule is clarified uniformly for every split:
exclude a relation class unless it has at least two distinct subjects and two
distinct values.  Record every excluded source row and reason in the manifest.
This removes the one validation singleton rather than fabricating a positive;
all remaining SemEval calibration/development relations and every FewRel
training relation satisfy the rule.

## Structured example construction

Render each source sentence with explicit `[SUBJECT]...[/SUBJECT]` and
`[VALUE]...[/VALUE]` spans.  For every eligible anchor construct exactly five
deterministic examples:

1. `duplicate`: the same fact twice (`E=1,R=1,V=1`);
2. `update`: replace only the value with a different value from the same
   relation (`E=1,R=1,V=0`);
3. `same_relation_other_entity`: a naturally different statement of the same
   relation (`E=0,R=1`, with a different value);
4. `same_entity_other_relation`: choose a lexically hard different-relation
   statement and replace only its subject with the anchor subject
   (`E=1,R=0`);
5. `unrelated`: a deterministic different-entity, different-relation pair
   (`E=0,R=0`).

The action is a fixed physical rule:

```text
duplicate iff E=1 and R=1 and V=1
update    iff E=1 and R=1 and V=0
retain    otherwise
```

Source relation/entity metadata constructs labels and audits only.  Model
input contains only the two marked natural statements and the output schema.
Every source instance occurs in one source split.  Pair IDs, donor choices,
and caps are SHA256 deterministic under seed `24117`.

## Model and training protocol

Load the final 1B-token Qwen3-4B checkpoint with native TTT disabled.  Train
LoRA adapters only, initially with rank 16, alpha 32, dropout 0.05, on
`q_proj`, `k_proj`, `v_proj`, and `o_proj`.  The supervised answer is canonical
single-line JSON with keys in this order:

```json
{"same_entity":true,"same_relation":true,"same_value":false,"operation":"update"}
```

The pilot uses seed `24117`, BF16, maximum length 384, effective batch size 64,
AdamW, learning rate `2e-4`, and checkpoints after 0.25, 0.5, 1.0, and 2.0
epochs.  Use weight decay 0.01, three-percent linear warmup followed by cosine
decay, gradient clipping at 1.0, per-GPU batch four on two GPUs, and eight-way
gradient accumulation.  Only the SemEval calibration split may select checkpoint and a minimum
generation-confidence threshold.  Below that threshold the policy defers to
`retain`.  Architecture, prompt, rank, optimizer, and field definitions may
not be changed after observing development.

Before calibration scoring, freeze the structured decoder as eight candidate
JSON strings, one for each Boolean `(E,R,V)` state with its operation fixed by
the physical rule. Select the minimum mean target-token NLL; confidence is the
NLL gap between the best and second candidate. This prevents arbitrary text
parsing from contaminating field metrics. Calibration selects the checkpoint
and minimum confidence at 99% deletion precision. On development, additionally
run unconstrained greedy generation over every example (`max_new_tokens=80`)
and compute JSON validity with exact key order, Boolean types and physical
operation consistency. Candidate likelihood supplies semantic and action
metrics; unconstrained generation supplies the registered format-validity
metric. The zero-shot baseline uses the identical candidate decoder and its own
calibration-only threshold.

If the pilot development gate passes, repeat the frozen configuration with
seeds `24117/24119/24121`.  The external test is opened once only after all
three runs pass the confirmation-development gate.

## Baselines and metrics

Report exact JSON validity, exact four-field accuracy, per-field precision,
recall/F1, operation confusion, and each of the five construction kinds.
Deletion metrics treat `duplicate` and `update` as delete-compatible and
`retain` as unsafe to delete.  Report relation-macro and instance-micro
metrics, calibration coverage, latency, peak allocated/reserved memory, and
all raw predictions.

Baselines are exact lexical identity, character-trigram similarity, zero-shot
Qwen3-4B with the same prompt, and the frozen FewRel cross-controller when its
input is available.  No baseline may set the S1 confidence threshold.

## One-shot development gate

The pilot passes only if all checks hold on SemEval validation:

- JSON validity at least 99.5%;
- exact four-field accuracy at least 85%;
- same-relation recall at least 80% and specificity at least 95%;
- delete-compatible precision at least 99.0% and recall at least 25%;
- `same_entity_other_relation` specificity at least 98%;
- relation-macro balanced accuracy at least 85%;
- delete recall exceeds the best calibrated non-adapted baseline by at least
  ten percentage points at the same 99% precision constraint.

The three-seed confirmation additionally requires every seed to pass, a
positive relation-bootstrap 95% lower bound for the recall gain, and a maximum
two-point range in deletion precision.

Failure stops S1 tuning on SemEval and keeps its test sealed.  The permitted
next experiment must introduce a new supervision source or an explicit
retrieval/reranking decomposition; it may not relax these thresholds or tune
the same S1 prompt/rank on validation errors.

## Conditional causal integration

Only a passed, independently audited external confirmation may enter the
explicit session-ledger replay.  Compare retain-all, global forgetting,
last-k, recency, similarity, V10, oracle structure, and S1 under matched write
streams.  Then report stale-fact removal, unrelated-memory retention, duplicate
deduplication, query accuracy, latency, and memory footprint.  The native dense
writer remains an ablation until it independently demonstrates content
binding; ledger success must not be described as dense-writer success.
