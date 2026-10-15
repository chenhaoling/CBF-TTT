# Qwen natural relation retriever and reranker S2

## Motivation and frozen decision

S1 passed its engineering audits but failed its one-shot SemEval development
gate. It reached 99.84% deletion precision and 99.96% recall while raw
same-relation specificity was only 59.04%. Exact lexical and trigram baselines
also reached 100% deletion precision and recall because every S1 update or
duplicate reused the anchor relation template. S1 therefore measured a
construction shortcut rather than semantic slot equivalence.

S2 changes the learning object and the data construction before reading a new
development source. It learns only whether two **distinct, natural relation
mentions** express the same relation. Subject and value strings are replaced
by placeholders before retrieval or reranking. An explicit session ledger
supplies normalized entity and value equality. The unchanged physical rule
maps those three signals to `duplicate`, `update`, or `retain`.

This is the explicit retrieval/reranking decomposition permitted by the S1
stop rule. S1 prompt, rank, threshold and SemEval validation errors are closed.
The native `ttt_proj`/`ttt_conv` writer remains a separate downstream ablation.

## Sources and split sealing

### Weight training

Use only the same 48 registered FewRel training relations from the official
THU FewRel repository at commit
`278a2315d2138810a379cd8d5718914dc56e2582`. The `train_wiki.json` SHA256 is
`8e9e58a05d559be773cbe18a8d4a16ba17e6d397663f9f05a056f894a94b92c1`.
No old V8--V10 prediction, feature, action or score may enter S2.

### Calibration

SemEval train remains calibration-only. It was already observed by S1 and is
not claimed as new evidence. Its SHA256 is
`5a53be5e2dfbf865f77c287deec7b9e8a12be333ffb6157bfb5a692250a06220`.
It may select a checkpoint, reranker threshold and retrieval `k` from the
finite registered grid, but may not change the architecture or construction.

### Fresh one-shot development

Use official FewRel 2.0 `val_pubmed.json` once. At this registration point only
its hash and byte count have been read; no JSON object has been parsed:

- SHA256: `fc0f13b534f6664fe0d8aec231d7590313bc6510d00dbc1b6746147f6a2b46e6`;
- bytes: 432,600;
- source cardinality after the first registered parse: 10 biomedical relations
  with 100 mentions each (1,000 total).

The preregistration initially described the source as 25 relations based on a
FewRel 2.0 domain-level summary. The first post-registration schema read showed
that this pinned validation file contains 10 relation keys. This source-count
correction was made before pair construction or any S2 model execution and
does not change the method, thresholds or release gates.

### Sealed external confirmation

SemEval test remains sealed at SHA256
`1f33efe3db4dc6ff99f52416331682f3ec6759de526786d3629e6b43f8020795`.
It is opened once only after the pilot and all three S2 seed confirmations pass
on PubMed. DialogRE is a possible later dialogue-domain replication and is not
part of S2 model selection.

## Natural pair construction

Every relation comparison uses two different source mentions. Replace the
annotated subject and value spans with `[SUBJECT]` and `[VALUE]` before they
reach either model. Pair labels and relation IDs are audit-only fields.

For each eligible anchor construct five ledger scenarios:

1. `paraphrased_duplicate`: a second natural mention of the same relation;
   ledger entity and value are equal;
2. `paraphrased_update`: a second natural mention of the same relation; ledger
   entity is equal and value differs;
3. `same_relation_other_entity`: a second natural mention of the same relation;
   ledger entity and value differ;
4. `same_entity_other_relation`: a natural mention of a different relation;
   ledger entity is equal and value differs;
5. `unrelated`: a natural mention of a different relation; ledger entity and
   value differ.

The model receives the two masked natural mentions only. The renderer must not
copy one relation surface into the other. Ledger equality bits are combined
with the relation decision only after scoring. The physical action remains:

```text
duplicate iff E=1 and R=1 and V=1
update    iff E=1 and R=1 and V=0
retain    otherwise
```

Candidate pair selection is SHA256 deterministic under seed `25117`. Positive
and negative relation pairs are matched by masked character-trigram similarity
bins of width 0.05. Within each split, retain the same number of positive and
negative pairs in every occupied bin. This prevents global lexical overlap
from becoming a label proxy. Source-pair IDs may occur in one split and one
direction only. The heterogeneous SemEval `Other` class is excluded.

## Pre-model shortcut gate

Before any S2 model forward pass, an independent auditor must verify:

- every pair uses two distinct source IDs and two non-identical masked texts;
- all five ledger scenarios exist in equal anchor groups;
- operation labels exactly follow the fixed physical rule;
- source relations and pair labels agree;
- no train/calibration/development source or pair overlap;
- positive and negative trigram-bin counts match exactly;
- exact masked-surface matching has zero positive coverage;
- a calibration-fitted trigram threshold has at most 50% relation recall and
  at most 50% delete recall at 99% precision on fresh PubMed development.

Failure stops before LoRA training. Construction rules may be corrected only
if the failure is an integrity bug; they may not be tuned to PubMed labels.

## Retrieval and reranking

### Retrieval

Obtain one frozen Qwen3-4B representation per masked mention from the final
1B-token checkpoint, with native TTT disabled. Mean-pool the final hidden state
over non-padding tokens, L2 normalize, and retrieve by cosine similarity.
Evaluate `k` in the frozen set `{4, 8, 16}` on calibration only. Retrieval is
candidate generation; it never authorizes deletion by itself.

### Reranking

The cross-encoder prompt contains two masked natural mentions and asks only for
canonical `{"same_relation":true}` or `{"same_relation":false}`. Train LoRA
on balanced FewRel relation pairs with rank 16, alpha 32 and dropout 0.05 over
`q_proj`, `k_proj`, `v_proj`, and `o_proj`. Use BF16, maximum length 384,
AdamW, learning rate `2e-4`, weight decay 0.01, three-percent warmup, cosine
decay, gradient clipping 1.0 and effective batch size 64 on two GPUs.

Save checkpoints after 0.25, 0.5 and 1.0 epochs. Score the two canonical
suffixes with a shared prompt cache. The signed NLL difference is the relation
score. Calibration selects the checkpoint and threshold that maximize
same-relation recall subject to 99% delete precision; ties use relation-macro
balanced accuracy and then the earlier checkpoint. No free generation is
needed for the deployed decision.

## Baselines and measurements

Compare exact masked surface, masked character trigram, BM25, frozen Qwen
cosine, zero-shot Qwen cross-encoder, S1, oracle relation, reranker without
retrieval and retrieval plus reranking. Report:

- retrieval recall@4/8/16 and candidate count;
- relation precision, recall, specificity, F1 and ROC-AUC;
- relation-macro balanced accuracy;
- operation confusion and all five scenario accuracies;
- delete precision, recall, false-positive count and coverage;
- latency, throughput, peak allocated/reserved memory and ledger footprint;
- per-relation bootstrap confidence intervals and all raw decisions.

## One-shot PubMed gate

The pilot passes only if every condition holds:

- retrieval recall@16 at least 95%;
- same-relation recall at least 80% and specificity at least 95%;
- relation-macro balanced accuracy at least 85%;
- delete precision at least 99%, delete recall at least 50%, and no more than
  four false-positive deletes;
- `same_entity_other_relation` deletion specificity at least 98%;
- delete recall exceeds every safe non-adapted baseline by at least ten points;
- the reranker improves delete recall by at least five points over frozen Qwen
  cosine at the same precision constraint.

If the pilot passes, repeat seeds `25117/25119/25121`. Every seed must pass;
the relation-bootstrap 95% lower bound of the recall gain must be positive and
the deletion-precision range must be at most two points. Only then may the
SemEval test be opened once.

Failure seals the external test. The next branch must add a new natural
supervision source or explicit entity-linking evidence; thresholds may not be
relaxed on PubMed errors.

## Conditional causal session experiment

Only an audited external pass permits session-ledger replay. Compare retain
all, global reset, last-k, recency, lexical similarity, frozen Qwen similarity,
V10, S1, S2, oracle relation and S2 without retrieval under identical event
streams. Measure stale-fact removal, unrelated-memory retention, duplicate
deduplication, downstream answer accuracy, long-session scaling and cost.
Repeat S2 actions with the original dense `ttt_proj`/`ttt_conv` writer only as
a named ablation; ledger evidence must not be attributed to dense fast weights.
