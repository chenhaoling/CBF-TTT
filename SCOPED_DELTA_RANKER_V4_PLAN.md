# Scoped Delta Ranker V4 preregistration

## Motivation

The best observed CounterFact development policy identifies the correct old
memory in 79.62% of update events. Its on-policy forgetting gate does not
improve causal query accuracy. Error inspection fixed before V4 found 115/785
updates selecting the wrong memory for the same subject and 45/785 selecting a
different subject. V4 therefore changes the scope representation and does not
tune another deletion threshold.

## Development data

- All already-observed ZsRE counterfactual splits may be used for fitting.
- CounterFact remains development-only and is evaluated with five subject/session
  disjoint outer folds.
- LongMemEval-S and LoCoMo remain sealed.
- The current final Qwen3-4B 1B-token checkpoint is frozen.

The corpus builder emits every sequential decision and labels candidate pairs
as `same_chain_positive`, `same_subject_different_chain`, or
`different_subject`. Chain identifiers are retained only for construction and
auditing. They are never model features.

## Registered representations

For each current-query / old-memory pair, V4 computes:

1. the existing frozen Qwen full-pair hidden state and lexical/cosine features;
2. a relation-focused pair hidden state after replacing the known subject span
   with `<ENTITY>` in both texts;
3. relation-template cosine and lexical features;
4. a one-bit subject-match feature.

The primary `structured_full` arm uses all four feature groups. Ablations remove
the relation-focused features or the subject-match feature. Subject masking is
an oracle-segmentation diagnostic: a later stage must replace supplied subject
spans with an automatic extractor before a final claim.

## Training and selection

- Candidate ranking is trained only on events whose desired action is local.
- Five CounterFact outer folds are deterministic by session ID.
- For each outer fold, one other fold selects an epoch; the model is then
  retrained on ZsRE plus all four non-outer CounterFact folds for that epoch.
- Loss is listwise cross entropy plus a margin loss against same-subject,
  different-chain negatives.
- Three initialization seeds are ensembled by mean logits.
- No deletion gate is trained in this stage.

## Primary metrics and gate

The primary arm must satisfy every check on CounterFact out-of-fold predictions:

- candidate top-1 accuracy at least 88%;
- MRR at least 93%;
- at least five percentage points top-1 gain over V2;
- session-paired bootstrap 95% confidence-interval lower bound above zero;
- target beats the same-subject hard negative in at least 85% of applicable
  update events;
- target beats every different-subject candidate in at least 95% of update
  events.

If the gate fails, causal gate training remains stopped. If it passes, the next
stage freezes the ranker and evaluates a selective low-rank delta bank with
causal on-policy forgetting.

