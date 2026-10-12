# Cross-domain forgetting reranker plan

## Motivation fixed before scoring

The audited CounterFact zero-shot run preserved a useful conflict signal but
failed the registered reliability gate. Its delete probability achieved ROC AUC
0.759, while only 49.08% of triggered CounterFact updates selected the correct
old memory. A static threshold cannot repair wrong candidate identity. The next
method therefore remains forgetting-only (`g_t = 1`) and decomposes

1. **which old memory is in the same factual scope**, and
2. **whether that candidate should be forgotten**.

It does not revive the current-input write gate and does not use hidden chain IDs
as model inputs.

## Development data status

All ZsRE calibration, development, and test results and the complete CounterFact
zero-shot result are already observed. They are development-only from this point.
They may be used to diagnose and train the next architecture, but no subsequent
score on either source is a final held-out claim.

The initial experiment has two arms:

- **ZsRE-only transfer:** train and tune only on observed ZsRE sessions, then
  evaluate on observed CounterFact. This is a diagnostic comparison with the
  earlier frozen controller.
- **CounterFact grouped OOF:** deterministic five-fold session-disjoint
  evaluation. Each outer fold is excluded from head fitting and threshold
  selection. The other folds plus ZsRE train the candidate ranker; one inner
  CounterFact fold selects checkpoints and the safe-delete threshold.

## Model

For every current input and old-memory candidate, reuse frozen Qwen3-4B features:

- three contextual-key cosine scores;
- lexical and relation-token overlap features;
- the frozen joint pair representation from the existing scope template.

A residual linear ranker starts from the ensemble-cosine ranking and learns a
pairwise correction using local-update events. A separate binary gate sees only
the selected pair and score geometry. It deletes the selected memory only when
its probability exceeds an inner-selected threshold. The current item is always
written. Candidate rank and gate checkpoint selection are nested inside each
outer fold.

## Controls and metrics

Compare against frozen cosine-local, fixed retain, the previous ZsRE controller,
and oracle local deletion. Report:

- old-chain top-1 accuracy on local updates;
- overall, hard-retain, and update action accuracy;
- candidate-memory query accuracy by exact, rephrase, and historical query;
- subject-paired bootstrap differences against the previous frozen controller;
- runtime and peak GPU memory.

The development gate for advancing to a new benchmark requires CounterFact OOF:

1. candidate top-1 accuracy at least 75%;
2. overall action accuracy at least 85%;
3. hard-retain accuracy at least 85%;
4. update action accuracy at least 75%;
5. query accuracy above fixed retain (49.44%).

Failure of the candidate threshold rejects this feature family. Failure only of
the gate thresholds motivates calibration changes without changing the ranker.

## Sealed final sources

No model score from the next method may be computed on these files during method
selection:

- LongMemEval-S cleaned, SHA256
  `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`;
- LoCoMo 10, SHA256
  `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.

Only schema and count inspection has occurred: LongMemEval-S contains 500
questions, including 78 `knowledge-update` examples; LoCoMo contains 10 source
conversations and 1,986 questions. If the development gate passes, freeze the
architecture and construct a chronological LongMemEval-S knowledge-update test
from `answer_session_ids`. LoCoMo remains an independent robustness source.

