# Preregistered causal-ledger lifecycle OOD V1

## Freeze point

This protocol completes the one-time OOD stage reserved in
`QWEN_CAUSAL_LEDGER_RESIDUAL_PLAN.md`. It is frozen after the matched
confirmation and native-writer diagnostic, before constructing any OOD row or
inspecting an OOD model score. The confirmation fixed residual scale 48. No
scale, prompt, policy, threshold or comparator may be tuned on this split.

## Corpus

Use reserved seed `202610162`, a new surface template, disjoint identifiers and
the fourth quarter of the opaque five-token value space. Construct exactly 64
counterfactual source groups per suite, two twins per group and three sparse
decision boundaries per episode. This yields 320 groups, 640 episodes and
5,760 query rows. The five frozen suites are:

1. sequence lengths 24 and 32;
2. two distant record revocations at one boundary;
3. revocation of the newest same-key assignment with older-value restoration;
4. reset followed by exact duplicate/idempotent writes;
5. reset/revoke/overwrite compositions unseen in development.

Each boundary queries the current key, a lifecycle-relevant historical key and
the protected anchor. Null answers use the same causal candidate rule as the
confirmed split. Teacher actions, future events and confirmation outcomes never
enter the method state.

## Frozen methods and gate

Replay No Memory, Retain Latest, Global Boundary Clear, Last 1, Last 2, Dense
Full, Full Causal Ledger and the independently supplied Semantic Oracle. Use
the unchanged Qwen3-4B checkpoint, frozen LM head, fresh attention KV and scale
48 structured residual.

The full ledger passes only if all hold:

- macro candidate accuracy is at least 85%;
- candidate accuracy is at least 75% in every suite;
- five-token exact accuracy is at least 80%;
- current and anchor exact accuracy are at least 90%;
- historical exact accuracy is at least 80%;
- absent/null exact accuracy is at least 85%;
- exact gain over the strongest non-oracle comparator is at least three points
  and its source-group bootstrap 95% lower bound is above zero;
- answer-NLL gain has a bootstrap 95% lower bound above zero;
- candidate prediction and generated IDs agree exactly with the semantic
  oracle on every row;
- corpus, dependencies, rows, hashes, resources, aggregate and terminal state
  pass an independent audit.

Pass releases natural-text scope and lifecycle transfer. Failure stops that
release and permits diagnosis only on a separately created development corpus;
this OOD split cannot be rerun with changed choices.
