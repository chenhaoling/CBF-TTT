# Preregistered scale-32 causal-ledger content validation V2

## Freeze point

This protocol is fixed after the V1 content gate failed, the LM-head geometry
diagnostic localized the error to contextual token competition, and the
independently audited fixed-grid diagnostic selected scale 32. No V2 corpus,
model score or ledger outcome existed at this freeze point. Development,
confirmation and OOD remain ungenerated.

## Fresh corpus

Use seed `202610163`, a new surface template and a disjoint `Y ...` five-token
value namespace. Construct exactly eight groups per lifecycle family and two
twins per group, preserving the seven V1 families and three sparse boundaries.
Every answer and candidate is five Qwen tokens. The V1 `Z ...` pilot is used
only to select scale 32 and is not pooled into V2 scores.

## Readout arms

All arms use fresh KV and the unchanged Qwen3-4B LM head at fixed scale 32.
Generation stops after exactly five tokens.

1. `single_correct`: one exact key/value component;
2. `single_wrong`: the same key with a deterministic wrong candidate;
3. `zero` and norm-matched sign-randomized controls;
4. `dense_visible`: sum Qwen-key cosine-weighted values over every causally
   visible fact;
5. `addressable_top1`: route an existing key to the most similar causally
   visible key and route an absent key to the fixed null code;
6. `exact_lookup`: dictionary-key semantic oracle.

The top-1 arm is primary. Its exact self-key should agree with `exact_lookup`;
any disagreement is an implementation failure. Dense similarity is a fixed
ablation, not a tunable competitor.

For queries whose paired twin has a different causal answer, additionally
score `twin_counterfactual`. When reset or a later write causally erases the
twin difference, record equality as a negative-control outcome rather than
calling the twin value wrong.

## Registered gate

V2 passes only if all conditions hold:

- primary candidate accuracy at least 95% and five-token exact at least 85%;
- every family primary exact at least 80%;
- current and anchor exact at least 90%, historical at least 80%, and absent
  exact at least 85%;
- exact agreement between `addressable_top1` and `exact_lookup` predictions
  and generated IDs;
- `single_correct` candidate/exact at least 95%/90%;
- `single_wrong` truth candidate/exact at most 10%, while generation matches
  the injected wrong value at least 85%;
- Zero and Random truth candidate accuracy at most 30%;
- on every available twin-counterfactual row, truth exact at most 10% and twin
  answer exact at least 85%;
- source-group bootstrap 95% lower bounds for correct-answer NLL gains over
  Wrong, Zero and Random are above zero;
- model provenance, corpus, row, state, generation, resource and sealed-split
  audits all pass.

Failure stops before development construction. Passing V2 permits fresh
development construction at scale 32 but does not itself establish gains over
fixed/window lifecycle comparators.

