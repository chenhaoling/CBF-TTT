# V20 Semantic-Alias Headroom Confirmation

## Freeze point and scope

This protocol and alias bank are frozen while V19 model scoring is still in
progress and before any V19 arm outcome is read. V20 can be constructed only
if the independently audited V19 causal-rule diagnostic concludes that an
explicit-key rule closes at least 90% of the registered V19 oracle gap.

V20 tests whether selective forgetting still has model-level headroom after
removing exact surface identity. It does not train or select a controller.
Session notes remain contextual evidence memory scored by the frozen Qwen3-4B
slow model with fresh query KV. This stage does not claim that the original
dense `ttt_conv/ttt_proj` writer learned a readable representation.

## Frozen construction

- 320 episodes, 80 subjects, 20 template groups and five outer folds;
- four balanced lengths (8/16/32/64) and four interference mechanisms;
- a new bank of 20 latent slots, each with historical, update and query surface
  forms that are pairwise different;
- fresh seed 2003, entity namespace and opaque-value range beginning at 16000,
  disjoint from V18 and V19;
- historical target facts use surface A, the current update uses surface B and
  the current query uses surface C;
- exact-string conflict deletion therefore removes zero target components;
- the latent semantic oracle removes every stale component for latent slot 0
  and always retains the current update;
- no external benchmark or future outcome is loaded during construction.

## Frozen arms and primary comparison

Score no-memory, retain-all, global-reset, last-1, last-2, exact-string
conflict and the latent semantic oracle. Duplicate memory states are evaluated
once. The fixed comparator is selected using the other four outer folds.

The sole primary comparison is latent semantic oracle versus exact-string
conflict. Report candidate and independent greedy joint success, update and
retained-key accuracy, locality, answer NLL, subject/template clustered 95%
intervals, all folds, latency and peak memory.

## Frozen success gate

Semantic selective-forgetting headroom is confirmed only if all hold:

1. semantic-oracle joint gain over exact matching is at least 10 points;
2. subject and template 95% interval lower bounds are above zero;
3. semantic-oracle update accuracy is at least 90%;
4. retained-key accuracy is at least 90%;
5. locality is at least 97%;
6. no outer fold has negative joint gain;
7. exact matching removes zero components and scores identically to retain-all;
8. corpus, raw outcome, model/tokenizer and recomputation audits all pass.

Passing releases training of a semantic conflict resolver and forgetting
policy with nested out-of-fold subject/template splits. Failure stops the
synthetic semantic-forgetting claim and blocks natural-benchmark expansion.
