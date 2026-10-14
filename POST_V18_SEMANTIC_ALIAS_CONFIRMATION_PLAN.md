# Post-V18 Semantic-Alias Confirmation Plan

This protocol is frozen while the V18 formal causal-headroom scorer is still
running and before any formal outcome is read. It is a one-shot confirmation
set, not a development set.

## Claim tested

The exact-key V18 set can validate selective conflict removal but cannot show
that a learned policy is needed. The semantic-alias set therefore assigns
three source-disjoint surface forms to one latent slot: a historical form, a
new-update form and a query form. Exact string matching cannot connect them.
A publishable learned-policy claim requires the frozen OOF controller to beat
the exact-key rule while retaining at least 90% update and retained-key
accuracy and 97% locality.

## Frozen construction

- 320 episodes, 80 subjects, four mechanisms and lengths 8/16/32/64;
- 20 alias-only template groups nested in five 64-row outer folds;
- alias-only entities, values, templates and 20 three-surface latent-slot
  families, disjoint from pilot and formal V18;
- the primary historical, update and current-query key strings are different;
- stale repetitions use the historical surface, the final update uses the
  update surface, and the two update queries use historical and third surfaces;
- retained keys remain literal controls; an irrelevant near-duplicate key adds
  a lexical suffix while belonging to a different latent slot;
- the newest component is never deletable.

The builder is hard-gated by the positive independently audited controlled
policy evaluation. The code and alias bank are committed before that result;
generation after release does not permit editing the bank or templates.

## Frozen comparisons

Score retain-all, global reset, exact-string conflict, Qwen-cosine conflict,
the frozen three-seed learned policy, matched random deletion and the latent
semantic oracle. Scale 8 remains primary. No alias result may select a model,
feature, epoch, threshold, residual scale or training loss.

Report subject- and template-cluster 95% intervals, all five folds, candidate
accuracy, answer NLL, independent greedy generation, latency and peak memory.
The learned-policy alias gate requires:

- joint gain over exact string matching at least five points with subject and
  template 95% lower bounds above zero;
- update and retained-key accuracy at least 90%, locality at least 97%;
- no negative outer fold and no seed below exact matching;
- at least 40% of the exact-to-semantic-oracle joint gap closed.

Failure preserves any controlled mechanism result but blocks a learned
semantic-forgetting claim and all natural benchmark confirmation.
