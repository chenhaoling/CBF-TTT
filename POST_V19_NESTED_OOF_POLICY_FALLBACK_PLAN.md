# V19 Nested-OOF Forgetting-Policy Fallback

## Freeze point and release boundary

This fallback was frozen while the two V19 Qwen shards were still running
(84/320 rows written) and before any V19 arm outcome, aggregate or causal-rule
result was read. It is released only when both conditions hold:

1. the independently audited V19 locality-constrained confirmation gate passes;
2. the independently audited V19 exact-key explanation gate fails and names
   `train_v19_nested_oof_causal_policy` as its next stage.

If the exact-key explanation passes, this branch is skipped in favor of V20.
If V19 itself fails, this branch stops without constructing labels.

## Claim boundary

The policy operates on contextual evidence memory and makes no claim that the
original dense `ttt_conv/ttt_proj` writer has readable content binding. The
current update is always retained. The controller decides only which earlier
session components to suppress after observing a new write.

## Frozen causal labels

For each historical component, use only V19 arms from training folds:

- its single-component deletion arm;
- its whole-key deletion arm when it belongs to an exact-key group;
- retain-all as the reference.

Deletion utility is 1 when joint success improves, 0 when it decreases and
0.5 when it ties. Answer NLL breaks training-ranking ties but cannot relabel
joint success. A separate binary risk target records whether the intervention
reduces locality relative to retain-all. Exact stale repetitions receive the
larger causal utility of single and group deletion, together with the risk of
the intervention supplying that utility.

Query text, answer candidates, future scores, row identity, fold identity and
the registered oracle selection are never controller inputs.

## Frozen write-time features

The feature vector is computed from history through the current update only:

- exact key, exact value and exact-conflict indicators with the newest write;
- normalized age;
- later exact-key, same-value and conflicting-value fractions;
- historical same-key and same-value group fractions;
- character 3-gram and common-prefix similarity to the newest key;
- maximum and mean key similarity to later components;
- maximum conflicting-later-key similarity;
- useful-repetition and near-duplicate indicators;
- normalized memory length.

## Frozen learner and nested selection

Use a shared two-layer MLP with 64 hidden units, GELU and layer normalization,
followed by utility and locality-risk logits. Seeds are 19031, 19037 and 19043.
Use AdamW with learning rate 1e-3, weight decay 1e-4 and gradient clipping at
1. Candidate epochs are 25/50/100. Candidate utility thresholds are
0.4/0.5/0.6 and risk penalties are 0/0.5/1/2.

The fixed training loss is smooth-L1 on sigmoid utility (weight 1), binary
cross-entropy on locality risk (weight 0.5), binary conservation loss toward
no deletion on target-0/0.5 components (weight 0.25), and within-episode
pairwise ranking of target-1 over target-0/0.5 utility (weight 0.25).

For outer fold `k`, fold `(k+1) mod 5` is the fixed inner validation fold.
Subjects and template groups never cross outer folds. Inner selection first
requires at least 80% positive-utility recall, 95% conservation specificity
and at most 3% predicted deletions on locality-risk examples. Feasible settings
maximize balanced utility accuracy, then risk specificity, then use fewer
deletions, fewer epochs and the more conservative threshold. If no setting is
feasible, record a failed constraint selection; do not silently relax it.

Refit on all four outer-training folds and emit three-seed out-of-fold memory
states for the untouched fold. Every combined state must be rescored with the
same frozen Qwen3-4B protocol because multi-component intervention outcomes
cannot be inferred by adding single-deletion labels.

## Frozen comparisons and gate

Compare the three-seed policy mean with outer-fold fixed, retain-all,
whole-key exact deletion, the strongest registered causal rule and the V19 3%
oracle. Report candidate scoring and independent greedy generation, update,
retention, locality, joint answer NLL, latency, peak memory and deletion rate.

Policy support requires all of:

1. at least five points of subject- and template-macro joint gain over the
   outer-fold fixed comparator;
2. both clustered 95% interval lower bounds above zero;
3. non-negative gain in every outer fold and every seed;
4. update and retained-key accuracy at least 90%;
5. locality at least 97%;
6. at least 40% of the positive fixed-to-V19-oracle joint gap closed;
7. at least two points of joint gain over whole-key exact deletion with a
   subject-cluster interval lower bound above zero;
8. all label, split, checkpoint, prompt, score, resource and provenance audits
   pass.

Passing releases a source-disjoint semantic and natural-data confirmation.
Failure ends learned-policy iteration on the V19 explicit-key corpus.
