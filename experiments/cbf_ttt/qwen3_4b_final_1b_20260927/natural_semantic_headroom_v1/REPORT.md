# Natural semantic-conflict headroom after V19

## Question

V19 showed that exact complete-key deletion explains most controlled selective
forgetting gains, while its learned controller adds no significant benefit.
This experiment asks whether natural paraphrases create a reproducible causal
gap that exact surface matching cannot solve.

It evaluates forgetting only: the current update is always retained. The
source is the audited CounterFact/MQuAKE development inventory with five
relation- and subject-disjoint folds. The memory substrate is contextual
evidence scored by the final 1B-token Qwen3-4B slow weights with fresh KV and
native TTT adapters excluded.

## Data and isolation

The deterministic builder created an 80-row pilot and a disjoint 320-row
sealed confirmation. Each fold contains 16 pilot and 64 confirmation rows.
Lengths 8/16/32/64 and the four registered mechanisms are exactly balanced.
The corpus covers 36 relations and contains 73/7 CounterFact/MQuAKE pilot rows
and 299/21 confirmation rows.

The exact-surface rule deletes zero components by construction. The semantic
oracle deletes stale components only when subject and source relation agree;
same-relation other-subject distractors remain. Pilot/formal subjects and
source rows are disjoint. Nine independent construction checks pass.

## Pilot

| Arm | Joint | Update | Retention | Locality | Generation joint |
|---|---:|---:|---:|---:|---:|
| semantic oracle | 95.00% | 97.50% | 100.00% | 100.00% | 23.75% |
| exact surface / retain all | 51.25% | 70.63% | 100.00% | 100.00% | 22.50% |
| outer-fold fixed | 57.50% | 99.38% | 74.38% | 100.00% | 1.25% |

Semantic deletion improves joint success over exact matching by 43.75 points.
The subject-cluster 95% interval is [32.50, 55.00] points and every fold is
positive. All six pilot checks pass. The seven MQuAKE pilot rows have zero
gain, so the pilot alone cannot support a cross-source claim.

The first GPU attempt stopped before one completed row because the variable
answer scorer inherited a multi-token-only path. A one-token answer called the
model on an empty float tensor. The fixed scorer directly uses the prompt's
last hidden state for one-token answers. The successful scoring run completed
all 80 rows. Its original runner then exited after summary creation because
the independent auditor missed one constant import. A read-only audit recovery
copied the immutable raw rows and summary, reran no model, and passed all eight
checks. The recovered summary SHA256 is
`64c9d8c7025ad48c0847a4fb2035634b4bd2fa4218a253431f2a45f4ddd587dc`.

## Sealed confirmation

| Arm | Joint | Update | Retention | Locality | Generation joint | Joint answer NLL |
|---|---:|---:|---:|---:|---:|---:|
| semantic oracle | **92.19%** | 95.31% | 99.22% | 100.00% | 32.19% | **0.4361** |
| exact surface / retain all | 36.56% | 59.22% | 98.91% | 100.00% | 17.19% | 0.7094 |
| outer-fold fixed | 48.75% | 99.22% | 71.25% | 100.00% | 1.56% | 1.8576 |

The semantic-minus-exact joint gain is **55.63 points**. The subject-cluster
95% interval is **[50.00, 61.25]** points and the relation-cluster interval is
**[49.17, 61.74]** points. Fold gains are 53.13/62.50/54.69/54.69/53.13
points. CounterFact and MQuAKE gains are 56.86 and 38.10 points, respectively.
The semantic arm preserves 78.09% of the exact-to-semantic gap after replacing
exact with the strongest non-oracle comparator.

All ten preregistered scientific checks and all eight independent outcome
checks pass. The formal summary/audit SHA256 values are
`5301773ddf293959ac7c9ac0fd015cc3dfc504f1d44ac0aef821c1db5c13a830`
and `00e8ca36d7275014fb939fc71a7e5bffeff010dc3d446a99ee0410a89f31549d`.

## Resources

The confirmation scored 1,600 unique memory states and 8,000 queries. The
slower shard took 677.73 seconds. Mean/median/p95 query latency was
0.1645/0.1480/0.2968 seconds. Peak allocated/reserved memory was
8.096/8.527 GiB. The pilot used 400 unique states, 2,000 queries and 170.55
seconds on its slower shard.

## Supported conclusion

Natural surface variation creates a large, independently replicated gap
between exact matching and relation-aware selective forgetting. The effect is
positive across every fold and both data sources, while retained facts and
locality remain near perfect. This supplies the missing motivation for a
learned semantic resolver that V19's explicit-key corpus could not test.

The semantic arm uses source relation IDs and is an oracle. It is not yet a
deployable learned controller, an end-to-end native dense-writer result, or a
natural benchmark improvement. The registered next stage is a relation- and
subject-disjoint exact-rule residual controller. LoCoMo, RULER/RULER-QA and
MMLU remain locked until that learned stage passes.

## Reproduction

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_post_v19_natural_semantic_data_repro \
  bash scripts/run_post_v19_natural_semantic_headroom_data.sh

ROOT=/home/ctj/cbf_ttt_post_v19_natural_semantic_pilot_repro \
  DATA_ROOT=/home/ctj/cbf_ttt_post_v19_natural_semantic_data_repro/data \
  bash scripts/run_post_v19_natural_semantic_pilot.sh

ROOT=/home/ctj/cbf_ttt_post_v19_natural_semantic_formal_repro \
  DATA_ROOT=/home/ctj/cbf_ttt_post_v19_natural_semantic_data_repro/data \
  PILOT_RELEASE_ROOT=/home/ctj/cbf_ttt_post_v19_natural_semantic_pilot_repro \
  bash scripts/run_post_v19_natural_semantic_formal.sh
```

