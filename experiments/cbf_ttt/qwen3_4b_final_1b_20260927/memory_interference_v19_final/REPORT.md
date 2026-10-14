# V19 selective-forgetting confirmation and learned-policy result

## Scope

This archive closes the V19 controlled contextual-memory experiment. The
current update is always retained. Every learned or rule-based action only
decides which earlier session-memory components survive, so these results are
evidence about selective forgetting rather than an update gate.

The scorer uses the 1B-token Qwen3-4B checkpoint with native TTT adapters
excluded and fresh attention KV. Historical statements are supplied as
contextual evidence. The result therefore does not establish readable content
binding in the original dense `ttt_conv`/`ttt_proj` writer.

## Independently confirmed causal headroom

The formal confirmation contains 320 episodes, 80 subjects, 20 template
groups and five subject/template-disjoint folds. The fixed comparator and the
registered 3% locality-budget oracle were selected without access to the
held-out fold.

| Policy | Joint success | Update | Retention | Locality | Generation joint |
|---|---:|---:|---:|---:|---:|
| outer-fold fixed / retain-all | 39.375% | 66.250% | 100.000% | 100.000% | 14.375% |
| registered oracle | 96.250% | 100.000% | 96.719% | 100.000% | 65.313% |

The oracle gain is 56.875 points. The subject-cluster 95% interval is
[52.813, 61.250] points and the template-cluster interval is
[52.188, 61.875] points. Every fold is positive and no locality failure was
observed. The preregistered confirmation gate passed.

## Exact causal rule

Deleting every stale component with the same complete explicit key reached
86.563% joint success, 89.688% update accuracy, 100% retention and 100%
locality. Its gain over fixed retain-all is 47.188 points, with subject and
template cluster intervals [42.500, 51.875] and [41.875, 52.813] points. It
closes 82.97% of the positive oracle gap.

This is strong mechanistic evidence that selective removal of stale memory is
responsible for most of the controlled gain. It did not meet the registered
90% gap-explanation threshold, leaving a residual opportunity for a learned
policy.

## Nested out-of-fold policy

The causal-label builder produced 9,280 examples on the `alpha` grid
`{0, 0.5, 1}`. It contains 1,280 exact-conflict rows and 7,662 conservation
rows. No example had a positive locality-risk label, so the risk head is not
identified by this corpus.

The two-head MLP was selected with nested five-fold cross fitting and three
seeds. Only one of five inner selections met the registered positive-recall
and conservation constraints. The model-level result was:

| Policy | Joint success | Update | Retention | Locality | Generation joint |
|---|---:|---:|---:|---:|---:|
| learned policy, three-seed mean | 86.667% | 91.094% | 100.000% | 100.000% | 46.250% |
| exact complete-key deletion | 86.563% | 89.688% | 100.000% | 100.000% | 45.313% |

The learned-minus-exact gain is 0.104 points. Its subject-cluster 95% interval
is [-1.563, 1.875] points and its template-cluster interval is
[-1.458, 1.667] points. The registered two-point improvement, positive
subject-cluster lower bound and all-inner-constraints checks failed. The
terminal scientific status is `v19_nested_policy_not_supported`; V19 policy
iteration must stop.

## Audit recovery

The first model-evaluation runner exited after writing the complete raw rows
and unchanged scientific summary because the independent auditor counted
shard states in policy-file order. Policy states are serialized by outer fold,
while collection shards the original episode order. The auditor was fixed to
join by `episode_id` and count in immutable episode order.

No model was rescored. The recovered summary SHA256 remains
`a5d7659ea585994ac8731742792bf4e4e324ab8523304c21e3f924abd012227d`.
The corrected audit SHA256 is
`437942b6151c53998d5716429c15b1deab03f7104cb62d0eafda786bc073b00b`,
all twelve independent checks pass, and the recovery runner exited zero.

## Resources

- Confirmation: 57,400 query measurements, 11,480 unique states, 0.2734 s
  mean query latency, 0.3458 s p95, and 8.361/8.938 GiB peak
  allocated/reserved GPU memory.
- Learned-policy scoring: 1,880 query measurements, 376 unique states, 0.2509
  s mean query latency, 0.3430 s p95, and 8.361/8.939 GiB peak
  allocated/reserved GPU memory.
- Controller training: 5.31 s and 0.032/0.045 GiB peak allocated/reserved
  memory.

## Claim and next experiment

V19 supports a large, independently replicated causal benefit of selective
forgetting on explicit contextual memory. It also shows that learning the same
explicit-key decision adds no measurable value over a direct causal rule. This
is not evidence that learning is generally useless: the corpus does not
require semantic relation matching and has no positive locality-risk examples.

The next experiment is frozen in
`POST_V19_NATURAL_SEMANTIC_HEADROOM_PLAN.md`. It uses disjoint natural
CounterFact/MQuAKE relation surfaces to test whether semantic conflict
resolution has measurable headroom over exact strings before any new
controller is trained. Native dense-writer experiments are not repeated;
their audited negative content-binding result remains part of the paper's
claim boundary.

## Archived files

- `v19_confirmation_summary.json` and `v19_confirmation_audit.json`
- `v19_exact_rule_summary.json` and `v19_exact_rule_audit.json`
- `v19_policy_examples_manifest.json` and `v19_policy_examples_audit.json`
- `v19_policy_training_summary.json` and `v19_policy_training_audit.json`
- `v19_policy_eval_summary.json` and `v19_policy_eval_audit.json`

