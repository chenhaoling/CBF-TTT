# Cross-domain forgetting replication synthesis v1

## Purpose

Combine the final frozen CounterFact V13 result and the previously completed
one-time ZsRE held-out result without fitting a model, selecting a threshold, or
rescoring either source. Domain-specific clustered confidence intervals remain
the statistical evidence; pooled counts are descriptive because the two query
tasks have different readouts and difficulty.

## Replication result

| Domain | Status | Actions | Overall action | Hard retain | Primary update |
|---|---|---:|---:|---:|---:|
| CounterFact | grouped OOF, observed development | 2,355 | 95.29% | 96.56% | 89.55% |
| ZsRE | one-time subject-disjoint held-out test | 1,548 | 95.16% | 96.12% | 89.34% |

The absolute domain differences are 0.132 points for overall action, 0.436
points for hard retain, and 0.213 points for primary update. All are below the
registered one-point replication tolerance.

The descriptive pooled table contains 3,903 actions:

| Event | Correct / total | Accuracy |
|---|---:|---:|
| Primary initial | 1,299 / 1,301 | 99.85% |
| Hard retain | 1,254 / 1,301 | 96.39% |
| Primary update | 1,164 / 1,301 | 89.47% |
| All | 3,717 / 3,903 | 95.23% |

Both domains independently have positive clustered-bootstrap lower bounds for
query and action gains over their registered baselines:

- CounterFact query gain: +1.949 points, 95% CI `[1.547,2.368]`;
  action gain: +28.620 points, CI `[27.601,29.597]`.
- ZsRE query gain: +6.587 points, 95% CI `[5.430,7.790]`;
  action gain: +12.726 points, CI `[10.271,15.245]`.

All five replication checks pass.

## Interpretation

The nearly identical event decomposition on two different factual-update
corpora is evidence that semantic-scope forgetting is reproducible rather than
a CounterFact-specific threshold artifact. It also exposes a reproducible
bottleneck: about 10.5% of required old-version deletions are still missed,
while false deletion of hard-retain facts is below 4%.

This synthesis does not turn CounterFact into a held-out test, combine unlike
query accuracies into a meta-analytic effect, or repair either experiment's
strict 90% update-floor failure. It does not establish native dense fast-weight
writing. The previous dense-writer content audit failed, whereas the contextual
low-rank Qwen memory passed readability and source-disjoint forgetting gates.

## Next stage

Freeze the current scope/forgetting recipe and validate natural sequential
memory using the contextual low-rank substrate. LongMemEval-S and LoCoMo remain
unscored until their construction, prompts, baselines, judging, cluster units,
and release rule are preregistered. MMLU and RULER are non-regression and
long-context controls rather than primary forgetting evidence.

## Reproduction

```bash
python -m tasks.synthesize_crossdomain_forgetting \
  --counterfact experiments/cbf_ttt/qwen3_4b_final_1b_20260927/causal_onpolicy_scope_v13/summary.json \
  --zsre experiments/cbf_ttt/qwen3_4b_final_1b_20260927/zsre_forgetting_final_test_v1/summary.json \
  --output experiments/cbf_ttt/qwen3_4b_final_1b_20260927/crossdomain_replication_synthesis_v1/summary.json
```

The deterministic synthesis and its three unit tests require only Python's
standard library.
