# Qwen sparse-reader ceiling diagnostic

## Status and purpose

This diagnostic is frozen after the first projection seed of the registered
continuous-cohort experiment exposed a reader ceiling, and before observing
any outcome from the diagnostic itself.  It is a development-only mechanism
diagnostic.  It cannot release, encode, score, or select on the sealed test
split.

On projection seed 2301, continuous cohort forgetting improved query accuracy
from 66.30% for the strongest fixed policy to 89.00% and recovered 98.1% of
the available 27-grid oracle improvement.  The oracle itself reached only
89.43%, with 68.8% historical accuracy.  The next causal question is therefore
whether dense summation of every active rank-one write limits historical
readability.

## Frozen dependencies

Use the same artifacts as `QWEN_COHORT_FORGETTING_PLAN.md`:

- final 1B-token Qwen3-4B checkpoint;
- audited `context_last_r0.01`, rank-256 contextual geometry;
- fresh event-curriculum seed `202610151`;
- train and development splits only;
- projection seeds `2301/2307/2311`.

The diagnostic must verify the contextual-geometry hashes and the corpus
manifest before loading any model or data.  No representation, temperature,
unknown score, retention action, or query target may be fitted.

## Reader variants

The existing dense reader sums every active rank-one component and computes
`M q`.  A sparse reader instead ranks active fact components by the cosine of
their stored entity key and the query key, keeps the highest scoring `k`
components, and sums only their weighted value contributions.  Its ranking is
causal and uses only the current query and already stored keys.

Evaluate these fixed variants:

1. `dense_all`: the unchanged current matrix reader;
2. `component_top1`;
3. `component_top2`;
4. `component_top4`;
5. `component_top8`;
6. `component_top16`;
7. `exact_key_upper`: retain every active component whose entity string is
   exactly the query key.  This is a diagnostic upper bound and is never
   eligible for selection.

For sparse readers, rank by descending key cosine, then by newer write, then
by source fact order.  Do not use answer values, query targets, event family,
future blocks, oracle actions, or batch identities in routing.  Preserve each
write's original block-strength coefficient and current retention weight.

## Policies and measurements

For every reader and projection seed, replay identical development sessions
under:

- fixed retain-all;
- the explicit cohort physics rule;
- exact referenced-batch deletion;
- the causal 27-grid oracle.

Recompute the oracle separately for each reader.  Report query accuracy, NLL,
current/historical/anchor accuracy, per-family accuracy, retained mass, reader
component count, latency, and peak GPU memory.  Also report improvement over
`dense_all` and the gap from `exact_key_upper`.

## Development decision

An inference-safe sparse reader establishes sufficient headroom only if its
three-seed oracle satisfies all of:

1. mean query accuracy at least 95%;
2. current and anchor accuracy at least 95%;
3. historical accuracy at least 90%;
4. minimum family accuracy at least 85%;
5. every projection seed improves over the corresponding dense oracle;
6. all source, replay, row, aggregate and sealing audits pass.

Among passing variants, select the smallest `k`; when variants differ by less
than 0.5 accuracy points, the smaller `k` wins.  The selected reader is only a
candidate for a separately frozen experiment on a new corpus split.  This
diagnostic never changes the failed/pass status of the continuous-cohort run.

If only `exact_key_upper` passes, the remaining bottleneck is representation
routing.  If it also fails, the bottleneck is value geometry, readout, or
curriculum ambiguity, and sparse-reader work stops before controller training.

