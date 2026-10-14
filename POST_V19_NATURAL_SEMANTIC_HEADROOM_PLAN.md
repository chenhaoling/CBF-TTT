# Post-V19 natural semantic-conflict headroom

## Freeze point and rationale

This protocol is frozen after the terminal V19 controlled results and before
constructing or scoring any row in this new experiment. V19 established a
large benefit over retain-all, but the learned policy did not improve an exact
complete-key deletion rule: +0.104 points with a subject-cluster 95% interval
crossing zero. Continuing to tune that policy on V19 is forbidden.

The native dense `ttt_conv`/`ttt_proj` writer has already undergone
Correct/Wrong/Zero interventions, accumulation control and balanced writer
fine-tuning. It did not learn reliable content binding. This study therefore
uses the already validated contextual-evidence memory substrate and makes no
native-writer claim. Its purpose is to determine whether natural paraphrases
create a causal gap between exact-string and semantic forgetting. A learned
semantic controller is conditional on that gap.

## Source lock and leakage boundary

Consume only the independently audited inventory at
`/home/ctj/cbf_ttt_semantic_controller_inventory_20261014/inventory`:

- CounterFact records already designated development-only;
- MQuAKE-T train and tune records;
- the registered five relation- and subject-disjoint folds;
- source manifest SHA256
  `df3002f4913663ef829b865b3d1502305b712f5a96c401568c413a9fa2862836`;
- semantic-row SHA256
  `222f7c51d384a9e26da93ecfd0dda91dd0add148c9901e5765f2fe4458614a99`.

Do not read MQuAKE dev, CounterFact outcome traces, ZsRE dev/test,
LongMemEval-S outcomes, LoCoMo, RULER or MMLU. ZsRE calibration remains
readout-only and is not used in headroom construction. V19 episodes, labels,
features and learned predictions cannot select a new row, relation, threshold
or action.

## Deterministic pilot and confirmation split

Rank eligible source rows inside each registered outer fold by SHA256 of the
protocol seed and `source_row_sha256`. Require distinct normalized subjects
and nonempty old/new values. Allocate without replacement:

- pilot: 80 rows, 16 per fold;
- sealed confirmation: 320 different rows, 64 per fold.

Within each fold use deterministic round-robin allocation over relations and
then sources. No subject, source-row hash or relation surface instance may
cross pilot and confirmation. Relations remain in their registered fold; the
experiment reports the unavoidable CounterFact/MQuAKE source imbalance.

Each target record supplies three natural surfaces of one relation and two
values. Surface 0 writes the stale value, surface 1 writes the current value,
and surface 2 queries the effective value. The stale and current surface
strings must differ after normalization. Deterministically selected donor
records provide retained facts and distractors. Each episode is assigned one
of four mechanisms and one of lengths 8, 16, 32 and 64:

1. one semantic conflict;
2. repeated stale evidence;
3. repeated useful retained evidence;
4. a lexically similar but relation-distinct distractor.

The current update is always the last component and can never be deleted.
Every episode includes two update queries on different surfaces, two retained
fact queries and one self-contained locality query. Candidate answers contain
the old value, current value and three donor values; free generation is also
recorded.

## Frozen causal arms

Score the final 1B-token Qwen3-4B checkpoint with native TTT disabled and
fresh query KV under:

- no memory;
- retain all;
- global reset;
- last one and last two components;
- exact surface-conflict deletion;
- relation-aware semantic-oracle deletion.

Exact deletion may use normalized subject plus normalized surface string. It
cannot use relation ID. The semantic oracle removes only stale historical
components with the target subject and source-provided relation ID. It does
not remove another subject, another relation, a same-value repetition or the
current update. The scorer deduplicates identical memory states but preserves
arm identities.

## Pilot gate and resource decision

Score all 80 pilot episodes once on two GPUs. Report per-query latency and
peak allocated/reserved memory. Release the sealed 320-row confirmation only
if all conditions hold:

1. semantic oracle improves subject-macro joint success over exact deletion by
   at least five points;
2. update accuracy is at least 85%;
3. retained-fact accuracy is at least 90%;
4. locality is at least 97%;
5. every fold has nonnegative joint gain;
6. all construction, model-provenance, prompt, state and resource audits pass.

The pilot estimates runtime only; it cannot change rows, arms, prompts,
metrics or confirmation thresholds. Based on V19 latency, the pilot is
expected to require roughly 10--20 minutes and the confirmation roughly
40--80 minutes on two RTX 5090 cards, subject to natural-answer token length.

## Confirmation gate

The 320 sealed rows use the same code and fixed settings. A semantic-headroom
claim requires:

1. at least ten points subject-macro joint gain over exact deletion;
2. subject- and relation-cluster bootstrap 95% lower bounds above zero;
3. at least 90% update and retained-fact accuracy;
4. at least 97% locality and unrelated-session retention;
5. no negative outer fold or source-domain stratum;
6. at least 40% of the positive exact-to-oracle gap closed relative to the
   strongest non-oracle comparator;
7. exact deletion is verified to use no relation ID;
8. the independent outcome audit passes every registered check.

If the pilot or confirmation fails, stop this semantic-controller branch.
Natural benchmarks remain sealed and no threshold or representation may be
retuned on the failed rows.

## Conditional learned residual

Only a passing confirmation releases controller training. The model is an
exact-rule residual rather than a replacement:

1. exact string matches are handled deterministically;
2. a frozen-Qwen masked relation representation handles unresolved pairs;
3. a rank-16 contrastive projection and two-head 64-unit utility/risk MLP are
   trained with nested relation/subject OOF splits;
4. source group weights use the already audited inverse-square-root rule;
5. three seeds, epochs 25/50/100, thresholds 0.4/0.5/0.6 and risk penalties
   0/0.5/1/2 are fixed now;
6. positive locality-risk examples must exist in every inner selection fold;
   otherwise the learned stage is invalid rather than treated as zero risk.

The learned residual must exceed both exact deletion and projected-cosine
deletion by at least two points with a positive relation-cluster lower bound,
while retaining the confirmation safety thresholds. Failing this gate stops
training on the observed sources.

## Conditional natural confirmation

One-time LoCoMo, RULER/RULER-QA and MMLU evaluation is allowed only after all
development gates and independent audits pass. LongMemEval-S remains an
already observed diagnostic set. A natural result must report generated
answers, session/conversation clustered intervals, deletion-matched random
controls, latency, memory and MMLU non-regression. No failed development stage
can be rescued by inspecting these benchmarks.

