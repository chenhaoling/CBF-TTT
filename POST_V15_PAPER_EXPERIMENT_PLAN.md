# Post-V15 paper experiment plan

## Status and evidence boundary

This plan was frozen while the formal V15 run was still computing its first
outer fold and before any V15 `inner_complete` or `outer_complete` result was
available.  It therefore defines a result-conditional next step rather than
selecting a method after seeing V15 outcomes.

V15 operates on the audited contextual additive Qwen memory, not on a proven
native dense In-Place TTT memory.  The distinction is material:

- `ttt_conv` and `ttt_proj` are the original In-Place TTT writer modules;
- the final-checkpoint native writer failed the source-disjoint Correct versus
  Wrong/Zero/Random content-binding gate;
- several direct writer-training variants also failed to generalize random
  entity--value binding;
- the frozen contextual Qwen key/value substrate passed its own readability
  gate and is the memory representation used by the natural forgetting line.

Consequently, a successful V15 result supports a forgetting policy on a
readable additive memory.  It does not by itself authorize a claim that the
native `ttt_conv`/`ttt_proj` fast weights store readable facts.  The V15
`real_fast_weight_memory_integration` terminal label denotes entry into the
integration track, whose first experiment is the causal Qwen readout below.

## Research question

Can a causal, history-conditioned policy delete stale components from a
readable Qwen-grounded session memory and improve actual frozen-Qwen answer
likelihood and accuracy over every fixed forgetting schedule, without harming
current or unrelated memories?

The paper method fixes admission of the current update to one.  It learns only
retention/deletion of historical session memory:

\[
M_t = \sum_{i < t} \alpha_{t,i}\,\Delta_i + \Delta_t,\qquad g_t=1.
\]

`alpha` acts only on session-local fast memory.  Slow model parameters and
pretrained knowledge are never deleted.

## Stage 0: adjudicate V15 exactly as registered

V15 is accepted only if its independent audit passes and all registered pair
and safety checks pass:

- stale deletion and pair state at least 90%;
- subject-macro pair at least 85%;
- minimum outer-fold pair at least 85%;
- current retention at least 99%;
- non-target and hard-retain at least 97%;
- subject-cluster safety LCB95 at least 95%;
- matched-random improvement CI lower bound above zero.

The frozen candidate resolver's approximately 98.93% top-1 is reported as a
separate known ceiling.  It must not be hidden by a controller-only pass.

## Branch A: V15 passes -- frozen-Qwen downstream outcome experiment

### Readout prerequisite and fixed realization

The contextual memory currently exposes a 256-dimensional candidate-space
readout.  It has no existing, audited decoder into Qwen vocabulary logits.
Candidate-space accuracy must therefore not be renamed Qwen answer accuracy.

The first downstream realization uses no learned decoder.  A policy's retained
session components are serialized back to their original, immutable
`Memory statement`/`Value` text in chronological order, followed by a fixed
instruction, the primary current question and an answer prefix.  Frozen Qwen
scores the complete current answer and stale answer autoregressively from a
fresh KV cache.  All policies use byte-identical text for every component they
share.  Deletion changes only whether the selected historical session appears
in the serialized memory.

Across all 5,976 MQuAKE train+tune episodes, the complete unfiltered session
text is 1,202--6,137 tokenizer tokens (64-session maximum 6,137).  The frozen
checkpoint declares 40,960 positions, so this protocol uses every retained
unit without truncation.  Prompt scaffolding and both answer candidates must
still be checked to remain below the declared limit for every row before any
model score is accepted.

This experiment measures whether the learned memory-selection policy improves
actual frozen-Qwen use of retained textual evidence.  It is a downstream
utility bridge for the contextual memory and is reported as such; it is not a
native dense-fast-weight result.

### Queries and fixed sampling

For every episode, the primary-old and primary-current sessions are already
identified by the corpus.  Their matched unit pair supplies two query
phrasings, the old and current `Memory statement`, and two candidate answers,
the stale and current `Value`.  The primary metric averages both phrasings and
requires the current answer to outrank the stale answer.  One deterministic
hard-retain unit and one deterministic unrelated unit provide the two safety
queries; their own serialized values remain the targets.  The sampler prefers
a same-relation hard-retain episode in every `subject x length` cell and uses
the corpus builder's explicitly marked hard fallback only when that cell has
no same-relation example.  Matched and fallback safety results are reported
separately.

The resource pilot uses 40 episodes: two hash-minimal episode IDs in each of
the five subject folds and four registered lengths.  It reports only runtime,
peak memory, token lengths and the oracle/no-memory readability controls; V15
versus fixed-policy outcomes remain masked.  Formal scoring is unlocked only
if the oracle arm prefers the current answer on at least 80% of primary
queries, every prompt is within the checkpoint limit and no row fails or
produces a non-finite score.

The formal internal set uses one hash-minimal episode for every available
`subject x length` cell across train+tune, after preferring same-relation hard
episodes as described above.  The current corpus has 68 subjects and four
lengths, so the expected size is 272 episodes, subject-balanced by
construction.  A source-only audit found 252 cells with a same-relation hard
episode and 20 cells requiring the marked fallback.  The exact ordered episode
IDs are written and hashed before formal model scoring.  Confidence intervals
resample subjects, keeping the four lengths and all query types for a sampled
subject together.

### Intervention

Replay every held-out subject episode with the exact contextual memory
components.  At each event, the deployed cross-fitted V15 policy may remove
only a component visible at that time.  The current component is then always
admitted.  The registered query point is immediately after the primary-current
session is admitted; no later session is visible or used as a safety query.
At that point, retained sessions are serialized by
the fixed realization above and every answer is scored from a fresh query KV
cache so that no attention state leaks across arms.

### Fixed arms

All arms use identical episodes, component values, frozen Qwen weights,
candidate resolver, queries and answer tokens:

1. no session memory;
2. retain all components;
3. global reset before every new component;
4. last-1 fixed window;
5. last-2 fixed window;
6. matched-random deletion count;
7. V14 single-calibration policy;
8. V15 jackknife-consensus policy;
9. causal oracle component deletion, reported only as an upper bound.

The primary comparison is V15 versus the strongest fixed arm selected without
using the held-out outer fold.  V14 is an architecture ablation, not a tuning
source.

The fixed-arm selection rule is frozen before outcome scoring: for outer fold
`k`, select among no-memory, retain-all, global-reset, last-1 and last-2 using
the other four folds, ordered by primary-query candidate accuracy, then lower
answer NLL, then the registered arm order.  Apply that selected arm only to
fold `k`.  Accuracy and NLL differences use subject-cluster bootstrap with
20,000 draws and seeds 1611 and 1612, respectively.

Before any formal Qwen score, a state-freezing pass reconstructs the retained
session IDs for all nine arms at the registered query time and hashes the
ordered 272-row matrix.  V14 and V15 states come only from their audited OOF
decision traces; matched random comes from V15's exact deletion-count-matched
OOF trace.  Fixed windows and the causal oracle are rebuilt directly from the
source episode.  This pass is unlocked only by a passed and audited resource
pilot and loads no model.  It makes memory selection auditable independently
of the expensive Qwen readout.

Candidate NLL uses one prompt forward followed by independent answer-cache
branches.  The same immutable prompt cache also starts an independent greedy
decode, capped at the longest candidate token length plus eight tokens (and at
most 64 tokens), stopping at EOS, newline or the cap.  Strict and conservative
case/whitespace-normalized exact match are both reported.  Identical retained
session-ID sequences within an episode are scored once and copied bit-for-bit
to their duplicate arms; the first arm in the registered order is recorded as
the score source and the auditor verifies every reuse.

### Primary outcomes and release gate

- current-answer exact generation, current-versus-stale preference and answer
  NLL;
- stale-answer suppression and old/current pair accuracy;
- non-target and hard-retain query accuracy;
- subject-macro values and the minimum outer-fold value;
- matched subject-cluster bootstrap intervals;
- deletion count, retained-memory size, latency and peak GPU memory.

The stage passes only when all of the following hold:

1. V15 improves query accuracy over the strongest fixed arm and the clustered
   95% CI lower bound is above zero;
2. V15 reduces answer NLL over the strongest fixed arm and the clustered 95%
   CI lower bound for fixed-minus-V15 NLL is above zero;
3. current, non-target and hard-retain query accuracy are at least 97%;
4. no outer fold has a negative V15-minus-strongest-fixed accuracy difference;
5. the independent audit reconstructs every memory state, serialized prompt,
   token-length check, Qwen score, cross-fit identity, paired comparison and
   artifact hash.

Action-label accuracy remains secondary.  A controller that predicts symbolic
deletions but does not improve Qwen answers does not pass.

### External confirmation

Only a passed Qwen-outcome stage releases one-time evaluation in this order:

1. MQuAKE-CF for independently sourced temporal conflicts;
2. LongMemEval-S and LoCoMo for natural long-session memory;
3. MMLU and RULER/RULER-QA for slow-knowledge and long-context preservation.

No external set is used for threshold, feature, epoch or architecture choice.
The paper reports each domain separately and uses domain-specific clustered
confidence intervals rather than pooling incompatible query metrics.

## Branch B: V15 fails -- stop scalar-threshold calibration

A V15 pair or safety failure closes the current listwise-score plus scalar
threshold family.  There is no V16 threshold grid, normalizer or voting-quorum
search.

The replacement experiment first measures causal headroom.  For each visible
candidate historical component, exact replay evaluates retain versus delete
under identical future queries.  If the causal oracle does not improve query
accuracy by at least five points over the strongest fixed arm, the corpus does
not contain sufficient controllable forgetting headroom and controller
training stops.

This diagnostic uses the already frozen 40-row pilot and 272-row formal sample.
At the registered query point it materializes no-memory, retain-all,
global-reset, last-1 and last-2 plus one state for deleting each visible
historical session; the current session is never a deletion candidate.  The
per-episode oracle chooses by mean primary-query candidate accuracy, then lower
primary answer NLL, then retain-all, then chronological deletion order.  The
strongest fixed comparator uses the same leave-one-outer-fold-out rule as
Branch A.  Formal scoring is allowed only after a resource pilot and a
model-free independent reconstruction audit.  The five-point gate is applied
to the subject-macro oracle-minus-fixed accuracy and its subject-cluster 95%
interval is reported; this is explicitly a diagnostic upper bound.

The frozen 40-row resource pilot scores all four registered query kinds for
every unique fixed or single-deletion state with the final Qwen3-4B slow
checkpoint and native TTT adapters disabled. Duplicate fixed states are
evaluated once and copied with an explicit source-arm identity. Candidate NLL
uses one prompt forward followed by independent answer-cache branches; no
generation result enters oracle selection. Each query records wall time and
CUDA start/peak allocated and reserved memory. The pilot releases formal
scoring only if all 40 rows and scores are finite and within the 40,960-token
context, every resource record is complete, peak reserved memory is at most
31.5 GiB, the two-GPU formal-time projection is at most 48 hours, and the
independent audit passes. Pilot headroom is descriptive and cannot block or
release formal scoring. The time projection scales the slower pilot shard by
the registered formal-to-pilot unique-state ratio.

If the headroom gate passes, train a readout-utility policy with subject-level
cross-fitting.  Its target is the actual downstream change

\[
U_i = L_{future}(M_t-\Delta_i)-L_{future}(M_t),
\]

while its inputs remain causal: current query/event representation, candidate
component readout, component interactions and past scores.  Future losses,
truth labels, subject ID, absolute position and episode length are unavailable
to the deployed policy.  The same Branch-A fixed arms and Qwen-outcome gate are
used; only the training target changes.

## Native In-Place TTT writer bridge

Native dense integration is a separate prerequisite, not an automatic sequel
to a controller pass.  Before the forgetting policy can be advertised as an
end-to-end improvement to the original In-Place TTT writer, a source-disjoint
writer experiment must establish both single-component and multi-component
content binding.

The next admissible writer study distils the already readable contextual
key/value component into the existing `ttt_conv`/`ttt_proj` path while freezing
the Qwen backbone.  It compares the original writer with a writer trained by:

- answer-token likelihood;
- Correct-versus-Wrong component contrast;
- contextual-memory logit/readout distillation;
- an explicit per-layer delta-norm constraint.

Admission requires Correct to beat Wrong, Zero and norm-matched Random on new
sources with clustered CI lower bounds above zero, followed by a registered
2/4/8/16-component load test.  Until both gates pass, native dense results stay
a negative boundary result and are not mixed with the contextual-memory main
table.

## Statistical and reporting requirements

- subject-disjoint cross-fitting and no test refit;
- at least three controller seeds for a final positive result;
- subject/source-cluster bootstrap with fixed seeds and at least 20,000 draws;
- per-fold values, macro values and worst-fold values;
- exact denominators and confidence intervals for every headline difference;
- complete negative and stopped-stage reporting;
- independent aggregation and replay auditor;
- source, checkpoint, raw-output and code hashes;
- wall time, peak allocated/reserved memory and per-query latency;
- explicit separation of controlled forgetting evidence, Qwen contextual
  memory evidence and native dense writer evidence.

## Stop rules

- Do not open external confirmation sets after a failed internal Qwen-outcome
  gate.
- Do not tune native writer or controller architecture on MQuAKE-CF,
  LongMemEval-S, LoCoMo, MMLU or RULER.
- Do not claim end-to-end native In-Place TTT benefit before the writer content
  and multi-component gates pass.
- Do not train a new deletion controller when the exact causal oracle lacks
  sufficient query-level headroom.

