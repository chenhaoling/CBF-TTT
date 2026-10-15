# Preregistered Qwen natural-value residual bridge V1

## Freeze point and question

The causal ledger passed matched confirmation and lifecycle OOD at scale 48,
but every successful value was an artificial code of exactly five tokens. The
original native `ttt_conv`/`ttt_proj` writer failed its favorable single-record
bridge and remains stopped. Before natural lifecycle sequences are built, this
pilot asks whether the structured Qwen LM-head residual itself supports real,
variable-length counterfactual answers under natural exact and paraphrased
queries.

This protocol is frozen before producing any natural-value residual score. It
uses only the already observed ZsRE contextual-readout calibration and
development splits. ZsRE test, LongMemEval-S outcomes, LoCoMo, MMLU, RULER and
RULER/QA remain unopened by this experiment.

## Fixed data and isolation

Use the hash-locked `zsre_contextual_readout_v1` corpus derived from
`ZsRE-test-all.json` SHA256
`0e0214dda853a906ef02bef58c7ce2978af5a962e91fc4a10827e0125459905c`.
Keep only facts whose normalized original and counterfactual updated values
differ. This fixes 258 calibration facts and 256 subject-disjoint development
facts. For every fact score its exact prompt and registered rephrase. Do not
load the 534-subject test file.

The updated value is the correct counterfactual content. The original value is
the deliberately wrong injected content. Add four deterministic, distinct
within-split answer distractors. Candidate construction cannot use model
scores. Report answer token length and exclude no nonempty value by length;
the observed calibration/development maxima are 18/12 tokens.

## Fixed writer and readout

For a natural value with token IDs `(v_1, ..., v_L)`, construct an ordered
residual whose position `j` is the normalized frozen Qwen LM-head direction of
`v_j`. The content-only bridge assumes the record key has already been selected
and therefore uses exact unit key self-similarity. This isolates value binding
from the separately measured natural scope resolver.

At each answer position, add the corresponding residual row to the fresh-KV
Qwen hidden state and score the unchanged full-vocabulary LM head. Candidate
scores are mean token NLL, which permits different candidate lengths. Greedy
generation is conditioned on the registered correct answer length and is
reported explicitly as a diagnostic rather than unrestricted generation.

Four arms share the identical prompt and candidates:

1. `correct`: residual from the updated value;
2. `wrong`: residual from the original value;
3. `zero`: no residual;
4. `random`: deterministic element-wise sign randomization of every Correct
   residual row, preserving row norms.

Calibration scores scales `4/8/16/32/48`. Select the smallest scale satisfying
all calibration gates. Development uses that scale once; it cannot change any
prompt, candidate, arm, metric or threshold.

## Calibration and development gates

Calibration releases development only if one scale satisfies:

- Correct candidate accuracy at least 90% overall and 85% for both exact and
  rephrase queries;
- Correct length-conditioned exact generation at least 70% overall and 65%
  for both query types;
- Wrong updated-value candidate accuracy at most 20% and injected-original
  candidate accuracy at least 70%;
- Zero updated-value candidate accuracy at most 35%;
- source-fact bootstrap 95% lower bounds for Correct updated-answer NLL gain
  over Wrong, Zero and Random are all above zero.

The one-time subject-disjoint development split applies the identical checks,
plus requires no negative exact/rephrase candidate-accuracy gain over the
strongest control. All raw rows, prompts, token IDs, hashes, scale choice,
aggregates, intervals, resources and conditional access must pass an
independent audit.

## Decision rule and claim boundary

A passing development gate releases construction of natural-answer causal
ledger sequences with explicit lifecycle operations and automatic query scope.
Failure stops this LM-head value-writer form; scale or prompt retuning on
development is prohibited and the next method must learn a position-aware
value writer.

A pass establishes content binding and fresh-KV generation for natural
multi-token counterfactual values under an already selected record key. It does
not establish automatic write-side fact extraction, lifecycle discovery,
unrestricted generation, native dense-writer content binding or gains on an
external long-memory benchmark.
