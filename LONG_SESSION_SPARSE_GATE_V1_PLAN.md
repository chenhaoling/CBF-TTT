# Sparse long-session forgetting gate V1 preregistration

## Evidence and scope

The one-time LongMemEval-S transfer showed a useful but unsafe behavior: the
frozen short-fact controller reduced stale hit@5 from 61.54% to 7.69%, while
retaining only 50.00% of current answer sessions and 41.69% of unrelated
sessions. The answer-ID oracle reached 100% stale/current state and improved
current hit@5 to 66.67%. The mechanism is viable, but the controller is
miscalibrated under sparse, long-session updates.

The 78 LongMemEval outcomes are locked. They do not train, select, or calibrate
V1. LoCoMo remains unopened and is the next independent source only if this
development gate passes.

## Constructed development corpus

Use only the existing subject-disjoint ZsRE contextual-readout calibration and
development splits. Calibration source sessions are deterministically divided
into fit and tune groups; the original ZsRE development subjects are the final
development gate.

For each primary fact and each length in `{16,32,48,64}`, build one chronological
episode with:

- one old fact session at one-quarter of the trajectory;
- one paraphrased updated fact session at three-quarters;
- two same-subject locality prompts as hard-retain sessions;
- unrelated factual sessions filling the remaining positions;
- four fact units per session, with three deterministic distractor units.

Exactly one event per episode requires deletion. All other events require
retain, matching the sparse update rate that exposed the zero-shot failure.
Only the updated event may delete the old session. Source subject, chain ID,
event kind, position, and target session are retained for labels and auditing;
none is a model feature.

## Representation and controller

Freeze the final Qwen3-4B checkpoint and the three ZsRE contextual transforms.
Each fact unit is embedded independently. For every new session and active old
session, compute maximum unit-pair cosine in each projection. Select the old
session with the highest mean maximum cosine.

The binary sparse deletion gate receives only observable geometry:

- three maximum cosines, their mean and standard deviation;
- top-candidate margin and projection agreement;
- fractions of unit pairs above fixed cosine levels;
- lexical and relation-token overlap of the best unit pair;
- normalized candidate age, memory size, and session unit counts.

A rank-16 nonlinear gate is trained with three fixed seeds. Fit/tune selection
chooses epoch and threshold by causal replay. The development split is evaluated
once after all fit/tune choices are frozen.

## Controls

- retain all;
- always delete the most similar session;
- deterministic action-rate-matched random deletion;
- cosine threshold selected on tune without the learned gate;
- frozen learned sparse gate;
- oracle local deletion.

## Development gate

The learned arm passes only if:

- positive-event candidate top-1 is at least 95%;
- current-session retention is at least 95%;
- non-target session retention is at least 95%;
- stale-session deletion is at least 90%;
- strict final stale/current pair state is at least 90%;
- every length has at least 85% strict pair state;
- strict pair state exceeds action-rate-matched random with a subject-cluster
  bootstrap lower bound above 5 points;
- current-query proxy hit@5 is not used for selection because this constructed
  corpus has no independent natural question.

If the gate fails, LoCoMo remains sealed and the result is an architecture
diagnostic. If it passes, freeze the session encoder, controller, and threshold,
then preregister a one-time LoCoMo evidence-retrieval and official QA run.
