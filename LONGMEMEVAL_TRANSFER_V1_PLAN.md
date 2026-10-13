# LongMemEval-S zero-shot forgetting transfer V1 preregistration

## Motivation

CounterFact V13 and the one-time ZsRE test independently reproduce almost the
same retain/update action decomposition. CounterFact tuning now stops. The next
question is whether the frozen ZsRE contextual representation and forgetting
head transfer to long, natural multi-session updates without fitting on the
target benchmark.

The native dense `ttt_proj`/`ttt_conv` writer is excluded: its preregistered
content-binding audit failed. This experiment uses the readable, session-local,
contextual low-rank Qwen memory selected before the ZsRE test.

## Data and release

- Source: LongMemEval-S cleaned, SHA-256
  `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
- Primary subset: all 78 examples whose frozen `question_type` is
  `knowledge-update`.
- Schema and aggregate counts have been inspected. No model prediction or
  answer score has been produced before this registration.
- Each example has exactly two `answer_session_ids`. Their earlier occurrence
  in `haystack_session_ids` is the stale evidence; the later occurrence is the
  current evidence. These IDs are used only after policy replay for evaluation.
- LoCoMo remains unscored.

This is a mechanism-transfer evaluation, not the official LongMemEval answer
generation score. It will not be presented as official benchmark accuracy.

## Frozen representation and controller

- Qwen3-4B checkpoint after the 1B-token continual-pretraining run.
- ZsRE contextual `context_last` transforms and three seeds 5101/5102/5103.
- ZsRE final binary forgetting head and the independently selected threshold
  0.675 used in the one-time held-out ZsRE test.
- No LongMemEval text, session ID, date, answer, or outcome trains or calibrates
  any parameter.

Every session is rendered with its date and role-labelled turns, tokenized, and
split into non-overlapping 1024-token windows. All windows are encoded; no
session is truncated. A session-to-session score is the maximum contextual
cosine over window pairs, averaged across the three frozen projections. The
same maximum-window construction scores the final question against retained
sessions.

At each new session, the frozen controller evaluates the highest-cosine active
session using its unchanged cosine, lexical, and relation-overlap features. It
either retains all old sessions or deletes that single candidate, then appends
the current session.

## Arms

1. `retain_all`: append every session.
2. `window8`: retain only the eight most recent sessions.
3. `fixed_top`: always delete the most similar active session before appending.
4. `random_matched`: at every step where the learned controller deletes, delete
   one deterministic random active session instead.
5. `frozen_controller`: primary zero-shot transfer arm.
6. `oracle_pair`: delete the older answer session when the newer answer session
   arrives; this evaluation-only arm is an upper bound.

## Metrics

The primary state metric is strict pair correctness: the current answer session
is retained and the stale answer session is deleted. Also report current-session
retention, stale-session deletion, non-answer retention, memory fraction,
question retrieval hit@1/hit@5, stale hit@1/hit@5, and whether current evidence
ranks above stale evidence.

All comparisons are paired over the 78 questions. Confidence intervals use a
deterministic question-cluster bootstrap with 10,000 resamples. Descriptive
turns, sessions, encoded windows, wall time, and peak CUDA memory are recorded.

## Gate

The frozen controller passes only if all checks hold:

- strict pair-state accuracy exceeds `random_matched` with bootstrap lower
  bound above zero;
- current hit@5 is non-inferior to `retain_all`, with bootstrap lower bound
  above -2 percentage points;
- stale hit@5 is lower than `retain_all`, with the lower bound of the reduction
  above zero;
- current answer-session retention is at least 95%;
- non-answer session retention is at least 80%.

No threshold, window size, input rendering, metric, or gate may change after
scoring. A failed gate ends zero-shot LongMemEval transfer; follow-up development
must use a separate constructed/ZsRE development corpus, not repeated inspection
of these 78 outcomes.

## Interpretation

A pass would establish zero-shot transfer of selective forgetting to natural
long-session updates and authorize a separately preregistered official answer
generation evaluation. A failure would localize the remaining gap to session
representation/controller transfer. Neither outcome proves the failed native
dense writer.
