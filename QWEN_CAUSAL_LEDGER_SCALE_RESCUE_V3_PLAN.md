# Preregistered causal-ledger scale-rescue V3 and release path

## Evidence at the freeze point

The independently audited development V2 result is fixed before this plan is
executed. The primary `full_ledger` policy reached 100% candidate accuracy,
exactly agreed with the independently supplied semantic oracle, and improved
five-token exact match over the strongest non-oracle policy (`retain_latest`)
by 9.9206 points. The source-group bootstrap 95% interval was
`[8.6806, 11.1855]` points and the answer-NLL gain interval was
`[1.3696, 1.7714]`. Every family and length-bucket gain was non-negative.

Development V2 failed exactly one registered check: current and anchor
five-token exact match were below 95% (91.37% and 89.21%). Historical exact
was 93.45%, every family exceeded 85%, overall exact was 91.34%, and candidate
accuracy was 100%. Therefore the observed failure is assigned to contextual
generation margin at the fixed scale of 32, not to ledger state transitions or
record selection.

Confirmation and OOD data remain ungenerated and sealed at this freeze point.

## Stage 1: one-shot scale-48 development replay

Run exactly one matched replay of the complete development V2 protocol with
residual scale 48. Scale 48 is fixed because it was already part of the
preregistered contextual scale grid and reached 100% five-token exact on the
preceding audited diagnostic. No new scale sweep is permitted in this stage.

Everything other than the scalar is held fixed:

- final 1B-token Qwen3-4B checkpoint and tokenizer;
- development seed `202610160`, prompts, episodes, policy traces and answer
  candidates;
- all eight policies and the strongest-comparator selection rule;
- exact five-step generation with fresh KV state;
- source-group bootstrap with 20,000 resamples;
- all eleven V2 release checks and the independent execution audit.

The replay passes only if all original checks pass. If it fails, scalar repair
stops. Scale 64, prompt changes, decoder changes and post-hoc subset selection
are not allowed as further rescue attempts.

## Stage 2: one-time sealed confirmation

If and only if Stage 1 passes its independent audit, generate one fresh
confirmation corpus from the reserved seed and unseen surface templates. Freeze
scale 48, policies, comparator selection, five-step decoder and all thresholds.
The confirmation reports the same aggregate, role, family, sequence-length,
absence, oracle-agreement, NLL and group-bootstrap statistics.

The main controlled claim is accepted only when confirmation independently
satisfies the unchanged gate. A failed confirmation is reported as a failed
replication; it does not trigger confirmation-set tuning.

## Stage 3: native In-Place TTT writer bridge

After controlled confirmation, replace the externally constructed value
residual with the baseline writer output from `ttt_proj`/`ttt_conv`. Preserve
the causal ledger as the forgetting and retrieval policy. Train only the
forgetting policy first; keep the writer and Qwen backbone frozen so that the
source of improvement is identifiable.

Use three matched arms:

1. baseline In-Place TTT writer and baseline memory update;
2. the same writer with global/session-level forgetting;
3. the same writer with the proposed record-level causal-ledger forgetting.

Log writer/read residual cosine, norm, selected record, active-record count,
query loss, generation result and memory state transition for every query.
This stage determines whether the controlled ledger result transfers to the
actual fast-parameter path.

## Stage 4: natural-data effectiveness and safety

Run the frozen native method and matched baselines on:

- ZsRE counterfactual updates for overwrite, revoke, duplicate and restoration;
- LongMemEval-S and LoCoMo for session-memory retention and selective removal;
- RULER and RULER/QA for long-context retrieval and interference;
- MMLU before/after adaptation as a general-knowledge retention control.

Primary outcomes are update accuracy, historical-query accuracy, revoked-fact
suppression, restoration accuracy and macro exact/F1 as appropriate. Report
mean and 95% source/session bootstrap intervals, paired differences against
the strongest baseline, family-level effects, memory size, tokens/s, peak
memory and added latency.

## Required ablations

- no forgetting;
- global reset only;
- last-1 and last-2 recency;
- retain-latest without revoke/restoration semantics;
- dense read versus addressable sparse read;
- learned forgetting without lifecycle supervision;
- full causal-ledger forgetting;
- oracle lifecycle labels as an upper bound.

The paper-level conclusion requires both a positive native-writer transfer and
a positive natural-data result. The current controlled residual experiments
alone establish causal state-management and readout behavior, not an end-to-end
In-Place TTT improvement.

## Stop and decision rules

- Stage 1 pass: release the one-time confirmation.
- Stage 1 fail: stop scale tuning and diagnose the residual decoder geometry.
- Stage 2 pass: freeze the controlled mechanism and begin the native bridge.
- Stage 2 fail: report non-replication and do not claim a robust mechanism.
- Stage 3 fail: keep the controlled result as analysis evidence, but do not
  claim benefit for the baseline In-Place TTT system.
- Stage 3 pass but Stage 4 fail: report a synthetic-to-natural transfer gap.
- Stages 2--4 pass: proceed to multi-seed scaling, full benchmark tables and
  paper figures.
