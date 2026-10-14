# Causal-ledger residual resource/content pilot

## Outcome

The frozen pilot completed on both RTX 5090 cards and passed its independent
artifact audit, but failed the registered content gate. Development,
confirmation and OOD were not generated.

The failure is sharply localized. The correct residual ranked the answer first
among six candidates for all 1,008 queries at scales 4, 8 and 16, including
100% accuracy in each of the seven lifecycle families. At scale 8 its mean
answer NLL was 1.2121, compared with 5.1971/5.8774/5.8901 for the
Wrong/Zero/Random controls. Source-group bootstrap lower bounds for the three
NLL gains were 3.9382, 4.6413 and 4.6534. Registered greedy exact match was
0%, below the 85% requirement, so the complete gate failed.

## Data and execution

- Seed: `202610159`.
- Seven families, eight source groups per family, two counterfactual twins per
  group: 56 groups and 112 episodes.
- Three sparse boundaries per episode: 336 label rows.
- Three queries per boundary: 1,008 content rows.
- Scales: 4, 8 and 16.
- Valid action grids: 6 boundaries with 3 actions, 184 with 9 actions and 146
  with 27 actions.
- Two-GPU critical path: 369.5284 seconds.
- Mean/p95 label time: 0.0460/0.0840 seconds per boundary.
- Mean score time: 0.00282 seconds per valid action.
- Peak allocated/reserved CUDA memory: 7.9704/8.3633 GiB.

The model was the final 1B-token Qwen3-4B checkpoint at
`global_step_81381/hf_ckpt`; native TTT parameters loaded with no missing,
unexpected or mismatched keys. This experiment did not execute a native dense
TTT write.

## Content results

| Scale | Correct candidate | Correct NLL | Wrong candidate | Zero candidate | Random candidate | Registered greedy EM |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 100.00% | 2.9058 | 0.00% | 15.18% | 14.98% | 0.00% |
| 8 | 100.00% | 1.2121 | 0.00% | 15.18% | 16.07% | 0.00% |
| 16 | 100.00% | 0.3788 | 0.00% | 15.18% | 18.25% | 0.00% |

Every family reached 100% Correct candidate accuracy at scale 8. The gate
passed six of seven checks; only `correct_greedy_at_least_point_85` failed.
The selected development size is therefore `null`, even though the measured
runtime would have allowed 64 groups per family in about 2,956 seconds.

## Post-failure generation checks

The collector inherited a maximum of 13 generated tokens and waited for EOS or
newline even though this protocol fixes every answer at five tokens. The
registered result remains unchanged. A hash-bound post-hoc diagnostic decoded
only the first five saved IDs: exact match was 0% at scales 4 and 8 and 27.08%
at scale 16. The gate failure therefore cannot be explained by the stopping
rule alone.

The subsequently frozen LM-head geometry diagnostic found 17 unique answer
tokens. Every `normalize(W[token])` direction ranked its target first over the
full vocabulary, both with the real output head and with a diagnostic
cosine-normalized head. At scale 16, rollout position accuracy was
`[100%, 27.08%, 96.23%, 99.21%, 100%]`; 735/1,008 sequences first failed at
position one (the second output token). Errors overwhelmingly replaced a
space-prefixed token such as `" G"` with its no-space counterpart `"G"`.
The target residual-only margin over that competitor was about 0.25--0.32, so
the frozen scale was insufficient to overcome the contextual logit preference.

## Integrity audit

`execution_audit.json` independently passed all seven checks:

1. terminal status and runner exit;
2. deterministic corpus construction and full coverage;
3. shard hashes, model loading and resource records;
4. action enumeration and lexicographic label selection;
5. query truth, control scores and generation traces;
6. complete metric, bootstrap and gate recomputation;
7. downstream split sealing.

The audit establishes internal consistency of the recorded implementation. It
does not erase two protocol deviations discovered after execution:

- `Wrong` used the first deterministic nontruth/non-null candidate, rather
  than guaranteeing the paired twin value for every query;
- the registered generator used a 13-token termination budget rather than
  stopping after the fixed five-token answer.

Neither deviation can turn this run into a pass: the wrong control already has
0% answer accuracy, and the five-token recomputation still fails. Both must be
corrected in any versioned successor.

## Decision and next experiment

This V1 branch stops before development as registered. The supported result is
that a causal ledger and rank-one Qwen residual provide perfect controlled
six-candidate retrieval across reset, revoke, shadow restoration and no-op
families, with strong content interventions, but the selected residual scale
does not support reliable free generation.

`QWEN_CAUSAL_LEDGER_CONTEXT_SCALE_PLAN.md` freezes a post-failure diagnostic on
the same pilot at scales 16/20/24/32/48/64. It measures teacher-forced and free
full-vocabulary generation and may nominate only the smallest scale with at
least 85% free exact and 95% teacher-forced sequence accuracy. Any nominated
scale must then pass a new resource/content protocol with corrected five-token
termination and an explicit twin-aware wrong control before fresh development
data can be constructed.

