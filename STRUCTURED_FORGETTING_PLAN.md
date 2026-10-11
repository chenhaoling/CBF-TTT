# Physics-Informed Structured Forgetting Pilot

## Objective

The dense TTT writer has not supplied a readable entity-value memory, and the pooled retrieval head is invalid even under Full-KV. This pilot isolates the forgetting hypothesis on a readable structured session memory before attempting distillation back into fast weights. The update gate is fixed on (`g=1`); the only learned decision is how much session history to forget.

## Corpus and actions

`build_cbf_event_curriculum.py` generates 64 train, 16 development, and 16 sealed test worlds with disjoint identifiers and wording templates. Every world contains retain, revoke-last-one, revoke-last-two, and reset episodes, one to three decision boundaries, twin initial values, chronological assignments, cancellations, and session changes.

At each boundary, four counterfactual actions are evaluated against the three current queries:

1. `retain`: keep all existing batches;
2. `last1`: remove the most recent batch;
3. `last2`: remove the second-most-recent batch;
4. `global`: clear all session batches.

The minimal-forgetting maximizer is the supervision label. The controller receives only visible structural invariants: memory size, whether a referenced batch is at recency one/two/deeper, lines before the new assignment, block position, fact count, and active-key count. It never receives family, split, identifiers, answers, or future blocks. A history-blind ablation removes the recency features.

## Comparisons and gate

Compare learned full controller, history-blind controller, all four fixed actions, deterministic random, and a hand-coded physical-rule upper bound. Report exact query accuracy by family and role, action accuracy, average retained batches, paired source-group bootstrap intervals, label ties, time, and memory.

The development gate requires:

- learned action accuracy at least 95%;
- learned query accuracy at least 98%;
- at least 15 percentage points over the best fixed action;
- paired source-group bootstrap lower bound over the best fixed action above 10 points.

Only a passing development gate authorizes one test evaluation with frozen controllers. A pass validates the forgetting controller on structured memory, not end-to-end TTT fast-weight memory. The next step would distill the same actions into a differentiable TTT-compatible memory representation.

## Command

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_structured_forgetting_v1 \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
TEST_PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python \
bash scripts/run_cbf_structured_forgetting.sh
```

All components are opt in and require no model checkpoint or extra dependency.
