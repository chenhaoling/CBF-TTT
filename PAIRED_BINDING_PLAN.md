# Within Update Counterfactual Pairing

## Motivation

The first episodic reassignment experiment showed that fixed assignments fit substantially better than balanced reassignments: final window four candidate accuracy was 36.72% versus 26.56%. On development, episodic Correct and Counterfactual top 1 were 25.00% and 22.66%, and the NLL confidence interval crossed zero. Reassignment removed the slow query to value shortcut, but assignments presented in separate optimizer steps still gave a noisy, changing gradient.

This experiment places two opposite assignments of the same context and queries in every optimizer step. Both arms see identical pairs, writes, reads, exposure, initialization, and order:

- `paired_listwise`: mean four candidate retrieval CE for both assignments;
- `paired_swap`: the same CE plus a symmetric swap term. For each query and each assignment, its target digit NLL must be lower under its own memory than under the paired memory: `0.5 × mean softplus(0.2 + own_nll - cross_nll)`.

No extra forward pass is needed for the swap term because every memory already scores all four values. Both arms omit LM loss, so the paired comparison isolates associative retrieval and has matched compute.

## Protocol

- Same 32 train / 8 development worlds and sealed test as the episodic experiment.
- Assignment pairs `(0,2)` and `(1,3)`; every context sees all four assignments exactly once.
- 128 optimizer updates, two 4,096 token writes and eight supervised query exposures per update, for 1,024 question exposures per arm.
- Original Qwen3-4B frozen; seven writer pairs initialized with seed 301.
- AdamW FP32, BF16 forward, learning rate `1e-7`, clip 1, no weight decay, fast memory cap 0.3 with fixed 0.005 BF16 audit tolerance.
- Checkpoints and evaluation at 0 and 1,024 exposures. Evaluation and the development gate are unchanged from `EPISODIC_BINDING_PLAN.md`.
- Test is neither tokenized nor scored.

This remains a writer prerequisite. Passing the causal retrieval gate would lead to a fresh permutation family and seed confirmation, then generation and forgetting experiments. Failure would motivate changing the memory representation/readout rather than adding more fixed assignment training.

## Command

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_paired_binding_v1 \
DATA=/home/ctj/cbf_ttt_dynamic_binding_v1/data \
MODEL=/home/ctj/models/Qwen3-4B \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
TEST_PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python \
bash scripts/run_cbf_paired_binding.sh
```
