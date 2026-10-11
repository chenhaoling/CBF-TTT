# Explicit Candidate Retrieval Head Pilot

## Motivation

Balanced episodic assignments and within-update swap supervision both failed with the frozen LM head. The paired swap term stayed at its no-separation value and memory-path digit NLL deteriorated severely without LM regularization. Repeating loss variants on the same readout is no longer informative.

This pilot adds a small associative readout while retaining the original context-generated in-place writer. The final query hidden state and frozen mean embedding of each candidate's digit tokens are projected to 256 dimensions, L2 normalized, and scored by cosine similarity with temperature 0.1. The candidate head and writer are trained on balanced paired assignments.

## Arms and controls

- `head_only`: train the retrieval head while the zero-initialized writer is frozen. Because every query is equally paired with all four values, this is a direct query-prior shortcut control.
- `joint_head_writer`: jointly train the same head and the seven writer conv/proj pairs.

Both arms share head seed 1301, writer seed 301, data order, four balanced cycles, two context writes per update, and 4,096 question exposures. Head learning rate is `1e-4`; writer learning rate is `1e-7`; each group is clipped separately at 1. Fast memory cap remains 0.3. Backbone and candidate embeddings are frozen.

Evaluation uses shifts 0 and 2 on the first eight train and all eight development worlds. Correct is compared with same-query Counterfactual, Wrong, and Empty memories over context4 and global16 candidates. A Full-KV retrieval-head score checks whether the head can use the ordinary attention path. Test remains sealed.

The joint arm passes only if development has Correct context4 top 1 at least 60%, Correct minus Counterfactual at least 25 points, a strictly positive group-bootstrap Counterfactual-minus-Correct NLL interval, Correct global16 top 1 at least 25%, and Full-KV global16 top 1 at least 50%. A pass is a retrieval prerequisite and still requires fresh-permutation confirmation and a generation bridge before forgetting-policy claims.

## Command

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_retrieval_head_v1 \
DATA=/home/ctj/cbf_ttt_dynamic_binding_v1/data \
MODEL=/home/ctj/models/Qwen3-4B \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
TEST_PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python \
bash scripts/run_cbf_retrieval_head.sh
```

The head is an opt-in research module used only by this task. Baseline logits, model classes, pretraining, and default TTT behavior are unchanged.
