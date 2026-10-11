# Counterfactually Reassigned Episodic Binding

## Question

The norm constrained four candidate writer reduced its training retrieval loss below chance, but its final train Correct versus permuted top 1 gap was only 3.125 points and its development top 1 was 7.8125% on 16 candidates. The fixed entity to value relation permits a slow parameter shortcut: repeated optimizer updates can associate a training query with one answer without reading the current fast memory.

This experiment tests that explanation by presenting the same packed context and the same four queries under four coherent cyclic value assignments. Every query is paired equally with all four values in the episodic arm, so its correct answer can only be inferred from the current context generated memory.

## Fixed comparison

- `fixed_assignment`: sees only the original assignment on every round.
- `episodic_assignment`: sees all four cyclic assignments exactly once per context over four rounds.
- Both arms use the same 32 train and 8 development worlds, initialization seed 301, context order seed 1021, 1,024 question exposures, 256 updates, one 4,096 token write per update, four four candidate query reads, AdamW learning rate `1e-7`, BF16 forward, FP32 writer and optimizer state, clip 1, and detached per layer fast memory cap 0.3.
- Both optimize four candidate retrieval CE plus `0.25 ×` the balanced answer language modeling loss.
- The original packed context length is preserved. Only the four audited equal length answer spans in its fact suffix are cyclically replaced. Queries, natural background, and all other tokens remain unchanged.
- Test remains sealed. Evaluation uses the first eight train groups and all eight selected development groups at assignment shifts 0 and 2.

## Causal controls and endpoints

For every evaluated query and assignment:

- `Correct`: memory written from that assignment;
- `Counterfactual`: memory from the next cyclic assignment for the same entities and queries;
- `Wrong`: same shift from another world;
- `Empty`: no fast memory;
- `Full-KV`: complete reassigned 4,096 token context retained as the readability control.

Candidate ranking is reported both within the four values assigned to that context and over the established 16 candidate set. Generation metrics remain secondary. The final endpoint is fixed at 1,024 exposures; no midpoint selection is used.

The episodic arm passes the preliminary binding gate only if development satisfies all of:

1. Full-KV complete code accuracy at least 80%;
2. Correct memory four candidate top 1 at least 60%;
3. Correct minus Counterfactual four candidate top 1 at least 25 points;
4. the group bootstrap 95% interval for Counterfactual minus Correct true digit NLL is strictly positive;
5. Correct memory 16 candidate top 1 at least 25%.

This gate is deliberately about causal retrieval. Passing it would justify a subsequent generation and forgetting experiment; it would not by itself establish end to end selective forgetting.

## Command

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_episodic_binding_v1 \
DATA=/home/ctj/cbf_ttt_dynamic_binding_v1/data \
MODEL=/home/ctj/models/Qwen3-4B \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
TEST_PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python \
bash scripts/run_cbf_episodic_binding.sh
```

The runner refuses to overwrite an existing output root, runs focused tests, performs one update smoke tests on both GPUs, enforces the 28 GiB smoke ceiling, runs the matched arms in parallel, summarizes, and independently audits the result.

## Scope and risks

The cyclic episodes are derived from train/dev data already used in earlier diagnostics; they are not new blind test examples. Four cyclic permutations are a small subset of all assignments and may allow a rotation specific shortcut, although held out worlds and exact counterfactual memories still test content dependence. The evaluation has eight development source groups and one writer seed. A passing result therefore needs a fresh assignment family and seed confirmation before a paper claim. A failing result would reject the slow parameter shortcut as the sole explanation but would not prove that the frozen Qwen writer architecture cannot learn associative memory at larger scale.

The cap is computed in FP32 and applied to a BF16 delta. Resource and artifact checks allow a fixed 0.005 absolute norm-ratio tolerance for BF16 rounding; the independent safety stop remains 1.0. This tolerance changes neither the generated memory nor an optimization or efficacy threshold.
