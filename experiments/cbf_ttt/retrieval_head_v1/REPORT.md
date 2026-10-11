# Explicit Candidate Retrieval Head Pilot Result

## Result

The experiment completed at commit `3d3457f`. All 25 focused server tests and the independent artifact audit passed. Both arms used the same head and writer initialization, balanced all four episodic assignments, and completed 512 optimizer updates / 4,096 question exposures. The backbone hash was unchanged and test remained sealed.

| Arm | First to last 32 loss | First to last 32 accuracy | Mean update | Peak reserved | Max relative delta |
|---|---:|---:|---:|---:|---:|
| Head only | 1.3949 to 1.3868 | 25.78% to 22.66% | 0.610 s | 9.643 GiB | 0 |
| Joint head and writer | 1.3979 to 1.3847 | 25.00% to 26.56% | 0.887 s | 15.830 GiB | 0.30103 |

The four candidate random loss is `ln(4)=1.38629`. Neither arm learned a stable training association beyond chance.

Final development top 1:

| Arm / memory | Context4 | Global16 |
|---|---:|---:|
| Head only / Correct | 23.44% | 2.34% |
| Head only / Counterfactual | 23.44% | 2.34% |
| Head only / Wrong | 23.44% | 2.34% |
| Head only / Empty | 23.44% | 2.34% |
| Joint / Correct | 25.00% | 13.28% |
| Joint / Counterfactual | 25.00% | 13.28% |
| Joint / Wrong | 24.22% | 7.81% |
| Joint / Empty | 25.00% | 3.12% |

For the primary joint arm, Counterfactual minus Correct NLL was `0.00001196`, with source group bootstrap 95% CI `[-0.00000062, 0.00002883]`; Correct minus Counterfactual top 1 was exactly zero. The global16 increase over Empty therefore does not identify the current session assignment: Correct and same query Counterfactual memories produce the same ranking.

The final head's Full-KV global16 top 1 was 3.91% on development, below the 6.25% chance rate and far below the registered 50% positive control threshold. Because this head was trained only through the compressed memory path, this observation does not by itself show that the head architecture is unable to read Full-KV representations. A direct Full-KV trained positive control is required before attributing failure solely to the writer.

All five registered joint-arm checks failed. No test examples were tokenized or scored.

## Interpretation and decision

The experiment rejects the proposed simple mean-digit embedding cosine head as an immediately usable bridge from the current dense TTT delta. It does not yet distinguish two explanations:

1. the last query hidden state plus mean digit embedding objective is not a learnable readout for multi-token codes; or
2. the context-generated dense fast memory does not preserve entity to value binding in a form available at the query state.

The next experiment directly trains the identical candidate head on Full-KV contexts, with the writer frozen and absent. If it reaches the preregistered Full-KV threshold, the writer path is isolated as the bottleneck. If it remains at chance, this retrieval head is invalid and the next implementation should use candidate-conditioned autoregressive states or an explicit structured key-value memory instead of increasing the writer budget.

## Reproduction

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_retrieval_head_v1 \
DATA=/home/ctj/cbf_ttt_dynamic_binding_v1/data \
MODEL=/home/ctj/models/Qwen3-4B \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
TEST_PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python \
bash scripts/run_cbf_retrieval_head.sh
```

## Artifacts and compatibility

Small manifests, checkpoint summaries, `summary.json`, and `execution_audit.json` are archived here. Raw step logs, 512 MB class checkpoints, optimizer states, evaluation JSONL, and the run log remain at `/home/ctj/cbf_ttt_retrieval_head_v1`.

The retrieval head exists only in the opt-in research task. Baseline logits, model classes, pretraining entry points, datasets, and default TTT behavior are unchanged. The pilot uses one initialization and eight development worlds; it is a mechanism screen rather than a final effectiveness claim.
