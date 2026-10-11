# Full-KV Retrieval Head Positive Control Result

## Result

The formal run completed at commit `c2e37ae`. Twelve focused server tests, two real Full-KV smoke steps, both 512-update arms, fixed evaluations, and the independent artifact audit passed. Both arms used byte-identical frozen feature artifacts and identical head initialization. Test remained sealed.

| Arm | First to last 32 loss | First to last 32 accuracy | Train-probe context4 / global16 | Dev context4 / global16 |
|---|---:|---:|---:|---:|
| Same LR (`1e-4`) | 1.3905 to 1.1801 | 26.56% to 48.83% | 42.97% / 16.41% | 28.13% / 4.69% |
| High LR (`1e-3`) | 1.4129 to 1.2589 | 25.78% to 38.67% | 38.28% / 10.16% | 24.22% / 7.81% |

The same-learning-rate arm partially fit training features but did not reach the 80% optimization threshold. Neither arm generalized to the eight disjoint development worlds. The high-learning-rate development global16 group-bootstrap 95% interval was `[2.34%, 13.28%]`; the same-learning-rate interval was `[0.78%, 9.38%]`. Both include or approach the 6.25% chance rate and are far below the registered 50% threshold. All four checks failed for both arms.

Each arm extracted 1,152 frozen Full-KV features in 271–279 seconds, peak allocated/reserved 8.597/8.865 GiB. The feature files have the same SHA-256 `b56f30948540a3bf8b5ae19829d4fbcba744f36fd97a4a6e4827368100779572`. Head-only updates averaged about 1.3 ms, confirming that the outcome is dominated by representation quality rather than compute shortage.

## Interpretation and decision

This result rejects the pilot readout: a single final-query hidden state aligned to an average of five digit embeddings does not preserve enough candidate-conditioned autoregressive information for unseen codes, even with full attention KV. Raising its learning rate does not repair the problem.

Consequently, the preceding retrieval-head pilot cannot identify the dense writer as its sole bottleneck. The established candidate-conditioned LM scorer remains the valid readout positive control: it achieves over 95% Full-KV code accuracy but failed on compressed fast memory. Further tuning of the pooled cosine head is stopped.

The next method step is a structured session key-value memory with a separately trained forgetting controller. It first establishes a readable memory substrate and directly tests the four physical transition actions: retain, forget last one, forget last two, and global reset. A later bridge can distill the validated policy into differentiable TTT memory.

## Preflight failure

The first formal launch at commit `dbabbca` completed both 1,152-row feature extractions but stopped before checkpoint-0 scoring because `_scores` indexed a dictionary with the candidate list. No optimizer update or effectiveness result was produced. The fix stacks candidate vectors in registered order and adds a direct regression test. The failed directory is preserved at `/home/ctj/cbf_ttt_fullkv_head_control_v1_preflight_dbabbca`; no threshold, data, or hyperparameter changed.

## Reproduction and compatibility

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_fullkv_head_control_v1 \
DATA=/home/ctj/cbf_ttt_dynamic_binding_v1/data \
MODEL=/home/ctj/models/Qwen3-4B \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
TEST_PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python \
bash scripts/run_cbf_fullkv_head_control.sh
```

Small summaries, manifests, and audit files are archived here. Features, raw rows, checkpoints, and logs remain under `/home/ctj/cbf_ttt_fullkv_head_control_v1`. All code is opt in; baseline model, pretraining, evaluation, and default TTT behavior are unchanged.
