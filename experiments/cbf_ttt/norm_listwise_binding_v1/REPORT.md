# Norm Constrained Listwise Binding Result

## Goal and execution

This run recovered the candidate aligned experiment after the unbounded predecessor crossed the fast memory safety limit. Every generated layer delta was differentiably capped at `0.3 × ||W||`; the original baseline path remains unchanged when the cap is omitted. Two matched arms trained the seven Qwen3-4B writer pairs for 1,024 question exposures on 32 train worlds and evaluated 8 development worlds. Test was not scored.

The two GPU run completed all 256 updates per arm. Fourteen focused server tests passed and the independent artifact audit passed. Both arms shared exact initialization, retained the frozen backbone hash, and saved FP32 writer and AdamW state.

## Optimization result

| Arm | First 32 retrieval loss | Last 32 | First 32 accuracy | Last 32 | Mean update | Peak allocated / reserved |
|---|---:|---:|---:|---:|---:|---:|
| Retrieval only | 1.5732 | 1.2629 | 26.56% | 46.09% | 0.693 s | 18.816 / 19.105 GiB |
| Retrieval + LM | 1.5010 | 1.3198 | 23.44% | 37.50% | 0.681 s | 18.820 / 19.115 GiB |

The four way chance loss is `log(4)=1.3863`. Both arms ended below chance, so the listwise target supplied a usable optimization signal. Neither reached the registered 50% last window accuracy threshold. Maximum measured relative deltas were 0.30109 and 0.30087; the small overshoot reflects BF16 norm measurement around the 0.3 detached scale and remained below the independent safety tripwire of 1.

## Final causal retrieval result

The primary arm was Retrieval + LM.

| Split / memory | Correct | Permuted | Twin | Wrong | Empty |
|---|---:|---:|---:|---:|---:|
| Train probe, 16 candidate top 1 | 18.75% | 15.62% | 17.19% | 6.25% | 3.12% |
| Development, 16 candidate top 1 | 7.81% | 7.81% | 7.81% | 3.12% | 6.25% |

On development, Permuted minus Correct true digit NLL was `+0.0200`, bootstrap 95% CI `[0.0013, 0.0411]`. This is a small likelihood effect, while top 1 was identical. Wrong minus Correct was `-0.0442`, CI `[-0.4666, 0.3835]`; Empty minus Correct was `-0.5377`, CI `[-0.9533, -0.1723]`. The latter means the populated learned memory made the true digits less likely than Empty on average. Correct memory complete code generation remained zero. Full-KV development code accuracy was 92.19%, confirming that the reassigned facts were readable from the original context.

Retrieval only ended with train Correct and Permuted both at 20.31%, while development Correct was 4.69% and Permuted 3.12%. Its large negative Correct versus Empty NLL reflects severe language modeling degradation, which is why it was not the primary arm.

## Conclusion

The norm constraint solved the numerical failure and candidate aligned training genuinely optimized its four way objective. The registered stage gate still failed: neither arm reached the training accuracy threshold, neither formed a train Correct versus Permuted separation, and the primary development ranking gate failed. The small significant development NLL difference is insufficient because it did not change top 1 and did not generalize to Wrong or Empty controls.

Inspection of the protocol reveals a slow parameter shortcut: the same training query is repeatedly paired with the same value. The writer parameters can learn entity to answer associations without using the current session memory. The next experiment therefore counterfactually reassigns all four values across repeated episodes while holding questions and backgrounds fixed, and compares that episodic curriculum to this fixed assignment baseline.

## Files, reproduction, and compatibility

- `cbf_ttt/event_writer.py`: optional differentiable per layer memory cap.
- `tasks/train_cbf_listwise_binding.py`: four candidate objective, training, evaluation, summary, and fixed gates.
- `scripts/run_cbf_norm_listwise_binding.sh`: tests, two GPU smoke/formal execution, summary, and audit.
- `scripts/audit_cbf_listwise_binding.py`: independent schedule, checkpoint, metric, and test seal checks.
- `tests/test_cbf_listwise_binding.py`: cap gradient, objective, candidates, and gate tests.
- `summary.json`, `execution_audit.json`, `RUN_STATUS.json`, and `git_commit.txt`: downloaded compact artifacts. Raw steps, evaluations, checkpoints, and optimizer states remain under `/home/ctj/cbf_ttt_norm_listwise_binding_v1` on `hku-gpu2`.

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_norm_listwise_binding_v1 \
DATA=/home/ctj/cbf_ttt_dynamic_binding_v1/data \
MODEL=/home/ctj/models/Qwen3-4B \
PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_train_py311/bin/python \
TEST_PYTHON=/home/ctj/miniconda3/envs/cbf_ttt_verify_20260923/bin/python \
bash scripts/run_cbf_norm_listwise_binding.sh
```

All additions are separate opt in entry points. The baseline model, pretraining configuration, runtime, and default memory behavior are unchanged. The result uses one writer seed, eight development source groups, and synthetic random values; it is a diagnostic result rather than a paper claim.
