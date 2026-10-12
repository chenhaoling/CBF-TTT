# Low-Rank Fast-Weight Forgetting Result

## Main result

The experiment completed at commit `3f66da9`. Thirteen server tests, a fresh 128/32/32-world corpus, three independent low-rank projection seeds, the aggregate development gate, one-time test release, and independent full recomputation audit passed.

Each assignment was encoded as an entity-key/value outer product. Batch deltas were additive components of a 64 by 64 fast-weight matrix. At each decision the experiment compared global alpha 0, 0.5, and 1 against removing the referenced component, while adding the same current delta.

| Policy | Dev top 1 | Dev NLL | Test top 1 | Test NLL | Worst test seed |
|---|---:|---:|---:|---:|---:|
| Fixed alpha 0 | 65.65% | 1.4798 | 65.76% | 1.4722 | 65.30% |
| Fixed alpha 0.5 | 57.93% | 1.2702 | 57.88% | 1.2721 | 57.68% |
| Fixed local | **82.07%** | 1.1002 | **82.13%** | 1.1039 | 81.53% |
| Fixed alpha 1 | 47.80% | 1.8984 | 47.61% | 1.9081 | 47.14% |
| Global-only learned | 78.77% | 0.7959 | 78.36% | 0.7965 | 77.52% |
| Reference-blind | 82.08% | 1.0996 | 82.13% | 1.1039 | 81.53% |
| **Full learned** | **96.82%** | **0.3842** | **96.68%** | **0.3895** | **95.75%** |
| Visible physical rule | 96.82% | 0.3842 | 96.68% | 0.3895 | 95.75% |
| Counterfactual oracle | 96.93% | 0.3837 | 96.85% | 0.3890 | 96.07% |

All six preregistered development checks passed. Relative to the strongest control, fixed local deletion, the learned controller improved development top 1 by 14.75 points, paired source-world bootstrap 95% CI `[14.06, 15.35]`, and reduced NLL by 0.7159, 95% CI `[0.6927, 0.7405]`.

The one-time test result reproduced the effect: +14.55 top-1 points over fixed local, 95% CI `[13.98, 15.12]`, and NLL reduction 0.7143, 95% CI `[0.6913, 0.7371]`. Relative to the learned controller restricted to global alpha 0/0.5/1, the test gain was 18.32 points, 95% CI `[17.80, 18.85]`.

The three test projection seeds obtained 97.85%, 95.75%, and 96.74% for the learned controller. Their corresponding oracle results were 97.96%, 96.07%, and 96.88%, so the learned policy stayed within 0.32 points of oracle in every representation.

## Interpretation

This result extends the exact-memory pilot to an additive low-rank fast-weight substrate. It provides evidence that:

- learned state-dependent forgetting remains effective when entity/value associations interfere in a shared matrix;
- local component removal is necessary: a controller restricted to a single global alpha cannot match it;
- the effect is stable across three independent memory projections and a fresh held-out corpus;
- the controller nearly matches the counterfactual oracle.

This is closer to the TTT update equation than dictionary memory, but it remains a controlled surrogate. The current Qwen dense writer has not yet demonstrated a readable decomposition into these components.

## Continuous-alpha limitation

The training label counts expose a critical limitation. Across 6,144 decisions per projection seed, alpha 0.5 was selected only 14–17 times, while local deletion appeared 3,058–3,068 times, alpha 0 about 1,523–1,534 times, and alpha 1 exactly 1,536 times. Thus this experiment strongly supports selective local/global forgetting but does not substantiate the stronger claim that intermediate continuous alpha values are broadly useful.

The next registered experiment must deliberately create conflicts that trade off preserving old facts against accepting current updates, evaluate a denser alpha grid, and show that intermediate optima occur often enough to train and test a continuous regressor.

## Resources and provenance

The experiment used 4,608 episodes and 27,648 queries. Controller construction, three-seed training, development, and conditional test took 139.3 seconds on CPU. Test was read only after the aggregate development gate passed. Raw development rollouts, test recomputation inputs, controllers, and the generated corpus remain under `/home/ctj/cbf_ttt_lowrank_forgetting_v1`; compact manifests, summaries, training metadata, and audit results are archived here.

All code is opt in. Baseline Qwen, pretraining, runtime, and default In-Place TTT behavior are unchanged.
