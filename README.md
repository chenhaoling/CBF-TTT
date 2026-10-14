<div align="center">
 👋 Hi, everyone!
  <br>
  We are <b>ByteDance Seed team.</b>
</div>

<p align="center">
  You can get to know us better through the following channels👇
  <br>
  <a href="https://seed.bytedance.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/5793e67c-79bb-4a59-811a-fcc7ed510bd4">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
  <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
</p>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)

# In-Place Test-Time Training

**Seamlessly Endowing LLMs with Test-Time Training Ability**

Guhao Feng\*, Shengjie Luo\*, Kai Hua, Ge Zhang, Wenhao Huang, Di He, Tianle Cai

<p align="center">
  <a href="https://openreview.net/forum?id=dTWfCLSoyl">
    <img src="https://img.shields.io/badge/ICLR%202026-Oral-b31b1b?style=for-the-badge"></a>
  <a href="https://arxiv.org/abs/2604.06169">
    <img src="https://img.shields.io/badge/Paper-PDF-blue?style=for-the-badge&logo=adobeacrobatreader"></a>
  <a href="#license">
    <img src="https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge"></a>
</p>

In-Place TTT is a drop-in test-time training method for Transformer LLMs. This repository provides the training, checkpoint conversion, inference, and evaluation stack built on VeOmni, together with recommended configs for Qwen3-8B and LLaMA-3.1-8B.

## News

- CBF-TTT experiment notes: [native TTT accumulation diagnostic](NATIVE_ACCUMULATION_PLAN.md) and [completed report](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/native_accumulation/REPORT.md).
- Next causal gate: [single-update fast-weight content protocol](FASTWEIGHT_CONTENT_PLAN.md).
- Readout-calibrated follow-up: [fast-weight content v2 protocol](FASTWEIGHT_CONTENT_V2_PLAN.md).
- Completed content result: [single-update fast-weight content v2 report](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/fastweight_content_v2/REPORT.md).
- Final-checkpoint writer recovery: [pre-registered binding protocol](FINAL_WRITER_BINDING_PLAN.md).
- Final-checkpoint writer result: [audited Q-gate report](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/final_writer_binding_v1/REPORT.md).
- Low-rank Qwen memory bridge: [pre-registered protocol](QWEN_LOWRANK_BRIDGE_PLAN.md).
- Low-rank Qwen bridge result: [audited Q/V report](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/qwen_lowrank_bridge_v1/REPORT.md).
- Qwen-grounded multi-component forgetting: [pre-registered protocol](QWEN_LOWRANK_FORGETTING_PLAN.md).
- MQuAKE subject-risk result: [V12 report](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/mquake_subject_risk_v12/REPORT.md).
- Final raw-ensemble calibration: [V13 pre-registered protocol](MQUAKE_ENSEMBLE_CALIBRATION_V13_PLAN.md).
- Final raw-ensemble result: [V13 audited report](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/mquake_ensemble_calibration_v13/REPORT.md).
- Session-relative forgetting: [V14 pre-registered protocol](MQUAKE_ONLINE_MARGIN_V14_PLAN.md).

[2026/03] The codebase is open-sourced.
<br>
[2026/02] In-Place TTT is accepted to ICLR 2026 as an Oral presentation.

## Table of Contents

- [In-Place Test-Time Training](#in-place-test-time-training)
  - [News](#news)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Getting Started](#getting-started)
    - [Environment Setup](#environment-setup)
    - [Data Preparation](#data-preparation)
    - [Recommended Config](#recommended-config)
    - [Training](#training)
    - [Checkpoint Conversion](#checkpoint-conversion)
    - [Evaluation](#evaluation)
  - [Features](#features)
  - [License](#license)
  - [Citation](#citation)
  - [About ByteDance Seed Team](#about-bytedance-seed-team)

## Introduction

Current large language models follow a static "train then deploy" paradigm. Once deployed, model weights are frozen and cannot adapt to new information encountered during inference. This limits long-context reasoning, where useful information arrives progressively and the model would benefit from updating itself as it reads.

**In-Place Test-Time Training (In-Place TTT)** addresses this by updating a subset of model parameters, the MLP down-projection fast weights, during inference. Unlike prior TTT approaches that require architectural side modules or external memory, In-Place TTT stays inside the standard Transformer block and remains compatible with off-the-shelf autoregressive LLMs.

The method is centered around three ideas:

1. **Architectural compatibility.** Fast weights live in the existing MLP down-projection matrix, so no extra attention heads or memory modules are introduced.
2. **LM-aligned objective.** The fast-weight update is aligned with next-token prediction instead of a generic reconstruction target.
3. **Chunk-wise update.** Long sequences are split into chunks so updates can be computed efficiently and scaled to long contexts.

![In-Place TTT Method Overview](assets/pipeline.png)

As used in this repo, the end-to-end workflow is:

1. Provide your own VeOmni-compatible processed dataset and base model assets.
2. Launch continual pretraining with VeOmni through `train.sh` and `tasks/train_torch.py`.
3. Export DCP checkpoints into HuggingFace format with `scripts/merge_dcp_to_hf.py`.
4. Run TTT-aware inference and RULER evaluation with `inference_model/`, `eval.sh`, and `eval_config/`.

The repository includes recommended training configs for Qwen3-8B and LLaMA-3.1-8B, checkpoint conversion utilities, and a full RULER evaluation pipeline via OpenCompass from 4K to 256K context lengths.

## Getting Started

### Environment Setup

**Step 1.** Install PyTorch and FlashAttention:

```bash
pip3 install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip3 install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
rm flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

**Step 2.** Install VeOmni from the validated commit:

```bash
pip3 install "veomni @ git+https://github.com/ByteDance-Seed/VeOmni.git@9b91e164bea9e17f17ed490aab5e076c2335ca25"
```

**Step 3.** Install the remaining dependencies:

```bash
pip3 install liger-kernel
pip3 install byted-wandb torchdata blobfile datasets diffusers tiktoken timm
pip3 install transformers==4.57.3
pip3 install opt_einsum einops

pip3 uninstall -y byted-wandb wandb
pip3 install byted-wandb
```

**Step 4.** Optionally verify the installed VeOmni source:

```bash
python3 - <<'PY'
import json, pathlib, veomni
p = pathlib.Path(veomni.__file__).resolve().parents[1] / "veomni-0.1.0.dist-info" / "direct_url.json"
print("veomni file:", veomni.__file__)
print("direct_url:", json.loads(p.read_text()) if p.exists() else "not found")
PY
```

### Data Preparation

The baseline expects a processed dataset through `data.train_path`. For the optional CBF-TTT Qwen3-4B FineWeb-Edu/LongCrawl64 preparation and public evaluation downloads, see [DATA_DOWNLOAD_GUIDE.md](DATA_DOWNLOAD_GUIDE.md).

For the optional selective-forgetting experiment (global history plus the last two update contributions), see [SELECTIVE_FORGETTING_PLAN.md](SELECTIVE_FORGETTING_PLAN.md). Its standalone data builder and runner preserve the original baseline; implementation changes and experimental status are recorded in [MODIFICATION_LOG.md](MODIFICATION_LOG.md).

For the subsequent native/CBF multi-chunk numerical-path diagnostic, see [MEMORY_STABILITY_PLAN.md](MEMORY_STABILITY_PLAN.md) and its [completed report](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/memory_stability/REPORT.md). This diagnostic preserves the baseline and does not train a new controller.

For the training/inference forward comparison and bounded writer training resource checks, see [TRAIN_INFER_PARITY_PLAN.md](TRAIN_INFER_PARITY_PLAN.md) and the [results and optimization risks](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/train_infer_parity/REPORT.md). These checks do not replace the pretrained checkpoint.

The subsequent [optimizer audit](OPTIMIZER_AUDIT_PLAN.md) compares fused/standard CE gradients and isolates writer updates using fixed parameter snapshots. Its [completed report](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/optimizer_audit/REPORT.md) includes actual relative parameter displacements, local interventions, and reproducibility limits.

The [multi-decision forgetting study](MULTIDECISION_FORGETTING_PLAN.md) directly compares one versus two/three forgetting opportunities, exhaustive global schedules, sampled local schedules, and fixed controls. See its [launch record](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/multidecision_forgetting/RUN_STATUS.md) for the frozen design and current execution stage.

The next [memory value validation plan](MEMORY_VALUE_VALIDATION_PLAN.md) specifies fresh paired facts, four fixed policies, full-KV and memory-only reads, and correct/zero/twin memory controls. It separates content value, forgetting tradeoffs, and optional local selection. Its standalone builder and V/F runner are implemented. The first run stopped at the preregistered readability gate; see the [completed Q report](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/memory_value_validation/REPORT.md). V/F and confirmation remain untested.

The subsequent [readability calibration](READABILITY_CALIBRATION_PLAN.md) uses only observed development sources to compare fact distance, key load, query format and record order, with an exact bridge to the original Q scores.

The recommended configs assume:

- `data.data_type=plaintext`
- `data.datasets_type=iterable`
- `data.text_keys=content_split`

For dataset argument definitions and supported loading modes, refer to the official VeOmni docs:

- [Arguments API Reference](https://veomni.readthedocs.io/en/latest/usage/arguments.html)
- [Basic Modules / Dataset & DataLoader](https://veomni.readthedocs.io/en/latest/usage/basic_modules.html)

Example:

```bash
bash train.sh tasks/train_torch.py configs/pretrain/qwen3_longct.yaml \
  --data.train_path /path/to/your_data \
  --train.output_dir /path/to/your_output_dir
```

### Recommended Config

Below is the recommended model config pattern used in the provided Qwen and LLaMA examples.

> **Tip:** Set `ttt_target: input_embed` for from-scratch pretraining, or `ttt_target: hidden_states` for continual training.

```yaml
model:
  model_path: /path/to/your_base_model
  foundation:
    ttt_layers: [0, 6, 12, 18, 24, 30, 36]
    ttt_mode: true
    ttt_proj: true
    ttt_lr: 3
    ttt_chunk: 4096

data:
  train_path: /path/to/your_data
  train_size: 20000000000
  dataloader_type: native
  datasets_type: iterable
  data_type: plaintext
  max_seq_len: 65536
  text_keys: content_split
  drop_last: true

train:
  output_dir: /path/to/your_output_dir
  data_parallel_mode: fsdp2
  global_batch_size: 64
  micro_batch_size: 1
  optimizer: adamw
  lr: 5.0e-6
  lr_warmup_ratio: 0.02
  lr_decay_style: cosine
  lr_decay_ratio: 0.90
  weight_decay: 0.1
  max_grad_norm: 1.0
  max_steps: 5000
  enable_mixed_precision: true
  enable_gradient_checkpointing: true
  enable_full_shard: true
  init_device: meta
  ckpt_manager: dcp
  save_steps: 500
  save_hf_weights: true
  use_wandb: true
```

The corresponding recommended config files are:

- `configs/pretrain/qwen3_longct.yaml`
- `configs/pretrain/llama3_longct.yaml`

### Training

Quick smoke run:

```bash
bash train.sh tasks/train_torch.py configs/pretrain/qwen3_longct.yaml \
  --train.output_dir /path/to/your_output_dir \
  --train.max_steps 1 \
  --train.use_wandb false
```

Recommended Qwen config override:

```bash
bash train.sh tasks/train_torch.py configs/pretrain/qwen3_longct.yaml \
  --train.wandb_project your_wandb_project \
  --train.wandb_name your_run_name \
  --train.output_dir /path/to/your_output_dir \
  --model.foundation '{"ttt_layers":[0,6,12,18,24,30,36],"ttt_mode":true,"ttt_proj":true,"ttt_lr":3,"ttt_chunk":4096}'
```

Recommended LLaMA config override:

```bash
bash train.sh tasks/train_torch.py configs/pretrain/llama3_longct.yaml \
  --train.wandb_project your_wandb_project \
  --train.wandb_name your_run_name \
  --train.output_dir /path/to/your_output_dir \
  --model.foundation '{"ttt_layers":[0,6,12,18,24,30,36],"ttt_mode":true,"ttt_proj":true,"ttt_lr":3,"ttt_chunk":4096}'
```

### Checkpoint Conversion

Convert VeOmni DCP checkpoints into HuggingFace format:

```bash
python scripts/merge_dcp_to_hf.py \
  --load-dir /path/to/your_checkpoint_dir

python scripts/merge_dcp_to_hf.py \
  --load-dir /path/to/your_checkpoint_dir \
  --save-dir /path/to/your_hf_checkpoint_dir \
  --model-assets-dir /path/to/your_base_model \
  --shard-size 5000000000
```

### Evaluation

Run the default RULER evaluation sweep:

```bash
bash eval.sh
```

Single-config smoke run:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -c \
  "import inference_model; from opencompass.cli.main import main; import sys; sys.argv=['opencompass','eval_config/ruler_4k.py','--debug']; main()"
```

To evaluate your own checkpoints, update `eval_config/models.py` with your model name and HuggingFace checkpoint path.

## Features

- **Drop-in TTT for standard Transformers.** In-Place TTT updates the MLP down-projection fast weights without introducing extra architectural side modules.
- **LM-aligned fast-weight updates.** The optimization target is derived for autoregressive language modeling instead of a generic reconstruction objective.
- **Long-context continual pretraining stack.** The repo includes recommended Qwen3-8B and LLaMA-3.1-8B configs built on VeOmni and FSDP2.
- **Checkpoint export path.** `scripts/merge_dcp_to_hf.py` converts VeOmni DCP checkpoints into HuggingFace format.
- **TTT-aware inference and evaluation.** `inference_model/`, `eval.sh`, and `eval_config/` cover inference and RULER evaluation through OpenCompass.
- **Long-context coverage.** The evaluation setup spans 4K, 8K, 16K, 32K, 64K, 128K, and includes a 256K config.

## License

This project is licensed under the [Apache License 2.0](./LICENSE).

## Citation

If you find this work useful for your research and applications, feel free to give us a star or cite us using:

```bibtex
@inproceedings{feng2026inplace,
  title     = {In-Place Test-Time Training},
  author    = {Feng, Guhao and Luo, Shengjie and Hua, Kai and Zhang, Ge and Huang, Wenhao and He, Di and Cai, Tianle},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {Oral Presentation},
  url       = {https://arxiv.org/abs/2604.06169}
}
```

## Contact

If you have any questions about this project, please feel free to contact us:

> Guhao Feng (fenguhao@stu.pku.edu.cn)  
> Shengjie Luo (shengjieluo@bytedance.com)

## CBF-TTT latest experiment

The audited Qwen-grounded multi-component forgetting run completed on the final 1B-token Qwen3-4B checkpoint. Development accuracy was 27.73% for the full learned policy versus 32.20% for fixed global reset; the greedy oracle reached only 28.49%. The preregistered readability and policy gates failed, so test remained sealed. See the [complete report](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/qwen_lowrank_forgetting_v1/REPORT.md).

The next experiment follows the [Qwen memory geometry and readability plan](QWEN_LOWRANK_GEOMETRY_PLAN.md): compare the current projection against 128/256-dimensional whitening under a perfect semantic state, quantify cross-talk, and retrain the forgetting controller only after a representation passes the fixed readability gate.

That geometry experiment is now [complete](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/qwen_lowrank_geometry_v1/REPORT.md). Whitening raised the generic 16-component sweep from 69.16% to 87.03%, but perfect-state event accuracy remained near 32%; random codes reached 100%. The next registered step is the [Qwen contextual geometry study](QWEN_CONTEXTUAL_GEOMETRY_PLAN.md), which isolates key versus value collisions and evaluates frozen final-layer contextual identifiers before releasing controller v2.

The contextual study also [completed and passed](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/qwen_contextual_geometry_v1/REPORT.md). Random-key/Qwen-value reached 98.18%, while Qwen-key/random-value reached 54.88%, localizing the old failure primarily to keys. Frozen final-layer `context_last` reached 100% across every seed, family and 1–32 component load; the unchanged forgetting-controller v2 is now the active experiment.

The contextual forgetting v2 [passed its development gate and one-time sealed test](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/qwen_contextual_forgetting_v2/REPORT.md). Full learned reached 89.05% versus 79.01% for fixed local, a +10.04 point gain with 95% CI `[9.35,10.82]`. The explicit physics rule matched learned accuracy, so the next paper-level experiment must remove symbolic-reference privilege and test automatic evidence on natural sequential updates.

The first subject-disjoint [ZsRE contextual readout](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/zsre_contextual_readout_v1/REPORT.md) reached 100% exact and about 98.7% rephrase accuracy, but same-subject different-relation locality reached 88.26%, below the fixed 90% development gate. Test remained sealed. The registered [contrastive key study](ZSRE_CONTRASTIVE_KEY_PLAN.md) now isolates whether calibration-only prompt/rephrase alignment with explicit locality hard negatives fixes that natural-key failure without changing values, memory, backbone, or the release gate.

That [contrastive key study is complete](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/zsre_contrastive_keys_v1/REPORT.md). Relation-aware alignment raised locality from 88.26% to 99.09%, but unseen-subject rephrase accuracy fell to about 69.5%; positive-only alignment showed the same generalization problem. The gate failed and test remained sealed. The next iteration replaces the global key warp with a pairwise query–memory scope resolver and explicitly treats locality as “defer to the frozen model.”

The first [pairwise scope resolver](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/zsre_scope_resolver_v1/REPORT.md) also completed with test sealed. A structured cosine control reached 94.51% overall, 100% exact, 98.38% rephrase and 89.83% locality. A frozen cross-linear nine-way head improved locality to 95.13% but created many false deferrals, reducing overall accuracy to 82.12%. The next iteration freezes cosine key selection and learns only the binary retrieve/defer decision.

The decomposed retrieve/defer gate [passed development and its one-time test](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/zsre_decomposed_scope_gate_v1/REPORT.md). Test accuracy was 98.47% versus 88.58% for an ensemble cosine threshold, a +9.90 point gain with 67-session bootstrap 95% CI `[8.57,11.24]`. Exact/rephrase/locality were 100.00/97.75/98.06%, and the minimum gate seed reached 98.46%. This validates automatic update-scope evidence in structured natural memory; the next experiment must use it for causal retain/local/global forgetting over natural sequences.

The first calibration-only [natural sequential forgetting pilot](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/zsre_natural_forgetting_pilot_v1/REPORT.md) found that simple cosine-local deletion already reached 97.21%, near the 97.77% oracle, while direct learned-gate transfer reached only 84.43%. Development and test remained unopened. The next calibration pilot adds same-subject locality facts as hard retain events so semantic scope, rather than only subject novelty, is required.

The expanded subject-disjoint [ZsRE learned-forgetting test](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/zsre_forgetting_final_test_v1/REPORT.md) is complete. At the development-selected threshold, learned forgetting reached 77.23% query and 95.16% action accuracy versus cosine-local 70.67/82.43%, with paired gains of +6.59 points (95% CI `[5.43,7.79]`) and +12.73 points (CI `[10.27,15.25]`). The independent audit passed. The strict absolute gate failed because update action was 89.34%; the supported claim is robust relative improvement in constructed candidate memory. Native dense-writer content binding remains unsupported, so the next architecture step uses the validated Qwen low-rank fast memory with automatic natural evidence and a new untouched benchmark.

The preregistered [CounterFact zero-shot transfer](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/counterfact_zero_shot_forgetting_v1/REPORT.md) also completed and passed independent audit. The frozen ZsRE controller beat cosine by +4.57 query points (95% CI `[3.90,5.24]`) and +11.51 action points (CI `[9.51,13.50]`), but reached only 67.77% action and 30.70% update accuracy; fixed retain had higher query accuracy. A hash-verified diagnostic found probability AUC 0.759 but only 49.08% correct old-chain selection among triggered updates. This rejects reliable zero-shot transfer and identifies pairwise candidate ranking plus conflict calibration as the next method target.

Cross-domain reranker [V1](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/crossdomain_forgetting_reranker_v1/REPORT.md) raised observed CounterFact OOF candidate top-1 to 78.47% and causal action to 84.93%, but update remained 63.69%. Class-balanced [V2](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/crossdomain_forgetting_reranker_v2/REPORT.md) improved candidate top-1 to 79.62% and oracle-state update to 78.22%, yet aggressive 0.20–0.25 thresholds caused causal rollout collapse to 78.26% action and 46.79% query. The V2 independent audit passed. This identifies irreversible state-distribution shift, rather than static candidate capacity alone, as the next forgetting-policy problem.

The no-weight-change [constraint replay](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/crossdomain_forgetting_constraint_v1/REPORT.md) recovered hard-retain/action to 88.66/84.29%, but its causal update lagged oracle state by 9.30 points. [On-policy V3](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/onpolicy_forgetting_gate_v3/REPORT.md) then trained on policy-induced states and selected thresholds by causal inner replay; it still reached only 59.75% update and 82.34% action. The registered stop rule ends scalar-gate tuning on this representation. LongMemEval-S and LoCoMo remain unscored while a stronger semantic scope representation is developed.

[Scoped Delta Ranker V4](SCOPED_DELTA_RANKER_V4_PLAN.md) is the next preregistered stage. It builds audited same-chain, same-subject/different-chain, and different-subject pair supervision, then evaluates a relation-focused Qwen3-4B candidate representation with five-fold CounterFact OOF predictions. The supplied subject span is used only as an oracle-segmentation diagnostic; chain IDs, relation IDs, candidate age, LongMemEval-S, and LoCoMo are excluded from model selection. A causal delta-bank experiment is allowed only if the fixed candidate-ranking gate passes.

The completed [V4 report](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/scoped_delta_ranker_v4/REPORT.md) shows a significant but insufficient improvement: candidate top-1 rose from 79.62% to 82.93% (gain 3.31 points, 95% CI `[0.64,5.86]`), while the 88%/93% top-1/MRR gate failed. All 95 remaining same-subject errors selected a different relation. [Relation-supervised V5](RELATION_SUPERVISED_SCOPE_V5_PLAN.md) therefore tests a frozen-Qwen relation classifier with subject-disjoint raw-CounterFact training; relation IDs are supervision only and are not inference features.

The completed [V5 report](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/relation_supervised_scope_v5/REPORT.md) reached 97.83% candidate top-1 and 98.92% MRR, a 14.90-point gain over V4 with CI `[12.23,17.58]`. Its exact relation-class auxiliary was 87.64%, below the preregistered 90% floor, even though pairwise compatibility remained correct for 82/97 class errors. [Automatic factorized V6](AUTOMATIC_FACTORIZED_SCOPE_V6_PLAN.md) now replaces the supplied subject span with an IDF-weighted shared-span rule and evaluates fully text-only entity/relation scope resolution before any causal deletion experiment.

The completed [V6 report](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/automatic_factorized_scope_v6/REPORT.md) reached 93.25% text-only top-1 and 96.37% MRR, improving over V4 by 10.32 points with CI `[7.52,13.25]`, but it missed the 95%/97% gate. Every inner fold selected the maximum registered combination weight and improved monotonically through that boundary. [V7](AUTOMATIC_SCOPE_BOUNDARY_V7_PLAN.md) therefore performs one fixed boundary extension; failure permanently stops scalar scope-weight tuning, while a pass freezes the resolver for causal evaluation.

[V7](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/automatic_scope_boundary_v7/REPORT.md) reached only 93.63% top-1 and 96.76% MRR, so scalar combination tuning has stopped. [Text-only calibrator V8](TEXT_ONLY_SCOPE_CALIBRATOR_V8_PLAN.md) now trains a nested-OOF rank-16 residual over fixed relation compatibility, automatic IDF-span evidence, and lexical similarity. It still receives no subject/relation/chain metadata at inference and must significantly beat V7 before causal forgetting resumes.

[V8](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/text_only_scope_calibrator_v8/REPORT.md) reached 94.01% top-1 and did not significantly improve V7, so calibration on the fixed feature family has stopped. [Learned entity scope V9](LEARNED_ENTITY_SCOPE_V9_PLAN.md) trains subject-disjoint binary same-entity heads from raw text pairs, using same-relation/different-subject hard negatives, and combines their probability with the frozen automatic relation posterior. Inference remains text-only.

[V9](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/learned_entity_scope_v9/REPORT.md) obtained 96.57% entity-pair AUC but only 92.48% candidate top-1; the fixed probability product was worse than V8. This rejects rigid entity/relation multiplication while leaving OOF entity probability as a potentially complementary feature. The next allowed experiment is one nested-OOF fusion with V8; failure ends CounterFact score calibration and moves to token-level entity-span extraction.

[V10](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/entity_fusion_scope_v10/REPORT.md) reached 94.65% top-1 and 97.29% MRR. Its +0.64-point gain over V8 had CI `[-0.25,1.66]`, so the registered gate failed and existing-score fusion is closed. A post-hoc audit found that 41/42 errors selected the correct subject but the wrong relation, while oracle segmentation fixed 32/42. [V11](TOKEN_SPAN_RELATION_SCOPE_V11_PLAN.md) therefore learns subject spans to construct de-identified relation views rather than adding another entity score.

[V11 completed](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/token_span_relation_scope_v11/REPORT.md) at 97.20% top-1 and 98.60% MRR. Its +2.55-point gain over V10 had paired-bootstrap CI `[1.02,4.20]`, span token F1 was 98.14%, and the complete registered gate passed. The resolver is now frozen for the [V12 causal sequential-memory experiment](CAUSAL_TOKEN_SPAN_FORGETTING_V12_PLAN.md).

[V12](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/causal_token_span_forgetting_v12/REPORT.md) improved causal query accuracy over fixed retain by 1.46 points with CI `[0.99,1.96]`, but its linear gate reached only 93.50% hard-retain and 80.00% update accuracy. The candidate resolver stayed at 97.07%. The final registered CounterFact iteration is [V13](CAUSAL_ONPOLICY_SCOPE_V13_PLAN.md): a nonlinear gate with richer posterior geometry and one policy-induced state aggregation round.

[V13 completed](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/causal_onpolicy_scope_v13/REPORT.md) at 95.29% overall action, 96.56% hard-retain, and 89.55% primary-update accuracy. It improved query accuracy over fixed retain by 1.95 points with CI `[1.55,2.37]` and over V12 by 0.49 points with CI `[0.24,0.75]`. Five of six registered checks passed; primary-update missed the 90% floor by 0.45 points, so CounterFact controller tuning stops. The on-policy arm was not significantly better than the oracle-state-only nonlinear ablation, making posterior geometry the supported change. These action rates independently match the held-out ZsRE result to within about 0.2 points, but neither result proves the previously failed native dense writer. External work therefore proceeds with the readable contextual low-rank Qwen memory rather than another `ttt_proj`/`ttt_conv` threshold search.

The deterministic [cross-domain synthesis](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/crossdomain_replication_synthesis_v1/REPORT.md) confirms that CounterFact and ZsRE agree within 0.44 points on every action slice and that both domain-specific query/action gains have positive clustered-bootstrap lower bounds. The next registered evaluation is [LongMemEval-S zero-shot transfer](LONGMEMEVAL_TRANSFER_V1_PLAN.md): it freezes the ZsRE contextual representation, forgetting head, and threshold, then scores the 78 knowledge-update examples exactly once. It measures stale/current session state and retrieval rather than claiming an official LongMemEval generation score; LoCoMo remains sealed.

[LongMemEval-S transfer V1](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/longmemeval_forgetting_transfer_v1/REPORT.md) failed its one-time gate. The frozen controller reduced stale hit@5 from 61.54% to 7.69%, but retained only 50.00% of current answer sessions and 41.69% of non-answer sessions; its strict pair state was 44.87% versus 51.28% for deletion-rate-matched random. In contrast, answer-ID oracle local deletion reached 100% pair state, increased current hit@5 to 66.67%, and reduced stale hit@5 to zero. The supported diagnosis is a short-fact-to-long-session scope-calibration failure with a viable oracle opportunity. The 78 outcomes are locked against retuning; subsequent session-level development must use independently constructed ZsRE/CounterFact episodes before LoCoMo is opened.

[Sparse long-session gate V1](LONG_SESSION_SPARSE_GATE_V1_PLAN.md) is the registered corrective development stage. `tasks/build_long_session_sparse_corpus.py` turns the already observed, subject-disjoint ZsRE calibration/dev facts into 16/32/48/64-session trajectories with exactly one update, two same-subject hard-retain events, unrelated sessions, and four fact units per session. LongMemEval outcomes, session IDs, and labels are excluded. Fit/tune selection must recover stale deletion while keeping current and unrelated sessions before LoCoMo can be scored.

The executable stage is `scripts/run_long_session_sparse_gate_v1.sh`. It embeds every fact unit once with the frozen final Qwen3-4B checkpoint, precomputes session-pair evidence for efficient causal replay, trains three rank-16 deletion gates on calibration-fit, selects epoch/threshold on calibration-tune, refits on fit+tune, and reads the subject-disjoint development split once. The deterministic random control matches the primary controller's exact deletion count without reading labels. Development candidate accuracy and every release check are computed on development rather than tune; LoCoMo remains sealed unless all registered checks pass.

[V1 completed](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/long_session_sparse_gate_v1/REPORT.md) with an independently audited gate failure. Maximum-cosine candidate top-1 was only 33.99%; the three-seed controller reached 55.18% strict pair state, 73.75% stale deletion, 77.81% current retention, and 58.43% non-target retention. Oracle local deletion was 100% on every length, so the opportunity remains but the flat maximum over multi-fact sessions is rejected. LoCoMo remains sealed; the next iteration must solve hierarchical unit-to-session scope before another deletion gate is evaluated.

[Hierarchical candidate V2](LONG_SESSION_CANDIDATE_V2_PLAN.md) is the frozen next stage. It trains a rank-16 listwise session resolver over order-neutral multi-instance geometry and lexical/relation evidence, with no age, unit-position, chain, subject, or target features. Fit/tune choices are frozen before the already designated subject-disjoint development split is materialized. Passing a five-check 95%-top1 release gate is required before any new deletion controller; an independent auditor recomputes all selections, ranks, clustered bootstrap results, hashes, and claim locks.

[V2 completed](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/long_session_candidate_v2/REPORT.md) at 67.38% top-1 and 73.16% MRR. Its 33.33-point gain over V1 was significant (95% CI `[28.93,37.64]`) and stable across seeds, but the 95% release gate failed. The [factorized scope-key V3 plan](FACTORIZED_SCOPE_CANDIDATE_V3_PLAN.md) now removes changed value payloads from candidate keys while preserving them as memory contents. It repeats the frozen split/listwise protocol and requires both a 95% learned result and a 30-point fixed scope-key mechanism gain before causal deletion can resume.

[V3 completed](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/factorized_scope_candidate_v3/REPORT.md) and passed all six checks: learned top-1/MRR were 97.07/98.38%, every length exceeded 95.6% top-1, and paired gain over V2 was 29.40 points with CI `[24.72,34.18]`. Fixed scope-only maximum cosine independently rose by 59.64 points over the full-text key. [Factorized causal forgetting V4](FACTORIZED_CAUSAL_FORGETTING_V4_PLAN.md) now freezes that resolver and trains a high-specificity scope-plus-payload-conflict gate. It uses complete causal tune replay and cumulative memory-retention checks before any LoCoMo release.

[V4 completed](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/factorized_causal_forgetting_v4/REPORT.md) with 98.85% current retention, 97.34% non-target retention, and 97.32% hard-retain accuracy, but only 83.71% stale deletion and 82.74% pair state averaged across seeds. One seed exceeded 91% pair state while another reached 70%, and no fit-only tune point jointly met update and safety constraints; refitting shifted probability calibration. [Cross-fit V5](CROSSFIT_CALIBRATED_FORGETTING_V5_PLAN.md) replaces that split/refit procedure with nested subject folds, fixed-epoch training, per-model causal calibration, and an unrefit 15-model median-margin ensemble. Development is conditionally sealed behind an OOF safety/efficacy gate.

[V5 completed](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/crossfit_calibrated_forgetting_v5/REPORT.md) on 1,068 strictly held-out OOF episodes. It reached 99.81% current, 99.70% non-target, and 99.86% hard-retain retention, but stale deletion/pair state were 83.99/83.90%, about one point below the fixed 85% prerequisite; dev was never loaded. Threshold search is closed. [Episode-barrier V6](EPISODE_BARRIER_FORGETTING_V6_PLAN.md) keeps the entire cross-fit protocol fixed and replaces independent BCE with a fixed-margin objective that ranks each episode's update above its maximum retain event.

[V6 completed](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/episode_barrier_forgetting_v6/REPORT.md) at 99.81% current, 99.98% non-target, and 100% hard-retain safety, but stale deletion/pair state were 83.52/83.43%; the fixed barrier did not improve V5. All audits passed and dev remained absent. The registered stop rule now closes loss, epoch, and threshold iteration on this constructed ZsRE corpus. The next stage requires a separate natural long-session training/development source while preserving LoCoMo as confirmation.

[MQuAKE temporal transfer V7](MQUAKE_TEMPORAL_TRANSFER_V7_PLAN.md) is the independent
natural-session stage. It converts official MQuAKE-T real temporal updates into
entity-disjoint 16/32/48/64-event sessions and first evaluates the frozen V3 scope
resolver plus V5 forgetting ensemble with no MQuAKE fitting or threshold selection.
Only a preregistered zero-shot transfer pass may advance to one-time LoCoMo confirmation;
otherwise MQuAKE train/tune adaptation is allowed while its dev split remains locked.

[V7 completed](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/mquake_temporal_transfer_v7/REPORT.md)
and failed its independently audited zero-shot gate. Candidate top-1 was 26.34%,
current/non-target retention was 31.82/29.16%, and strict pair state was 9.83% versus
51.14% for exact deletion-count matched random. The 100% oracle leaves a valid local
deletion opportunity, but universal zero-shot transfer is rejected. The registered
[V8 adaptation](MQUAKE_TEMPORAL_ADAPTATION_V8_PLAN.md) trains only on MQuAKE train,
calibrates only on tune without refitting, and conditionally reads entity-disjoint dev.
LoCoMo remains sealed.

[V8 completed](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/mquake_temporal_adaptation_v8/REPORT.md)
and stopped before dev. Tune safety recovered to 100.00% current, 98.86% non-target,
and 97.49% hard-retain accuracy, but shortlist/top-1 remained 28.82% and pair state was
15.88%. The next [V9 candidate diagnostic](MQUAKE_CANDIDATE_DIAGNOSTIC_V9_PLAN.md)
compares full-list and 4/8/16/32 shortlist ranking with explicit scope anchors on tune;
no deletion head is retrained and dev remains absent.

[V9 completed](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/mquake_candidate_diagnostic_v9/REPORT.md)
with a passing audit. Raw explicit-anchor cosine and `anchor_4` both reached 100% tune
top-1 at every length, while raw all-unit maximum was 31.84% with mean target rank
7.25. [V10](MQUAKE_ANCHOR_FORGETTING_V10_PLAN.md) freezes `anchor_4`, retrains only
the causal deletion heads on train, calibrates on tune without refitting, and permits
dev only after a six-check prerequisite. LoCoMo remains sealed.

[V10 completed](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/mquake_anchor_forgetting_v10/REPORT.md)
with 99.67% candidate top-1, 96.19% stale deletion, 99.87% current retention, and
96.12% strict pair state on entity-disjoint dev. Its +84.49-point pair-state gain over
matched random had 95% CI `[81.71,88.18]`. The formal gate still failed because
non-target and hard-retain safety were 91.80/91.81% versus the 95% floors.
[V11](MQUAKE_ROBUST_CALIBRATION_V11_PLAN.md) therefore freezes the resolver and uses
subject-level OOF conservative calibration before any independent MQuAKE-CF stress
test. V10 dev will not be rescored; LoCoMo remains sealed.

V11 is implemented by `tasks/train_mquake_robust_calibration_v11.py` and
`scripts/run_mquake_robust_calibration_v11.sh`. Its independent auditor recomputes
fold assignments, every calibrated threshold, pooled and per-fold metrics, the exact
deletion-count random comparison, artifact hashes, and benchmark-access locks.

[V11 completed](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/mquake_robust_calibration_v11/REPORT.md)
with an independently audited negative release result. OOF current/non-target/
hard-retain reached 100.00/99.05/99.51%, while stale deletion and strict pair state
were 83.08%; fold pair state ranged from 69.88% to 96.91%. Read-only attribution found
13.92% correct-candidate threshold rejections and candidate failure concentrated in
one 700-episode subject. The next registered experiment is
[V12 subject-balanced risk control](MQUAKE_SUBJECT_RISK_V12_PLAN.md): nested candidate
training, a fixed anchor-16 shortlist, subject-equal losses, and cluster-level safety
calibration. External MQuAKE-CF and LoCoMo remain sealed.

V12 is implemented by `tasks/train_mquake_subject_risk_v12.py`, its independent
auditor, runner, and focused tests. The five-fold formal run uses a separate output
root and does not modify baseline training or inference defaults.

[V14 online margin](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/mquake_online_margin_v14/REPORT.md)
completed with an independent audit. It improved strict OOF pair state from V13's
37.05% to 49.87% while retaining 99.87% of current and 99.85% of non-target memory,
but fold pair state ranged from 1.34% to 94.73%. Post-hoc locked diagnostics reject
relation scarcity and additional z-normalization; the failure is single-fold threshold
calibration under unseen-subject score shift. The registered
[V15 jackknife consensus](MQUAKE_JACKKNIFE_CONSENSUS_V15_PLAN.md) keeps the same causal
median margin and tests four independently calibrated inner models with a three-of-four
forgetting vote. External confirmation sets remain sealed.

[Entity-fusion V10](ENTITY_FUSION_SCOPE_V10_PLAN.md) is that final registered fusion. It reuses frozen fold-specific V9 entity heads, appends entity probability/logit to V8's features, and retains the same nested five-fold/three-seed protocol. A failure permanently ends CounterFact score fusion; a pass freezes the resolver for causal forgetting.

## About [ByteDance Seed Team](https://seed.bytedance.com/)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society.

Readability calibration is [complete](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/readability_calibration/REPORT.md): 128 contexts / 384 queries, exact original-Q bridge. Rebuilt QA targets reached 100% eight-choice accuracy, while cloze failed on near dual-key facts; this is development-only protocol evidence, not memory or forgetting efficacy. The original confirmation split remains unscored.

The next [fixed-slot record interference experiment](RECORD_INTERFERENCE_PLAN.md) keeps the original QA, target facts and background prefixes fixed while varying distractor slots, with natural-token and newline controls and two exact reproduction bridges.

The [completed record interference report](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/record_interference/REPORT.md) documents 160 contexts / 320 queries: removing distractor records raised target eight-choice accuracy from 31.25% to 100% under both fillers, with exact bridges and fixed surrounding tokens. Position/load effects were non-monotonic and filler-sensitive. This is evidence of input-reading interference, not fast-memory forgetting efficacy.

The next [identity-matched position experiment](MATCHED_POSITION_PLAN.md) moves exactly the same two distracting records across three positions, using two donor pairs, fixed surrounding slots and three exact bridges to the prior run.

The [completed matched-position report](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/matched_position/REPORT.md) contains 112 contexts / 224 queries with 96 exact bridge checks. Fixed distracting identities did not support a uniform later-is-worse rule. An explicitly exploratory label-overlap diagnostic also flags accidental correct-color matches; fresh four-regime readability and memory-value tests remain pending.

The next [disjoint-color four-regime readout calibration](DISJOINT_READOUT_PLAN.md) retains distracting records while excluding both target variants, anchor and old correction values from distractor colors. It compares two fixed query formats under a per-domain, per-regime, per-query gate.

The [completed disjoint-readout report](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/disjoint_readout/REPORT.md) documents 80 contexts / 288 queries with 32 exact bridge checks and an independent construction/score audit. Neither fixed format passed the preregistered 16-cell readability gate (QA: 14 failed cells; binding: 12). The run stopped before fresh-source construction, memory writes or V/F. This M=0 diagnostic does not establish forgetting efficacy or failure; an original-Qwen3-4B versus final-checkpoint comparison is the proposed next diagnostic.

The next [original versus final checkpoint diagnostic](CHECKPOINT_READOUT_PLAN.md) freezes the same disjoint inputs and compares plain M=0 inference, with full final-checkpoint reproduction and a native-Transformers original-model path check. It does not train or advance to V/F.

The [completed checkpoint comparison](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/checkpoint_readout/REPORT.md) includes 176 contexts / 608 queries and an independent audit. Both models failed the fixed readability gate; under the binding prompt, mean target accuracy fell from 73.44% (original) to 48.44% (final), and anchor accuracy from 100% to 59.38%. Native/original and prior/final path checks were exact. These M=0 results separate a shared task difficulty from additional checkpoint-related degradation; they do not establish forgetting efficacy or a specific training failure mechanism.

The next [fixed-record length diagnostic](LENGTH_READOUT_PLAN.md) compares 6K/12K/24K nested background windows for both checkpoints. Device records and queries stay fixed, with exact long-input reproduction and audited reuse of the full 24K results. This is an M=0 readability diagnostic; removing background also changes content and distances.

The [completed length diagnostic](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/length_readout/REPORT.md) contains 288 newly evaluated contexts / 1152 queries, with exact 24K replay and independent suffix/record/score audits. Original Qwen3-4B passed the 16-cell gate at 12K with the binding prompt; the final checkpoint passed neither 6K nor 12K. The run stopped before label expansion or V/F. These M=0 results do not establish native TTT performance or forgetting efficacy; the readable original-model condition can anchor a checkpoint and native-TTT diagnostic.

固定12K的中间checkpoint与原生TTT对照：见 [实验计划](CHECKPOINT_NATIVE_PLAN.md) 与 [完整实验结果](experiments/cbf_ttt/qwen3_4b_final_1b_20260927/checkpoint_native/REPORT.md)。

任务训练语料方向：[可验证事件流训练计划](TRAINING_CORPUS_PLAN.md)，以及[已生成的288会话原型](experiments/cbf_ttt/event_curriculum_v1/REPORT.md)。

原始Qwen3-4B冻结骨干的[原生writer首次100步训练报告](experiments/cbf_ttt/event_writer_100step_v1/REPORT.md)：已在双5090完成训练、四点评估与独立审计。正确记忆NLL下降，但完整答案EM为0、预登记记忆门槛未通过；尚未进入控制器或联合训练。

阶段A第二轮：[少量世界的成对内容训练计划](EVENT_CONTENT_TRAINING_PLAN.md)，固定两个训练世界、CE/content_pair各400步，分别检查内容记忆和输出格式。[执行状态](experiments/cbf_ttt/event_content_v1/RUN_STATUS.md)。

已完成[双臂400步内容训练与独立审计](experiments/cbf_ttt/event_content_v1/REPORT.md)：训练集完整编号CE4/16、content_pair7/16，开发集均0/16；两臂均未通过固定内容门槛，未进入遗忘控制器/联合训练。

下一轮[事实优化干扰实验](FACT_INTERFERENCE_PLAN.md)：逐问题、同context四问题联合、跨context批量对照，相同问题暴露，并做可回滚的一步更新干预。[运行状态](experiments/cbf_ttt/fact_interference_v1/RUN_STATUS.md)。

该[干扰实验已完成](experiments/cbf_ttt/fact_interference_v1/REPORT.md)：发现局部“学一题伤其他题”事件，joint/mixed通过小样本训练内容门槛，但三臂开发均0/16。逐问题正确/错配均14/16，联合14/16对7/16，mixed13/16对5/16；尚未验证同context分组的独特泛化优势。

动态世界记忆写入[实验计划](DYNAMIC_MEMORY_WRITER_PLAN.md)与[完成报告](experiments/cbf_ttt/dynamic_memory_writer_v1/REPORT.md)：64/16/16个独立随机世界、FineWeb-Edu/LongCrawl64自然背景。Joint完成2048问题暴露但dev Correct/Wrong/Empty编号均为0，内容门槛失败；sequential在第1095步触发delta安全阈值。当前不进入遗忘控制器训练。

已完成[动态随机值绑定实验](experiments/cbf_ttt/dynamic_binding_v1/REPORT.md)：128/32/32世界、每臂4096问题暴露，独立审计通过。Binding相对joint CE将dev Correct-memory digit NLL降低0.0965，但两臂完整编号正确率仍为0；Binding的Wrong-minus-Correct增益仅0.0394且95% CI跨0。Full-KV为95.70%，说明writer仍未可靠绑定随机值。预登记内容门槛失败，test未评分，未进入遗忘控制器训练。

已完成[候选排序定位实验](experiments/cbf_ttt/candidate_ranking_v1/REPORT.md)：复用六个writer checkpoint，在Correct/twin/Wrong/Empty记忆下固定比较2/5/16候选，独立审计通过。最终binding臂的Correct-memory `global16` top-1为6.25%，正好等于随机水平，并与twin/Wrong相同；Correct相对Wrong的digit-NLL增益0.0401且95% CI跨0。固定结论为`no_localized_binding`：瓶颈在实体—数值绑定，而不是只在自由生成。下一步先改为四事实全量实体条件检索与within-world permutation对比，内容门槛通过后再恢复遗忘控制器。

已完成[全事实关联绑定pilot](experiments/cbf_ttt/associative_binding_v1/REPORT.md)：32/8世界、两臂各1024问题暴露，独立审计通过。新方法在中点产生Correct相对permutation的0.0125 digit-NLL收益且95% CI大于0，但终点缩至0.0017、区间跨0；终点Correct `global16` top-1为4.69%，低于6.25%随机水平，完整code仍为0。训练contrast从首段0.7989到末段0.8072，没有被优化。下一步应把同一Correct memory下的四值listwise实体检索设为主要目标，并加入梯度配平或去除LM正项消融。

下一步[候选对齐listwise绑定实验](LISTWISE_BINDING_PLAN.md)将同一Correct memory下的四值实体选择直接设为主要交叉熵目标，比较retrieval-only与`retrieval + 0.25 * LM`。使用相同32/8世界、固定0/512/1024三点、真实GPU smoke及Correct/permuted/Wrong/Empty门槛；训练probe拟合与开发泛化必须同时通过，test继续封存。

范数约束恢复实验已[完成并独立审计](experiments/cbf_ttt/norm_listwise_binding_v1/REPORT.md)：cap 0.3解决了无界fast memory安全停止，两臂末段四分类loss均低于随机损失；primary训练准确率37.5%，但dev 16候选Correct/Permuted top-1同为7.81%，完整code仍为0，正式门槛失败。小幅permutation NLL差异没有转化为排序或Wrong/Empty内容区分。

下一步[反事实重赋值episodic binding实验](EPISODIC_BINDING_PLAN.md)让同一问题在四个episode中分别对应四个不同值，消除慢参数直接记忆固定实体—编号的捷径；与只看原始赋值的matched baseline比较，并以Correct/同实体Counterfactual/跨世界Wrong/Empty及4/16候选指标检验是否真正读取当前session memory。

反事实重赋值实验已[完成并独立审计](experiments/cbf_ttt/episodic_binding_v1/REPORT.md)：固定映射末段四分类准确率36.72%，平衡重赋值仅26.56%，说明固定query→value关系夸大训练拟合；episodic开发Correct/Counterfactual四候选top-1为25.00/22.66%，NLL区间跨0，仍没有可靠current-memory绑定。

下一步[同一步反事实配对实验](PAIRED_BINDING_PLAN.md)把相反赋值放进同一个optimizer step，比较对称listwise与显式memory-swap loss；两臂复用相同两个write和全部候选score，使每个目标必须在自己的memory下优于配对memory。

同一步反事实配对已[完成并独立审计](experiments/cbf_ttt/paired_binding_v1/REPORT.md)：swap loss 0.79807→0.79864未下降，dev Correct/Counterfactual context4 top-1为21.09/21.88%，NLL差异显著朝错误方向；去掉LM后memory digit-NLL升至15.81而Empty为2.71。结果排除“仅缺同一步反事实loss”，停止继续调同一LM-head目标。

下一步[显式候选检索头pilot](RETRIEVAL_HEAD_PLAN.md)以256维归一化query/value投影读取候选，比较冻结writer的head-only捷径控制与joint head+writer；仍用平衡重赋值、Correct/Counterfactual/Wrong/Empty和4/16候选门槛。
