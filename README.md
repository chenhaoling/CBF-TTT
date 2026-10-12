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
