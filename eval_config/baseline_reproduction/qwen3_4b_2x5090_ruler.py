"""Registered 13-task RULER evaluation for the 2x5090 Qwen3-4B arms."""

import os

from mmengine.config import read_base
from opencompass.models import HuggingFaceBaseModel
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

with read_base():
    from opencompass.configs.datasets.ruler.ruler_cwe_gen import cwe_datasets
    from opencompass.configs.datasets.ruler.ruler_fwe_gen import fwe_datasets
    from opencompass.configs.datasets.ruler.ruler_niah_gen import niah_datasets
    from opencompass.configs.datasets.ruler.ruler_qa_gen import qa_datasets
    from opencompass.configs.datasets.ruler.ruler_vt_gen import vt_datasets
    from opencompass.configs.summarizers.groups.ruler import ruler_summary_groups


CONTEXT_SUFFIX = {
    4096: "4k",
    8192: "8k",
    16384: "16k",
    32768: "32k",
    65536: "64k",
    131072: "128k",
    262144: "256k",
}

model_path = os.environ["BASELINE_EVAL_MODEL"]
tokenizer_path = os.environ["BASELINE_EVAL_TOKENIZER"]
model_name = os.environ["BASELINE_EVAL_NAME"]
context_length = int(os.environ["BASELINE_EVAL_CONTEXT"])
num_samples = int(os.environ.get("BASELINE_EVAL_NUM_SAMPLES", "100"))
if context_length not in CONTEXT_SUFFIX:
    raise ValueError(f"unregistered RULER context: {context_length}")
if num_samples not in {1, 100}:
    raise ValueError("2x5090 RULER permits only 1-sample resource or 100-sample formal runs")
suffix = CONTEXT_SUFFIX[context_length]

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr=model_name,
        path=model_path,
        tokenizer_path=tokenizer_path,
        tokenizer_kwargs=dict(fix_mistral_regex=False),
        engine_config=dict(session_len=context_length, max_batch_size=1, tp=1),
        gen_config=dict(
            top_k=20,
            temperature=0,
            top_p=0.95,
            do_sample=False,
            enable_thinking=False,
            max_new_tokens=64,
        ),
        max_seq_len=context_length,
        max_out_len=64,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]

import_datasets = sum(
    [qa_datasets, niah_datasets, vt_datasets, fwe_datasets, cwe_datasets], []
)
datasets = []
for dataset in import_datasets:
    item = dataset.deepcopy()
    item["tokenizer_model"] = tokenizer_path
    item["abbr"] = item["abbr"] + "_" + suffix
    item["num_samples"] = num_samples
    item["max_seq_length"] = context_length
    item["random_seed"] = 42
    datasets.append(item)

model_dataset_combinations = [dict(models=models, datasets=datasets)]
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        task=dict(type=OpenICLInferTask),
        retry=1,
    ),
)
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=1, task=dict(type=OpenICLEvalTask)),
)
summarizer = dict(
    dataset_abbrs=["ruler_" + suffix],
    summary_groups=sum([ruler_summary_groups], []),
)
