"""One-example-per-task RULER 4K pipeline smoke for unmodified Qwen3-4B."""

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

model_path = __import__("os").environ["BASELINE_EVAL_MODEL"]
tokenizer_path = __import__("os").environ.get(
    "BASELINE_EVAL_TOKENIZER", model_path
)
model_name = __import__("os").environ.get("BASELINE_EVAL_NAME", "qwen3_4b_base")
num_samples = int(__import__("os").environ.get("BASELINE_EVAL_NUM_SAMPLES", "1"))

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr=model_name,
        path=model_path,
        tokenizer_path=tokenizer_path,
        # The converted local Qwen config triggers a false-positive Mistral
        # regex warning in Transformers 4.57.3. Keep Qwen's native regex.
        tokenizer_kwargs=dict(fix_mistral_regex=False),
        engine_config=dict(session_len=4096, max_batch_size=1, tp=1),
        gen_config=dict(
            top_k=20,
            temperature=0,
            top_p=0.95,
            do_sample=False,
            enable_thinking=False,
            max_new_tokens=64,
        ),
        max_seq_len=4096,
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
    item["abbr"] = item["abbr"] + "_4k"
    item["num_samples"] = num_samples
    item["max_seq_length"] = 4096
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
# OpenCompass 0.5.4 names the generated group ``ruler_4k``.  Selecting the
# full group name makes the compact table expose the aggregate score.
summarizer = dict(
    dataset_abbrs=["ruler_4k"], summary_groups=sum([ruler_summary_groups], [])
)
