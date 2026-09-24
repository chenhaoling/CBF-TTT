# Run only the QA subset of the baseline OpenCompass RULER 4k generator.
from mmengine.config import read_base

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.ruler.ruler_qa_gen import qa_datasets
    from .models import models as qwen3_models

NUM_SAMPLES = int(__import__("os").environ.get("CBF_RULER_NUM_SAMPLES", "100"))
if NUM_SAMPLES < 1:
    raise ValueError("CBF_RULER_NUM_SAMPLES must be positive")
max_seq_lens = [4096]
abbr_suffixs = ["4k"]
work_dir = "./results/ruler_qa/4k"
model_settings = [[model, model["path"]] for model in qwen3_models]
datasets = []
models = []
model_dataset_combinations = []
for max_seq_len, abbr_suffix in zip(max_seq_lens, abbr_suffixs):
    for model, model_path in model_settings:
        selected = []
        for dataset in qa_datasets:
            item = dataset.deepcopy()
            item["tokenizer_model"] = model_path
            item["abbr"] = item["abbr"] + "_" + abbr_suffix
            item["num_samples"] = NUM_SAMPLES
            item["max_seq_length"] = max_seq_len
            selected.append(item)
        model_dataset_combinations.append(dict(models=[model], datasets=selected))
        models.append(model)
        datasets.extend(selected)

infer = dict(partitioner=dict(type=NumWorkerPartitioner),
             runner=dict(type=LocalRunner, max_num_workers=1,
                         task=dict(type=OpenICLInferTask), retry=5))
eval = dict(partitioner=dict(type=NaivePartitioner),
            runner=dict(type=LocalRunner, max_num_workers=1,
                        task=dict(type=OpenICLEvalTask)))
