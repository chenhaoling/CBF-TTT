import json

from scripts.summarize_in_place_ttt_ruler_smoke import EXPECTED_TASKS, summarize


def test_summarize_complete_smoke(tmp_path):
    root = tmp_path
    (root / "run_metadata.json").write_text(
        json.dumps(
            {
                "protocol": "in_place_ttt_ruler_4k_smoke_v1",
                "upstream_commit": "abc",
                "context_length": 4096,
                "num_samples_per_task": 1,
                "plan_sha256": "plan",
                "config_sha256": "config",
                "model_config_sha256": "model",
            }
        )
    )
    (root / "exit_code.txt").write_text("0\n")
    (root / "eval.log").write_text(
        "Elapsed (wall clock) time (h:mm:ss or m:ss): 2:34.87\n"
    )
    (root / "gpu.csv").write_text(
        "timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu\n"
        "1,0,100,32000,50\n2,0,125,32000,60\n"
    )
    result_dir = root / "opencompass" / "stamp" / "results" / "model"
    result_dir.mkdir(parents=True)
    summary_dir = root / "opencompass" / "stamp" / "summary"
    summary_dir.mkdir()
    (summary_dir / "summary_stamp.txt").write_text(
        "ruler_4k: {'score': 100.0, 'naive_average': 100.0}\n"
    )
    for task in EXPECTED_TASKS:
        (result_dir / f"{task}.json").write_text(json.dumps({"score": 100.0}))

    payload = summarize(root)

    assert payload["completed"] is True
    assert payload["task_count"] == 13
    assert payload["macro_accuracy"] == 100.0
    assert payload["macro_matches_official"] is True
    assert payload["peak_gpu_memory_mib"] == {"0": 125.0}
