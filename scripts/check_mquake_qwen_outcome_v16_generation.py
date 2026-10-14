"""No-data Qwen bridge for V16 greedy decoding from a shared prompt cache."""

from __future__ import annotations

import argparse
import json

from tasks.cbf_checkpoint_readout import load_plain
from tasks.evaluate_mquake_qwen_outcome_v16 import normalize_generation, score_query
from tasks.train_cbf_event_writer import write_json


def run(model_path, tokenizer_path, output):
    import torch
    from transformers import AutoTokenizer

    torch.manual_seed(1604)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, metadata = load_plain(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    spec = {"kind": "synthetic_generation_bridge",
            "statement": "The capital of France is", "answer": "Paris",
            "candidates": ["Paris", "London"]}
    measured = score_query(model, tokenizer, [], {}, spec, generate=True)
    generation = measured["generation"]
    decoded = tokenizer.decode(generation["token_ids"], skip_special_tokens=True)
    text = decoded.replace("\r", "\n").split("\n", 1)[0].strip()
    checks = {
        "ttt_disabled": model.config.ttt_mode is False,
        "expected_adapters_excluded": len(metadata["excluded_adapters"]) == 14,
        "candidate_scores_complete": set(measured["scores"]) == set(spec["candidates"]),
        "generation_ids_decode_exactly": generation["raw_text"] == decoded
            and generation["text"] == text,
        "bounded_generation": generation["tokens"] == len(generation["token_ids"])
            and generation["tokens"] <= generation["max_new_tokens"] <= 64,
        "registered_stop_reason": generation["stop_reason"]
            in {"eos", "newline", "max_new_tokens"},
        "exact_metrics_recomputed": generation["exact"] == int(text == spec["answer"])
            and generation["normalized_exact"] == int(
                normalize_generation(text) == normalize_generation(spec["answer"])),
        "resource_measurement_complete": measured["resources"]["seconds"] > 0
            and measured["resources"]["gpu_peak_allocated_gib"]
                >= measured["resources"]["gpu_start_allocated_gib"]
            and measured["resources"]["gpu_peak_reserved_gib"]
                >= measured["resources"]["gpu_start_reserved_gib"],
    }
    result = {
        "protocol": "mquake_qwen_outcome_v16_generation_bridge",
        "checks": checks, "passed": all(checks.values()),
        "generation": generation, "candidate_prediction": measured["prediction"],
        "candidate_scores": measured["scores"], "prompt_tokens": measured["prompt_tokens"],
        "resources": measured["resources"], "model": model_path, "tokenizer": tokenizer_path,
        "synthetic_only": True, "experiment_rows_loaded": 0,
        "external_sets_loaded": False,
    }
    write_json(output, result)
    if not result["passed"]:
        raise RuntimeError("V16 synthetic generation bridge failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", dest="model_path", required=True)
    parser.add_argument("--tokenizer", dest="tokenizer_path", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
