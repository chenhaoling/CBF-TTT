"""No-data numerical bridge for V16 shared-prompt candidate scoring."""

from __future__ import annotations

import argparse
import json
import math

from tasks.cbf_checkpoint_readout import PlainSession, load_plain
from tasks.evaluate_mquake_qwen_outcome_v16 import (
    encode_candidates, render_prompt, score_query,
)
from tasks.train_cbf_event_writer import write_json


TOLERANCE = 2e-5


def run(model_path, tokenizer_path, output):
    import torch
    from transformers import AutoTokenizer
    from cbf_ttt.runtime import _nll_from_hidden

    torch.manual_seed(1602)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, metadata = load_plain(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    spec = {"kind": "synthetic_bridge", "statement": "The capital of France is",
            "answer": "Paris", "candidates": ["Paris", "London"]}
    measured = score_query(model, tokenizer, [], {}, spec)
    prompt = render_prompt([], {}, spec["statement"])
    prefix, encoded = encode_candidates(tokenizer, prompt, spec["candidates"])
    reference = {}
    for answer, answer_ids in encoded.items():
        session = PlainSession(model)
        hidden = session._forward(prefix + answer_ids[:-1])
        labels = torch.tensor(answer_ids, dtype=torch.long, device=session.device)
        start = len(prefix) - 1
        reference[answer] = _nll_from_hidden(
            model, hidden[:, start:start + len(answer_ids)], labels)
    errors = {answer: abs(measured["scores"][answer] - reference[answer])
              for answer in spec["candidates"]}
    reference_prediction = min(spec["candidates"],
                               key=lambda value: (reference[value], value))
    checks = {
        "ttt_disabled": model.config.ttt_mode is False,
        "expected_adapters_excluded": len(metadata["excluded_adapters"]) == 14,
        "finite_scores": all(math.isfinite(value) for value in
                             (*measured["scores"].values(), *reference.values())),
        "max_nll_error_at_most_2e_5": max(errors.values()) <= TOLERANCE,
        "prediction_identical": measured["prediction"] == reference_prediction,
        "shared_prompt_cache_recorded": measured["resources"]["candidate_scores"] == 2,
    }
    result = {
        "protocol": "mquake_qwen_outcome_v16_shared_prompt_bridge",
        "checks": checks, "passed": all(checks.values()), "tolerance": TOLERANCE,
        "candidate_nll_error": errors, "max_nll_error": max(errors.values()),
        "shared_prediction": measured["prediction"],
        "reference_prediction": reference_prediction,
        "prompt_tokens": measured["prompt_tokens"],
        "resources": measured["resources"],
        "model": model_path, "tokenizer": tokenizer_path,
        "synthetic_only": True, "pilot_rows_loaded": 0,
        "model_scored_on_experiment": False, "external_sets_loaded": False,
    }
    write_json(output, result)
    if not result["passed"]:
        raise RuntimeError("shared-prompt scoring bridge failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
