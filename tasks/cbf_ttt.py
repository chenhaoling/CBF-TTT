"""CLI for CBF-TTT counterfactual labels, controller fitting, and rollout evaluation."""

import argparse
import json
from pathlib import Path


def _load_model(path: str, device: str, dtype: str):
    import torch
    import inference_model  # noqa: F401  Register the repository's TTT AutoModel classes.
    from transformers import AutoModelForCausalLM

    resolved = torch.device(device)
    resolved_dtype = (
        torch.bfloat16 if dtype == "bfloat16" or (dtype == "auto" and resolved.type == "cuda") else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(path, dtype=resolved_dtype)
    model.to(resolved).eval()
    model.requires_grad_(False)
    return model


def _model_args(parser):
    parser.add_argument("--model", required=True, help="HF-format TTT checkpoint from the baseline conversion script")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="Generate offline counterfactual labels")
    _model_args(collect)
    collect.add_argument("--data", required=True, help="Pretokenized JSONL scenarios")
    collect.add_argument("--split", choices=("train", "dev", "test"), required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--grid", default="0,0.25,0.5,0.75,1")
    collect.add_argument("--state-policy", choices=("baseline", "controller"), default="baseline")
    collect.add_argument("--controller", help="Required for controller state resampling")
    collect.add_argument("--every", type=int, default=1, help="Collect every Nth complete chunk boundary")
    collect.add_argument("--tie-tolerance", type=float, default=1e-6)

    train = sub.add_parser("train", help="Fit controller on train labels, select epoch on dev labels")
    train.add_argument("--train-samples", nargs="+", required=True)
    train.add_argument("--dev-samples", nargs="+", required=True)
    train.add_argument("--layers", nargs="+", type=int, required=True, help="TTT layer IDs from model config")
    train.add_argument("--output", required=True)
    train.add_argument("--width", type=int, default=128)
    train.add_argument("--semantic-size", type=int, default=64)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=42)

    evaluate = sub.add_parser("eval", help="Full independent session rollout on dev or test")
    _model_args(evaluate)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--split", choices=("dev", "test"), required=True)
    evaluate.add_argument("--controller")
    evaluate.add_argument("--policies", nargs="+", default=("baseline", "controller"))
    evaluate.add_argument("--output", required=True)

    generate = sub.add_parser("generate", help="Greedy answer generation after CBF context adaptation")
    _model_args(generate)
    generate.add_argument("--context-ids", required=True, help="JSON list of context token IDs")
    generate.add_argument("--query-ids", required=True, help="JSON list of query token IDs")
    generate.add_argument("--controller", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--tokenizer", help="Optional tokenizer path for decoding")

    args = parser.parse_args()
    from cbf_ttt.experiment import (
        collect_labels, evaluate as run_evaluation, load_controller, load_scenarios, parse_grid,
        summarize_label_profile, train_controller,
    )
    from cbf_ttt.runtime import CBFSession

    if args.command == "train":
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result = train_controller(
            args.train_samples, args.dev_samples, args.output, tuple(sorted(args.layers)),
            args.width, args.semantic_size, args.epochs, args.batch_size, args.lr, args.seed,
        )
    else:
        model = _load_model(args.model, args.device, args.dtype)
        controller = (
            load_controller(args.controller, model, next(model.parameters()).device) if args.controller else None
        )
        if args.command == "collect":
            if args.state_policy == "controller" and controller is None:
                parser.error("--controller is required for controller state policy")
            scenarios = load_scenarios(args.data, args.split)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            count = collect_labels(
                model, scenarios, args.output, parse_grid(args.grid),
                args.state_policy, controller, args.every, args.tie_tolerance,
            )
            profile = summarize_label_profile(args.output)
            profile.update({"scenarios": len(scenarios), "grid": parse_grid(args.grid), "every": args.every})
            Path(args.output + ".summary.json").write_text(
                json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            result = {"states": count, "profile": profile}
        elif args.command == "eval":
            if "controller" in args.policies and controller is None:
                parser.error("--controller is required for controller evaluation")
            scenarios = load_scenarios(args.data, args.split)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            result = run_evaluation(model, scenarios, args.output, args.policies, controller)
        else:
            session = CBFSession(model, controller)
            alphas = session.consume(json.loads(args.context_ids))
            output_ids = session.generate(json.loads(args.query_ids), args.max_new_tokens)
            result = {"alphas": alphas, "generated_ids": output_ids}
            if args.tokenizer:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
                result["text"] = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
