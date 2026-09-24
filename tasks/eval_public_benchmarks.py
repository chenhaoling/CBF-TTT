"""Small, reproducible text-only probes for the requested public benchmarks.

MMLU reports multiple-choice accuracy. QA tasks report normalized exact match,
which is a diagnostic proxy, not each benchmark's official judging protocol.
Full histories are never silently truncated; over-limit examples are skipped.
"""

import argparse
import json
import re
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_answer(value):
    text = str(value).lower()
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def iter_examples(dataset, root, limit):
    root = Path(root)
    emitted = 0
    if dataset == "mmlu":
        dev = [json.loads(line) for line in (root / "mmlu/dev.jsonl").open(encoding="utf-8")]
        demos = {}
        for row in dev:
            demos.setdefault(row["subject"], []).append(row)
        source = (json.loads(line) for line in (root / "mmlu/test.jsonl").open(encoding="utf-8"))
    else:
        path = {
            "zsre": root / "zsre/benchmark/ZsRE/ZsRE-test-all.json",
            "longmemeval_s": root / "longmemeval_s/longmemeval_s_cleaned.json",
            "locomo": root / "locomo/locomo10.json",
        }[dataset]
        source = read_json(path)
    for index, row in enumerate(source):
        if dataset == "mmlu":
            examples = demos.get(row["subject"], [])[:5]
            shots = "".join(format_mmlu_question(item) + f" {chr(65 + item['answer'])}\n\n"
                            for item in examples)
            yield {"id": f"mmlu-{index}", "context": "", "query": shots + format_mmlu_question(row),
                   "choices": [" " + chr(65 + i) for i in range(4)], "correct": row["answer"],
                   "subject": row["subject"]}
            emitted += 1
        elif dataset == "zsre":
            # This tests in-context uptake of the requested edit, not an official
            # parameter-edit success/locality/portability aggregate.
            target = row["target_new"]
            yield {"id": f"zsre-{index}",
                   "context": f"Updated fact: {row['prompt']} {target}.\n",
                   "query": f"Question: {row.get('rephrase_prompt') or row['prompt']}\nAnswer:",
                   "answer": target}
            emitted += 1
        elif dataset == "longmemeval_s":
            sessions = []
            for date, turns in zip(row["haystack_dates"], row["haystack_sessions"]):
                sessions.append(f"Session date: {date}\n")
                sessions.extend(f"{turn['role']}: {turn['content']}\n" for turn in turns)
            yield {"id": row["question_id"], "context": "".join(sessions),
                   "query": f"Question: {row['question']}\nAnswer:", "answer": row["answer"],
                   "question_type": row["question_type"]}
            emitted += 1
        else:
            conversation = row["conversation"]
            sessions = []
            for key in sorted((key for key in conversation if re.fullmatch(r"session_\d+", key)),
                              key=lambda key: int(key.split("_")[1])):
                sessions.append(f"Session date: {conversation.get(key + '_date_time', '')}\n")
                for turn in conversation[key]:
                    sessions.append(f"{turn['speaker']}: {turn.get('text', '')}\n")
                    if turn.get("blip_caption"):
                        sessions.append(f"Image caption: {turn['blip_caption']}\n")
            history = "".join(sessions)
            for question_index, qa in enumerate(row["qa"]):
                if qa.get("answer") is None:
                    continue
                yield {"id": f"locomo-{row['sample_id']}-{question_index}",
                       "context": history, "query": f"Question: {qa['question']}\nAnswer:",
                       "answer": qa["answer"], "category": qa.get("category"),
                       "text_only": True}
                emitted += 1
                if limit and emitted >= limit:
                    return
        if limit and emitted >= limit:
            return


def format_mmlu_question(row):
    choices = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(row["choices"]))
    return f"Question: {row['question']}\n{choices}\nAnswer:"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mmlu", "zsre", "longmemeval_s", "locomo"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model", help="HF-format TTT checkpoint, required for scoring")
    parser.add_argument("--output", help="JSONL predictions, required for scoring")
    parser.add_argument("--policy", default="baseline", help="baseline or controller")
    parser.add_argument("--controller", help="Controller .pt if policy=controller")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--max-examples", type=int, default=0, help="0 means all")
    parser.add_argument("--max-context-tokens", type=int, default=0,
                        help="0 uses model max_position_embeddings; over-limit examples are skipped")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--inspect-only", action="store_true", help="Check source parsing without a model")
    args = parser.parse_args()
    if args.max_examples < 0 or args.max_context_tokens < 0 or args.max_new_tokens < 1:
        parser.error("limits must be nonnegative and max-new-tokens positive")
    examples = iter_examples(args.dataset, args.data_root, args.max_examples)
    if args.inspect_only:
        sample = list(examples)
        print(json.dumps({"dataset": args.dataset, "examples": len(sample),
                          "first_id": sample[0]["id"] if sample else None}, ensure_ascii=False))
        return
    if not args.model or not args.output:
        parser.error("--model and --output are required unless --inspect-only")
    if args.policy == "controller" and not args.controller:
        parser.error("--controller is required for controller policy")

    from transformers import AutoTokenizer
    from cbf_ttt.experiment import load_controller
    from cbf_ttt.runtime import CBFSession
    from tasks.cbf_ttt import _load_model

    model = _load_model(args.model, args.device, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    controller = load_controller(args.controller, model, next(model.parameters()).device) if args.controller else None
    context_limit = args.max_context_tokens or int(model.config.max_position_embeddings)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = correct = skipped = 0
    with output.open("w", encoding="utf-8") as sink:
        for example in examples:
            context_ids = tokenizer.encode(example["context"], add_special_tokens=False)
            query_ids = tokenizer.encode(example["query"], add_special_tokens=False)
            reserve = args.max_new_tokens if args.dataset != "mmlu" else 8
            record = {"id": example["id"], "dataset": args.dataset, "policy": args.policy,
                      "context_tokens": len(context_ids), "query_tokens": len(query_ids)}
            if len(context_ids) + len(query_ids) + reserve > context_limit:
                record["skipped"] = "context_over_limit"
                skipped += 1
            else:
                session = CBFSession(model, controller)
                if context_ids:
                    record["alphas"] = session.consume(context_ids, policy=args.policy)
                if args.dataset == "mmlu":
                    losses = [session.score_answer(query_ids, tokenizer.encode(choice,
                              add_special_tokens=False)) for choice in example["choices"]]
                    predicted = min(range(4), key=lambda index: losses[index])
                    record.update({"choice_losses": losses, "prediction": predicted,
                                   "answer": example["correct"],
                                   "correct": predicted == example["correct"],
                                   "subject": example["subject"]})
                else:
                    generated_ids = session.generate(query_ids, args.max_new_tokens)
                    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    record.update({"prediction": prediction, "answer": example["answer"],
                                   "correct": normalize_answer(prediction) == normalize_answer(example["answer"])})
                correct += int(record["correct"])
                count += 1
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {"dataset": args.dataset, "policy": args.policy, "scored": count,
               "skipped": skipped, "accuracy_or_exact_match": correct / count if count else None,
               "metric": "choice_accuracy" if args.dataset == "mmlu" else "normalized_exact_match_proxy",
               "full_context_only": True, "max_context_tokens": context_limit,
               "official_benchmark_score": False}
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
