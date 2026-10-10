"""Pack dynamic single-block facts into natural 4096-token contexts; keep test un-tokenized."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from tasks.pack_cbf_event_warmup import encode_answer, sha


SOURCES = ("fineweb_edu", "longcrawl64")


def _backgrounds(corpus: Path, tokenizer, needed: int, chunk: int) -> list[dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    target = (needed + 1) // 2
    seen = set()
    with corpus.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            source = row.get("source")
            text = row.get("content_split")
            if source not in SOURCES or not isinstance(text, str):
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < chunk:
                continue
            seen.add(digest)
            by_source[source].append({
                "source": source,
                "line_number": line_number,
                "text_sha256": digest,
                "token_count": len(ids),
                "ids": ids,
                "text": text,
            })
            if all(len(by_source[name]) >= target for name in SOURCES):
                break
    if any(len(by_source[name]) < target for name in SOURCES):
        raise ValueError("natural corpus does not contain enough long unique records")
    selected = []
    positions = {name: 0 for name in SOURCES}
    for index in range(needed):
        source = SOURCES[index % len(SOURCES)]
        selected.append(by_source[source][positions[source]])
        positions[source] += 1
    return selected


def pack(source: str | Path, output: str | Path, tokenizer, tokenizer_path: str,
         corpus: str | Path, chunk: int = 4096) -> dict:
    source, output, corpus = Path(source), Path(output), Path(corpus)
    if output.exists():
        raise FileExistsError(output)
    if chunk < 64:
        raise ValueError("chunk too short")
    source_manifest = json.loads((source / "manifest.json").read_text())
    counts = source_manifest["group_counts"]
    if any(counts.get(split, 0) < 1 for split in ("train", "dev", "test")):
        raise ValueError("semantic source must contain train/dev/test groups")

    warmup = {}
    groups = {}
    for split in ("train", "dev"):
        path = source / f"{split}.warmup.jsonl"
        if sha(path) != source_manifest["files_sha256"][path.name]:
            raise ValueError("semantic source hash mismatch")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(rows) != counts[split] * 8:
            raise ValueError("warmup row count differs from source manifest")
        warmup[split] = rows
        groups[split] = sorted({row["group_id"] for row in rows})
        if len(groups[split]) != counts[split]:
            raise ValueError("warmup group count differs from source manifest")
    if set(groups["train"]) & set(groups["dev"]):
        raise ValueError("train/dev source group overlap")

    ordered_groups = [(split, group) for split in ("train", "dev") for group in groups[split]]
    selected = _backgrounds(corpus, tokenizer, len(ordered_groups), chunk)
    background_by_group = dict(zip(ordered_groups, selected))
    packed = []
    contexts = {}
    for split in ("train", "dev"):
        for row in warmup[split]:
            if row["split"] != split or len(row["write_blocks"]) != 1 or row["decision_boundaries"]:
                raise ValueError("dynamic warmup is not a single causal write block")
            group = row["group_id"]
            twin = row["id"].split(".")[1]
            context_id = f"{group}.{twin}"
            background = background_by_group[(split, group)]
            suffix_text = "\n\nActive session records:\n" + row["write_blocks"][0] + "\n"
            if row["answer"] in background["text"]:
                raise ValueError("random answer unexpectedly occurs in natural background")
            suffix = tokenizer.encode(suffix_text, add_special_tokens=False)
            need = chunk - len(suffix)
            if need <= 0 or len(background["ids"]) < need:
                raise ValueError("dynamic facts do not fit the requested chunk")
            context_ids = background["ids"][:need] + suffix
            query_ids, answer_ids = encode_answer(tokenizer, row["read_query"], row["answer"])
            if context_id in contexts and contexts[context_id] != context_ids:
                raise ValueError("questions sharing a context produced different write tokens")
            contexts[context_id] = context_ids
            packed.append({
                "id": row["id"],
                "context_id": context_id,
                "group_id": group,
                "split": split,
                "context_ids": context_ids,
                "query_ids": query_ids,
                "answer_ids": answer_ids,
                "answer": row["answer"],
                "fact_suffix_tokens": len(suffix),
                "answer_start": len(query_ids) - 1,
                "background_source": background["source"],
                "background_line_number": background["line_number"],
                "background_sha256": background["text_sha256"],
            })

    by_id = {row["id"]: row for row in packed}
    for row in packed:
        split_groups = groups[row["split"]]
        other = split_groups[(split_groups.index(row["group_id"]) + 1) % len(split_groups)]
        twin = row["context_id"].rsplit(".", 1)[1]
        row["wrong_context_id"] = f"{other}.{twin}"
        row["twin_context_id"] = f"{row['group_id']}.{1 - int(twin)}"
        row["anchor_query"] = ".asset_" in row["id"]
        twin_id = row["id"].split(".", 1)[0] + f".{1 - int(twin)}." + row["id"].split(".", 2)[2]
        if twin_id not in by_id:
            raise ValueError("packed twin row is missing")
        row["twin_row_id"] = twin_id
        if by_id[twin_id]["query_ids"] != row["query_ids"]:
            raise ValueError("twin query changed")
        if (by_id[twin_id]["answer"] != row["answer"]) != row["anchor_query"]:
            raise ValueError("only anchor answers may differ across twin sessions")

    output.mkdir(parents=True)
    rows_path = output / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in packed))
    background_manifest = [{
        "split": split,
        "group_id": group,
        "source": background_by_group[(split, group)]["source"],
        "line_number": background_by_group[(split, group)]["line_number"],
        "text_sha256": background_by_group[(split, group)]["text_sha256"],
        "token_count": background_by_group[(split, group)]["token_count"],
    } for split, group in ordered_groups]
    background_path = output / "backgrounds.json"
    background_path.write_text(json.dumps(background_manifest, indent=2) + "\n")
    result = {
        "protocol": "dynamic_memory_writer_v1",
        "source": str(source),
        "source_manifest_sha256": sha(source / "manifest.json"),
        "data_sha256": sha(rows_path),
        "backgrounds_sha256": sha(background_path),
        "natural_corpus": str(corpus),
        "natural_corpus_size": corpus.stat().st_size,
        "chunk": chunk,
        "rows": {split: len(warmup[split]) for split in ("train", "dev")},
        "contexts": {split: 2 * counts[split] for split in ("train", "dev")},
        "groups": {split: counts[split] for split in ("train", "dev")},
        "test_tokenized": False,
        "background_sources": {name: sum(item["source"] == name for item in background_manifest)
                               for name in SOURCES},
        "tokenizer": tokenizer_path,
        "tokenizer_files_sha256": {
            path.name: sha(path) for path in Path(tokenizer_path).glob("*")
            if path.name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
        },
        "answer_tokens_range": [min(len(row["answer_ids"]) for row in packed),
                                max(len(row["answer_ids"]) for row in packed)],
        "fact_suffix_tokens_range": [min(row["fact_suffix_tokens"] for row in packed),
                                     max(row["fact_suffix_tokens"] for row in packed)],
        "eos_token_id": tokenizer.eos_token_id,
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--chunk", type=int, default=4096)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(json.dumps(pack(args.source, args.output, tokenizer, args.tokenizer,
                          args.corpus, args.chunk), indent=2))


if __name__ == "__main__":
    main()
