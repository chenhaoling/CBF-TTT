"""Build disjoint natural-source selective-forgetting scenes; no repeated filler."""

import argparse
import hashlib
import json
from pathlib import Path


REGIMES = ("stable", "recent1_distractor", "recent2_distractor", "topic_switch")


def make_scenes(documents, chunk_size=4096):
    if len(documents) != 16:
        raise ValueError("the preregistered pilot requires exactly 16 documents")
    ids = [row["sha256"] for row in documents]
    if len(set(ids)) != len(ids):
        raise ValueError("source documents must be distinct")
    scenes = []
    for group in range(8):
        a, b = documents[2*group:2*group+2]
        for row in (a, b):
            if len(row["ids"]) < 7*chunk_size + 160:
                raise ValueError("source too short; repeating filler is forbidden")
        def chunk(doc, index):
            return doc["ids"][index*chunk_size:(index+1)*chunk_size]
        layouts = (
            ([(a, 0), (a, 1), (a, 2), (a, 3)], a, 4),
            ([(a, 0), (a, 1), (b, 0), (a, 2)], a, 3),
            ([(a, 0), (b, 0), (a, 1), (a, 2)], a, 3),
            ([(a, 0), (a, 1), (b, 0), (b, 1)], b, 2),
        )
        for regime, (prefix, target, first) in zip(REGIMES, layouts):
            future = []
            for step in range(3):
                index = first + step
                start = (index + 1)*chunk_size
                future.append({"ids": chunk(target, index),
                               "query_ids": target["ids"][start:start+32],
                               "answer_ids": target["ids"][start+32:start+160]})
            scenes.append({"id": f"g{group:02d}_{regime}", "group": group,
                           "split": "pilot" if group < 4 else "confirm", "regime": regime,
                           "protocol": "selective_forgetting_v1", "chunk_size": chunk_size,
                           "sources": [a["sha256"], b["sha256"]],
                           "prefix": [chunk(doc, index) for doc, index in prefix], "future": future})
    return scenes


def normalize(text):
    return " ".join(text.split())


def packed_text(row):
    # VeOmni's plaintext adapter stores the packed string in content_split.
    text = row.get("content_split", row.get("text"))
    if not isinstance(text, str):
        raise ValueError("training data must expose content_split or text")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    parquet = pq.ParquetFile(args.parquet)
    if "text" not in parquet.schema.names:
        raise ValueError("this protocol requires the publisher's text column")
    documents, hashes, anchors = [], set(), []
    # Read the final row group in small batches; never load the full shard.
    rg = parquet.num_row_groups - 1
    offset = sum(parquet.metadata.row_group(i).num_rows for i in range(rg))
    for batch in parquet.iter_batches(batch_size=8, row_groups=[rg], columns=["text"]):
        for row in batch.to_pylist():
            row_index = offset
            offset += 1
            raw = row["text"]
            if not isinstance(raw, str):
                continue
            sha = hashlib.sha256(raw.encode()).hexdigest()
            if sha in hashes:
                continue
            tokens = tokenizer.encode(raw, add_special_tokens=False)
            if len(tokens) < 7*4096+160:
                continue
            used = tokens[:7*4096+160]
            # Multiple interior anchors survive a training packing boundary.
            doc_anchors = []
            for start in [c*4096 + 512 for c in range(7)] + [7*4096]:
                anchor = normalize(tokenizer.decode(used[start:start+128]))[:192]
                if len(anchor) < 96:
                    raise ValueError("source lacks sufficiently long overlap anchors")
                doc_anchors.append(anchor)
            anchors.extend(doc_anchors)
            hashes.add(sha)
            documents.append({"sha256": sha, "row_index": row_index, "ids": used,
                              "original_qwen_tokens": len(tokens)})
            print(json.dumps({"documents": len(documents), "row_index": row_index}), flush=True)
            if len(documents) == 16:
                break
        if len(documents) == 16:
            break
    if len(documents) != 16:
        raise RuntimeError("not enough long unique documents in final row group; protocol not changed")
    # Match against the exact packed training text, after whitespace normalization.
    unique_anchors = sorted(set(anchors))
    training_rows = matches = 0
    with Path(args.training_data).open() as stream:
        for line in stream:
            row = json.loads(line)
            packed = normalize(packed_text(row))
            matches += any(anchor in packed for anchor in unique_anchors)
            training_rows += 1
            if training_rows % 20000 == 0:
                print(json.dumps({"training_rows_scanned": training_rows, "matches": matches}), flush=True)
    metadata = {"protocol": "selective_forgetting_v1", "source": "manifestai/longcrawl64",
                "source_parquet": args.parquet, "parquet_rows": parquet.metadata.num_rows,
                "source_row_group": rg, "training_data": args.training_data,
                "training_rows_scanned": training_rows, "overlap_matching_rows": matches,
                "anchors": len(set(anchors)), "audit": "normalized long-anchor exact match, not semantic dedup",
                "documents": [{k: v for k, v in d.items() if k != "ids"} for d in documents]}
    Path(str(output)+".meta.json").write_text(json.dumps(metadata, indent=2)+"\n")
    if matches or not training_rows:
        raise RuntimeError("training overlap audit failed; no scenarios written")
    scenes = make_scenes(documents)
    output.write_text("".join(json.dumps(row)+"\n" for row in scenes))
    print(json.dumps({"scenarios": len(scenes), "groups": 8, "overlap_matching_rows": matches}), flush=True)


if __name__ == "__main__":
    main()
