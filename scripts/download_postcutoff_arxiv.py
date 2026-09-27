"""Download dated arXiv PDFs and extract long, provenance-tracked documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


ATOM = "{http://www.w3.org/2005/Atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"


def query_entries(category: str, start: str, end: str, max_results: int,
                  api_feed: Path | None = None) -> tuple[list[dict], int, str]:
    if not re.fullmatch(r"[a-z-]+\.[A-Z]{2}", category):
        raise ValueError("category must look like cs.CL")
    if not re.fullmatch(r"\d{12}", start) or not re.fullmatch(r"\d{12}", end) or start > end:
        raise ValueError("start/end must be ordered YYYYMMDDHHMM values")
    query = f"cat:{category} AND submittedDate:[{start} TO {end}]"
    params = urllib.parse.urlencode({"search_query": query, "start": 0,
                                     "max_results": max_results,
                                     "sortBy": "submittedDate", "sortOrder": "descending"})
    url = f"https://export.arxiv.org/api/query?{params}"
    if api_feed is None:
        request = urllib.request.Request(url, headers={"User-Agent": "CBF-TTT-research/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read()
    else:
        payload = api_feed.read_bytes()
    root = ET.fromstring(payload)
    title = root.findtext(f"{ATOM}title", "")
    if category not in title or start not in title or end not in title:
        raise ValueError("API feed does not match requested category and date range")
    total = int(root.findtext(f"{OPENSEARCH}totalResults", "0"))
    entries = []
    for entry in root.findall(f"{ATOM}entry"):
        source_url = entry.findtext(f"{ATOM}id", "")
        pdf_url = next((link.attrib["href"] for link in entry.findall(f"{ATOM}link")
                        if link.attrib.get("type") == "application/pdf"), None)
        if not source_url or not pdf_url:
            continue
        entries.append({
            "source_id": source_url.rstrip("/").split("/")[-1],
            "source_url": source_url.replace("http://", "https://"),
            "pdf_url": pdf_url, "title": " ".join(entry.findtext(f"{ATOM}title", "").split()),
            "published": entry.findtext(f"{ATOM}published", ""),
        })
    return entries, total, hashlib.sha256(payload).hexdigest()


def clean_pdf_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw).replace("\x0c", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.split(r"(?im)^\s*(?:references|bibliography)\s*$", text, maxsplit=1)[0]
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fetch_pdf(url: str, path: Path, timeout: int = 60) -> bytes:
    if path.exists():
        data = path.read_bytes()
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "CBF-TTT-research/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
        if not data.startswith(b"%PDF-"):
            raise ValueError("download is not a PDF")
        path.write_bytes(data)
    if not data.startswith(b"%PDF-"):
        raise ValueError("cached file is not a PDF")
    return data


def collect(entries: list[dict], target: int, min_tokens: int, tokenizer,
            pdf_dir: Path, delay_s: float) -> tuple[list[dict], dict]:
    if target < 1 or min_tokens < 1 or delay_s < 0:
        raise ValueError("target and min_tokens must be positive; delay must be nonnegative")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    accepted, source_meta = [], []
    rejected = Counter()
    seen_ids = set()
    for entry in entries:
        if len(accepted) >= target:
            break
        source_id = entry["source_id"]
        if source_id in seen_ids or not re.fullmatch(r"\d{4}\.\d{4,5}v\d+", source_id):
            rejected["duplicate_or_invalid_id"] += 1
            continue
        seen_ids.add(source_id)
        try:
            pdf = fetch_pdf(entry["pdf_url"], pdf_dir / f"{source_id}.pdf")
            if delay_s:
                time.sleep(delay_s)
            raw = subprocess.run(["pdftotext", "-enc", "UTF-8", "-nopgbrk",
                                  str(pdf_dir / f"{source_id}.pdf"), "-"],
                                 check=True, capture_output=True, text=True, timeout=60).stdout
            text = clean_pdf_text(raw)
            count = len(tokenizer.encode(text, add_special_tokens=False))
            if count < min_tokens:
                rejected["too_short"] += 1
                print(json.dumps({"source_id": source_id, "status": "too_short", "tokens": count}), flush=True)
                continue
        except (OSError, ValueError, subprocess.SubprocessError, UnicodeError) as exc:
            rejected["download_or_extract_error"] += 1
            print(json.dumps({"source_id": source_id, "status": "error",
                              "error_type": type(exc).__name__}), flush=True)
            continue
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        accepted.append({"content_split": text, "source_id": source_id,
                         "source_url": entry["source_url"], "published": entry["published"]})
        source_meta.append({**entry, "pdf_sha256": hashlib.sha256(pdf).hexdigest(),
                            "text_sha256": content_hash, "qwen_tokens": count})
        print(json.dumps({"source_id": source_id, "status": "accepted",
                          "tokens": count, "accepted": len(accepted)}), flush=True)
    return accepted, {"attempted": len(seen_ids), "rejected": dict(rejected),
                      "accepted": source_meta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--category", default="cs.CL")
    parser.add_argument("--start", default="202609010000")
    parser.add_argument("--end", default="202609262359")
    parser.add_argument("--target", type=int, default=48)
    parser.add_argument("--max-results", type=int, default=200)
    parser.add_argument("--min-tokens", type=int, default=4256)
    parser.add_argument("--delay-s", type=float, default=3.0)
    parser.add_argument("--selection-seed", type=int, default=118)
    parser.add_argument("--api-feed", type=Path,
                        help="saved official Atom feed when the GPU host cannot reach the API")
    parser.add_argument("--pdf-cache", type=Path,
                        help="reuse a PDF cache across repeat data builds")
    args = parser.parse_args()
    if args.max_results < args.target:
        parser.error("max-results must be at least target")
    from transformers import AutoTokenizer

    entries, total, feed_hash = query_entries(args.category, args.start, args.end,
                                              args.max_results, args.api_feed)
    random.Random(args.selection_seed).shuffle(entries)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    accepted, funnel = collect(entries, args.target, args.min_tokens, tokenizer,
                               args.pdf_cache or args.output.parent / "pdf_cache", args.delay_s)
    metadata = {"protocol": "postcutoff_arxiv_v1", "category": args.category,
                "start": args.start, "end": args.end, "total_api_results": total,
                "max_results": args.max_results, "target": args.target,
                "min_tokens": args.min_tokens, "tokenizer": args.tokenizer,
                "api_feed_sha256": feed_hash,
                "selection_seed": args.selection_seed,
                "selection_method": "shuffle_API_top_max_results_then_first_eligible",
                **funnel}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                    for row in accepted), encoding="utf-8")
    Path(str(args.output) + ".meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if len(accepted) < args.target:
        raise RuntimeError(f"only {len(accepted)}/{args.target} eligible documents; see metadata")
    print(json.dumps({"output": str(args.output), "accepted": len(accepted),
                      "attempted": funnel["attempted"], "rejected": funnel["rejected"]}))


if __name__ == "__main__":
    main()
