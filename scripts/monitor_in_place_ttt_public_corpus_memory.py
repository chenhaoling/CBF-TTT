#!/usr/bin/env python3
"""Read-only memory monitor for the active full public-corpus builder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_public_corpus_memory_monitor_v1"
BUILDER_NAME = "build_in_place_ttt_public_corpus.py"
PARTIAL_FILES = (
    "stage1_32k.jsonl",
    "stage1_32k.provenance.jsonl",
    "stage2_128k.jsonl",
    "stage2_128k.provenance.jsonl",
)
STATUS_FIELDS = (
    "VmPeak",
    "VmSize",
    "VmHWM",
    "VmRSS",
    "RssAnon",
    "RssFile",
    "VmData",
    "VmSwap",
    "Threads",
)
CSV_FIELDS = (
    "time_unix",
    "builder_pid",
    "vm_peak_kib",
    "vm_size_kib",
    "vm_hwm_kib",
    "vm_rss_kib",
    "rss_anon_kib",
    "rss_file_kib",
    "vm_data_kib",
    "vm_swap_kib",
    "threads",
    "host_mem_total_kib",
    "host_mem_available_kib",
    "host_swap_total_kib",
    "host_swap_used_kib",
    "disk_total_bytes",
    "disk_free_bytes",
    "stage1_data_bytes",
    "stage1_provenance_bytes",
    "stage2_data_bytes",
    "stage2_provenance_bytes",
    "wrapper_status",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(value, encoding="utf-8")
    os.replace(partial, path)


def read_key_values(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, remainder = line.split(":", 1)
        parts = remainder.split()
        if parts and parts[0].isdigit():
            values[key] = int(parts[0])
    return values


def read_cmdline(path: Path) -> list[str]:
    return [
        value.decode(errors="replace")
        for value in path.read_bytes().split(b"\0")
        if value
    ]


def matching_builder_pids(proc_root: Path, output_root: Path) -> list[int]:
    expected = str(output_root.resolve())
    matches: list[int] = []
    for process in proc_root.glob("[0-9]*"):
        try:
            argv = read_cmdline(process / "cmdline")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not any(Path(value).name == BUILDER_NAME for value in argv):
            continue
        for index, value in enumerate(argv[:-1]):
            if value == "--output-root" and str(Path(argv[index + 1]).resolve()) == expected:
                matches.append(int(process.name))
                break
    return sorted(matches)


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def wrapper_status(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return "MISSING"
    return lines[0].strip() if lines else "EMPTY"


def collect_sample(args: argparse.Namespace, pid: int) -> dict[str, Any]:
    process = read_key_values(args.proc_root / str(pid) / "status")
    host = read_key_values(args.meminfo)
    disk = shutil.disk_usage(args.partial_root.parent)
    sizes = {name: file_size(args.partial_root / name) for name in PARTIAL_FILES}
    return {
        "time_unix": f"{time.time():.6f}",
        "builder_pid": pid,
        "vm_peak_kib": process.get("VmPeak", 0),
        "vm_size_kib": process.get("VmSize", 0),
        "vm_hwm_kib": process.get("VmHWM", 0),
        "vm_rss_kib": process.get("VmRSS", 0),
        "rss_anon_kib": process.get("RssAnon", 0),
        "rss_file_kib": process.get("RssFile", 0),
        "vm_data_kib": process.get("VmData", 0),
        "vm_swap_kib": process.get("VmSwap", 0),
        "threads": process.get("Threads", 0),
        "host_mem_total_kib": host.get("MemTotal", 0),
        "host_mem_available_kib": host.get("MemAvailable", 0),
        "host_swap_total_kib": host.get("SwapTotal", 0),
        "host_swap_used_kib": host.get("SwapTotal", 0) - host.get("SwapFree", 0),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "stage1_data_bytes": sizes["stage1_32k.jsonl"],
        "stage1_provenance_bytes": sizes["stage1_32k.provenance.jsonl"],
        "stage2_data_bytes": sizes["stage2_128k.jsonl"],
        "stage2_provenance_bytes": sizes["stage2_128k.provenance.jsonl"],
        "wrapper_status": wrapper_status(args.wrapper_status),
    }


def write_state(root: Path, state: str, **details: Any) -> None:
    payload = {
        "protocol": PROTOCOL,
        "state": state,
        "time_unix": time.time(),
        **details,
    }
    atomic_text(root / "state.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    atomic_text(root / "state.txt", state + "\n")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    if socket.gethostname() != args.expected_hostname:
        raise RuntimeError("memory monitor is restricted to hku-gpu2")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise RuntimeError("memory monitor must hide CUDA")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse {args.output_root}")
    if args.poll_seconds < 60 or args.max_wait_seconds < args.poll_seconds:
        raise ValueError("invalid polling interval or wait budget")
    if subprocess.check_output(
        ["git", "-C", str(args.tool_root), "rev-parse", "HEAD"], text=True
    ).strip() != args.tool_commit:
        raise RuntimeError("tool commit differs")
    if subprocess.check_output(
        ["git", "-C", str(args.tool_root), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("tool checkout is not clean")
    for path in (args.plan, args.builder, args.wrapper_status, args.partial_root):
        if not path.exists():
            raise FileNotFoundError(path)
    return {
        "protocol": PROTOCOL,
        "hostname": socket.gethostname(),
        "tool_commit": args.tool_commit,
        "monitor_path": str(Path(__file__).resolve()),
        "monitor_sha256": sha256(Path(__file__).resolve()),
        "plan_path": str(args.plan.resolve()),
        "plan_sha256": sha256(args.plan),
        "builder_path": str(args.builder.resolve()),
        "builder_sha256": sha256(args.builder),
        "partial_root": str(args.partial_root.resolve()),
        "wrapper_status_path": str(args.wrapper_status.resolve()),
        "poll_seconds": args.poll_seconds,
        "max_wait_seconds": args.max_wait_seconds,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def run(args: argparse.Namespace) -> int:
    identity = validate(args)
    args.output_root.mkdir(parents=True)
    atomic_text(
        args.output_root / "identity.json",
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
    )
    csv_path = args.output_root / "samples.csv"
    started = time.monotonic()
    samples = 0
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        handle.flush()
        os.fsync(handle.fileno())
        while True:
            pids = matching_builder_pids(args.proc_root, args.corpus_root)
            status = wrapper_status(args.wrapper_status)
            if len(pids) > 1:
                write_state(args.output_root, "FAILED_MULTIPLE_BUILDERS", pids=pids, samples=samples)
                return 2
            if not pids:
                write_state(
                    args.output_root,
                    "BUILDER_EXITED",
                    wrapper_status=status,
                    samples=samples,
                    elapsed_seconds=time.monotonic() - started,
                )
                return 0
            row = collect_sample(args, pids[0])
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
            samples += 1
            write_state(
                args.output_root,
                "MONITORING",
                builder_pid=pids[0],
                wrapper_status=status,
                samples=samples,
                latest=row,
            )
            elapsed = time.monotonic() - started
            if elapsed >= args.max_wait_seconds:
                write_state(
                    args.output_root,
                    "MONITOR_TIMEOUT",
                    builder_pid=pids[0],
                    wrapper_status=status,
                    samples=samples,
                    elapsed_seconds=elapsed,
                )
                return 0
            time.sleep(args.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-root", type=Path, required=True)
    parser.add_argument("--tool-commit", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--partial-root", type=Path, required=True)
    parser.add_argument("--wrapper-status", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-hostname", default="user-Super-Server")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-wait-seconds", type=int, default=172800)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--meminfo", type=Path, default=Path("/proc/meminfo"))
    return parser.parse_args()


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
