"""Safely fall back to the frozen streaming full-corpus auditor.

This watcher is intentionally specific to the active 2026-10-18 public-corpus
build. It never replaces an existing audit. The fallback is released only
after the original wrapper reports failure, the corpus was atomically
published, and no builder or original auditor remains alive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROTOCOL = "in_place_ttt_public_corpus_streaming_audit_watcher_v1"
CORPUS_PROTOCOL = "in_place_ttt_public_corpus_v1"
AUDIT_PROTOCOL = CORPUS_PROTOCOL + "_independent_audit"
EXPECTED_TOTALS = {
    "records": 724_800,
    "training_tokens": 34_999_369_728,
}
EXPECTED_HASHES = {
    "plan": "c5019a28b9fa7fa27052ddf1d69eb029a2ec5b3b6446f1527e9ceb7db1a12bab",
    "builder": "925ac99e701d9aa6444e8ec0c2f329cb8200d58155e672018d6ea3c4a044ec59",
    "original_auditor": "9c5dbadba76c4ebbba5248952455b3bfe4860b9b2b79bdbb555034dcc76c7e31",
    "streaming_auditor": "5ec753a29996f6abc386d7e16529218bbaaf6c1b1ff271638db730b9c41f8b87",
    "inventory": "c7e6761dc5d27013c09d6002de5c3e2b2185492b18100fba4b966c855debf7df",
    "inventory_audit": "32daacb7eee2c9536ab8c06c01ac7c423d429f04cdf512a958f9ffd201487507",
    "scalability_addendum": "c8641bbfa83f79a5838f13cf906ef0fdbd2053c080cbab987aa20f798423fdf1",
    "watcher_addendum": "14de0eeca54915d5a6c2d3daeda16679dbf8ecd055fb8b016d97e32193b40d3e",
}
TOKENIZER_ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "config.json",
)
TARGET_PROCESSES = {
    "build_in_place_ttt_public_corpus.py",
    "audit_in_place_ttt_public_corpus.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def tokenizer_assets(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in TOKENIZER_ASSETS:
        path = root / name
        if path.is_file():
            result[name] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    return result


def atomic_text(path: Path, value: str) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(value, encoding="utf-8")
    os.replace(partial, path)


def write_state(root: Path, state: str, **details: Any) -> None:
    payload = {
        "protocol": PROTOCOL,
        "state": state,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **details,
    }
    atomic_text(root / "state.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    atomic_text(root / "state.txt", state + "\n")


def git_value(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True
    ).strip()


def corpus_processes() -> list[dict[str, Any]]:
    rows = []
    for process in Path("/proc").glob("[0-9]*"):
        if int(process.name) == os.getpid():
            continue
        try:
            argv = [
                item.decode(errors="replace")
                for item in (process / "cmdline").read_bytes().split(b"\0")
                if item
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(Path(argument).name in TARGET_PROCESSES for argument in argv):
            rows.append({"pid": int(process.name), "argv": argv})
    return sorted(rows, key=lambda row: row["pid"])


def wrapper_state(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[0].strip() if lines else "EMPTY"


def validate_manifest(args: argparse.Namespace) -> tuple[list[str], dict[str, Any] | None]:
    manifest_path = args.corpus_root / "manifest.json"
    errors: list[str] = []
    if not manifest_path.is_file():
        return ["published manifest is missing"], None
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [f"cannot read published manifest: {error}"], None
    if manifest.get("protocol") != CORPUS_PROTOCOL or manifest.get("profile") != "full":
        errors.append("manifest protocol/profile differs from frozen full corpus")
    totals = manifest.get("totals", {})
    for name, expected in EXPECTED_TOTALS.items():
        if totals.get(name) != expected:
            errors.append(f"manifest total {name} differs from {expected}")
    bindings = manifest.get("bindings", {})
    expected_bindings = {
        "plan": sha256(args.plan),
        "builder": sha256(args.builder),
        "inventory": sha256(args.inventory),
        "inventory_audit": sha256(args.inventory_audit),
    }
    for name, expected in expected_bindings.items():
        if bindings.get(name, {}).get("sha256") != expected:
            errors.append(f"manifest {name} binding differs from frozen input")
    if bindings.get("tokenizer_path") != str(args.tokenizer.resolve()):
        errors.append("manifest tokenizer path differs from frozen tokenizer")
    observed_assets = tokenizer_assets(args.tokenizer)
    if bindings.get("tokenizer_assets") != observed_assets:
        errors.append("manifest tokenizer assets differ from current frozen assets")
    expected_artifacts = (
        "stage1_32k.jsonl",
        "stage1_32k.provenance.jsonl",
        "stage2_128k.jsonl",
        "stage2_128k.provenance.jsonl",
    )
    for name in expected_artifacts:
        path = args.corpus_root / name
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"published corpus artifact is missing or empty: {name}")
    if args.corpus_root.with_name(args.corpus_root.name + ".partial").exists():
        errors.append("partial corpus directory still exists after publication")
    return errors, manifest


def validate_audit(args: argparse.Namespace) -> list[str]:
    audit_path = args.corpus_root / "independent_audit.json"
    manifest_path = args.corpus_root / "manifest.json"
    if not manifest_path.is_file():
        return ["published manifest is missing"]
    if not audit_path.is_file():
        return ["canonical independent audit is missing"]
    try:
        audit = read_json(audit_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [f"cannot read canonical independent audit: {error}"]
    errors = []
    if audit.get("protocol") != AUDIT_PROTOCOL or audit.get("profile") != "full":
        errors.append("audit protocol/profile differs from frozen full-corpus audit")
    if audit.get("passed") is not True:
        errors.append("canonical independent audit did not pass")
    checks = audit.get("checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        errors.append("canonical independent audit checks are incomplete or failed")
    if audit.get("bindings", {}).get("manifest_sha256") != sha256(manifest_path):
        errors.append("canonical independent audit binds another manifest")
    auditor_hash = audit.get("bindings", {}).get("auditor_sha256")
    if auditor_hash not in {
        EXPECTED_HASHES["original_auditor"],
        EXPECTED_HASHES["streaming_auditor"],
    }:
        errors.append("canonical independent audit used an unregistered auditor")
    return errors


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    details: dict[str, Any]


def decide(args: argparse.Namespace, processes: list[dict[str, Any]] | None = None) -> Decision:
    audit_path = args.corpus_root / "independent_audit.json"
    status = wrapper_state(args.original_status)
    active = corpus_processes() if processes is None else processes
    if audit_path.exists():
        errors = validate_audit(args)
        if errors:
            return Decision("refuse", "existing canonical audit is invalid", {"errors": errors})
        return Decision("accept_existing", "canonical audit already passed", {})
    manifest_path = args.corpus_root / "manifest.json"
    if not manifest_path.is_file():
        if status == "FAILED":
            return Decision("refuse", "original wrapper failed before atomic corpus publication", {})
        return Decision("wait", "full corpus is not atomically published", {"wrapper_state": status})
    manifest_errors, _ = validate_manifest(args)
    if manifest_errors:
        return Decision("refuse", "published manifest failed fallback preflight", {"errors": manifest_errors})
    if active:
        return Decision("wait", "original builder or auditor is still active", {"processes": active})
    if status == "FAILED":
        return Decision("run_fallback", "original wrapper failed after atomic publication without an audit", {})
    if status == "COMPLETED":
        return Decision("refuse", "original wrapper completed without a canonical audit", {})
    return Decision("wait", "waiting for the original audit to reach a terminal state", {"wrapper_state": status})


def validate_static(args: argparse.Namespace) -> dict[str, Any]:
    if socket.gethostname() != args.expected_hostname:
        raise RuntimeError(f"watcher is restricted to {args.expected_hostname}")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse watcher output: {args.output_root}")
    paths = {
        "plan": args.plan,
        "builder": args.builder,
        "original_auditor": args.original_auditor,
        "streaming_auditor": args.streaming_auditor,
        "inventory": args.inventory,
        "inventory_audit": args.inventory_audit,
        "scalability_addendum": args.scalability_addendum,
        "watcher_addendum": args.watcher_addendum,
    }
    observed = {name: sha256(path) for name, path in paths.items()}
    for name, expected in EXPECTED_HASHES.items():
        if observed.get(name) != expected:
            raise RuntimeError(f"{name} SHA256 differs from frozen fallback input")
    if git_value(args.tool_repo, "rev-parse", "HEAD") != args.tool_commit:
        raise RuntimeError("tool repository is not at the frozen commit")
    if git_value(args.tool_repo, "status", "--short"):
        raise RuntimeError("tool repository is not clean")
    if args.streaming_auditor.resolve() != (
        args.tool_repo / "scripts/audit_in_place_ttt_public_corpus.py"
    ).resolve():
        raise RuntimeError("streaming auditor is outside the frozen tool repository")
    if not args.python_bin.is_file() or not os.access(args.python_bin, os.X_OK):
        raise RuntimeError("verification Python is missing or not executable")
    assets = tokenizer_assets(args.tokenizer)
    for required in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if required not in assets:
            raise RuntimeError(f"tokenizer asset is missing: {required}")
    return {
        "protocol": PROTOCOL,
        "expected_hostname": args.expected_hostname,
        "tool_commit": args.tool_commit,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": observed[name]}
            for name, path in paths.items()
        },
        "tokenizer_path": str(args.tokenizer.resolve()),
        "tokenizer_assets": assets,
    }


def run_fallback(args: argparse.Namespace) -> int:
    audit_path = args.corpus_root / "independent_audit.json"
    if audit_path.exists() or audit_path.with_name(audit_path.name + ".partial").exists():
        raise FileExistsError("refusing to overwrite canonical audit or its partial file")
    command = [
        "/usr/bin/time", "-v", str(args.python_bin), str(args.streaming_auditor),
        "--root", str(args.corpus_root),
        "--inventory", str(args.inventory),
        "--inventory-audit", str(args.inventory_audit),
        "--tokenizer", str(args.tokenizer),
        "--output", str(audit_path),
        "--plan", str(args.plan),
        "--builder", str(args.builder),
    ]
    (args.output_root / "fallback_command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_root / "fallback.stdout.log").open("x", encoding="utf-8") as stdout, \
            (args.output_root / "fallback.time_and_stderr.log").open("x", encoding="utf-8") as stderr:
        return subprocess.run(command, stdout=stdout, stderr=stderr, check=False).returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-repo", type=Path, required=True)
    parser.add_argument("--tool-commit", required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-audit", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--original-auditor", type=Path, required=True)
    parser.add_argument("--streaming-auditor", type=Path, required=True)
    parser.add_argument("--scalability-addendum", type=Path, required=True)
    parser.add_argument("--watcher-addendum", type=Path, required=True)
    parser.add_argument("--original-status", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-hostname", default="user-Super-Server")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-wait-seconds", type=int, default=604800)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for name in vars(args):
        value = getattr(args, name)
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    if args.poll_seconds < 10 or args.max_wait_seconds < args.poll_seconds:
        raise ValueError("invalid polling interval or wait budget")
    static = validate_static(args)
    args.output_root.mkdir(parents=True)
    (args.output_root / "frozen_inputs.json").write_text(
        json.dumps(static, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    started = time.monotonic()
    while True:
        decision = decide(args)
        write_state(args.output_root, decision.action, reason=decision.reason, **decision.details)
        if decision.action == "accept_existing":
            write_state(args.output_root, "completed_existing_audit", reason=decision.reason)
            return 0
        if decision.action == "refuse":
            write_state(args.output_root, "refused_fallback", reason=decision.reason, **decision.details)
            return 7
        if decision.action == "run_fallback":
            time.sleep(5)
            confirmation = decide(args)
            if confirmation.action != "run_fallback":
                write_state(
                    args.output_root,
                    "fallback_preflight_changed",
                    first=decision.__dict__,
                    second=confirmation.__dict__,
                )
                return 8
            break
        if time.monotonic() - started >= args.max_wait_seconds:
            write_state(args.output_root, "timed_out_waiting_for_original_audit")
            return 9
        time.sleep(args.poll_seconds)
    manifest_errors, manifest = validate_manifest(args)
    if manifest_errors:
        write_state(args.output_root, "fallback_manifest_recheck_failed", errors=manifest_errors)
        return 10
    (args.output_root / "fallback_preflight.json").write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "manifest_sha256": sha256(args.corpus_root / "manifest.json"),
                "manifest_totals": manifest.get("totals", {}) if manifest else None,
                "original_wrapper_state": wrapper_state(args.original_status),
                "active_corpus_processes": corpus_processes(),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    write_state(args.output_root, "running_streaming_fallback_audit")
    status = run_fallback(args)
    (args.output_root / "fallback_exit_code.txt").write_text(f"{status}\n", encoding="utf-8")
    if status != 0:
        write_state(args.output_root, "failed_streaming_fallback_audit", exit_code=status)
        return status or 11
    errors = validate_audit(args)
    if errors:
        write_state(args.output_root, "invalid_streaming_fallback_audit", errors=errors)
        return 12
    write_state(
        args.output_root,
        "completed_streaming_fallback_audit",
        audit_path=str(args.corpus_root / "independent_audit.json"),
        audit_sha256=sha256(args.corpus_root / "independent_audit.json"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
