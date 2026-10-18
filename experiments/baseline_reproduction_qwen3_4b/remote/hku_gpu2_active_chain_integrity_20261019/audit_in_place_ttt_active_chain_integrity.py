from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


PROTOCOL = "in_place_ttt_active_chain_integrity_v1"
EXPECTED_HOSTNAME = "user-Super-Server"
UPSTREAM_COMMIT = "be2324829b0e91c8fd10a74d4b43714fde6676e1"

CHECKOUTS = {
    "upstream": ("/home/ctj/in_place_ttt_official_be232482_resource_clean", UPSTREAM_COMMIT),
    "activation": ("/home/ctj/cbf_ttt_tool_f14cb715", "f14cb7152bc7607e0b8555d449bcf2b28853eb1c"),
    "stage1_ladder": ("/home/ctj/cbf_ttt_tool_bae87e6c", "bae87e6c75183a73f5ed2cd8150e0cc228ed19f0"),
    "stage1_timing": ("/home/ctj/cbf_ttt_tool_56094b89", "56094b8906ad670ec2be0d21ab75d698216082d4"),
    "stage1_formal": ("/home/ctj/cbf_ttt_tool_3c7e29fc", "3c7e29fc33cbb781090db6792e4fef42dec6dbcf"),
    "stage2_ladder": ("/home/ctj/cbf_ttt_tool_1b04d8cf", "1b04d8cf55821056e1dfb3a724d3ded1a87ebffc"),
    "stage2_timing": ("/home/ctj/cbf_ttt_tool_fbc02a6e", "fbc02a6e6ad88834b2bb2533906264ddd670d2f3"),
    "stage2_formal": ("/home/ctj/cbf_ttt_tool_73950ecd", "73950ecd1d0a8c2163863dce48de272e5db7d8d8"),
    "ruler_primary": ("/home/ctj/cbf_ttt_tool_439fcea1", "439fcea142c4d3bb51413d75480d5231d0c62cab"),
    "ruler_256k_resource": ("/home/ctj/cbf_ttt_tool_b33b93ac", "b33b93ace74240b2bb2332368647c4adda31b7dc"),
    "ruler_256k_formal": ("/home/ctj/cbf_ttt_tool_a0375e93", "a0375e93a530591ca3e36d6d4c8498c4772d8784"),
    "corpus_audit_fallback": ("/home/ctj/cbf_ttt_tool_990da145", "990da145d814aded46a560f64947bc17e7e89fb6"),
}

WATCHERS = {
    "activation": (1408519, "/home/ctj/cbf_ttt_tool_f14cb715/scripts/watch_and_run_in_place_ttt_2x5090_activation.sh"),
    "stage1_ladder": (1411193, "/home/ctj/cbf_ttt_tool_bae87e6c/scripts/watch_and_run_in_place_ttt_2x5090_stage1_ladder.sh"),
    "stage1_timing": (1412088, "/home/ctj/cbf_ttt_tool_56094b89/scripts/watch_and_run_in_place_ttt_2x5090_stage1_timing.sh"),
    "stage1_formal": (1422180, "/home/ctj/cbf_ttt_tool_3c7e29fc/scripts/watch_and_run_in_place_ttt_2x5090_100m_stage1.sh"),
    "stage2_ladder": (1424042, "/home/ctj/cbf_ttt_tool_1b04d8cf/scripts/watch_and_run_in_place_ttt_2x5090_stage2_ladder.sh"),
    "stage2_timing": (1425561, "/home/ctj/cbf_ttt_tool_fbc02a6e/scripts/watch_and_run_in_place_ttt_2x5090_stage2_timing.sh"),
    "stage2_formal": (1426725, "/home/ctj/cbf_ttt_tool_73950ecd/scripts/watch_and_run_in_place_ttt_2x5090_100m_stage2.sh"),
    "ruler_resource": (1430098, "/home/ctj/cbf_ttt_tool_439fcea1/scripts/watch_and_run_in_place_ttt_2x5090_ruler_resource_ladder.sh"),
    "ruler_matrix": (1430137, "/home/ctj/cbf_ttt_tool_439fcea1/scripts/watch_and_run_in_place_ttt_2x5090_ruler_matrix.sh"),
    "ruler_256k_resource": (1431971, "/home/ctj/cbf_ttt_tool_b33b93ac/scripts/watch_and_run_in_place_ttt_2x5090_ruler_256k_resource.sh"),
    "ruler_256k_formal": (1435576, "/home/ctj/cbf_ttt_tool_a0375e93/scripts/watch_and_run_in_place_ttt_2x5090_ruler_256k.sh"),
    "corpus_audit_fallback": (1605091, "/home/ctj/cbf_ttt_tool_990da145/scripts/watch_and_run_in_place_ttt_public_corpus_streaming_audit.py"),
    "stage2_sparse90": (1666185, "/home/ctj/in_place_ttt_public_corpus_full_run_20261018/audit_stage2_sparse_90.py"),
}

RELEASES = {
    "stage1": "/home/ctj/in_place_ttt_2x5090_100m_stage1_release_3c7e29fc_20261018.json",
    "stage2": "/home/ctj/in_place_ttt_2x5090_100m_stage2_release_73950ecd_20261018.json",
    "ruler_256k": "/home/ctj/in_place_ttt_2x5090_ruler_256k_formal_release_a0375e93_20261018.json",
}


def run(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    output = Path(os.environ["OUTPUT"])
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise FileExistsError(output)

    checks: dict[str, bool] = {"hostname_exact": platform.node() == EXPECTED_HOSTNAME}
    checkout_rows = {}
    for name, (raw_root, expected_commit) in CHECKOUTS.items():
        root = Path(raw_root)
        commit = run("git", "-C", str(root), "rev-parse", "HEAD")
        status = run("git", "-C", str(root), "status", "--porcelain")
        checkout_rows[name] = {
            "root": str(root),
            "expected_commit": expected_commit,
            "actual_commit": commit,
            "clean": status == "",
        }
        checks[f"checkout_{name}_commit_exact"] = commit == expected_commit
        checks[f"checkout_{name}_clean"] = status == ""

    watcher_rows = {}
    for name, (pid, raw_script) in WATCHERS.items():
        script = Path(raw_script)
        proc = Path(f"/proc/{pid}")
        live = proc.is_dir()
        argv = []
        if live:
            argv = [item.decode(errors="replace") for item in
                    (proc / "cmdline").read_bytes().split(b"\0") if item]
        watcher_rows[name] = {
            "pid": pid,
            "live": live,
            "script": str(script),
            "script_sha256": sha256(script),
            "script_in_argv": str(script) in argv,
        }
        checks[f"watcher_{name}_live"] = live
        checks[f"watcher_{name}_script_exact"] = live and str(script) in argv

    release_rows = {}
    for name, raw_path in RELEASES.items():
        path = Path(raw_path)
        release_rows[name] = {"path": str(path), "exists": path.exists()}
        checks[f"release_{name}_sealed_absent"] = not path.exists()

    gpu_lines = run(
        "nvidia-smi", "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits"
    ).splitlines()
    gpu_rows = []
    for line in gpu_lines:
        index, name, total = [part.strip() for part in line.split(",")]
        gpu_rows.append({"index": int(index), "name": name, "memory_total_mib": int(total)})
    checks["two_expected_5090_devices"] = (
        len(gpu_rows) == 2
        and [row["index"] for row in gpu_rows] == [0, 1]
        and all("RTX 5090" in row["name"] and row["memory_total_mib"] >= 32_000
                for row in gpu_rows)
    )

    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": platform.node(),
        "checkouts": checkout_rows,
        "watchers": watcher_rows,
        "sealed_releases": release_rows,
        "gpus": gpu_rows,
        "checks": checks,
        "passed": all(checks.values()),
        "script_sha256": sha256(Path(__file__)),
    }
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, output)


if __name__ == "__main__":
    main()
