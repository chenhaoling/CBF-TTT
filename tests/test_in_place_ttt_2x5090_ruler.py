from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import scripts.audit_in_place_ttt_2x5090_ruler as audit_module
import scripts.audit_in_place_ttt_2x5090_ruler_resource as resource_audit_module
from scripts.audit_in_place_ttt_2x5090_ruler import audit
from scripts.audit_in_place_ttt_2x5090_ruler_matrix import audit as audit_matrix
from scripts.audit_in_place_ttt_2x5090_ruler_resource import audit as audit_resource
from scripts.audit_in_place_ttt_2x5090_ruler_resource_ladder import (
    BOUNDARIES,
    audit as audit_resource_ladder,
)
from scripts.build_in_place_ttt_paper_inference_overlay import (
    UPSTREAM_COMMIT,
    build_overlay,
)
from scripts.summarize_in_place_ttt_2x5090_ruler import summarize
from scripts.summarize_in_place_ttt_2x5090_ruler_matrix import (
    CONTEXTS,
    PAPER_TABLE1,
    summarize as summarize_matrix,
)
from scripts.summarize_in_place_ttt_level_b_ruler import (
    TASK_PREFIXES,
    evaluation_input_sha256,
)
from tests.test_in_place_ttt_level_b_ruler import make_raw_run, sha, write_json


def add_paper_inference_fixture(base: Path, root: Path, metadata: dict) -> Path:
    upstream = base / "paper_upstream"
    subprocess.run(
        ["git", "clone", "--no-hardlinks", "--quiet", str(Path.cwd()),
         str(upstream)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(upstream), "checkout", "--detach", "--quiet",
         UPSTREAM_COMMIT],
        check=True,
    )
    overlay = root / "paper_inference_overlay"
    build_overlay(upstream, overlay)
    manifest = overlay / "paper_inference_overlay_manifest.json"
    validation = root / "paper_inference_overlay_validation.json"
    write_json(validation, {
        "protocol": "in_place_ttt_paper_inference_overlay_validation_v1",
        "passed": True,
        "device": "cpu",
        "cuda_initialized": False,
        "clip_norm": 1e-5,
        "clipped_delta_frobenius_norm": 1.0000000656873453e-5,
        "unclipped_delta_frobenius_norm": 24.207931518554688,
        "zero_delta_unchanged": True,
        "config_default_disabled": True,
        "config_round_trip": True,
        "from_pretrained_override": True,
        "invalid_values_rejected": ["0.0", "-1.0", "inf", "nan"],
        "overlay_manifest_sha256": sha(manifest),
        "modeling_module_path": str(
            overlay / "inference_model/hf_qwen3/modeling_qwen3.py"
        ),
    })
    addendum = base / "paper_inference_addendum.md"
    addendum.write_text("paper inference clip addendum\n")
    metadata.update({
        "paper_inference_addendum_sha256": sha(addendum),
        "paper_inference_semantics": "per_update_delta_frobenius_clip",
        "ttt_update_clip_norm": 1e-5,
        "paper_inference_overlay_manifest_relative_path":
            "paper_inference_overlay/paper_inference_overlay_manifest.json",
        "paper_inference_overlay_manifest_sha256": sha(manifest),
        "paper_inference_overlay_validation_relative_path":
            "paper_inference_overlay_validation.json",
        "paper_inference_overlay_validation_sha256": sha(validation),
    })
    return addendum


class TwoRTX5090RulerSingleTest(unittest.TestCase):
    def test_summary_registers_100m_tier(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_raw_run(root)
            result = summarize(root)
            self.assertTrue(result["completed"])
            self.assertEqual(result["protocol"],
                             "in_place_ttt_2x5090_ruler_summary_v1")
            self.assertEqual(result["baseline_tier"], "2x5090_100m")

    def test_complete_single_context_fixture_and_model_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "run"
            root.mkdir()
            make_raw_run(root)

            corpus_plan = base / "corpus_plan.md"
            evaluation_plan = base / "ruler_plan.md"
            corpus_plan.write_text("corpus\n")
            evaluation_plan.write_text("ruler\n")

            training = base / "training"
            model = training / "no_ttt/model/hf_ckpt"
            model.mkdir(parents=True)
            tokenizer = base / "tokenizer"
            tokenizer.mkdir()
            write_json(model / "config.json", {
                "ttt_mode": False, "max_position_embeddings": 131072,
            })
            (model / "tokenizer.json").write_text("tokenizer\n")
            (model / "model.safetensors").write_bytes(b"weights")
            (tokenizer / "tokenizer.json").write_text("tokenizer\n")
            write_json(tokenizer / "tokenizer_config.json", {
                "model_max_length": 131072,
            })
            model_files = {
                str(path.relative_to(model)): {
                    "size_bytes": path.stat().st_size,
                    "sha256": sha(path),
                }
                for path in sorted(model.iterdir()) if path.is_file()
            }
            training_audit = training / "independent_audit.json"
            write_json(training_audit, {
                "protocol": "in_place_ttt_2x5090_100m_stage2_audit_v1",
                "passed": True,
                "arms": {"no_ttt": {
                    "final_hf_relative_path": "model/hf_ckpt",
                    "hf_manifest": model_files,
                }},
            })

            generator_modules = {
                f"opencompass.{prefix}.ruler_{task}{suffix}"
                for prefix, suffix in (
                    ("configs.datasets.ruler", "_gen"),
                    ("datasets.ruler", ""),
                )
                for task in ("qa", "niah", "vt", "fwe", "cwe")
            }
            generator_files = {}
            for index, module in enumerate(sorted(generator_modules)):
                path = base / "dependencies" / f"generator_{index}.py"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(module + "\n")
                generator_files[module] = {"path": str(path), "sha256": sha(path)}
            source_files = {}
            for relative in (
                "data/ruler/dev-v2.0.json",
                "data/ruler/hotpotqa.json",
                "data/ruler/PaulGrahamEssays.jsonl",
            ):
                path = base / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative + "\n")
                source_files[relative] = {"path": str(path), "sha256": sha(path)}

            metadata = json.loads((root / "run_metadata.json").read_text())
            metadata.update({
                "protocol": "in_place_ttt_2x5090_ruler_run_v1",
                "baseline_tier": "2x5090_100m",
                "run_kind": "formal",
                "hostname": "user-Super-Server",
                "upstream_commit": audit_module.UPSTREAM_COMMIT,
                "upstream_clean": True,
                "tool_commit": "a" * 40,
                "tool_clean": True,
                "public_corpus_plan_sha256": sha(corpus_plan),
                "evaluation_plan_sha256": sha(evaluation_plan),
                "training_audit_path": str(training_audit),
                "training_audit_sha256": sha(training_audit),
                "model_path": str(model),
                "model_config_sha256": sha(model / "config.json"),
                "model_files": model_files,
                "tokenizer_path": str(tokenizer),
                "tokenizer_json_sha256": sha(tokenizer / "tokenizer.json"),
                "tokenizer_config_sha256": sha(tokenizer / "tokenizer_config.json"),
                "physical_gpu": 0,
                "ruler_random_seed": 42,
                "context_suffix": "4k",
                "extrapolation": False,
                "tokenizer_policy": "qwen_original_tokenizer_mistral_fix_false_v1",
                "fix_mistral_regex": False,
                "config_sha256": sha(Path(
                    "eval_config/baseline_reproduction/qwen3_4b_2x5090_ruler.py"
                )),
                "tool_artifacts": {
                    relative: sha(Path(relative))
                    for relative in (
                        "eval_config/baseline_reproduction/qwen3_4b_2x5090_ruler.py",
                        "scripts/run_in_place_ttt_2x5090_ruler_single.sh",
                        "scripts/build_in_place_ttt_paper_inference_overlay.py",
                        "scripts/validate_in_place_ttt_paper_inference_overlay.py",
                        "scripts/audit_in_place_ttt_paper_inference_overlay.py",
                        "scripts/summarize_in_place_ttt_2x5090_ruler.py",
                        "scripts/audit_in_place_ttt_2x5090_ruler.py",
                    )
                },
                "dependency_versions": {
                    "opencompass": "0.5.4",
                    "transformers": "4.57.3",
                    "datasets": "2.21.0",
                    "torch": "2.8.0+cu128",
                },
                "ruler_generator_files": generator_files,
                "ruler_source_files": source_files,
            })
            paper_addendum = add_paper_inference_fixture(base, root, metadata)
            write_json(root / "run_metadata.json", metadata)
            (root / "gpu.csv").write_text(
                "timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu\n"
                "1,0,20000,32607,50\n"
            )
            write_json(root / "summary.json", summarize(root))
            (root / "gpu_before.csv").write_text(
                "0, NVIDIA GeForce RTX 5090, 32607, 46, 0\n"
            )
            (root / "preflight_host.txt").write_text(
                "MemTotal:       65011712 kB\nMemAvailable:   50000000 kB\n"
            )
            (root / "preflight_conflicts.txt").write_text("")

            expected_generators = {
                module: record["sha256"] for module, record in generator_files.items()
            }
            expected_sources = {
                relative: record["sha256"] for relative, record in source_files.items()
            }
            with mock.patch.object(
                    audit_module, "EXPECTED_GENERATOR_SHA256", expected_generators), \
                    mock.patch.object(
                        audit_module, "EXPECTED_SOURCE_SHA256", expected_sources):
                passed = audit(
                    root, corpus_plan, evaluation_plan, paper_addendum,
                )
                self.assertTrue(passed["passed"], passed["errors"])
                (model / "model.safetensors").write_bytes(b"changed")
                failed = audit(
                    root, corpus_plan, evaluation_plan, paper_addendum,
                )
            self.assertFalse(failed["passed"])
            self.assertTrue(any("model artifact differs" in item
                                for item in failed["errors"]))


class TwoRTX5090RulerResourceTest(unittest.TestCase):
    def test_complete_one_sample_resource_fixture_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "run"
            model = base / "training/no_ttt/final/hf_ckpt"
            tokenizer = base / "tokenizer"
            model.mkdir(parents=True)
            tokenizer.mkdir()
            write_json(model / "config.json", {
                "ttt_mode": False, "max_position_embeddings": 131072,
            })
            (model / "tokenizer.json").write_text("tokenizer\n")
            (model / "model.safetensors").write_bytes(b"weights")
            (tokenizer / "tokenizer.json").write_text("tokenizer\n")
            write_json(tokenizer / "tokenizer_config.json", {
                "model_max_length": 131072,
            })
            model_files = {
                str(path.relative_to(model)): {
                    "size_bytes": path.stat().st_size, "sha256": sha(path),
                }
                for path in model.iterdir() if path.is_file()
            }
            training_audit = base / "training/independent_audit.json"
            write_json(training_audit, {
                "protocol": "in_place_ttt_2x5090_100m_stage2_audit_v1",
                "passed": True,
                "arms": {"no_ttt": {
                    "final_hf_relative_path": "final/hf_ckpt",
                    "hf_manifest": model_files,
                }},
            })
            result_dir = root / "opencompass/date/results/model"
            prediction_dir = root / "opencompass/date/predictions/model"
            for prefix in TASK_PREFIXES:
                task = f"{prefix}_4k"
                write_json(result_dir / f"{task}.json", {
                    "score": 50.0,
                    "details": {"type": "GEN", "0": {"prediction": "x"}},
                })
                write_json(prediction_dir / f"{task}.json", {
                    "0": {
                        "origin_prompt": "prompt",
                        "references": "answer",
                        "prediction": "x",
                    },
                })
            corpus_plan = base / "corpus.md"
            evaluation_plan = base / "evaluation.md"
            corpus_plan.write_text("corpus\n")
            evaluation_plan.write_text("evaluation\n")

            generator_files = {}
            expected_generators = {}
            for index, name in enumerate(resource_audit_module.EXPECTED_GENERATOR_SHA256):
                path = base / "deps" / f"generator_{index}.py"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name + "\n")
                digest = sha(path)
                generator_files[name] = {"path": str(path), "sha256": digest}
                expected_generators[name] = digest
            source_files = {}
            expected_sources = {}
            for index, name in enumerate(resource_audit_module.EXPECTED_SOURCE_SHA256):
                path = base / "deps" / f"source_{index}.json"
                path.write_text(name + "\n")
                digest = sha(path)
                source_files[name] = {"path": str(path), "sha256": digest}
                expected_sources[name] = digest
            config = Path(
                "eval_config/baseline_reproduction/qwen3_4b_2x5090_ruler.py"
            )
            resource_auditor = Path(
                "scripts/audit_in_place_ttt_2x5090_ruler_resource.py"
            )
            single_runner = Path(
                "scripts/run_in_place_ttt_2x5090_ruler_single.sh"
            )
            metadata = {
                "protocol": "in_place_ttt_2x5090_ruler_run_v1",
                "baseline_tier": "2x5090_100m",
                "run_kind": "resource",
                "num_samples_per_task": 1,
                "hostname": "user-Super-Server",
                "physical_gpu": 0,
                "arm": "no_ttt",
                "context_length": 4096,
                "upstream_commit": resource_audit_module.UPSTREAM_COMMIT,
                "upstream_clean": True,
                "tool_commit": "a" * 40,
                "tool_clean": True,
                "public_corpus_plan_sha256": sha(corpus_plan),
                "evaluation_plan_sha256": sha(evaluation_plan),
                "config_sha256": sha(config),
                "tool_artifacts": {
                    str(config): sha(config),
                    str(single_runner): sha(single_runner),
                    "scripts/build_in_place_ttt_paper_inference_overlay.py": sha(Path(
                        "scripts/build_in_place_ttt_paper_inference_overlay.py"
                    )),
                    "scripts/validate_in_place_ttt_paper_inference_overlay.py": sha(Path(
                        "scripts/validate_in_place_ttt_paper_inference_overlay.py"
                    )),
                    "scripts/audit_in_place_ttt_paper_inference_overlay.py": sha(Path(
                        "scripts/audit_in_place_ttt_paper_inference_overlay.py"
                    )),
                    str(resource_auditor): sha(resource_auditor),
                },
                "ruler_generator_files": generator_files,
                "ruler_source_files": source_files,
                "training_audit_path": str(training_audit),
                "training_audit_sha256": sha(training_audit),
                "model_path": str(model),
                "model_config_sha256": sha(model / "config.json"),
                "model_files": model_files,
                "tokenizer_path": str(tokenizer),
                "tokenizer_json_sha256": sha(tokenizer / "tokenizer.json"),
                "tokenizer_config_sha256": sha(tokenizer / "tokenizer_config.json"),
            }
            paper_addendum = add_paper_inference_fixture(base, root, metadata)
            write_json(root / "run_metadata.json", metadata)
            (root / "eval.log").write_text(
                "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:10.00\n"
            )
            (root / "exit_code.txt").write_text("0\n")
            (root / "gpu.csv").write_text(
                "timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu\n"
                "1,0,20000,32607,80\n"
            )
            (root / "gpu_before.csv").write_text(
                "0, NVIDIA GeForce RTX 5090, 32607, 46, 0\n"
            )
            (root / "preflight_host.txt").write_text(
                "MemTotal: 65011712 kB\nMemAvailable: 50000000 kB\n"
            )
            (root / "preflight_conflicts.txt").write_text("")
            with mock.patch.object(
                    resource_audit_module, "EXPECTED_GENERATOR_SHA256",
                    expected_generators), mock.patch.object(
                        resource_audit_module, "EXPECTED_SOURCE_SHA256",
                        expected_sources):
                result = audit_resource(
                    root, corpus_plan, evaluation_plan, paper_addendum,
                )
            self.assertTrue(result["passed"], result["errors"])
            self.assertTrue(result["scores_discarded"])

    def test_resource_ladder_checks_paired_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for context, suffix in BOUNDARIES:
                for arm in ("no_ttt", "ttt"):
                    run = root / arm / suffix
                    prediction_dir = run / "opencompass/date/predictions/model"
                    payload = {
                        "0": {
                            "origin_prompt": "shared prompt",
                            "references": "shared answer",
                            "prediction": arm,
                        }
                    }
                    for prefix in TASK_PREFIXES:
                        write_json(prediction_dir / f"{prefix}_{suffix}.json", payload)
                    write_json(run / "run_metadata.json", {
                        "protocol": "in_place_ttt_2x5090_ruler_run_v1",
                        "run_kind": "resource",
                        "num_samples_per_task": 1,
                        "arm": arm,
                        "context_length": context,
                        "paper_inference_addendum_sha256": "pending",
                    })
                    write_json(run / "independent_resource_audit.json", {
                        "protocol": "in_place_ttt_2x5090_ruler_resource_audit_v1",
                        "passed": True,
                        "arm": arm,
                        "context_length": context,
                        "gpu": {
                            "peak_memory_used_mib": 20000,
                            "memory_total_mib": 32607,
                        },
                        "bindings": {
                            "paper_inference_addendum_sha256": "pending",
                        },
                    })
            training = root / "training_dependency"
            training.mkdir()
            training_audit = training / "independent_audit.json"
            write_json(training_audit, {
                "protocol": "in_place_ttt_2x5090_100m_stage2_audit_v1",
                "passed": True,
            })
            public_plan = root / "public.md"
            evaluation_plan = root / "evaluation.md"
            paper_addendum = root / "paper_inference.md"
            runner = root / "runner.sh"
            public_plan.write_text("public\n")
            evaluation_plan.write_text("evaluation\n")
            paper_addendum.write_text("paper inference\n")
            runner.write_text("runner\n")
            for run_metadata in root.glob("*/*/run_metadata.json"):
                payload = json.loads(run_metadata.read_text())
                payload["paper_inference_addendum_sha256"] = sha(paper_addendum)
                write_json(run_metadata, payload)
            for checked_path in root.glob("*/*/independent_resource_audit.json"):
                payload = json.loads(checked_path.read_text())
                payload["bindings"]["paper_inference_addendum_sha256"] = sha(
                    paper_addendum
                )
                write_json(checked_path, payload)
            single_runner = Path(
                "scripts/run_in_place_ttt_2x5090_ruler_single.sh"
            ).resolve()
            ladder_auditor = Path(
                "scripts/audit_in_place_ttt_2x5090_ruler_resource_ladder.py"
            ).resolve()
            write_json(root / "ladder_metadata.json", {
                "protocol": "in_place_ttt_2x5090_ruler_resource_ladder_run_v1",
                "hostname": "user-Super-Server",
                "tool_clean": True,
                "tool_commit": "a" * 40,
                "training_root": str(training.resolve()),
                "boundaries": [4096, 32768, 65536, 131072],
                "training_audit_sha256": sha(training_audit),
                "public_corpus_plan_sha256": sha(public_plan),
                "evaluation_plan_sha256": sha(evaluation_plan),
                "paper_inference_addendum_sha256": sha(paper_addendum),
                "runner_sha256": sha(runner),
                "single_runner_sha256": sha(single_runner),
                "auditor_sha256": sha(ladder_auditor),
            })
            kwargs = {
                "public_corpus_plan": public_plan,
                "evaluation_plan": evaluation_plan,
                "paper_inference_addendum": paper_addendum,
                "training_root": training,
                "runner": runner,
            }
            checked = audit_resource_ladder(root, **kwargs)
            self.assertTrue(checked["passed"], checked["errors"])
            changed = root / "ttt/64k/opencompass/date/predictions/model/ruler_vt_64k.json"
            payload = json.loads(changed.read_text())
            payload["0"]["origin_prompt"] = "different"
            write_json(changed, payload)
            failed = audit_resource_ladder(root, **kwargs)
            self.assertFalse(failed["passed"])
            self.assertIn("paired resource inputs differ: 64k", failed["errors"])


class TwoRTX5090RulerMatrixTest(unittest.TestCase):
    def _fixture(self, root: Path) -> None:
        for index, (context, suffix) in enumerate(CONTEXTS):
            for arm, offset in (("no_ttt", 0.0), ("ttt", 1.0)):
                run = root / arm / suffix
                score = 40.0 + index + offset
                result_dir = run / "opencompass/20261018/results/model"
                prediction_dir = run / "opencompass/20261018/predictions/model"
                task_scores = {}
                input_hashes = {}
                result_hashes = {}
                prediction_hashes = {}
                for prefix in TASK_PREFIXES:
                    task = f"{prefix}_{suffix}"
                    result = result_dir / f"{task}.json"
                    prediction = prediction_dir / f"{task}.json"
                    write_json(result, {"score": score})
                    payload = {
                        str(sample): {
                            "origin_prompt": f"prompt {sample}",
                            "references": f"answer {sample}",
                            "prediction": arm,
                        }
                        for sample in range(100)
                    }
                    write_json(prediction, payload)
                    task_scores[task] = score
                    input_hashes[task] = evaluation_input_sha256(payload)
                    result_hashes[result.name] = sha(result)
                    prediction_hashes[prediction.name] = sha(prediction)
                write_json(run / "run_metadata.json", {
                    "protocol": "in_place_ttt_2x5090_ruler_run_v1",
                    "baseline_tier": "2x5090_100m",
                    "run_kind": "formal",
                    "arm": arm,
                    "context_length": context,
                    "paper_inference_addendum_sha256": "pending",
                })
                write_json(run / "summary.json", {
                    "protocol": "in_place_ttt_2x5090_ruler_summary_v1",
                    "baseline_tier": "2x5090_100m",
                    "completed": True,
                    "arm": arm,
                    "context_length": context,
                    "macro_accuracy": score,
                    "task_scores": task_scores,
                    "evaluation_input_sha256": input_hashes,
                })
                write_json(run / "independent_audit.json", {
                    "protocol": "in_place_ttt_2x5090_ruler_audit_v1",
                    "passed": True,
                    "arm": arm,
                    "context_length": context,
                    "bindings": {
                        "paper_inference_addendum_sha256": "pending",
                        "raw_result_sha256": result_hashes,
                        "raw_prediction_sha256": prediction_hashes,
                    },
                })

    def test_six_context_matrix_and_independent_recompute(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            matrix = summarize_matrix(root)
            self.assertTrue(matrix["completed"], matrix["errors"])
            self.assertEqual(len(matrix["context_scores"]), 6)
            self.assertEqual(
                matrix["registered_4k_to_128k_mean"]["ttt_minus_no_ttt"], 1.0
            )
            self.assertEqual(matrix["paper_table1_reference"], PAPER_TABLE1)
            self.assertEqual(
                matrix["public_replication_minus_paper"]["4k"]
                ["public_gain_minus_paper_gain"],
                1.5,
            )
            self.assertEqual(
                matrix["claim_boundary"],
                "public_data_matched_replication_not_paper_exact",
            )
            write_json(root / "matrix_summary.json", matrix)
            training = root / "training_dependency"
            resource = root / "resource_dependency"
            training.mkdir()
            resource.mkdir()
            training_audit = training / "independent_audit.json"
            resource_audit = resource / "independent_resource_ladder_audit.json"
            write_json(training_audit, {
                "protocol": "in_place_ttt_2x5090_100m_stage2_audit_v1",
                "passed": True,
            })
            write_json(resource_audit, {
                "protocol": "in_place_ttt_2x5090_ruler_resource_ladder_audit_v1",
                "passed": True,
                "boundaries": {
                    f"{suffix}_{arm}": {}
                    for suffix in ("4k", "32k", "64k", "128k")
                    for arm in ("no_ttt", "ttt")
                },
                "bindings": {
                    "paper_inference_addendum_sha256": "pending",
                },
            })
            public_plan = root / "public.md"
            evaluation_plan = root / "evaluation.md"
            paper_addendum = root / "paper_inference.md"
            runner = root / "runner.sh"
            public_plan.write_text("public\n")
            evaluation_plan.write_text("evaluation\n")
            paper_addendum.write_text("paper inference\n")
            runner.write_text("runner\n")
            for run_metadata in root.glob("*/*/run_metadata.json"):
                payload = json.loads(run_metadata.read_text())
                payload["paper_inference_addendum_sha256"] = sha(paper_addendum)
                write_json(run_metadata, payload)
            for checked_path in root.glob("*/*/independent_audit.json"):
                payload = json.loads(checked_path.read_text())
                payload["bindings"]["paper_inference_addendum_sha256"] = sha(
                    paper_addendum
                )
                write_json(checked_path, payload)
            resource_payload = json.loads(resource_audit.read_text())
            resource_payload["bindings"][
                "paper_inference_addendum_sha256"
            ] = sha(paper_addendum)
            write_json(resource_audit, resource_payload)
            auditor_path = Path(
                "scripts/audit_in_place_ttt_2x5090_ruler_matrix.py"
            ).resolve()
            summarizer_path = Path(
                "scripts/summarize_in_place_ttt_2x5090_ruler_matrix.py"
            ).resolve()
            write_json(root / "matrix_metadata.json", {
                "protocol": "in_place_ttt_2x5090_ruler_matrix_run_v1",
                "hostname": "user-Super-Server",
                "tool_clean": True,
                "tool_commit": "a" * 40,
                "training_root": str(training.resolve()),
                "resource_root": str(resource.resolve()),
                "contexts": [context for context, _ in CONTEXTS],
                "training_audit_sha256": sha(training_audit),
                "resource_audit_sha256": sha(resource_audit),
                "public_corpus_plan_sha256": sha(public_plan),
                "evaluation_plan_sha256": sha(evaluation_plan),
                "paper_inference_addendum_sha256": sha(paper_addendum),
                "runner_sha256": sha(runner),
                "summarizer_sha256": sha(summarizer_path),
                "auditor_sha256": sha(auditor_path),
            })
            kwargs = {
                "public_corpus_plan": public_plan,
                "evaluation_plan": evaluation_plan,
                "paper_inference_addendum": paper_addendum,
                "training_root": training,
                "resource_root": resource,
                "runner": runner,
            }
            checked = audit_matrix(root, **kwargs)
            self.assertTrue(checked["passed"], checked["errors"])
            raw = next((root / "ttt/64k/opencompass").glob("*/results/*/*.json"))
            payload = json.loads(raw.read_text())
            payload["score"] += 5
            write_json(raw, payload)
            self.assertFalse(audit_matrix(root, **kwargs)["passed"])

    def test_paired_input_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._fixture(root)
            summary_path = root / "ttt/32k/summary.json"
            summary = json.loads(summary_path.read_text())
            task = next(iter(summary["evaluation_input_sha256"]))
            summary["evaluation_input_sha256"][task] = "0" * 64
            write_json(summary_path, summary)
            result = summarize_matrix(root)
            self.assertFalse(result["completed"])
            self.assertIn("paired RULER inputs differ: 32k", result["errors"])


if __name__ == "__main__":
    unittest.main()
