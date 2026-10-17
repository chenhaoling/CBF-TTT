# In-Place TTT baseline tokenizer audit addendum

Date: 2026-10-17<br>
Parent plan: `IN_PLACE_TTT_BASELINE_REPRODUCTION_PLAN.md`<br>
Scope: tokenizer-equivalence audit and replacement smoke only

## 1. Motivation

The converted no-TTT and native-TTT one-step checkpoints each completed the
13-task RULER-4K smoke, but every log emitted 27 copies of the Transformers
`incorrect regex pattern` warning. No 100-example RULER score may be released
until the warning is explained and the evaluation path is made deterministic.

Read-only diagnosis established the following facts:

- the original Qwen3-4B and both converted checkpoints have byte-identical
  `tokenizer.json` files with SHA256
  `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`;
- the original Qwen config records `model_type=qwen3` and
  `transformers_version=4.51.0`;
- the official `save_pretrained()` conversion records the same model type but
  rewrites `transformers_version=4.57.3`;
- Transformers 4.57.3 only skips its Mistral-regex patch for a local non-Mistral
  config whose recorded version is at most 4.57.2. It therefore misclassifies
  the converted local Qwen config;
- explicitly setting `fix_mistral_regex=True` changes the converted
  checkpoint's pre-tokenizer state from SHA256
  `b49305328826a107166f6254ba88e17fd2433b260db352cd9e1932b8e8bc1815`
  to `86b1e8a1b128b5d10a05ff691121b7bf616d4a56a9698012e661508c98c4ce03`.
  The same flag leaves the original Qwen tokenizer unchanged.

The warning is therefore a Transformers 4.57.3 local-config detection false
positive for this converted Qwen checkpoint. Applying the Mistral correction
would change the tokenizer and is prohibited.

## 2. Frozen remediation

The replacement smoke must meet all of the following rules:

1. Model weights and `config.json` are loaded from the converted checkpoint.
2. Tokenization for both the OpenCompass model wrapper and all RULER dataset
   generators uses the unmodified original Qwen3-4B tokenizer directory.
3. Before launch, the runner verifies that the original and converted
   `tokenizer.json` SHA256 values are identical.
4. The model-wrapper load explicitly passes `fix_mistral_regex=False`. This
   suppresses the false-positive patch while preserving the original Qwen
   pre-tokenizer. The RULER dataset loaders receive the original tokenizer
   path, whose Qwen config already bypasses the patch.
5. Run metadata binds the tokenizer path, `tokenizer.json` SHA256,
   `tokenizer_config.json` SHA256, and the policy name
   `qwen_original_tokenizer_mistral_fix_false_v1`.
6. The old smoke output is immutable. Replacement outputs use new directories
   and a new state file.

No upstream source, Qwen backbone, native TTT writer, converted checkpoint, or
OpenCompass installation may be edited.

## 3. Replacement smoke

Run no-TTT and native-TTT converted checkpoints sequentially with:

- the same 13 official RULER-4K tasks;
- one deterministic example per task;
- context length 4,096 and batch size 1;
- the same generation and postprocessing settings as the completed smoke;
- the original Qwen3-4B tokenizer for both arms.

For each arm, require:

- process exit code 0;
- exactly 13 expected task scores and no unexpected task;
- equality of the official group score and independently recomputed macro;
- zero `incorrect regex pattern` warnings;
- metadata hashes that bind both model and tokenizer inputs.

The two arms must bind the same tokenizer hashes. The one-example results are
execution checks only and cannot support an efficacy claim.

## 4. Stop conditions

Stop and do not open formal 100-example RULER evaluation if any condition
below occurs:

- original and converted `tokenizer.json` hashes differ;
- any regex warning remains;
- either arm fails to load or completes fewer than 13 tasks;
- score extraction disagrees with the official summary;
- either arm uses a tokenizer hash different from the other;
- any proposed workaround requires `fix_mistral_regex=True` or modifies the
  original/converted tokenizer files.

If the replacement smoke passes, the tokenizer warning is closed as an
evaluation-plumbing issue. Formal public-data matched training still requires
an exclusive larger-GPU host under the parent plan.
