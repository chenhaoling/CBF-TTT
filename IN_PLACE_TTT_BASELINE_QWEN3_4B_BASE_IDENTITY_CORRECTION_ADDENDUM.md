# In-Place TTT Qwen3-4B-Base identity correction addendum

Status: **frozen before any corrected Base-model activation or training run**

Freeze date: 2026-10-19 (Asia/Shanghai)

## 1. Error and claim impact

The parent reproduction plan explicitly registers the paper's
`Qwen3-4B-Base` checkpoint. The checkpoint staged at
`/home/ctj/models/Qwen3-4B` was subsequently identified as the different
official `Qwen/Qwen3-4B` post-trained checkpoint:

- local `config.json` SHA256 is
  `8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a`;
- it has `max_position_embeddings=40960`, `eos_token_id=151645`, a chat
  template, and a sampling `generation_config.json`;
- these fields and file sizes match the official `Qwen/Qwen3-4B` repository;
- the paper and its Table 1 experiment use `Qwen3-4B-Base`, whose official
  config has `max_position_embeddings=32768` and `eos_token_id=151643`.

The old checkpoint therefore cannot support the intended paper-baseline
claim. Existing engineering smoke results remain labelled as engineering
evidence only. No activation-offload GPU child, formal 100M training run or
formal RULER matrix has started, so no formal score is being reinterpreted.

Primary references:

- paper: <https://openreview.net/pdf?id=dTWfCLSoyl>;
- required checkpoint: <https://huggingface.co/Qwen/Qwen3-4B-Base>;
- mistakenly staged checkpoint: <https://huggingface.co/Qwen/Qwen3-4B>.

## 2. Immediate containment

Terminate the old activation watcher before it can launch. Its downstream
watchers must either exit on the missing prerequisite or be terminated without
starting GPU work. Preserve the corpus builder, its sparse provenance watcher,
and the independent corpus-audit fallback while tokenizer compatibility is
assessed. Do not modify or overwrite any old checkpoint, corpus, result root or
release file.

All old activation, Stage-1, Stage-2 and RULER roots are superseded for the
paper-baseline claim. They remain immutable evidence of the discovered setup.

## 3. Correct model acquisition and identity gate

Download `Qwen/Qwen3-4B-Base` at an explicit Hugging Face commit into a new
directory, `/home/ctj/models/Qwen3-4B-Base`. Record the repository ID,
revision, complete file manifest and SHA256 values. The identity audit must
require at least:

- architecture `Qwen3ForCausalLM` and 36 decoder layers;
- hidden size 2560 and vocabulary size 151936;
- `max_position_embeddings=32768`;
- `eos_token_id=151643`;
- no substitution with `Qwen/Qwen3-4B`, an Instruct/Thinking checkpoint, a
  quantized checkpoint or a converted derivative;
- successful CPU config/tokenizer loading from the exact local directory.

No corrected GPU run may begin before this gate passes.

## 4. Corpus compatibility decision

The active public corpus was packed with the mistakenly staged tokenizer and
therefore currently appends EOS 151645 in its registered training-token hashes.
It is invalid for Base training unless a new, pre-outcome migration audit
proves compatibility.

After the Base tokenizer is available, compare every tokenizer asset and run
fixed raw-text and packed-row probes. A reusable text payload requires that
ordinary `add_special_tokens=false` encoding be byte-for-byte identical and
that every existing `content_split` still encodes to exactly
`sequence_length - 1` tokens. The Base dataloader will append EOS 151643, so
the old training-token hashes, manifest tokenizer binding and independent
audit cannot be reused even if text payloads are compatible.

If ordinary encoding differs at any registered probe, rebuild both stages from
the source inventory with the Base tokenizer. If it is identical, create a new
immutable corpus root with Base-bound provenance, manifest and independent
audit. Hard-linking unchanged JSONL text is permitted only after inode/content
hashes and exact per-row Base lengths are audited; the old manifest and
provenance must never be presented as Base evidence.

## 5. Replacement execution chain

After the model and corpus gates pass, freeze new execution roots and launch a
replacement chain from the corrected Base checkpoint. Re-run activation,
Stage-1 and Stage-2 resource/timing gates, both matched 100M arms, checkpoint
conversion and RULER. No old activation or training child result may satisfy a
new prerequisite.

The evaluation remains the official 13-task RULER matrix with 100 examples per
task and seed 42 at 4K, 8K, 16K, 32K, 64K and 128K. The official Table 1
reference rows are baseline `[96.6, 94.1, 92.1, 88.7, 74.3, 74.8]` and
In-Place TTT `[96.1, 95.6, 92.7, 89.3, 78.7, 77.0]`; 256K remains a separately
labelled extrapolation with 41.7/43.9 references.

All GPU work remains restricted to `hku-gpu2`. Update-admission experiments
remain sealed until the corrected matched baseline and its independent RULER
audit complete.
