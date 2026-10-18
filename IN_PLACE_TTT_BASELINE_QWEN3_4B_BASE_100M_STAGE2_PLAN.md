# Corrected Qwen3-4B-Base formal 100M-tier Stage-2 plan

Status: **frozen before corrected formal Stage-2 training**

Freeze date: 2026-10-19 (Asia/Shanghai)

## Scope

Train the matched no-TTT and native-TTT Stage-2 arms of the public-data
100M-tier baseline. All GPU work is restricted to `hku-gpu2` and its two RTX
5090 devices. RULER and update-admission experiments remain sealed until this
paired training root and its corrected independent audit pass.

Each arm starts directly from its corresponding, unchanged corrected formal
Stage-1 HF checkpoint. Use a fresh Stage-2 optimizer and scheduler; resource
or timing checkpoints may never initialize formal training. Run no-TTT before
native TTT. Both arms consume the same audited Base-tokenizer
`stage2_128k.jsonl`, use sequence length 131,072, global batch 64, microbatch
1, seed 42 and exactly five optimizer steps, or 41,943,040 training tokens per
arm.

Keep the frozen YaRN factor 4, AdamW `5e-6`, cosine schedule, weight decay
`0.1`, BF16 FSDP2, gradient checkpointing, activation offload and original
`ttt_proj`/`ttt_conv` writer. Save only `global_step_5`, convert it to HF
format, and preserve all raw logs and resource traces.

## Release prerequisites

Before CUDA initialization require all of the following:

1. the complete Base-tokenizer corpus and canonical audit pass;
2. corrected formal Stage-1 and both its underlying/wrapper audits pass;
3. corrected 64K/128K resource ladder and both audits pass;
4. corrected paired three-step 128K timing and both audits pass with finite
   positive per-arm and sequential projections;
5. a human-reviewed release created after measurement explicitly sets
   `accepted=true` and binds both underlying and corrected timing audits;
6. exact clean upstream/tool revisions, at least 160 GB free disk, no corpus
   or CUDA process, at most 128 MiB used on each GPU and at least 48 GiB
   available host memory.

The release is never inferred or created by a watcher. Review without
`--accept`, a missing reviewer or an empty decision note cannot write it.

## Stop and evidence rules

Stop on nonzero exit, OOM/nonfinite output, an incomplete five-step trace,
less than 5% GPU headroom, less than 8 GiB available host memory, more than
3.5 GiB swap use, disk shortage, missing/undersized final DCP or HF output,
dependency drift, or any failed independent audit. A retained single arm is
not a matched result.

The terminal corrected audit must bind the Base identity/tokenizer, complete
corpus, corrected Stage-1, resource ladder, both timing audits, explicit
release, plans, configs, existing runner and paired formal checkpoint audit.
Only a paired passing root may release the paper-facing RULER resource ladder
and matrix. The result remains a public-data 100M-tier reproduction and does
not establish the paper's private-data full-scale training result.
