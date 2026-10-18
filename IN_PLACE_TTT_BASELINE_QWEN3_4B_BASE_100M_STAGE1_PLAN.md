# Corrected Qwen3-4B-Base formal 100M-tier Stage-1 plan

Status: **frozen before corrected formal Stage-1 training**

Freeze date: 2026-10-19 (Asia/Shanghai)

## Scope

Train the matched no-TTT and native-TTT Stage-1 arms of the public-data
98,566,144-token resource tier. All GPU work is restricted to `hku-gpu2` and
its two RTX 5090 devices. This plan does not release Stage 2, the primary
RULER matrix, 256K extrapolation or update-admission experiments.

Both arms start from exact revision
`906bfd4b4dc7f14ee4320094d8b41684abff8539` of
`Qwen/Qwen3-4B-Base`, consume the same audited Base-tokenizer
`stage1_32k.jsonl`, and differ only in native `ttt_mode`. Run no-TTT before
TTT. Each arm uses sequence length 32,768, global batch 64, microbatch 1,
seed 42 and exactly 27 optimizer steps, or 56,623,104 training tokens.

Keep the frozen AdamW `5e-6`, cosine schedule, weight decay `0.1`, BF16 FSDP2,
gradient checkpointing, activation offload and original `ttt_proj`/`ttt_conv`
writer. Save only `global_step_27`, convert it to HF format, and preserve all
raw logs and resource traces.

## Release prerequisites

Before CUDA initialization require all of the following:

1. the corrected full Base-tokenizer corpus and canonical audit pass;
2. corrected Level A and the 16K/32K resource ladder pass;
3. the corrected paired three-step 32K timing wrapper audit passes;
4. the measured per-arm and sequential 27-step projections are finite;
5. a human-reviewed release created after measurement explicitly sets
   `accepted=true` and binds both underlying and corrected timing audits;
6. exact clean upstream/tool revisions, no corpus or CUDA process, at most
   128 MiB used on each GPU, 48 GiB available host memory and 160 GB free disk.

The release is never inferred or created by a watcher. Review without
`--accept`, a missing reviewer, or an empty decision note cannot write it.

## Stop and evidence rules

Stop on nonzero exit, OOM/nonfinite output, an incomplete 27-step trace, less
than 5% GPU headroom, less than 8 GiB available host memory, more than 3.5 GiB
swap use, disk shortage, missing/undersized DCP or HF conversion, dependency
drift, or any failed independent audit. A retained single arm is not a matched
result.

The terminal corrected audit must re-hash the Base identity, tokenizer,
corpus, timing audits, release, plans, configs, runner, both raw arms and both
final checkpoint manifests. Only a paired passing root may release the
separate corrected Stage-2 resource plan.
