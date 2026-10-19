# Live public-corpus memory-monitor addendum

Status: **frozen before persistent monitoring**

Freeze date: 2026-10-19 (Asia/Shanghai)

Parent: `IN_PLACE_TTT_BASELINE_PUBLIC_CORPUS_PLAN.md`

## Purpose

The full Base-tokenizer corpus builder has bounded explicit packer state, but
its process RSS increased during the long Stage-1 run. Register a read-only
monitor for the active full-corpus build so Stage-1 and Stage-2 memory trends
can be evaluated from persistent evidence without changing or restarting the
builder.

## Frozen scope

- Run only on `hku-gpu2` (`user-Super-Server`) with CUDA hidden.
- Bind the clean committed monitor source, this addendum, the frozen corpus
  builder and the exact active output root.
- Poll every 300 seconds for at most 172800 seconds.
- Discover the builder through `/proc/*/cmdline` by requiring both the exact
  builder filename and exact `--output-root` value.

Each sample may read only:

- selected fields from the builder's `/proc/<pid>/status`;
- aggregate host `/proc/meminfo` values;
- filesystem capacity through `shutil.disk_usage`;
- byte sizes of the four possible partial data/provenance files;
- the original wrapper's status text.

The monitor must not scan JSONL records, read source text, tokenize data,
access CUDA, send a signal, alter process priority, write below the corpus or
wrapper roots, or launch any audit/training process. Its output uses a new
standalone directory.

## Termination and evidence

Write an append-only CSV plus atomic state JSON/text. Stop successfully when
the bound builder is no longer present, recording the wrapper state observed
at that moment. Also stop non-destructively on the frozen monitor timeout. More
than one matching builder or an identity/host mismatch is a monitor failure;
it never changes the corpus process.

This monitor provides resource telemetry only. It cannot pass or fail the
corpus, relax the canonical independent audit, release Level-A, or authorize
training or update-admission experiments.
