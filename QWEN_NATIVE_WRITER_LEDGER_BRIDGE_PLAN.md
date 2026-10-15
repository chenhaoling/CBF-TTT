# Preregistered native-writer causal-ledger bridge pilot

## Freeze point and question

The scale-48 causal-ledger development and one-time confirmation both passed
all frozen gates at 100% primary five-token exact. Their value residuals were
externally constructed, so they do not establish that the baseline
`ttt_conv`/`ttt_proj` writer stores query-readable factual content.

This pilot asks the necessary single-record question before any multi-record
forgetting controller is attached: when the final 1B-token Qwen3-4B checkpoint
sees one complete native 4,096-token chunk that repeatedly states a key/value
mapping, does its original writer produce a fast-weight delta that identifies
that value under fresh-KV readout?

## Frozen diagnostic data

Use two development-only `resource_v2` groups per each of the seven lifecycle
families, twin zero only: 14 source groups. Select the first current, non-absent
query at the first decision boundary. The correct write chunk repeatedly uses
the selected key and causal value; the wrong chunk uses the same key and the
first deterministic non-answer, non-null candidate. Token repetition is
truncated to exactly 4,096 tokens. This deliberately favorable upper bound
maximizes exposure to the fact and cannot be interpreted as natural text.

No confirmation row or natural benchmark is used. The passing confirmation
summary and audit hashes are mandatory launch dependencies.

## Arms and readout

For each source group, run the unchanged checkpoint and writer in four arms:

1. `zero`: no fast memory;
2. `correct_writer`: original writer delta from the correct mapping;
3. `wrong_writer`: original writer delta from the wrong-value mapping;
4. `external_residual`: the already validated Qwen LM-head value residual at
   scale 48, reported only as a task/readout positive control.

The writer arms use the checkpoint's seven registered TTT layers,
`ttt_chunk=4096`, `ttt_lr=0.3`, and unmodified trained `ttt_conv`/`ttt_proj`.
After writing, discard all attention KV, retain only session fast weights and
score the same six candidates. Also generate exactly five tokens greedily.
Record per-arm NLL, prediction, exact IDs, memory/base relative norm, write and
read time, and peak allocated/reserved CUDA memory.

## Registered gate

The native bridge passes only if all hold:

- `correct_writer` candidate accuracy at least 80% and five-token exact at
  least 60%;
- `wrong_writer` truth candidate accuracy at most 20% and its injected wrong
  value is selected at least 80%;
- `zero` truth candidate accuracy at most 30%;
- source-group bootstrap 95% lower bounds for correct-answer NLL improvement
  over both Wrong and Zero are above zero;
- `external_residual` reaches at least 95% candidate and 85% exact;
- all 14 source groups, model parameters, seven layer memories, hashes,
  fixed-length traces and resources pass independent audit.

Pass releases a multi-record native-writer lifecycle experiment. Failure stops
the unmodified native-writer branch: sparse routing cannot repair a writer that
does not bind one explicitly repeated value. A failure does not invalidate the
confirmed external-residual causal ledger; it limits the end-to-end baseline
transfer claim and motivates a learned or structured writer replacement.
