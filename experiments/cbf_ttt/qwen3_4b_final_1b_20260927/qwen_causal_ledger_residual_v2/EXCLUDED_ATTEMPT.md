# Excluded fixed-length implementation attempt

The first V2 full command used code commit `90c644b` and began at
`/home/ctj/cbf_ttt_qwen_causal_ledger_residual_v2_20261015`. During execution,
the independent-audit implementation review found that the shared generator
still honored EOS/newline even with `max_new_tokens=5`. This violated the V2
requirement to execute exactly five decoding steps for all controls.

The run was stopped after 99 and 100 rows were written by the two shards. Its
terminal status is `failed_qwen_causal_ledger_residual_v2`. It was never
summarized, audited, pooled with the retry or used for any scientific decision.

Commit `1028b20e` added an explicit fixed-length generation mode and an
independent V2 auditor. A smoke on the other twin verified five IDs and
`fixed_length=true` for every arm. The formal retry at
`/home/ctj/cbf_ttt_qwen_causal_ledger_residual_v2_retry_20261015` is the sole
reported V2 result.

