# In-Place TTT full-corpus streaming-audit fallback watcher addendum

Status: **frozen before the fallback watcher is launched**

Freeze date: 2026-10-19 (Asia/Shanghai)

Parent public-corpus plan SHA256:
`c5019a28b9fa7fa27052ddf1d69eb029a2ec5b3b6446f1527e9ceb7db1a12bab`

Parent audit-scalability addendum SHA256:
`c8641bbfa83f79a5838f13cf906ef0fdbd2053c080cbab987aa20f798423fdf1`

## 1. Purpose and scope

The active full-corpus wrapper froze the original independent auditor before
the formal-scale memory risk was discovered. That auditor retains all prior
document-continuity entries and may be killed by host-memory exhaustion after
the builder has already atomically published a valid corpus. The frozen
streaming auditor preserves every registered check while retaining only live
continuity state.

This addendum automates the already registered fallback. It changes no source
document, token, packing rule, stage boundary, manifest, training config,
checkpoint, model, RULER input or metric. It uses CPU only on `hku-gpu2`.

## 2. Frozen inputs

| Input | SHA256 |
|---|---|
| public-corpus plan | `c5019a28b9fa7fa27052ddf1d69eb029a2ec5b3b6446f1527e9ceb7db1a12bab` |
| active builder | `925ac99e701d9aa6444e8ec0c2f329cb8200d58155e672018d6ea3c4a044ec59` |
| original auditor | `9c5dbadba76c4ebbba5248952455b3bfe4860b9b2b79bdbb555034dcc76c7e31` |
| streaming auditor | `5ec753a29996f6abc386d7e16529218bbaaf6c1b1ff271638db730b9c41f8b87` |
| source inventory | `c7e6761dc5d27013c09d6002de5c3e2b2185492b18100fba4b966c855debf7df` |
| inventory audit | `32daacb7eee2c9536ab8c06c01ac7c423d429f04cdf512a958f9ffd201487507` |

The watcher also freezes the clean tool commit, verification Python, current
Qwen3-4B tokenizer assets and its own arguments before waiting.

## 3. State machine

The watcher accepts an existing canonical `independent_audit.json` only when
it passes, binds the exact published manifest, contains all-true checks and
identifies either the registered original or streaming auditor.

It releases the streaming fallback only when all conditions hold:

1. the final corpus directory and full-profile manifest were atomically
   published and the `.partial` corpus directory is absent;
2. exact totals are 724,800 records and 34,999,369,728 training tokens;
3. manifest bindings match the frozen plan, builder, inventory, inventory
   audit and current tokenizer assets;
4. all four expected data/provenance artifacts exist and are nonempty;
5. the original wrapper state is exactly `FAILED`;
6. no public-corpus builder or auditor process remains active;
7. canonical `independent_audit.json` and its partial file do not exist.

The decision is evaluated twice five seconds apart before launch. The fallback
writes the canonical audit directly through the auditor's atomic partial-file
protocol because no prior canonical artifact exists.

## 4. Refusal and stop conditions

The watcher refuses to run when an existing audit is failed, malformed or
bound to another manifest. It also refuses a failed wrapper without atomic
corpus publication, a completed wrapper without an audit, dependency/hash
drift, a dirty tool checkout, wrong host, unexpected totals, tokenizer drift,
or any missing artifact. It never deletes, renames or overwrites an existing
audit.

Fallback nonzero exit or independent validation failure is terminal and does
not release training. Waiting times out after the registered budget without
starting an audit.

## 5. Acceptance and downstream release

Success requires the streaming audit to exit zero and the watcher to
independently recheck its protocol, full profile, all checks, manifest binding
and registered auditor SHA256. Only then may the canonical audit satisfy the
existing Stage-1 timing and formal-training gates.

This fallback cannot convert a partial or invalid corpus into a passing one.
It addresses only the original auditor's host-memory representation. The
result remains a public-data matched replication, not an exact reproduction
of the paper's private corpus.
