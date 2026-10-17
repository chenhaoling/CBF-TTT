# In-Place TTT Level-B corpus-audit scalability addendum

Status: **frozen before any optimized full-corpus audit**

Freeze date: 2026-10-18 (Asia/Shanghai)

Parent public-corpus plan SHA256: `c5019a28b9fa7fa27052ddf1d69eb029a2ec5b3b6446f1527e9ceb7db1a12bab`

## 1. Scope and active-run boundary

This addendum changes only the in-memory representation used by the independent
public-corpus auditor. It does not change the active full-corpus builder, source
files, source order, tokenization, packing, stage boundary, artifact format,
manifest, expected counts, hashes or acceptance checks.

The already running build froze builder SHA256
`925ac99e701d9aa6444e8ec0c2f329cb8200d58155e672018d6ea3c4a044ec59`
and original auditor SHA256
`9c5dbadba76c4ebbba5248952455b3bfe4860b9b2b79bdbb555034dcc76c7e31`
in `/home/ctj/in_place_ttt_public_corpus_full_run_20261018/input_sha256.txt`.
That process must not be restarted or have files replaced while active. Its
automatic original audit remains valid if it completes and passes.

The optimized auditor is an independently frozen fallback if the original
formal audit fails from host-memory exhaustion, and may also be run under a
new output name as a secondary audit. It must never overwrite an existing
audit artifact.

## 2. Scaling risk

The original `SourceContinuity` stored every document entry and cursor in two
dictionaries until the entire two-stage audit ended. The continuous pilot
loads 2,919 FineWeb documents for 3,145,728 FineWeb training tokens. Scaling
that observed density to the registered 17,499,684,864 FineWeb tokens implies
roughly 16.2 million document entries. Retaining nested locator dictionaries,
hash strings and cursors for all of them makes host memory grow with the full
source history even though a completed document can no longer be referenced
by a valid later span.

## 3. Streaming continuity state

The optimized auditor keeps:

- a monotonic `documents_seen` counter;
- the document-stream SHA256 state;
- the last consumed document index and source locator;
- entries and cursors only for the current completed/partial document and
  documents prefetched into the saved pending-token snapshot.

When consumption advances to the next consecutive document, the previous
fully consumed entry is checked and retired. A skipped document, repeated old
document, non-monotonic locator, discontinuous cursor, changed metadata or
snapshot mismatch still fails. `documents_loaded` is compared with
`documents_seen`, not with the bounded active dictionary size.

The audit result additionally reports `continuity_peak_active_documents` and
`continuity_documents_seen` for both sources. These diagnostics do not enter
the pass decision and do not change any registered corpus hash.

## 4. Verification and frozen files

The existing continuous two-stage packer test still passes. New tests stream
10,000 complete documents and require no more than two simultaneously active
entries, and preload two snapshot documents before proving that consumption
cannot skip the first one.

| File | SHA256 |
|---|---|
| `scripts/audit_in_place_ttt_public_corpus.py` | `5ec753a29996f6abc386d7e16529218bbaaf6c1b1ff271638db730b9c41f8b87` |
| `tests/test_in_place_ttt_public_corpus.py` | `30e2704b8bcfbdf7248e0a1875e6377d432d16a81a21b572c3048836dbc2f86b` |

Before use on the formal corpus, this version must pass the frozen pilot again
and reproduce all ten registered audit checks and observed totals. A full-run
fallback is allowed only when the published corpus directory and manifest are
present and the original builder completed successfully; a failed or partial
build cannot be rescued by this addendum.
