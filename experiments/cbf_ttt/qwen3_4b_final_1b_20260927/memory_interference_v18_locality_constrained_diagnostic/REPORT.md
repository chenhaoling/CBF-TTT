# V18-LC locality-constrained diagnostic

This directory archives the completed, independently audited post-hoc
diagnostic run from `hku-gpu2`.

- Terminal status: `completed_v18_locality_constrained_diagnostic`
- Runner and audit exit codes: 0
- Summary SHA-256:
  `54a87c57c65c3b36fc3fb13fef7b6c1c1e940cfc8765b5c507572c2f261ccc96`
- Audit SHA-256:
  `abe9b2078d31fee137a60f0ad648dfe8b5189e3b11b35b6196c3b47d89050677`
- Independent audit: passed all four checks
- Primary budget: 3%, at most 9 locality failures
- Observed locality failures: 7/320
- Joint success: 90.625%
- Update accuracy: 100%
- Retained-key accuracy: 91.25%
- Locality accuracy: 97.8125%
- Generation joint success: 73.125%
- Joint gain over the outer-fold fixed policy: 46.875 percentage points

This result is exploratory because the locality budget was introduced after
the original V18 outcome was observed. It leaves the original V18 failure and
`stage_b_released=false` unchanged. It releases only construction of the
preregistered, source-disjoint V19 confirmation corpus.
