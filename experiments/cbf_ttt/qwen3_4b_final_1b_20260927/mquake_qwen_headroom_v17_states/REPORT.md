# MQuAKE-T V17 Frozen Causal States

The independently audited retry produced the exact state matrix used by V17.

- Pilot: 40 rows, 1,200 historical single deletions, 200 fixed-arm entries.
- Formal: 272 rows, 8,160 historical single deletions, 1,360 fixed-arm entries.
- Current-session deletion: prohibited.
- Model scoring and external data access: absent.
- Terminal marker: `completed_mquake_qwen_headroom_v17_states`.
- Independent audit: passed all 10 checks.

Registered oracle order: primary candidate accuracy, lower primary answer NLL,
retain-all, then chronological single-deletion order.

## Frozen hashes

- Manifest: `2c7e9eba8aa0aef5e7438a6ffd29f7fa41040403cb9b1942fafb6c14d1449b83`
- Pilot states: `95b8d297b02cb0613503eeb1a0e82013896feda4e8c0a765e3092e70d02a0064`
- Formal states: `d5ad7c91d01421cdea1d77b45dd4825f9dbd9fdaf55d90a45e4d1b406763340e`
- State audit: `c9b8c6a4ebfb4029e863e50a8c36637b7ad3eeb5ea04664912d647f843772867`

The first state run is preserved remotely as a failed audit attempt caused by
an auditor assumption about JSON object iteration order. The retry rebuilt the
same state hashes and passed after object-key order was correctly treated as
non-semantic while exact key membership and registered policy order remained
checked independently.
