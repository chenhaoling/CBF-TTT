# V19 independent confirmation data

This directory archives the official frozen V19 corpus constructed on
`hku-gpu2` before any V19 model outcome was observed.

- Remote root:
  `/home/ctj/cbf_ttt_controlled_interference_v19_data_20261014_retry1`
- Code commit: `94dcff9c4ee163be535d211ae7ce2470baa4c648`
- Terminal status: `completed_controlled_interference_v19_data`
- Runner exit code: 0
- Independent corpus audit: all 11 checks passed
- Rows / subjects / template groups / folds: 320 / 80 / 20 / 5
- Primary locality budget: 3%, at most 9 failures
- Manifest SHA-256:
  `e620e75f1e73d1ec3f09510c784785a53c09a4209ce80fb767b259d7b0abbb6b`
- Data SHA-256:
  `384d94f24820bc11244d6d0c2525fc02887d456a066d68e0cbade0bcd7763cb9`
- State SHA-256:
  `2459ca4c48d2a341e277b875150cf532736d6fbf5ccd7a4e91c736ee9ec0bf3f`

An initial root without `_retry1` exited before corpus creation because the
runner referred to a non-existent legacy V18 data path. Its exit code was 1.
The runner path was corrected and committed before this official corpus was
created. The failed root was never scored or used as a dependency.
