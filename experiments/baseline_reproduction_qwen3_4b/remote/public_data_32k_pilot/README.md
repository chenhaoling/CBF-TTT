# Public-data 32K pilot evidence

This directory archives the preregistered data-only pilot for the public-data
In-Place TTT baseline. No model training ran in this stage.

- `in_place_ttt_public_data_pilot_20261017/` is the retained failed attempt.
  FineWeb rows were written, but the Datasets/PyArrow fragment reader aborted
  during interpreter shutdown with exit 134. It was not audited or counted.
- `in_place_ttt_public_data_pilot_20261017_r2/` is the repaired run. FineWeb
  uses pinned publisher parquet range reads; FineWeb and LongCrawl both pass
  all independent audit checks.
- `fineweb_range_probe.*` records the clean direct-parquet probe used before
  the repair.
- `in_place_ttt_public_data_pilot_20261017_evidence.tar.gz` is the exact remote
  transfer archive. Its SHA256 is recorded in `archive.sha256`.
- `local_archive_verification.json` rechecks the extracted audit, manifest,
  data and provenance hashes against the transferred summary.

The formal public-data mixture and matched no-TTT/TTT training are still
pending. These artifacts validate only the 8-record-per-source 32K packing
path and its provenance/resource accounting.
