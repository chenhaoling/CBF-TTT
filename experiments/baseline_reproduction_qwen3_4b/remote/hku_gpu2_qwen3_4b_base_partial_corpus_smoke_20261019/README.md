# Live Base partial-corpus smoke evidence

This directory records the read-only partial-corpus smoke at commit
`feeef4600bb13d3956409d9dbccf1493e7d83b6e` on `hku-gpu2`.

- `audit.json` contains the passing no-plaintext audit artifact. It binds the
  live builder PID and source hash, frozen smoke plan, auditor and exact
  Qwen3-4B-Base tokenizer assets.
- `remote_tests.txt` records the exact remote commit, hidden CUDA environment
  and 22 passing focused tests.

The smoke re-tokenized the first 16 Stage-1 records and checked the latest 32
complete provenance rows through index 65,354. It did not modify the growing
corpus, create a final manifest, authorize training or use a GPU. The complete
post-publication independent corpus audit remains mandatory.
