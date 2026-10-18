# Qwen3-4B-Base paper and RULER alignment preflight

Date: 2026-10-19 (Asia/Shanghai)

Host: `hku-gpu2` (`user-Super-Server`)

GPU use: none (`CUDA_VISIBLE_DEVICES=`)

Result: **PASS**

## Paper requirements used by this replication

The primary source is the April 2026 version of
[In-Place Test-Time Training](https://arxiv.org/abs/2604.06169).

- Section 4.1 and Table 1 compare matched Qwen3-4B-Base baseline and
  In-Place-TTT arms on RULER average accuracy at 4K, 8K, 16K, 32K, 64K,
  128K and 256K. The paper labels 256K as extrapolation.
- The continual curriculum uses about 20B tokens at 32K followed by about
  15B tokens at 128K.
- Table 6 specifies AdamW, learning rate `5e-6`, weight decay `0.1`, no RoPE
  extension in Stage 1, YaRN in Stage 2 and convolution size 5.
- Appendix 9.2 specifies per-update Frobenius clipping with `tau=1e-5` for
  every reported Qwen3-4B evaluation.
- Appendix 9.3 places In-Place TTT in every sixth layer.

The frozen local plans and audits encode these requirements. The public run
uses FineWeb-Edu and LongCrawl64 because the paper describes its continual
corpus by category but does not release its exact mixture or processed
records. Results therefore remain a public-data matched replication.

## Runtime configuration preflight

The remote check loaded
`eval_config/baseline_reproduction/qwen3_4b_2x5090_ruler.py` through
`mmengine.Config.fromfile`, the same configuration route used by OpenCompass.
It repeated the parse for all seven registered lengths and verified:

- exactly 13 tasks at every length: two QA, eight NIAH, variable tracking,
  frequent-word extraction and common-word extraction;
- exactly 100 examples per task and random seed 42;
- one official summary group containing the same 13 tasks;
- deterministic generation, thinking disabled and 64 output tokens;
- the original Qwen3-4B-Base tokenizer with `fix_mistral_regex=false`;
- `ttt_update_clip_norm=1e-5` and one-example batch execution;
- matching context, session and maximum-sequence lengths;
- exact hashes of all ten OpenCompass RULER generator/evaluator modules and
  the three source-data files already frozen in the independent auditor.

Every per-context check passed. The machine-readable evidence is
`remote_preflight.json`; `remote_preflight.raw.txt` preserves the three
OpenCompass data-resolution messages followed by the same JSON payload.

The earlier 2x5090 plan recorded config hash `4c67903e...`, before the paper's
inference clipping was implemented. The pre-score
`IN_PLACE_TTT_BASELINE_PAPER_INFERENCE_CLIP_ADDENDUM.md` explicitly replaces
that config binding with `1859bb2d...`; the only config diff adds the frozen
`1e-5` field and passes it through `model_kwargs`. The clean remote checkout
used by the live RULER watchers has exactly the replacement hash and commit
`c89c6f8c91e2b6bb435142e6fc87a221018be585`.

## Claim boundary

This preflight proves evaluation-definition alignment. It does not prove a
model score, successful long-context resource execution, full 35B-token
training or equivalence to the paper's private data and H800 hardware. Those
claims remain gated by the registered training, resource, matrix and final
Table 1 audits.
