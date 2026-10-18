# In-Place TTT Qwen3-4B paper-inference clipping addendum

Status: **frozen before any corrected-Base RULER score**

Freeze date: 2026-10-19 (Asia/Shanghai)

Upstream repository: `ByteDance-Seed/In-Place-TTT`
Upstream commit: `be2324829b0e91c8fd10a74d4b43714fde6676e1`

## 1. Reason for this correction

Appendix C of the paper specifies that Qwen3-4B inference clips every
fast-weight update delta whose Frobenius norm exceeds `tau=1e-5`, rescaling
that delta to norm `tau` before applying it. The released inference source at
the frozen commit computes `current_w = current_w + dw` directly and exposes
no clipping configuration. A formal RULER run through that released path
would therefore test different inference semantics from the paper.

This discrepancy was found before any corrected Qwen3-4B-Base formal RULER
score existed. Prior post-trained-Qwen or resource-smoke outputs remain
superseded and cannot be reinterpreted as paper-inference results.

Primary sources:

- paper: <https://arxiv.org/abs/2604.06169>;
- official repository: <https://github.com/ByteDance-Seed/In-Place-TTT>.

## 2. Evidence levels

Released-code Level A remains an unmodified-code execution check. It may
validate loading, conversion and metric plumbing, but its native-TTT score
does not implement the paper's inference clip and cannot be used as the
paper-facing efficacy result.

Public-data Level B uses the corrected paper-inference semantics defined here.
It remains a matched public-data replication because the private continual
training mixture and several full-scale settings are unavailable. The clip
correction does not upgrade Level B to an exact reproduction of Table 1.

Only the `hku-gpu2` 2xRTX5090 path is authorized for GPU execution. The older
80-GB multi-GPU runner is outside the active execution path and cannot release
a result under this addendum.

## 3. Frozen numerical rule

For every native Qwen3 TTT layer and every committed chunk update `dw`:

```text
n = ||dw||_F
scale = min(1, 1e-5 / max(n, tiny_float32))
dw = dw * scale
current_w = current_w + dw
```

The norm is computed over the complete two-dimensional update matrix in
float32. The scale is cast back to the update dtype before multiplication.
Clipping occurs after the original `ttt_conv`, optional `ttt_proj` and
`ttt_lr` computation and immediately before the existing native addition.
It is applied independently to every update delta; it does not clip the
accumulated fast weight.

The new configuration field is `ttt_update_clip_norm`. Its default is `None`,
so the baseline repository behavior is unchanged unless a caller opts in.
Paper-facing Qwen3-4B RULER configs pass exactly `1e-5` through OpenCompass
`model_kwargs`. Nonfinite, zero and negative values are rejected. For the
no-TTT arm the same configuration is passed, but no native TTT update exists,
so it has no numerical effect.

## 4. Source isolation and construction

The clean official checkout and every checkpoint remain immutable. Before
each resource or formal RULER run,
`scripts/build_in_place_ttt_paper_inference_overlay.py` must:

1. require the exact clean upstream commit;
2. require exactly the seven tracked `inference_model` files;
3. verify every input size and SHA256;
4. copy those files to a new output-local overlay;
5. modify only Qwen3 configuration and inference modeling files;
6. record source, generated-file and builder hashes in an immutable manifest;
7. refuse an existing output or partial output.

The overlay precedes the clean upstream checkout and tool checkout on
`PYTHONPATH`. It contains no CBF module and does not import the later
CBF-modified tool copy. The run retains the complete overlay and manifest.

Frozen clean-source SHA256 values:

| File | SHA256 |
|---|---|
| `inference_model/__init__.py` | `54ae124b5c993dff594b041dc4375407fbf1c016a689f9dac805e0043e86e52a` |
| `inference_model/hf_llama3/__init__.py` | `f084e24edbc2bb8cebe74f3ded368d40e1c6a848274f06a8077d09135567a1c9` |
| `inference_model/hf_llama3/configuration_llama.py` | `13e03dfcc525f809b215a0effae2837c82da612203f40d645b449e740650f8e8` |
| `inference_model/hf_llama3/modeling_llama.py` | `78e95c6cc3a64d4fc629ca6a9abb135883d1c0a6ca5184a2a0b8283f06802d85` |
| `inference_model/hf_qwen3/__init__.py` | `a38d76873e560e9c034258408e1c67fdfc08f2be846da1937cea1434f6e10e80` |
| `inference_model/hf_qwen3/configuration_qwen3.py` | `39ef69d7c79114ae9bd2710320c4260796fc5dbed9f725e18b5a0f5771114ec2` |
| `inference_model/hf_qwen3/modeling_qwen3.py` | `98ec94bcf1780635d7817b993a0bf4fa1f1a7e3a35415dce8a46c345a52a2858` |

Frozen generated Qwen3 SHA256 values:

| File | SHA256 |
|---|---|
| `inference_model/hf_qwen3/configuration_qwen3.py` | `c23540e8c00d659cef9cb810df7f75efa3389662736bf5570bc1d8f61bab86a6` |
| `inference_model/hf_qwen3/modeling_qwen3.py` | `a212c35ca9a5471f8ff9a1ef70c9cb0ec452c87eaac50dd4da51c8ce8d045103` |

All other generated Python files must remain byte-identical to their clean
source counterparts.

## 5. Pre-score validation and independent audit

Before OpenCompass starts, the run must execute
`scripts/validate_in_place_ttt_paper_inference_overlay.py` with CUDA hidden.
The CPU validation must prove:

- the imported Qwen model file resolves inside the new overlay;
- an above-boundary update is rescaled to Frobenius norm `1e-5`;
- an all-zero update is unchanged;
- the default is disabled and the enabled setting serializes;
- zero, negative, infinite and NaN settings are rejected;
- the real `from_pretrained` override reaches both model config and TTT MLP;
- CUDA was never initialized.

Every single resource/formal audit must independently rehash the clean-source
manifest, generated files, builder, validation record, this addendum and the
registered OpenCompass config. Resource-ladder and matrix audits must require
the same addendum hash from every child run. A stale child audit is invalid.

## 6. Stop conditions and claim boundary

Stop before scoring, or invalidate the affected run, if any source/overlay
hash differs, the checkout is dirty, another model module is imported, the
configured value differs from `1e-5`, CPU validation fails, CUDA is touched
during validation, or any child/parent audit has a different addendum hash.

The correction changes only paper-facing inference. It does not change the
training data, writer, checkpoints, optimizer, RULER inputs, decoding or
metrics. Update-admission experiments remain sealed until the matched
corrected-Base Level-B training and full independently audited RULER matrix
finish.

## 7. Frozen implementation bindings

The implementation was finalized and tested before this document was frozen:

- overlay builder: `efe5167baf623d311fb1856d1c7ea7ccabfaf45a3b37297cbf5cfafbd9814d2a`;
- CPU validator: `4c5ac959d9b7ae69c65013213089be4e03ee16521cec4682b840a564a088ea5f`;
- independent overlay auditor: `f7fcd1ae62df658e9aeed33fc9aa1a4a881027f1979578928e5f7e915de54ddf`;
- 2x5090 single runner: `a55634ec8de94e946c992188edb80fe941e237deb605f179f2e80f5f3311c0da`;
- formal single auditor: `ce20738276321c4193dd3151876694ee1272dc4b580c817d2b063c02cb13af13`;
- resource single auditor: `5d9c7b34ac9cac0e03a01491600b3b249be4e6e833274fb6e3c076a3d02ea4c6`;
- resource-ladder runner: `9ce38a3d00bed8944b11ffcfe4152013dcaba7206b5c753a5ba7bfb022448f8f`;
- resource-ladder auditor: `7e43fb4f1eb18d510fcafc34764b60f4ac3392cf39762b60ec9542722d9c1d23`;
- matrix runner: `ac8a567b40932ea2ffad2241d0924e7e81f8995a6cc6638b2c92dcc8a2bfa785`;
- matrix auditor: `edd439e1be3b2b215438480b5ae880fece2e1e3d8a7cc9f4998d4e4270934d8b`;
- 2x5090 OpenCompass config: `1859bb2d2e43f2a715feaa1171b51b9b1a0432e810115a411b5b81e60e48954d`.

On `hku-gpu2`, with all GPUs hidden, the focused runtime and audit tests
passed `9/9`, and the complete `tests/test_in_place_ttt*.py` baseline suite
passed `94/94`. The frozen official evaluation environment separately
validated an unclipped fixture norm of `24.207931518554688`, a clipped norm
of `1.0000000656873453e-5`, a zero-update no-op, configuration validation and
the `from_pretrained` override without initializing CUDA.
