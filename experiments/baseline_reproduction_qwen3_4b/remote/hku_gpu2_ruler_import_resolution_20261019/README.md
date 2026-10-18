# RULER import-resolution audit

This directory records a CPU-only import-resolution audit executed in the
frozen RULER environment on `hku-gpu2`. The process used an empty
`CUDA_VISIBLE_DEVICES` and did not load model weights or RULER examples.

The development checkout contains later CBF changes in its copy of
`inference_model`. The formal RULER runner places the clean upstream checkout
before the tool checkout in `PYTHONPATH`, and its registered entry imports
`inference_model` before OpenCompass constructs a model. The runtime probe
confirmed that both `Qwen3Config` and `Qwen3ForCausalLM` resolve from the clean
upstream checkout at commit
`be2324829b0e91c8fd10a74d4b43714fde6676e1`.

The resolved model file contains native `TTT_update` support and no import of
`cbf_ttt.runtime`. The tool checkout's contrasting model file does contain the
CBF import, so the path check is capable of detecting the relevant pollution.

The structured audit passed and is stored in `audit.json`, SHA256
`28445763f9e8261fbac7f32c3880e76b6afc696bdd58f3c2babed1e8d8313781`.
It binds the clean upstream and frozen tool commits, registered OpenCompass
entry, RULER runner, RULER config and resolved inference files. It proves
class resolution only; later resource scoring must still load the complete
Stage-2 HF checkpoint and produce audited predictions.
