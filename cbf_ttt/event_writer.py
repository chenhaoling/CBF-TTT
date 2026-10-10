"""Differentiable original conv/proj writer, with fresh-KV answer-only readout.

Single complete write block: its frozen backbone activations do not depend on
any candidate in that block. Thus discarding its KV does not detach the writer.
No inference_mode sessions or cached/detached candidate transforms are used.
"""
import hashlib
import torch
import torch.nn.functional as F

LAYERS=(0,6,12,18,24,30,35)


def cache_for(model, memory=None, collect=False):
    from inference_model.hf_qwen3.modeling_qwen3 import TTTDynamicCache
    cache=TTTDynamicCache(config=model.config)
    cache.cbf_enabled=True;cache.cbf_collect=collect;cache.cbf_candidates={}
    cache.cbf_memory={} if memory is None else memory
    return cache


def forward(model, ids, cache):
    tokens=torch.tensor([ids],device=next(model.parameters()).device)
    return model.model(input_ids=tokens,past_key_values=cache,use_cache=True).last_hidden_state


def write_memory(model, ids):
    if len(ids)!=model.config.ttt_chunk:raise ValueError('writer requires one complete chunk')
    cache=cache_for(model,collect=True)
    forward(model,ids,cache)
    memory=cache.cbf_candidates
    if set(memory)!=set(model.config.ttt_layers):raise ValueError('missing writer layer')
    if any(not torch.isfinite(v).all() for v in memory.values()):raise RuntimeError('nonfinite delta')
    return memory


def answer_logits(model,memory,query,answer,cache=None):
    if not query or not answer:raise ValueError('empty supervision')
    cache=cache if cache is not None else cache_for(model,memory)
    hidden=forward(model,query+answer[:-1],cache)
    # hidden at the final prompt position predicts the first answer token.
    return model.lm_head(hidden[:,len(query)-1:]).float().reshape(len(answer),-1)


def answer_loss(model,memory,row,eos):
    target=row['answer_ids']+[eos]
    logits=answer_logits(model,memory,row['query_ids'],target)
    y=torch.tensor(target,device=logits.device)
    losses=F.cross_entropy(logits,y,reduction='none')
    return losses.mean(),losses[:-1].mean(),losses[-1]


def backbone_digest(model):
    from tasks.cbf_checkpoint_readout import is_adapter
    h=hashlib.sha256()
    for n,p in sorted(model.named_parameters()):
        if is_adapter(n):continue
        h.update(n.encode());h.update(str((tuple(p.shape),str(p.dtype))).encode())
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def trainable_writer(model):
    from scripts.probe_ttt_writer_training import writer_parameters
    return writer_parameters(model)


def load_original(path,seed=301):
    from inference_model.hf_qwen3.configuration_qwen3 import Qwen3Config
    from inference_model.hf_qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from tasks.cbf_checkpoint_readout import is_adapter
    torch.manual_seed(seed)
    config=Qwen3Config.from_pretrained(path)
    config.ttt_mode=True;config.ttt_layers=list(LAYERS);config.ttt_chunk=4096
    config.ttt_lr=.3;config.ttt_proj=True;config.ttt_target='hidden_states'
    model,info=Qwen3ForCausalLM.from_pretrained(path,config=config,dtype=torch.bfloat16,
        attn_implementation='sdpa',output_loading_info=True)
    expected={n for n,p in model.named_parameters() if is_adapter(n)}
    if set(info['missing_keys'])!=expected or len(expected)!=14 or any(info.get(k) for k in ('unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('base loading mismatch; expected exactly new conv/proj parameters')
    model=model.to('cuda').eval();params=trainable_writer(model)
    # Match baseline zero-conv initialization explicitly; projection starts nonzero.
    torch.manual_seed(seed)
    with torch.no_grad():
        for n,p in sorted(params.items()):
            if '.ttt_conv.' in n:p.zero_()
            else:p.normal_(0,config.initializer_range)
    return model,params,info
