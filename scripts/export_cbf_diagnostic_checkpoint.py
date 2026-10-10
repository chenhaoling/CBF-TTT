"""Export a DCP model without changing its source; verify every exported tensor."""
import argparse
import gc
import json
from pathlib import Path
import shutil

from tasks.build_cbf_memory_value import file_digest


def export(source, output, config):
    import torch
    from safetensors import safe_open
    from torch.distributed.checkpoint import FileSystemReader, load
    from scripts.merge_dcp_to_hf import save_model_weights, _normalize_key
    source, output, config = Path(source), Path(output), Path(config)
    if output.exists():
        raise FileExistsError(output)
    def inventory():
        return {p.name: {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                for p in sorted(source.iterdir()) if p.is_file()}
    before = inventory()
    metadata_sha = file_digest(source/'.metadata')
    metadata = FileSystemReader(str(source)).read_metadata()
    mapping = {_normalize_key(k): k for k in metadata.state_dict_metadata if _normalize_key(k)}
    if not mapping or not any('ttt_conv' in k for k in mapping):
        raise ValueError('no trained TTT weights')
    save_model_weights(str(output), str(source))
    shutil.copyfile(config, output/'config.json')
    checked = set()
    for p in sorted(output.glob('*.safetensors')):
        with safe_open(str(p), framework='pt', device='cpu') as f:
            state = {mapping[k]: torch.empty(metadata.state_dict_metadata[mapping[k]].size,
                     dtype=metadata.state_dict_metadata[mapping[k]].properties.dtype) for k in f.keys()}
            load(state, checkpoint_id=str(source), storage_reader=FileSystemReader(str(source)))
            for k in f.keys():
                if k in checked or not torch.equal(f.get_tensor(k), state[mapping[k]].to(torch.bfloat16)):
                    raise ValueError('DCP/export tensor mismatch: '+k)
                checked.add(k)
            del state
        gc.collect()
    if checked != set(mapping) or before != inventory() or metadata_sha != file_digest(source/'.metadata'):
        raise ValueError('incomplete export or source changed')
    audit = {'source': str(source), 'output': str(output), 'tensors_equal_to_dcp': len(checked),
             'source_inventory_unchanged': True, 'source_inventory': before, 'metadata_sha256': metadata_sha,
             'config_source': str(config), 'config_sha256': file_digest(output/'config.json'),
             'source_files_sha256': {p.name: file_digest(p) for p in sorted(output.glob('*.safetensors'))}}
    (output/'export_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k != 'source_inventory'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'config'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args(); export(a.source, a.output, a.config)
