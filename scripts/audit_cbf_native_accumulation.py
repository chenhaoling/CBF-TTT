"""Independent completion audit for native_accumulation_v1."""
import argparse
import json
from pathlib import Path

from tasks.build_cbf_memory_value import file_digest
from tasks.cbf_native_accumulation import ARMS, SOURCE_SUMMARY_SHA, load_data, summarize


def audit(root):
    root=Path(root); rows,design=load_data(root)
    stored=json.loads((root/'summary.json').read_text()); recomputed=summarize(root)
    if stored != recomputed: raise ValueError('stored summary differs from recomputation')
    source=Path(design['source_root'])
    source_audit=json.loads((source/'execution_audit.json').read_text())
    if not source_audit['audit_passed'] or source_audit['summary_sha256'] != SOURCE_SUMMARY_SHA:
        raise ValueError('source audit changed')
    if file_digest(source/'summary.json') != SOURCE_SUMMARY_SHA:
        raise ValueError('source summary changed')
    expected={(r['id'],r['cell']) for r in rows}
    arm_hashes={}
    for arm in ARMS:
        seen=set()
        for shard in (0,1):
            path=root/f'{arm}_{shard}.jsonl'; audit_path=Path(str(path)+'.audit.json')
            part=list(map(json.loads,path.read_text().splitlines())); meta=json.loads(audit_path.read_text())
            if meta['rows_sha256'] != file_digest(path) or meta['rows'] != len(part): raise ValueError('row hash/count')
            seen.update((r['id'],r['cell']) for r in part)
            arm_hashes[path.name]=file_digest(path)
        if seen != expected: raise ValueError('incomplete arm contexts')
    return {'protocol':'native_accumulation_v1','audit_passed':True,
            'data_sha256':design['data_sha256'],'design_sha256':file_digest(root/'data/design.json'),
            'summary_sha256':file_digest(root/'summary.json'),'audit_script_sha256':file_digest(Path(__file__)),
            'source_summary_sha256':SOURCE_SUMMARY_SHA,'rows_sha256':arm_hashes,
            'terminal_status':stored['terminal_status'],'decision':stored['decision']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args(); result=audit(args.root);Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

