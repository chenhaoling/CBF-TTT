"""Regression checks for source reuse across legacy and current scene schemas."""
import json
from pathlib import Path
import tempfile
import unittest
from tasks.build_cbf_memory_value import old_manifests,excluded_values,context_texts


class Decoder:
    def decode(self,ids):return ' '.join(map(str,ids))


class SourceExclusionTests(unittest.TestCase):
    def test_old_and_new_manifests_include_nested_source_hash_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'cbf_ttt_new'/'data';output.mkdir(parents=True)
            legacy=root/'cbf_ttt_legacy'/'scenes.meta.json';legacy.parent.mkdir()
            legacy.write_text(json.dumps({'source_id':'legacy-id','sha256':'legacy-hash'}))
            design=root/'cbf_ttt_readability'/'data'/'design.json';design.parent.mkdir(parents=True)
            design.write_text(json.dumps({'context_hashes':[{'sources':['used-source-hash']}]}))
            current=output/'design.json';current.write_text(json.dumps({'sources':['current-output']}))
            paths=old_manifests(root,output)
            self.assertEqual(set(paths),{legacy,design})
            self.assertEqual(excluded_values(paths),{'legacy-id','legacy-hash','used-source-hash'})

    def test_current_chunks_and_legacy_contexts_are_decoded(self):
        scene={'chunks':[[11,12],[13,14]],'nested':{'prefix':[[21,22],{'ids':[23]}]},
               'context_ids':[31,32],'text':'raw source','choice_ids':[99,100]}
        self.assertEqual(list(context_texts(scene,Decoder())),
                         ['11 12','13 14','21 22','23','31 32','raw source'])
        self.assertEqual(list(context_texts({'chunks':[[],{'unrelated':[1,2]}]},Decoder())),[])


if __name__=='__main__':unittest.main()
