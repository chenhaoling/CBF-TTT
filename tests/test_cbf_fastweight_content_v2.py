"""Readout-template and three-way split tests for fastweight_content_v2."""
import copy,json,unittest
from tasks.build_cbf_memory_value import digest
from tasks.cbf_fastweight_content import Q_GROUPS,V_GROUPS
from tasks.cbf_fastweight_content_v2 import DEV_GROUPS,FORMATS,query_text,validate


def rows():
    result=[]
    for group in DEV_GROUPS+Q_GROUPS+V_GROUPS:
        domain='fineweb' if group%8 in (0,1,6,7) else 'longcrawl'
        for variant in (0,1):
            label=(group%8+4*variant)%8;chunks=[[0]*4096,[0]*4095+[label+1]];qs={fmt:{'ids':[len(query_text('device-1',fmt)),group],'label':label,'answer_id':10+label} for fmt in FORMATS}
            result.append({'id':f'g{group}_{variant}','twin_id':f'g{group}_{1-variant}','group':group,'split':'dev' if group in DEV_GROUPS else 'q' if group in Q_GROUPS else 'v',
                'domain':domain,'variant':variant,'sources':[f's{group}'],'choice_ids':list(range(10,18)),'chunks':chunks,'context_sha256':digest(json.dumps(chunks).encode()),'queries':{'target':qs}})
    return result


class V2Tests(unittest.TestCase):
    def test_formats_are_distinct_and_splits_valid(self):
        data=rows();validate(data)
        self.assertEqual(len({query_text('device-1',f) for f in FORMATS}),len(FORMATS))
        broken=copy.deepcopy(data);broken[-1]['sources']=data[0]['sources'];broken[-2]['sources']=data[0]['sources']
        with self.assertRaisesRegex(ValueError,'source reuse'):validate(broken)


if __name__=='__main__':unittest.main()
