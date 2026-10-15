from scripts.audit_qwen_utility_semantic_record_gate_v1_data import (
    input_fields_are_clean,
    query_structure_is_valid,
    twin_structure_is_valid,
)
from tests.test_qwen_utility_semantic_record_gate_v1 import material
from tasks.build_qwen_utility_semantic_record_gate_v1 import build_episode
import unittest


def episodes():
    return [
        build_episode(0, "semantic_overwrite", 16, twin, material())
        for twin in (0, 1)
    ]


def test_audit_helpers_accept_valid_rows():
    rows = episodes()
    assert input_fields_are_clean(rows)
    assert twin_structure_is_valid(rows)
    assert query_structure_is_valid(rows)


def test_audit_helpers_reject_leak_and_bad_twin():
    rows = episodes()
    rows[0]["boundaries"][0]["queries"][0]["input"]["relation_id"] = "P1"
    assert not input_fields_are_clean(rows)
    rows = episodes()
    rows[1]["events"][2]["record"]["input"]["value"] = "unexpected"
    assert not twin_structure_is_valid(rows)


class TestUtilitySemanticRecordDataAudit(unittest.TestCase):
    def test_valid_rows(self):
        test_audit_helpers_accept_valid_rows()

    def test_rejections(self):
        test_audit_helpers_reject_leak_and_bad_twin()
