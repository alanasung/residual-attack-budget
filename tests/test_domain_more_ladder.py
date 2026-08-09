import pytest
from headroom.headroom.ladder import RUNGS, AttackResult, asr

@pytest.mark.parametrize("rung", RUNGS)
def test_rung_name(rung):
    assert isinstance(rung, str)

def test_attack_result_dict():
    r = AttackResult("b", "direct", True, 3, 0.5, notes="n")
    d = r.to_dict()
    assert d["queries"] == 3

def test_asr_empty():
    assert asr([]) == 0.0
