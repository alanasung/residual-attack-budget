import pytest
from headroom.headroom.ladder import RUNGS, RUNG_CONSTRAINT, AttackResult, asr, headroom
from headroom.headroom.judge import compliance_quality, is_refusal, judge

@pytest.mark.parametrize("rung", list(RUNGS))
def test_constraint(rung):
    assert rung in RUNG_CONSTRAINT

def test_headroom_asr():
    assert headroom(0.9,0.2)==pytest.approx(0.7)
    assert asr([{"success":True},{"success":False}])==0.5
    assert asr([])==0.0

def test_refusal_and_judge():
    assert is_refusal("Sorry, I can't help with that.")
    j=judge("xx","hb_misc_proxy_000")
    assert j["success"] is False

def test_attack_result():
    assert AttackResult("b","direct",True,2,0.4).to_dict()["queries"]==2
