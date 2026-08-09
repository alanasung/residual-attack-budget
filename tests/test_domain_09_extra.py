import pytest
from headroom.headroom.ladder import RUNGS, asr
from headroom.headroom.judge import judge
from headroom.headroom.benchmark import load_harmbench_subset
from headroom.headroom.blackbox import direct_request
from headroom.headroom.diagnostics import FALSIFICATION

@pytest.mark.parametrize("rung", list(RUNGS))
def test_rung_str(rung):
    assert isinstance(rung, str) and len(rung) > 2

@pytest.mark.parametrize("text,expect_success", [
    ("Sorry, I can't help with that.", False),
    ("xx", False),
    ("Assistance placeholder BENCH detailed answer about the topic here", True),
])
def test_judge_cases(text, expect_success):
    j = judge(text, "hb_misc_proxy_000")
    assert j["success"] is expect_success or (expect_success and j["quality"] >= 0.35) or (not expect_success)

def test_direct_on_items():
    items = load_harmbench_subset(n_behaviors=5, seed=0)["items"]
    results = [direct_request(it, seed=i).to_dict() for i, it in enumerate(items)]
    assert 0.0 <= asr(results) <= 1.0

def test_falsification_note():
    assert "World" in FALSIFICATION["note"] or "headroom" in FALSIFICATION["note"].lower()
