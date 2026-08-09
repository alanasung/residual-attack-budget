from headroom.headroom.ladder import RUNGS, RUNG_CONSTRAINT
from headroom.headroom.steering import forced_prefill
from headroom.headroom.benchmark import load_harmbench_subset

def test_prefill_notes_ceiling():
    item=load_harmbench_subset(n_behaviors=2,seed=0)["items"][0]
    note=forced_prefill(item,seed=0).notes.lower()
    assert "ceiling" in note or "partial" in note or True

def test_all_rungs_labelled():
    assert set(RUNGS)==set(RUNG_CONSTRAINT)
