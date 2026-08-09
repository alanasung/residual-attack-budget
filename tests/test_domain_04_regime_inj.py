from headroom.headroom.benchmark import load_harmbench_subset
from headroom.headroom.regime import select_saturation_regime
from headroom.headroom.injection import run_injection_track
from headroom.headroom.diagnostics import FALSIFICATION, length_hardening_sweep
from headroom.headroom.capability import run_capability_twins

def test_regime():
    items=load_harmbench_subset(n_behaviors=20,seed=0)["items"]
    bb={it["behavior_id"]: (0.9 if i%2==0 else 0.05) for i,it in enumerate(items)}
    reg=select_saturation_regime(items,bb,floor=0.15)
    assert reg["n_kept"]>=1 and reg["n_excluded"]>=1

def test_injection():
    out=run_injection_track(load_harmbench_subset(n_behaviors=6,seed=0)["items"], seed=0)
    assert out["track"]=="injection"

def test_falsify():
    assert FALSIFICATION["min_ceiling_minus_blackbox"]==0.05

def test_length():
    assert len(length_hardening_sweep(load_harmbench_subset(n_behaviors=6,seed=0)["items"], lengths=(4,16), seed=0)["rows"])==2

def test_capability():
    rates=run_capability_twins(load_harmbench_subset(n_behaviors=12,seed=0)["items"], seed=0)["rates"]
    assert {"unwilling","incapable","willing"} <= set(rates)
