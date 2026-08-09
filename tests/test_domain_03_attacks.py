from headroom.headroom.benchmark import load_harmbench_subset
from headroom.headroom.blackbox import direct_request, pair_tap_search
from headroom.headroom.discrete import gcg_attack
from headroom.headroom.continuous import pgd_embedding_attack
from headroom.headroom.projection import project_to_manifold, projected_attack
from headroom.headroom.steering import extract_refusal_direction, forced_prefill, steer_attack

def test_suite():
    item=load_harmbench_subset(n_behaviors=3,seed=0)["items"][0]
    assert direct_request(item,seed=0).rung=="direct"
    assert pair_tap_search(item,budget=3,seed=0).queries>=1
    g=gcg_attack(item,steps=8,seed=0)
    assert "plateau_rate" in g["diagnostics"]
    c=pgd_embedding_attack(item,steps=4,seed=0)
    assert c["manifold_distance"]>=0
    assert "projection" in projected_attack(item,c,seed=0)
    d=extract_refusal_direction(dim=16,n_pairs=8,seed=0)
    assert len(d["direction"])==16
    assert steer_attack(item,d["direction"],seed=0).rung=="steering"
    assert forced_prefill(item,seed=0).rung=="forced_prefill"

def test_project_tokens():
    out=project_to_manifold([0.1]*64, n_tokens=4, seed=0)
    assert len(out["token_ids"])==4
