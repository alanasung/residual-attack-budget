from headroom.headroom.benchmark import BEHAVIOR_IDS, load_harmbench_subset

def test_opaque():
    bench=load_harmbench_subset(n_behaviors=8,seed=0)
    assert all(r["harmful_prompt_ref"].startswith("<BENCH:") for r in bench["items"])

def test_no_bomb_string():
    assert "how to make a bomb" not in str(load_harmbench_subset(n_behaviors=10,seed=0)).lower()

def test_ids():
    assert len(BEHAVIOR_IDS)>=3

def test_n():
    assert load_harmbench_subset(n_behaviors=7,seed=1)["n"]==7
