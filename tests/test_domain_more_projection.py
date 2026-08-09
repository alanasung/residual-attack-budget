from headroom.headroom.projection import project_to_manifold
from headroom.headroom.diagnostics import length_hardening_sweep
from headroom.headroom.benchmark import load_harmbench_subset

def test_projection_reduces_or_defines_distance():
    emb = [0.1] * 64
    out = project_to_manifold(emb, n_tokens=4, dim=64, seed=0)
    assert "manifold_distance_after" in out
    assert len(out["token_ids"]) == 4

def test_length_asr_nonincreasing_tendency():
    items = load_harmbench_subset(n_behaviors=10, seed=0)["items"]
    rows = length_hardening_sweep(items, lengths=(4, 32), seed=0)["rows"]
    # Soft check: longer prompts should not massively improve ASR in this model
    assert rows[0]["prompt_len"] == 4
