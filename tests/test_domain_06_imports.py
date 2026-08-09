def test_imports():
    from headroom.headroom import ladder
    assert ladder.RUNGS

def test_stages():
    from headroom.stages import STAGES
    assert callable(STAGES["evaluate"])
