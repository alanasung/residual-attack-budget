def test_imports():
    from headroom.headroom import benchmark, ladder, blackbox, discrete, continuous, steering, projection, diagnostics, judge, capability, injection, regime, pipeline
    assert ladder.RUNGS

def test_stages():
    from headroom.stages import STAGES
    assert callable(STAGES["evaluate"])
