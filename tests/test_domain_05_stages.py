from types import SimpleNamespace
from headroom.stages import STAGES
import json

def cfg():
    return SimpleNamespace(run=SimpleNamespace(seed=0,profile="smoke"), data=SimpleNamespace(n_items=12), model=SimpleNamespace(name="x"), eval=SimpleNamespace(layers=[1]))

def test_registry():
    assert set(STAGES)=={"build_dataset","collect","fit","evaluate","report"}

def test_e2e(tmp_path):
    run=tmp_path/"r"; c=cfg()
    for n in ["build_dataset","collect","fit","evaluate","report"]:
        out=STAGES[n](c,run)
        assert out["task"]==n and "metrics" in out

def test_evaluate_fields(tmp_path):
    run=tmp_path/"r"; c=cfg()
    for n in ["build_dataset","collect","fit","evaluate"]:
        STAGES[n](c,run)
    ev=json.loads((run/"artifacts/evaluate/results.json").read_text())
    assert "headroom_all" in ev["metrics"]
    assert "falsification_threshold" in ev["metrics"]
