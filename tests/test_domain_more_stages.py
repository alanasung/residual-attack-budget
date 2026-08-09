from types import SimpleNamespace
from headroom.stages import STAGES
import json

def cfg():
    return SimpleNamespace(
        run=SimpleNamespace(seed=0, profile="pilot"),
        data=SimpleNamespace(n_items=12),
        model=SimpleNamespace(name="x", revision=None),
        eval=SimpleNamespace(layers=[1]),
        force_synthetic=True,
    )

def test_build_safety_note(tmp_path):
    out = STAGES["build_dataset"](cfg(), tmp_path/"r")
    assert "harmful" in out["metrics"]["safety_note"].lower() or "no harmful" in out["metrics"]["safety_note"].lower()

def test_collect_regime(tmp_path):
    run = tmp_path/"r"
    c = cfg()
    STAGES["build_dataset"](c, run)
    STAGES["collect"](c, run)
    assert (run/"artifacts/collect/regime.json").is_file()

def test_fit_ladder_keys(tmp_path):
    run = tmp_path/"r"
    c = cfg()
    for s in ("build_dataset", "collect", "fit"):
        STAGES[s](c, run)
    ladder = json.loads((run/"artifacts/fit/ladder_results.json").read_text())
    assert {"discrete", "continuous", "projected", "steering", "forced_prefill"} <= set(ladder)

def test_evaluate_falsification_fields(tmp_path):
    run = tmp_path/"r"
    c = cfg()
    for s in ("build_dataset", "collect", "fit", "evaluate"):
        STAGES[s](c, run)
    ev = json.loads((run/"artifacts/evaluate/results.json").read_text())
    assert "falsification_threshold" in ev["metrics"]
    assert "headroom_all" in ev["metrics"]

def test_report(tmp_path):
    run = tmp_path/"r"
    c = cfg()
    for s in ("build_dataset", "collect", "fit", "evaluate", "report"):
        out = STAGES[s](c, run)
    assert out["task"] == "report"
