"""Eval-awareness cue arm stamps."""

from headroom.headroom.awareness import run_eval_awareness_arm, wrap_item
from headroom.headroom.benchmark import load_harmbench_subset
from headroom.headroom.diagnostics import length_hardening_sweep


def test_wrap_applies_cue():
    item = load_harmbench_subset(n_behaviors=2, seed=0)["items"][0]
    wrapped = wrap_item(item, cue="[SYSTEM NOTICE: eval]\n", tag="eval")
    assert wrapped["harmful_prompt_ref"].startswith("[SYSTEM NOTICE")
    assert wrapped["awareness_cue"] == "eval"
    assert wrapped["behavior_id"] == item["behavior_id"]


def test_awareness_arm_smoke():
    items = load_harmbench_subset(n_behaviors=6, seed=0)["items"]
    out = run_eval_awareness_arm(items, seed=0, force_synthetic=True, n_items=4)
    assert out["eval_awareness_arm"] == "applied"
    assert out["cues_applied"] is True
    assert out["mode"] == "synthetic"
    assert set(out["asr_by_rung"]) >= {"direct", "blackbox", "discrete"}


def test_length_sweep_accepts_runtime_kw():
    items = load_harmbench_subset(n_behaviors=4, seed=0)["items"]
    out = length_hardening_sweep(items, lengths=(4, 8), seed=0, force_synthetic=True)
    assert len(out["rows"]) == 2
    assert out["mode"] in {"synthetic", "measured"}
