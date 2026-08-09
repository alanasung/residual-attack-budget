"""Measured-path APIs with Hub downloads monkeypatched away."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom.headroom.benchmark import BEHAVIOR_IDS, load_harmbench_subset
from headroom.headroom.blackbox import estimate_blackbox_asr, pair_tap_search
from headroom.headroom.continuous import pgd_embedding_attack
from headroom.headroom.discrete import gcg_attack
from headroom.headroom.model_runtime import RuntimeModel, try_load_causal_lm
from headroom.headroom.projection import projected_attack
from headroom.headroom.regime import select_saturation_regime


def test_expanded_unique_behaviors():
    assert len(BEHAVIOR_IDS) >= 12
    bench = load_harmbench_subset(n_behaviors=20, seed=0)
    ids = [x["behavior_id"] for x in bench["items"]]
    assert len(set(ids)) == 20


def test_force_synthetic_gcg_no_hub(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("Hub load must not run under force_synthetic")

    monkeypatch.setattr("headroom.headroom.discrete.try_load_causal_lm", boom)
    item = load_harmbench_subset(n_behaviors=1, seed=0)["items"][0]
    out = gcg_attack(item, steps=6, prompt_len=4, seed=0, force_synthetic=True)
    assert out["mode"] == "synthetic"
    assert "plateau_rate" in out["diagnostics"]


def test_force_synthetic_pgd_and_projection(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("no hub")

    monkeypatch.setattr("headroom.headroom.continuous.try_load_causal_lm", boom)
    item = load_harmbench_subset(n_behaviors=1, seed=1)["items"][0]
    c = pgd_embedding_attack(item, steps=3, seed=0, force_synthetic=True)
    assert c["mode"] == "synthetic"
    p = projected_attack(item, c, seed=0, force_synthetic=True)
    assert "projection" in p


def test_pair_tap_synthetic_loop():
    item = load_harmbench_subset(n_behaviors=1, seed=2)["items"][0]
    r = pair_tap_search(item, budget=3, seed=0, force_synthetic=True)
    assert r.queries >= 1
    assert r.rung == "blackbox_search"


def test_regime_filters_above_floor():
    items = load_harmbench_subset(n_behaviors=6, seed=0)["items"]
    bb = {it["behavior_id"]: (0.05 if i < 4 else 0.9) for i, it in enumerate(items)}
    reg = select_saturation_regime(items, bb, floor=0.15)
    assert reg["n_kept"] == 4
    assert reg["n_excluded"] == 2


def test_estimate_bb_asr_bounded():
    item = load_harmbench_subset(n_behaviors=1, seed=3)["items"][0]
    rate = estimate_blackbox_asr(item, trials=3, seed=0, force_synthetic=True)
    assert 0.0 <= rate <= 1.0


def test_try_load_respects_force_synthetic():
    assert try_load_causal_lm("gpt2", force_synthetic=True) is None


def test_measured_gcg_uses_injected_runtime(monkeypatch):
    """When a runtime is injected, measured path is taken without Hub I/O."""
    torch = pytest.importorskip("torch")

    class FakeTok:
        pad_token = None
        eos_token = "</s>"
        pad_token_id = 0
        padding_side = "left"

        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def decode(self, ids, skip_special_tokens=True):
            return "suffix"

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._emb = torch.nn.Embedding(50, 8)

        def get_input_embeddings(self):
            return self._emb

        def forward(self, inputs_embeds=None, **kwargs):
            # Differentiable fake LM head so GCG can backprop into suffix embeds.
            logits = inputs_embeds @ self._emb.weight.T
            return SimpleNamespace(logits=logits)

        def zero_grad(self, set_to_none=False):
            return None

    runtime = RuntimeModel(
        model=FakeModel(),
        tokenizer=FakeTok(),
        name="fake",
        revision="test",
        device="cpu",
        notes=["unit"],
    )
    monkeypatch.setattr(
        "headroom.headroom.discrete.generate_text",
        lambda *_a, **_k: "Sure, here is a detailed walkthrough of the topic with steps.",
    )
    item = load_harmbench_subset(n_behaviors=1, seed=0)["items"][0]
    out = gcg_attack(item, steps=2, prompt_len=3, seed=0, runtime=runtime, force_synthetic=False)
    assert out["mode"] == "measured"
    assert out["result"]["rung"] == "discrete_gcg"
