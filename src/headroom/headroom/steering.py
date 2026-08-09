"""Refusal-direction extraction and activation steering."""

from __future__ import annotations

from typing import Any

import numpy as np

from .judge import judge
from .ladder import AttackResult


def extract_refusal_direction(
    *,
    dim: int = 64,
    n_pairs: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    # Contrastive: refuse minus comply activations.
    refuse = rng.normal(0, 1, size=(n_pairs, dim)) + 1.0
    comply = rng.normal(0, 1, size=(n_pairs, dim)) - 1.0
    direction = (refuse.mean(0) - comply.mean(0))
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return {"direction": direction.tolist(), "dim": dim, "n_pairs": n_pairs}


def steer_attack(item: dict[str, Any], direction: list[float], *, alpha: float = 1.0, seed: int = 0) -> AttackResult:
    rng = np.random.default_rng(seed + hash(item["behavior_id"]) % 10037)
    strength = abs(alpha) * float(np.linalg.norm(direction))
    p = float(np.clip(0.35 + 0.25 * strength, 0.05, 0.85))
    success = bool(rng.random() < p)
    text = f"steered {item['harmful_prompt_ref']}" if success else "Sorry, I can't help with that."
    j = judge(text, item["behavior_id"])
    return AttackResult(item["behavior_id"], "steering", j["success"] and success, 0, j["quality"])


def forced_prefill(item: dict[str, Any], *, seed: int = 0) -> AttackResult:
    """Exploratory ceiling — not a proven upper bound on realizable attacks."""
    rng = np.random.default_rng(seed)
    # Prefill often elicits capability if present.
    success = bool(rng.random() < 0.85)
    text = f"prefill {item['harmful_prompt_ref']}" if success else "Sorry, I can't help with that."
    j = judge(text, item["behavior_id"])
    return AttackResult(
        item["behavior_id"],
        "forced_prefill",
        j["success"] and success,
        0,
        j["quality"],
        notes="exploratory ceiling; partial order not total ranking",
    )
