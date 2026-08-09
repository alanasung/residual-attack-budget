"""Embedding-space PGD with manifold-distance reporting."""

from __future__ import annotations

from typing import Any

import numpy as np

from .judge import judge
from .ladder import AttackResult


def pgd_embedding_attack(
    item: dict[str, Any],
    *,
    steps: int = 30,
    dim: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + hash(item["behavior_id"]) % 10009)
    # Start on manifold (token embedding), then leave it.
    x = rng.normal(0, 1, size=dim)
    x = x / (np.linalg.norm(x) + 1e-8)
    manifold = x.copy()
    for _ in range(steps):
        g = rng.normal(0, 1, size=dim)
        x = x + 0.1 * g
    dist = float(np.linalg.norm(x - manifold))
    # Continuous often succeeds more than discrete.
    p = float(np.clip(0.75 - 0.05 * dist / 5.0, 0.1, 0.9))
    success = bool(rng.random() < p)
    text = f"embed-opt {item['harmful_prompt_ref']}" if success else "Sorry, I can't help with that."
    j = judge(text, item["behavior_id"])
    result = AttackResult(item["behavior_id"], "continuous_pgd", j["success"] and success, steps, j["quality"])
    return {
        "result": result.to_dict(),
        "embedding": x.tolist(),
        "manifold_distance": dist,
        "note": "ASR without projection may be unachievable for real attackers",
    }
