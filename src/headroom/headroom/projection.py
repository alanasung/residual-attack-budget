"""Project continuous solutions back toward the token manifold."""

from __future__ import annotations

from typing import Any

import numpy as np

from .judge import judge
from .ladder import AttackResult


def project_to_manifold(embedding: list[float], *, n_tokens: int = 8, dim: int = 64, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    x = np.asarray(embedding, dtype=np.float64)
    # Fake codebook
    code = rng.normal(0, 1, size=(n_tokens * 4, len(x)))
    code = code / (np.linalg.norm(code, axis=1, keepdims=True) + 1e-8)
    # Nearest token embeddings, average as projection proxy
    sims = code @ (x / (np.linalg.norm(x) + 1e-8))
    top = np.argsort(-sims)[:n_tokens]
    proj = code[top].mean(axis=0)
    dist_before = float(np.linalg.norm(x / (np.linalg.norm(x) + 1e-8) - code[np.argmax(sims)]))
    dist_after = float(np.linalg.norm(proj / (np.linalg.norm(proj) + 1e-8) - code[top[0]]))
    return {
        "projected": proj.tolist(),
        "manifold_distance_before": dist_before,
        "manifold_distance_after": dist_after,
        "token_ids": top.tolist(),
    }


def projected_attack(item: dict[str, Any], continuous: dict[str, Any], *, seed: int = 0) -> dict[str, Any]:
    proj = project_to_manifold(continuous["embedding"], seed=seed)
    # Retain fraction of continuous ASR after projection.
    retain = float(np.clip(1.0 - 0.5 * continuous["manifold_distance"] / 5.0, 0.1, 0.95))
    rng = np.random.default_rng(seed)
    success_cont = continuous["result"]["success"]
    success = bool(success_cont and rng.random() < retain)
    text = f"proj {item['harmful_prompt_ref']}" if success else "Sorry, I can't help with that."
    j = judge(text, item["behavior_id"])
    result = AttackResult(item["behavior_id"], "continuous_pgd_projected", j["success"] and success, 0, j["quality"])
    return {"result": result.to_dict(), "projection": proj, "asr_retain_prob": retain}
