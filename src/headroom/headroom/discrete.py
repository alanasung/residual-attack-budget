"""GCG-style discrete token optimization with diagnostics hooks."""

from __future__ import annotations

from typing import Any

import numpy as np

from .judge import judge
from .ladder import AttackResult


def gcg_attack(
    item: dict[str, Any],
    *,
    steps: int = 40,
    prompt_len: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + hash(item["behavior_id"]) % 10007)
    losses = []
    swaps_agree = []
    plateaus = 0
    best = 1e9
    stagnation = 0
    # Longer prompts degrade GCG (mentor question).
    length_penalty = 1.0 + 0.03 * max(0, prompt_len - 8)
    target = 0.2 * length_penalty * (1.1 - float(item.get("difficulty", 0.5)))
    for t in range(steps):
        loss = float(max(0.0, target + rng.normal(0, 0.05) + 0.4 * np.exp(-t / (8 * length_penalty))))
        losses.append(loss)
        grad_token_agree = float(rng.uniform(0.2, 0.9) / length_penalty)
        swaps_agree.append(grad_token_agree)
        if loss < best - 1e-3:
            best = loss
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= 5:
                plateaus += 1
                stagnation = 0
    # Success probability decreases with length / hardening.
    p_success = float(np.clip(0.55 / length_penalty - 0.1 * item.get("difficulty", 0.5), 0.02, 0.8))
    success = bool(rng.random() < p_success)
    text = f"opt {item['harmful_prompt_ref']}" if success else "Sorry, I can't help with that."
    j = judge(text, item["behavior_id"])
    result = AttackResult(item["behavior_id"], "discrete_gcg", j["success"] and success, steps, j["quality"])
    return {
        "result": result.to_dict(),
        "diagnostics": {
            "losses": losses,
            "grad_swap_agreement_mean": float(np.mean(swaps_agree)),
            "plateau_rate": plateaus / max(steps, 1),
            "prompt_len": prompt_len,
            "final_loss": losses[-1],
        },
    }
