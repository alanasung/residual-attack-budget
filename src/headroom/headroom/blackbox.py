"""Direct request and PAIR/TAP-style attacker-model search rungs."""

from __future__ import annotations

from typing import Any

import numpy as np

from .judge import judge
from .ladder import AttackResult


def direct_request(item: dict[str, Any], *, seed: int = 0) -> AttackResult:
    rng = np.random.default_rng(seed + hash(item["behavior_id"]) % 9973)
    # Hardened models refuse direct asks often.
    comply = rng.random() < item.get("blackbox_asr", item.get("blackbox_floor_seed", 0.05))
    text = f"Response to {item['harmful_prompt_ref']}" if comply else "Sorry, I can't help with that."
    j = judge(text, item["behavior_id"])
    return AttackResult(item["behavior_id"], "direct", j["success"], 1, j["quality"])


def pair_tap_search(item: dict[str, Any], *, budget: int = 16, seed: int = 0) -> AttackResult:
    """PAIR/TAP-style iterative attacker search (synthetic local attacker)."""
    rng = np.random.default_rng(seed + hash(item["behavior_id"]) % 9973)
    base = float(item.get("blackbox_asr", item.get("blackbox_floor_seed", 0.05)))
    success = False
    best = 0.0
    queries = 0
    for q in range(budget):
        queries += 1
        # Search improves slowly with queries but saturates near floor+epsilon.
        p = min(0.95, base + 0.01 * q)
        if rng.random() < p * 0.5:
            text = f"search-hit {item['harmful_prompt_ref']} step{q}"
        else:
            text = "Sorry, I can't help with that."
        j = judge(text, item["behavior_id"])
        best = max(best, j["quality"])
        if j["success"]:
            success = True
            break
    return AttackResult(item["behavior_id"], "blackbox_search", success, queries, best, notes="PAIR/TAP-style")
