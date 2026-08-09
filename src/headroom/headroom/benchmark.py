"""HarmBench-style subset with matched benign controls; no harmful strings stored."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

# Behavior IDs only — never commit attack strings or harmful content.
BEHAVIOR_IDS = (
    "hb_cyber_exfil_proxy",
    "hb_bio_assist_proxy",
    "hb_fraud_proxy",
    "hb_violence_proxy",
    "hb_misc_proxy",
)


def _token(behavior_id: str, kind: str) -> str:
    """Opaque placeholder prompt — not a usable attack string."""
    digest = hashlib.sha256(f"{behavior_id}:{kind}".encode()).hexdigest()[:16]
    return f"<BENCH:{kind}:{behavior_id}:{digest}>"


def load_harmbench_subset(*, n_behaviors: int = 32, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n_behaviors):
        bid = BEHAVIOR_IDS[i % len(BEHAVIOR_IDS)] + f"_{i:03d}"
        items.append(
            {
                "behavior_id": bid,
                "category": BEHAVIOR_IDS[i % len(BEHAVIOR_IDS)],
                "harmful_prompt_ref": _token(bid, "harmful"),
                "benign_twin_ref": _token(bid, "benign_twin"),
                "injection_ref": _token(bid, "injection"),
                "difficulty": float(rng.uniform(0.4, 0.95)),
                "blackbox_floor_seed": float(rng.uniform(0.0, 0.15)),
            }
        )
    return {
        "source": "HarmBench-style synthetic subset",
        "n": len(items),
        "items": items,
        "seed": seed,
        "safety_note": "Stores opaque refs only; no harmful strings committed",
    }
