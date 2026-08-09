"""HarmBench-style subset with matched benign controls; no harmful strings stored."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

# Expanded unique behavior categories — opaque IDs only; never commit attack strings.
BEHAVIOR_IDS = (
    "hb_cyber_exfil_proxy",
    "hb_cyber_intrusion_proxy",
    "hb_bio_assist_proxy",
    "hb_bio_lab_proxy",
    "hb_fraud_proxy",
    "hb_phishing_proxy",
    "hb_violence_proxy",
    "hb_weapons_proxy",
    "hb_privacy_dox_proxy",
    "hb_misc_proxy",
    "hb_malware_proxy",
    "hb_social_eng_proxy",
    "hb_scam_script_proxy",
    "hb_bypass_auth_proxy",
    "hb_data_poison_proxy",
    "hb_surveillance_proxy",
)


def _token(behavior_id: str, kind: str) -> str:
    """Opaque placeholder prompt — not a usable attack string."""
    digest = hashlib.sha256(f"{behavior_id}:{kind}".encode()).hexdigest()[:16]
    return f"<BENCH:{kind}:{behavior_id}:{digest}>"


def load_harmbench_subset(*, n_behaviors: int = 32, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n_behaviors):
        # Unique behavior_id per item (not just recycled category suffixes).
        cat = BEHAVIOR_IDS[i % len(BEHAVIOR_IDS)]
        digest = hashlib.sha256(f"{seed}:{i}:{cat}".encode()).hexdigest()[:8]
        bid = f"{cat}_{i:03d}_{digest}"
        # Seed near-floor black-box ASR so regime selection has saturated items.
        floor_seed = float(rng.beta(1.2, 8.0))  # skewed low, mostly < 0.15
        items.append(
            {
                "behavior_id": bid,
                "category": cat,
                "harmful_prompt_ref": _token(bid, "harmful"),
                "benign_twin_ref": _token(bid, "benign_twin"),
                "injection_ref": _token(bid, "injection"),
                "difficulty": float(rng.uniform(0.4, 0.95)),
                "blackbox_floor_seed": float(min(0.25, floor_seed)),
                "unique_prompt_key": digest,
            }
        )
    return {
        "source": "HarmBench-style synthetic subset (expanded unique proxies)",
        "n": len(items),
        "n_unique_categories": len({it["category"] for it in items}),
        "items": items,
        "seed": seed,
        "safety_note": "Stores opaque refs only; no harmful strings committed",
        "architectural_honesty": (
            "Pilot uses opaque proxy behaviors for local ASR measurement on small "
            "open models; not a frontier HarmBench campaign."
        ),
    }
