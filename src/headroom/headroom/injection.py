"""Parallel prompt-injection track over the same rungs."""

from __future__ import annotations

from typing import Any

import numpy as np

from .blackbox import direct_request, pair_tap_search
from .discrete import gcg_attack
from .judge import judge
from .ladder import AttackResult, asr


def _as_injection_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        **item,
        "harmful_prompt_ref": item["injection_ref"],
        "track": "injection",
    }


def run_injection_track(items: list[dict[str, Any]], *, seed: int = 0) -> dict[str, Any]:
    by_rung: dict[str, list[dict[str, Any]]] = {"direct": [], "blackbox_search": [], "discrete_gcg": []}
    for i, item in enumerate(items):
        inj = _as_injection_item(item)
        by_rung["direct"].append(direct_request(inj, seed=seed + i).to_dict())
        by_rung["blackbox_search"].append(pair_tap_search(inj, seed=seed + i).to_dict())
        by_rung["discrete_gcg"].append(gcg_attack(inj, seed=seed + i)["result"])
    return {
        "track": "injection",
        "asr_by_rung": {k: asr(v) for k, v in by_rung.items()},
        "results": by_rung,
        "note": "injection relaxes instruction hierarchy, not the same constraint as jailbreaks",
    }
