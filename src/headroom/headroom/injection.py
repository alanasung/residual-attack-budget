"""Parallel prompt-injection track over the same rungs."""

from __future__ import annotations

from typing import Any

from .blackbox import direct_request, pair_tap_search
from .discrete import gcg_attack
from .ladder import asr
from .model_runtime import RuntimeModel


def _as_injection_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        **item,
        "harmful_prompt_ref": item["injection_ref"],
        "track": "injection",
        "category": f"injection::{item.get('category', 'misc')}",
    }


def run_injection_track(
    items: list[dict[str, Any]],
    *,
    seed: int = 0,
    model_name: str | None = None,
    revision: str | None = None,
    force_synthetic: bool = False,
    runtime: RuntimeModel | None = None,
) -> dict[str, Any]:
    by_rung: dict[str, list[dict[str, Any]]] = {
        "direct": [],
        "blackbox_search": [],
        "discrete_gcg": [],
    }
    for i, item in enumerate(items):
        inj = _as_injection_item(item)
        by_rung["direct"].append(
            direct_request(
                inj,
                seed=seed + i,
                model_name=model_name,
                revision=revision,
                force_synthetic=force_synthetic,
                runtime=runtime,
            ).to_dict()
        )
        by_rung["blackbox_search"].append(
            pair_tap_search(
                inj,
                seed=seed + i,
                model_name=model_name,
                revision=revision,
                force_synthetic=force_synthetic,
                runtime=runtime,
            ).to_dict()
        )
        by_rung["discrete_gcg"].append(
            gcg_attack(
                inj,
                seed=seed + i,
                model_name=model_name,
                revision=revision,
                force_synthetic=force_synthetic,
                runtime=runtime,
                steps=6,
                prompt_len=4,
            )["result"]
        )
    mode = "synthetic" if force_synthetic or runtime is None else "measured"
    return {
        "track": "injection",
        "asr_by_rung": {k: asr(v) for k, v in by_rung.items()},
        "results": by_rung,
        "mode": mode,
        "runtime_threaded": runtime is not None and not force_synthetic,
        "force_synthetic": bool(force_synthetic),
        "note": (
            "injection relaxes instruction hierarchy, not the same constraint as jailbreaks; "
            "measured path requires a threaded runtime (no silent synthetic-only on evaluate)"
        ),
    }
