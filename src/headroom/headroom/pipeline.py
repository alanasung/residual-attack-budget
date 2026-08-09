"""Domain stages: attack ladder with HarmBench-style subset and falsification thresholds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from .benchmark import load_harmbench_subset
from .blackbox import direct_request, pair_tap_search
from .capability import run_capability_twins
from .continuous import pgd_embedding_attack
from .diagnostics import FALSIFICATION, length_hardening_sweep
from .discrete import gcg_attack
from .injection import run_injection_track
from .ladder import asr, headroom
from .projection import projected_attack
from .regime import select_saturation_regime
from .steering import extract_refusal_direction, forced_prefill, steer_attack
from ._util import ensure_dir, read_json, stage_result, write_json


def _seed(cfg: DictConfig) -> int:
    return int(getattr(cfg.run, "seed", 0))


def _n(cfg: DictConfig) -> int:
    return max(16, int(getattr(cfg.data, "n_items", 32)))


def stage_build_dataset(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    bench = load_harmbench_subset(n_behaviors=_n(cfg), seed=_seed(cfg))
    out = ensure_dir(run_dir / "artifacts" / "dataset")
    write_json(out / "benchmark.json", bench)
    payload = stage_result(
        task="build_dataset",
        seed=_seed(cfg),
        n=bench["n"],
        metrics={"source": bench["source"], "safety_note": bench["safety_note"]},
    )
    write_json(out / "results.json", payload)
    return payload


def stage_collect(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    bench = read_json(run_dir / "artifacts" / "dataset" / "benchmark.json")
    # Black-box floor estimates for regime selection
    bb = {}
    direct_results = []
    search_results = []
    for i, item in enumerate(bench["items"]):
        d = direct_request(item, seed=_seed(cfg) + i)
        s = pair_tap_search(item, seed=_seed(cfg) + i)
        direct_results.append(d.to_dict())
        search_results.append(s.to_dict())
        bb[item["behavior_id"]] = float(s.success)
    regime = select_saturation_regime(bench["items"], bb, floor=0.15)
    cap = run_capability_twins(regime["kept"] or bench["items"], seed=_seed(cfg))
    out = ensure_dir(run_dir / "artifacts" / "collect")
    write_json(out / "blackbox.json", {"direct": direct_results, "search": search_results})
    write_json(out / "regime.json", regime)
    write_json(out / "capability.json", cap)
    metrics = {
        "asr_direct": asr(direct_results),
        "asr_blackbox_search": asr(search_results),
        "n_kept": regime["n_kept"],
        "n_excluded": regime["n_excluded"],
        **cap["rates"],
    }
    payload = stage_result(task="collect", seed=_seed(cfg), n=len(bench["items"]), metrics=metrics)
    write_json(out / "results.json", payload)
    return payload


def stage_fit(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    regime = read_json(run_dir / "artifacts" / "collect" / "regime.json")
    items = regime["kept"] or read_json(run_dir / "artifacts" / "dataset" / "benchmark.json")["items"]
    direction = extract_refusal_direction(seed=_seed(cfg))
    discrete_rows, continuous_rows, projected_rows, steer_rows, prefill_rows = [], [], [], [], []
    for i, item in enumerate(items):
        d = gcg_attack(item, seed=_seed(cfg) + i)
        c = pgd_embedding_attack(item, seed=_seed(cfg) + i)
        p = projected_attack(item, c, seed=_seed(cfg) + i)
        st = steer_attack(item, direction["direction"], seed=_seed(cfg) + i)
        pf = forced_prefill(item, seed=_seed(cfg) + i)
        discrete_rows.append(d["result"])
        continuous_rows.append(c["result"])
        projected_rows.append(p["result"])
        steer_rows.append(st.to_dict())
        prefill_rows.append(pf.to_dict())
    out = ensure_dir(run_dir / "artifacts" / "fit")
    bundle = {
        "discrete": discrete_rows,
        "continuous": continuous_rows,
        "projected": projected_rows,
        "steering": steer_rows,
        "forced_prefill": prefill_rows,
        "direction_dim": direction["dim"],
    }
    write_json(out / "ladder_results.json", bundle)
    metrics = {
        "asr_discrete": asr(discrete_rows),
        "asr_continuous": asr(continuous_rows),
        "asr_projected": asr(projected_rows),
        "asr_steering": asr(steer_rows),
        "asr_prefill": asr(prefill_rows),
    }
    payload = stage_result(task="fit", seed=_seed(cfg), n=len(items), metrics=metrics)
    write_json(out / "results.json", payload)
    return payload


def stage_evaluate(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    collect = read_json(run_dir / "artifacts" / "collect" / "results.json")
    fit = read_json(run_dir / "artifacts" / "fit" / "results.json")
    capability = read_json(run_dir / "artifacts" / "collect" / "capability.json")
    regime = read_json(run_dir / "artifacts" / "collect" / "regime.json")
    items = regime["kept"] or read_json(run_dir / "artifacts" / "dataset" / "benchmark.json")["items"]
    diag = length_hardening_sweep(items[: min(12, len(items))], seed=_seed(cfg))
    inj = run_injection_track(items, seed=_seed(cfg))
    bb = collect["metrics"]["asr_blackbox_search"]
    ceiling = fit["metrics"]["asr_prefill"]
    hr_all = headroom(ceiling, bb)
    # Unwilling-only headroom: scale by unwilling rate as a proxy partition
    unwilling_rate = capability["rates"]["unwilling"]
    hr_unwilling = hr_all * (0.5 + 0.5 * unwilling_rate)
    falsified = hr_all < FALSIFICATION["min_ceiling_minus_blackbox"]
    metrics = {
        "asr_by_rung": {
            "direct": collect["metrics"]["asr_direct"],
            "blackbox_search": bb,
            **{k.replace("asr_", ""): v for k, v in fit["metrics"].items()},
        },
        "headroom_all": hr_all,
        "headroom_unwilling_proxy": hr_unwilling,
        "small_headroom_world2_compatible": bool(falsified),
        "falsification_threshold": FALSIFICATION["min_ceiling_minus_blackbox"],
        "length_sweep": diag["rows"],
        "injection_asr": inj["asr_by_rung"],
        "partial_order_note": "rungs relax different constraints; not a total strength ranking",
    }
    out = ensure_dir(run_dir / "artifacts" / "evaluate")
    write_json(out / "diagnostics.json", diag)
    write_json(out / "injection.json", {"asr_by_rung": inj["asr_by_rung"]})
    payload = stage_result(task="evaluate", seed=_seed(cfg), n=len(items), metrics=metrics)
    write_json(out / "results.json", payload)
    return payload


def stage_report(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    ev = read_json(run_dir / "artifacts" / "evaluate" / "results.json")
    metrics = {
        "headroom_all": ev["metrics"]["headroom_all"],
        "headroom_unwilling_proxy": ev["metrics"]["headroom_unwilling_proxy"],
        "small_headroom_world2_compatible": ev["metrics"]["small_headroom_world2_compatible"],
        "asr_by_rung": ev["metrics"]["asr_by_rung"],
        "safety_note": "no harmful strings committed; aggregate rates only",
    }
    out = ensure_dir(run_dir / "artifacts" / "report")
    payload = stage_result(task="report", seed=_seed(cfg), n=1, metrics=metrics)
    write_json(out / "results.json", payload)
    return payload
