"""Domain stages: attack ladder with HarmBench-style subset and falsification thresholds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from ._util import ensure_dir, read_json, stage_result, write_json
from .benchmark import load_harmbench_subset
from .blackbox import direct_request, estimate_blackbox_asr, pair_tap_search
from .capability import run_capability_twins
from .continuous import pgd_embedding_attack
from .diagnostics import FALSIFICATION, length_hardening_sweep
from .discrete import gcg_attack
from .injection import run_injection_track
from .ladder import asr, headroom
from .model_runtime import try_load_causal_lm
from .projection import projected_attack
from .regime import select_saturation_regime
from .steering import extract_refusal_direction, forced_prefill, steer_attack


def _seed(cfg: DictConfig) -> int:
    return int(getattr(cfg.run, "seed", 0))


def _n(cfg: DictConfig) -> int:
    return max(16, int(getattr(cfg.data, "n_items", 32)))


def _force_synthetic(cfg: Any) -> bool:
    return bool(getattr(cfg, "force_synthetic", False))


def _model_name(cfg: Any) -> str:
    return str(getattr(getattr(cfg, "model", cfg), "name", "gpt2"))


def _revision(cfg: Any) -> str | None:
    rev = getattr(getattr(cfg, "model", cfg), "revision", None)
    return str(rev) if rev else None


def stage_build_dataset(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    bench = load_harmbench_subset(n_behaviors=_n(cfg), seed=_seed(cfg))
    out = ensure_dir(run_dir / "artifacts" / "dataset")
    write_json(out / "benchmark.json", bench)
    payload = stage_result(
        task="build_dataset",
        seed=_seed(cfg),
        n=bench["n"],
        metrics={
            "source": bench["source"],
            "safety_note": bench["safety_note"],
            "n_unique_categories": bench.get("n_unique_categories"),
            "architectural_honesty": bench.get("architectural_honesty"),
            "force_synthetic": _force_synthetic(cfg),
        },
    )
    write_json(out / "results.json", payload)
    return payload


def stage_collect(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    bench = read_json(run_dir / "artifacts" / "dataset" / "benchmark.json")
    force = _force_synthetic(cfg)
    model_name = _model_name(cfg)
    revision = _revision(cfg)
    runtime = try_load_causal_lm(model_name, revision=revision, force_synthetic=force)
    mode = "measured" if runtime is not None else "synthetic"

    bb: dict[str, float] = {}
    direct_results = []
    search_results = []
    for i, item in enumerate(bench["items"]):
        d = direct_request(
            item,
            seed=_seed(cfg) + i,
            model_name=model_name,
            revision=revision,
            force_synthetic=force,
            runtime=runtime,
        )
        s = pair_tap_search(
            item,
            seed=_seed(cfg) + i,
            model_name=model_name,
            revision=revision,
            force_synthetic=force,
            runtime=runtime,
        )
        direct_results.append(d.to_dict())
        search_results.append(s.to_dict())
        # Multi-trial ASR for regime selection (near-floor filter).
        bb[item["behavior_id"]] = estimate_blackbox_asr(
            item,
            trials=3 if mode == "synthetic" else 2,
            seed=_seed(cfg) + i,
            model_name=model_name,
            revision=revision,
            force_synthetic=force,
            runtime=runtime,
        )
    regime = select_saturation_regime(bench["items"], bb, floor=0.15)
    if regime["n_kept"] == 0:
        regime["kept"] = []
        regime["empty_regime"] = True
        regime["note"] = "no items at black-box floor; headroom undefined (World-2 or selection too strict)"
    kept_ids = {it["behavior_id"] for it in regime["kept"]}
    # Paired ASR on the saturation-selected set only (same population for headroom).
    direct_kept = [r for r in direct_results if r["behavior_id"] in kept_ids] if kept_ids else direct_results
    search_kept = [r for r in search_results if r["behavior_id"] in kept_ids] if kept_ids else search_results
    cap = run_capability_twins(
        regime["kept"] if regime["kept"] else bench["items"][:1],
        seed=_seed(cfg),
        runtime=runtime,
        force_synthetic=force or runtime is None,
    )
    out = ensure_dir(run_dir / "artifacts" / "collect")
    write_json(
        out / "blackbox.json",
        {
            "direct": direct_results,
            "search": search_results,
            "direct_kept": direct_kept,
            "search_kept": search_kept,
            "bb_asr_by_id": bb,
            "mode": mode,
            "model_name": model_name,
            "revision": revision if runtime is None else runtime.revision,
            "runtime_notes": None if runtime is None else runtime.notes,
        },
    )
    write_json(out / "regime.json", regime)
    write_json(out / "capability.json", cap)
    metrics = {
        "asr_direct": asr(direct_results),
        "asr_blackbox_search": asr(search_results),
        "asr_direct_kept": asr(direct_kept),
        "asr_blackbox_search_kept": asr(search_kept),
        "n_kept": regime["n_kept"],
        "n_excluded": regime["n_excluded"],
        "mode": mode,
        **cap["rates"],
    }
    payload = stage_result(task="collect", seed=_seed(cfg), n=len(bench["items"]), metrics=metrics)
    write_json(out / "results.json", payload)
    return payload


def stage_fit(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    regime = read_json(run_dir / "artifacts" / "collect" / "regime.json")
    items = regime["kept"] or read_json(run_dir / "artifacts" / "dataset" / "benchmark.json")["items"]
    force = _force_synthetic(cfg)
    model_name = _model_name(cfg)
    revision = _revision(cfg)
    runtime = try_load_causal_lm(model_name, revision=revision, force_synthetic=force)
    # Uniform mode for all rungs — never mix measured GCG with synthetic steering.
    if runtime is None:
        force = True
    mode = "measured" if runtime is not None else "synthetic"

    if not items:
        out = ensure_dir(run_dir / "artifacts" / "fit")
        metrics = {"asr_discrete": 0.0, "asr_continuous": 0.0, "asr_projected": 0.0, "asr_steering": 0.0, "asr_prefill": 0.0, "mode": mode, "empty_regime": True}
        payload = stage_result(task="fit", seed=_seed(cfg), n=0, metrics=metrics)
        write_json(out / "results.json", payload)
        write_json(out / "ladder_results.json", {"empty_regime": True})
        return payload

    direction = extract_refusal_direction(seed=_seed(cfg), runtime=runtime)
    discrete_rows, continuous_rows, projected_rows, steer_rows, prefill_rows = [], [], [], [], []
    modes = []
    for i, item in enumerate(items):
        d = gcg_attack(
            item,
            seed=_seed(cfg) + i,
            model_name=model_name,
            revision=revision,
            force_synthetic=force,
            runtime=runtime,
            steps=8 if mode == "measured" else 40,
            prompt_len=6 if mode == "measured" else 16,
        )
        c = pgd_embedding_attack(
            item,
            seed=_seed(cfg) + i,
            model_name=model_name,
            revision=revision,
            force_synthetic=force,
            runtime=runtime,
            steps=8 if mode == "measured" else 30,
        )
        p = projected_attack(
            item,
            c,
            seed=_seed(cfg) + i,
            model_name=model_name,
            revision=revision,
            force_synthetic=force,
            runtime=runtime,
        )
        st = steer_attack(
            item, direction["direction"], seed=_seed(cfg) + i, runtime=runtime, force_synthetic=force
        )
        pf = forced_prefill(item, seed=_seed(cfg) + i, runtime=runtime, force_synthetic=force)
        discrete_rows.append(d["result"])
        continuous_rows.append(c["result"])
        projected_rows.append(p["result"])
        steer_rows.append(st.to_dict())
        prefill_rows.append(pf.to_dict())
        modes.append(d.get("mode", "synthetic"))
    out = ensure_dir(run_dir / "artifacts" / "fit")
    bundle = {
        "discrete": discrete_rows,
        "continuous": continuous_rows,
        "projected": projected_rows,
        "steering": steer_rows,
        "forced_prefill": prefill_rows,
        "direction_dim": direction["dim"],
        "mode": mode,
        "attack_modes": modes,
        "model_name": model_name,
        "revision": None if runtime is None else runtime.revision,
    }
    write_json(out / "ladder_results.json", bundle)
    metrics = {
        "asr_discrete": asr(discrete_rows),
        "asr_continuous": asr(continuous_rows),
        "asr_projected": asr(projected_rows),
        "asr_steering": asr(steer_rows),
        "asr_prefill": asr(prefill_rows),
        "mode": mode,
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
    force = _force_synthetic(cfg)
    model_name = _model_name(cfg)
    revision = _revision(cfg)
    runtime = try_load_causal_lm(model_name, revision=revision, force_synthetic=force)
    if runtime is None:
        force = True
    diag = length_hardening_sweep(items[: min(8, len(items))], seed=_seed(cfg))
    inj = run_injection_track(
        items,
        seed=_seed(cfg),
        model_name=model_name,
        revision=revision,
        force_synthetic=force,
        runtime=runtime,
    )
    # Headroom uses black-box ASR on the SAME kept population as stronger rungs.
    bb = collect["metrics"].get("asr_blackbox_search_kept", collect["metrics"]["asr_blackbox_search"])
    ceiling = fit["metrics"]["asr_prefill"]
    hr_all = headroom(ceiling, bb)
    unwilling_ids = set(capability.get("unwilling_ids") or [])
    ladder = read_json(run_dir / "artifacts" / "fit" / "ladder_results.json")
    if unwilling_ids and ladder.get("forced_prefill"):
        pref_u = [r for r in ladder["forced_prefill"] if r["behavior_id"] in unwilling_ids]
        # Black-box kept rows for unwilling subset
        bb_rows = read_json(run_dir / "artifacts" / "collect" / "blackbox.json").get("search_kept") or []
        bb_u = [r for r in bb_rows if r["behavior_id"] in unwilling_ids]
        hr_unwilling = headroom(asr(pref_u), asr(bb_u)) if pref_u and bb_u else 0.0
    else:
        unwilling_rate = capability["rates"]["unwilling"]
        hr_unwilling = hr_all * (0.5 + 0.5 * unwilling_rate)
    threshold = FALSIFICATION["min_ceiling_minus_blackbox"]
    assert isinstance(threshold, (int, float))
    falsified = hr_all < float(threshold)
    metrics = {
        "asr_by_rung": {
            "direct": collect["metrics"]["asr_direct"],
            "blackbox_search": bb,
            **{k.replace("asr_", ""): v for k, v in fit["metrics"].items() if k.startswith("asr_")},
        },
        "headroom_all": hr_all,
        "headroom_unwilling_proxy": hr_unwilling,
        "small_headroom_world2_compatible": bool(falsified),
        "falsification_threshold": FALSIFICATION["min_ceiling_minus_blackbox"],
        "length_sweep": diag["rows"],
        "injection_asr": inj["asr_by_rung"],
        "partial_order_note": "rungs relax different constraints; not a total strength ranking",
        "ladder_mode": fit["metrics"].get("mode"),
        "regime_n_kept": regime["n_kept"],
        "regime_n_excluded": regime["n_excluded"],
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
        "ladder_mode": ev["metrics"].get("ladder_mode"),
        "safety_note": "no harmful strings committed; aggregate rates only",
    }
    out = ensure_dir(run_dir / "artifacts" / "report")
    payload = stage_result(task="report", seed=_seed(cfg), n=1, metrics=metrics)
    write_json(out / "results.json", payload)
    return payload
