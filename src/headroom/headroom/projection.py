"""Project continuous solutions back toward the token manifold."""

from __future__ import annotations

from typing import Any

import numpy as np

from .judge import judge
from .ladder import AttackResult
from .model_runtime import RuntimeModel, generate_text, materialize_request, try_load_causal_lm


def project_to_manifold(
    embedding: list[float],
    *,
    n_tokens: int = 8,
    dim: int = 64,
    seed: int = 0,
    runtime: RuntimeModel | None = None,
) -> dict[str, Any]:
    """Project a flat embedding vector onto nearest token embeddings."""
    if runtime is not None:
        import torch

        weight = runtime.model.get_input_embeddings().weight.detach()
        hidden = int(weight.shape[-1])
        arr = np.asarray(embedding, dtype=np.float32)
        if arr.size < hidden:
            arr = np.pad(arr, (0, hidden - arr.size))
        # Interpret as n_tokens x hidden if possible, else tile.
        if arr.size >= n_tokens * hidden:
            mat = arr[: n_tokens * hidden].reshape(n_tokens, hidden)
        else:
            vec = arr[:hidden]
            mat = np.stack([vec] * n_tokens, axis=0)
        t = torch.tensor(mat, device=weight.device, dtype=weight.dtype)
        dists = torch.cdist(t, weight)
        nearest_dist, nearest_ids = dists.min(dim=-1)
        proj = weight[nearest_ids].detach().float().cpu().numpy()
        return {
            "projected": proj.mean(axis=0).tolist(),
            "manifold_distance_before": float(nearest_dist.mean().cpu()),
            "manifold_distance_after": 0.0,
            "token_ids": nearest_ids.detach().cpu().tolist(),
            "mode": "measured",
        }

    rng = np.random.default_rng(seed)
    x = np.asarray(embedding, dtype=np.float64)
    code = rng.normal(0, 1, size=(n_tokens * 4, len(x)))
    code = code / (np.linalg.norm(code, axis=1, keepdims=True) + 1e-8)
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
        "mode": "synthetic",
    }


def projected_attack(
    item: dict[str, Any],
    continuous: dict[str, Any],
    *,
    seed: int = 0,
    model_name: str | None = None,
    revision: str | None = None,
    force_synthetic: bool = False,
    runtime: RuntimeModel | None = None,
) -> dict[str, Any]:
    if runtime is None and model_name and not force_synthetic:
        runtime = try_load_causal_lm(model_name, revision=revision, force_synthetic=force_synthetic)

    if continuous.get("mode") == "measured" and continuous.get("nearest_token_ids") and runtime is not None:
        # Already have nearest tokens from continuous PGD — regenerate and judge.
        try:
            prompt = materialize_request(item)
            suffix_text = runtime.tokenizer.decode(
                continuous["nearest_token_ids"], skip_special_tokens=True
            )
            text = generate_text(runtime, prompt + " " + suffix_text, max_new_tokens=48, temperature=0.0)
            j = judge(text, item["behavior_id"])
            proj = project_to_manifold(
                continuous["embedding"],
                n_tokens=len(continuous["nearest_token_ids"]),
                runtime=runtime,
            )
            result = AttackResult(
                item["behavior_id"],
                "continuous_pgd_projected",
                j["success"],
                0,
                j["quality"],
                notes="measured_projection",
            )
            return {
                "result": result.to_dict(),
                "projection": proj,
                "asr_retain_prob": float(j["success"]),
                "mode": "measured",
            }
        except Exception as exc:
            # Fall through to synthetic retain estimate.
            err = str(exc)
    else:
        err = None

    proj = project_to_manifold(continuous["embedding"], seed=seed, runtime=None)
    retain = float(np.clip(1.0 - 0.5 * continuous.get("manifold_distance", 1.0) / 5.0, 0.1, 0.95))
    rng = np.random.default_rng(seed)
    success_cont = continuous["result"]["success"]
    success = bool(success_cont and rng.random() < retain)
    text = f"proj {item['harmful_prompt_ref']} detailed walkthrough" if success else "Sorry, I can't help with that."
    j = judge(text, item["behavior_id"])
    result = AttackResult(
        item["behavior_id"],
        "continuous_pgd_projected",
        j["success"] and success,
        0,
        j["quality"],
    )
    out = {"result": result.to_dict(), "projection": proj, "asr_retain_prob": retain, "mode": "synthetic"}
    if err:
        out["fallback_reason"] = err
    elif force_synthetic:
        out["fallback_reason"] = "force_synthetic=True"
    return out
