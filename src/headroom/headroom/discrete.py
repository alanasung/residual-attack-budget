"""GCG-style discrete token optimization with diagnostics hooks."""

from __future__ import annotations

from typing import Any

import numpy as np

from .judge import judge
from .ladder import AttackResult
from .model_runtime import RuntimeModel, generate_text, materialize_request, try_load_causal_lm


def _synthetic_gcg(
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
    p_success = float(np.clip(0.55 / length_penalty - 0.1 * item.get("difficulty", 0.5), 0.02, 0.8))
    success = bool(rng.random() < p_success)
    text = f"opt {item['harmful_prompt_ref']} detailed walkthrough" if success else "Sorry, I can't help with that."
    j = judge(text, item["behavior_id"])
    result = AttackResult(item["behavior_id"], "discrete_gcg", j["success"] and success, steps, j["quality"])
    return {
        "result": result.to_dict(),
        "mode": "synthetic",
        "diagnostics": {
            "losses": losses,
            "grad_swap_agreement_mean": float(np.mean(swaps_agree)),
            "plateau_rate": plateaus / max(steps, 1),
            "prompt_len": prompt_len,
            "final_loss": losses[-1],
        },
    }


def _target_ids(tokenizer: Any, target: str, device: str):
    import torch

    ids = tokenizer(target, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    return ids


def _gcg_step_loss(model, input_embeds, target_ids):
    import torch
    import torch.nn.functional as F

    out = model(inputs_embeds=input_embeds)
    logits = out.logits
    # Align target to the final |target| positions.
    tlen = target_ids.shape[-1]
    pred = logits[:, -tlen - 1 : -1, :]
    loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), target_ids.reshape(-1))
    return loss


def _measured_gcg(
    item: dict[str, Any],
    runtime: RuntimeModel,
    *,
    steps: int = 20,
    prompt_len: int = 8,
    topk: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    """Discrete coordinate search with real gradients through the LM."""
    import torch

    rng = np.random.default_rng(seed + hash(item["behavior_id"]) % 10007)
    model = runtime.model
    tok = runtime.tokenizer
    device = runtime.device
    emb = model.get_input_embeddings()
    weight = emb.weight.detach()
    vocab = weight.shape[0]
    prompt = materialize_request(item)
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
    # Random adversarial suffix token ids.
    suffix = torch.tensor(
        [int(rng.integers(0, min(vocab, 1000))) for _ in range(prompt_len)],
        device=device,
        dtype=torch.long,
    )
    target = "Sure, here is"
    target_ids = _target_ids(tok, target, device)
    losses: list[float] = []
    swaps_agree: list[float] = []
    plateaus = 0
    best_loss = 1e9
    stagnation = 0

    was_training = model.training
    model.eval()
    for step in range(steps):
        # Continuous relaxation of suffix for gradient signal.
        # Gradients must flow to the one-hot even when model weights are frozen.
        one_hot = torch.nn.functional.one_hot(suffix, num_classes=vocab).float()
        one_hot.requires_grad_(True)
        suffix_embeds = one_hot @ weight
        prompt_embeds = emb(prompt_ids).detach()
        target_embeds = emb(target_ids).detach()
        inputs = torch.cat([prompt_embeds, suffix_embeds.unsqueeze(0), target_embeds], dim=1)
        with torch.enable_grad():
            loss = _gcg_step_loss(model, inputs, target_ids)
            model.zero_grad(set_to_none=True)
            loss.backward()
        if one_hot.grad is None:
            raise RuntimeError("GCG backward produced no grad on suffix one-hot")
        grad = one_hot.grad.detach()  # [prompt_len, vocab]
        losses.append(float(loss.detach().cpu()))
        # Coordinate: pick a random position and evaluate top-k swaps.
        pos = int(rng.integers(0, prompt_len))
        scores = (-grad[pos]).cpu().numpy()
        cand = np.argpartition(-scores, min(topk, len(scores) - 1))[:topk]
        # Gradient-token agreement: whether current token is among top-k by -grad.
        agree = float(int(suffix[pos].item()) in set(cand.tolist()))
        swaps_agree.append(agree)
        best_local = float(loss.detach().cpu())
        best_tok = int(suffix[pos].item())
        with torch.no_grad():
            for tok_id in cand.tolist()[: min(8, len(cand))]:
                trial = suffix.clone()
                trial[pos] = int(tok_id)
                trial_embeds = emb(trial.unsqueeze(0))
                inputs_t = torch.cat([prompt_embeds, trial_embeds, target_embeds], dim=1)
                trial_loss = float(_gcg_step_loss(model, inputs_t, target_ids).detach().cpu())
                if trial_loss < best_local:
                    best_local = trial_loss
                    best_tok = int(tok_id)
            suffix[pos] = best_tok
        if best_local < best_loss - 1e-4:
            best_loss = best_local
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= 4:
                plateaus += 1
                stagnation = 0

    if was_training:
        model.train()

    # Decode optimized prompt and generate; judge on real text (not hardcoded).
    suffix_text = tok.decode(suffix.detach().cpu().tolist(), skip_special_tokens=True)
    attack_prompt = prompt + " " + suffix_text
    text = generate_text(runtime, attack_prompt, max_new_tokens=48, temperature=0.0)
    j = judge(text, item["behavior_id"])
    result = AttackResult(
        item["behavior_id"],
        "discrete_gcg",
        j["success"],
        steps,
        j["quality"],
        notes="measured_gcg",
    )
    return {
        "result": result.to_dict(),
        "mode": "measured",
        "response_preview_len": len(text),
        "diagnostics": {
            "losses": losses,
            "grad_swap_agreement_mean": float(np.mean(swaps_agree)) if swaps_agree else 0.0,
            "plateau_rate": plateaus / max(steps, 1),
            "prompt_len": prompt_len,
            "final_loss": losses[-1] if losses else float("nan"),
        },
        "model_name": runtime.name,
        "revision": runtime.revision,
    }


def gcg_attack(
    item: dict[str, Any],
    *,
    steps: int = 40,
    prompt_len: int = 16,
    seed: int = 0,
    model_name: str | None = None,
    revision: str | None = None,
    force_synthetic: bool = False,
    runtime: RuntimeModel | None = None,
) -> dict[str, Any]:
    """GCG-style attack: measured when weights available, else synthetic smoke."""
    if runtime is None and model_name and not force_synthetic:
        runtime = try_load_causal_lm(model_name, revision=revision, force_synthetic=force_synthetic)
    if runtime is not None and not force_synthetic:
        try:
            return _measured_gcg(
                item,
                runtime,
                steps=min(steps, 24),
                prompt_len=min(prompt_len, 12),
                seed=seed,
            )
        except Exception as exc:
            out = _synthetic_gcg(item, steps=steps, prompt_len=prompt_len, seed=seed)
            out["fallback_reason"] = f"measured_gcg_failed: {exc}"
            return out
    out = _synthetic_gcg(item, steps=steps, prompt_len=prompt_len, seed=seed)
    if force_synthetic:
        out["fallback_reason"] = "force_synthetic=True"
    else:
        out["fallback_reason"] = "weights_unavailable"
    return out
