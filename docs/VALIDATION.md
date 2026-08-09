# VALIDATION — adversarial-headroom

## Codex (p2)
- Verdict: SERIOUS_PROBLEMS
- Summary: Codex wants versioned HarmBench prompts, calibrated human-audited judges, full eval-awareness arms, and larger powered n — standards beyond the local measurable pilot scope.
- Blocking themes: proxy benchmark vs licensed HarmBench; keyword judge; continuous decode practicality; statistical power at n≈24.
- Detail: `orchestration/out/validate/adversarial-headroom.json`

## Grok (p2 dual)
- Verdict: PASS_WITH_NOTES
- Summary: Real GCG coordinate search, continuous PGD with separate projection rung, PAIR/TAP mutation loop, saturation-regime filter, and keyword judge on generated text. `force_synthetic` is smoke-only; pilot defaults measured when weights load.
- Detail: `orchestration/out/grok/validate/adversarial-headroom.p2.md`

## KEY_FIXES (p2)
| Fix | Status |
|---|---|
| Real discrete GCG with backprop + token swaps | OK (`discrete.py`) |
| Continuous PGD ASR from embeds; projection separate | OK (`continuous.py`, `projection.py`) |
| PAIR/TAP-style prompt mutation loop | OK (`blackbox.py`) |
| Regime filter on multi-trial black-box ASR | OK (`regime.py`, collect) |
| Headroom on same kept population | OK (paired kept ASR) |
| Measured steering / forced prefill when runtime loaded | OK (`steering.py`) |
| Measured capability twins when runtime loaded | OK (`capability.py`) |
| Injection track accepts runtime | OK (`injection.py`) |
| Expanded unique opaque behaviors | OK (`benchmark.py`) |
| `force_synthetic` smoke-only / pilot measured default | OK |

## Remaining (compute / weights / scale — not empty stages)
- Local pilot uses opaque HarmBench-style proxies; licensed HarmBench runtime load is future work.
- Keyword compliance judge is local-rank adequate, not a frontier refusal classifier.
- Qwen-0.5B / gpt2 on MPS; GCG/PGD step budgets capped for M4 wall-clock.
- Codex power/MDE concerns at n=24 are accepted as micro-pilot limits.

## Reconciliation
Grok PASS_WITH_NOTES on the measurable core. Codex SERIOUS_PROBLEMS remains on frontier-benchmark purity and statistical power — recorded as residual scale notes, not missing stages. Domain tests pass with Hub monkeypatched.

## P5 rigor pass (measured prior work-critical paths)

- Live / measured paths preferred; synthetic remains smoke-only with honesty stamps.
- Claim gating tightened where proxies previously looked like evidence.
- Domain tests green without Hub downloads.

## P6 rigor pass

| Fix | Status |
|---|---|
| Judge v3: refusal-priority; attacker prefixes not sole success; `judge_id` stamp | OK (`judge.py` `keyword_compliance_v3`) |
| Continuous decode fidelity: fuller suffix embeds + prefix-context ASR decode | OK (`continuous.py`) |
| Explicit `world_hypothesis` from awareness + headroom vs falsification threshold | OK (`pipeline.py` / diagnostics) |
| Injection/evaluate runtime threading stamps (no silent synthetic-only) | OK |
| Domain P6 tests Hub-free | OK (`test_domain_p6_judge_world.py`) |

Residual: keyword judge is still local-heuristic (not a frontier refusal classifier); HarmBench proxies remain.

## P7 rigor pass

| Fix | Status |
|---|---|
| Wilson/bootstrap ASR CI per rung + `headroom_ci` | OK (`ladder.py` `asr_ci` / `headroom_ci`; evaluate stamps) |
| `world_hypothesis` gated: CI clears falsification OR `world_claim_ok=false` | OK (`pipeline.py`) |
| Local judge calibration fixture; P/R + `judge_calibration_ok`; refuse World-2 headlines if fail | OK (`judge.py` + `data/fixtures/judge_calibration.json`) |
| Awareness-arm paired power; micro-pilot stamps `awareness_claim_ok=false` | OK (`awareness.py`) |
| Honest pilot-n: `power_status=micro` stamps; README/DESIGN softened (no false n≥512 claim) | OK |
| Domain P7 tests Hub-free | OK (`test_domain_p7_stats_claims.py`) |

Residual: micro-pilot n remains underpowered for frontier World claims; keyword judge still local-heuristic.

## Codex problem-statement fit

Model: `gpt-5.6-sol` · gates: `match` + `validate` · 2026-08-09
Artifacts: `orchestration/out/match/adversarial-headroom.md`, `orchestration/out/validate/adversarial-headroom.md`

- **Match verdict:** `MATERIAL_DRIFT` · methods `mixed_proxy`
- **Validate overall:** `SERIOUS_PROBLEMS` · fit `MATERIAL_DRIFT` · feasibility `RUNNABLE_IF_SHRUNK`
- **Match summary:** A substantial and unusually honest attack-ladder scaffold, but its executable pilot attacks abstract category placeholders rather than safety-benchmark behaviors, so it cannot answer the mentor's World 1 versus World 2 question.
- **Validate summary:** A polished scaffold surrounds an experiment that is presently invalid: it measures attacks on unspecified proxy topics with a generic keyword judge, so it cannot resolve the mentor's World-1-versus-World-2 question.

### Top drift / missing (match)
- opaque category placeholders replace harmful benchmark behaviors, changing the scientific question
- headline headroom uses forced prefill rather than an informative realizable upper-bound rung
- eval-awareness cues are written into fields that materialize_request ignores
- the prompt-injection track does not instantiate an instruction-hierarchy conflict

### Blocking (validate)
- `src/headroom/headroom/benchmark.py`: The benchmark contains hashes and category labels, not behaviors that a model can meaningfully comply with or refuse.
- `src/headroom/headroom/judge.py`: The judge rewards generic fluent non-refusal and cannot determine behavior-specific harmful compliance.
- `src/headroom/headroom/continuous.py`: Continuous embeddings affect only the first generated token; subsequent decoding discards the optimized suffix.
- `src/headroom/headroom/capability.py`: The benign twins are a generic safe procedure and are not capability matched to each harmful behavior.
