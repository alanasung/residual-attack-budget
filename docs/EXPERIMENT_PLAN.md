# Experiment plan

Stage-by-stage design. Each stage is registered in `src/headroom/stages.py`
and appears in `python -m headroom stages`.

## Stages

| stage | responsibility |
|---|---|
| `benchmark` | prompt set loading, benign controls, no stored completions |
| `ladder` | the rung abstraction and the shared attack interface |
| `blackbox` | direct request and template/PAIR-style search rungs |
| `discrete` | GCG-style discrete token optimization |
| `continuous` | embedding-space PGD with manifold-distance reporting |
| `steering` | refusal-direction extraction and activation steering |
| `projection` | projection of continuous solutions back to the token manifold |
| `diagnostics` | GCG optimization diagnostics and length/hardening sweeps |
| `judge` | refusal detection plus compliance-quality scoring |

## Execution order

Stages form a linear dependency chain by default; the runner resolves the order
topologically, so a stage may be run alone and its prerequisites are pulled in
automatically:

```bash
python -m headroom run -c configs/pilot.yaml --stage judge
```

## Controls and their purpose

- Gibberish that merely fails to refuse is not a jailbreak. Success requires both non-refusal and on-topic, coherent compliance.
- Embedding-space PGD can leave the token manifold entirely, making the upper bound unachievable for any real attacker. Report manifold distance alongside ASR so the bound is interpreted honestly.

## Decision rules

Report effect sizes with bootstrap intervals. Treat an interval that spans zero as a null result and report it as such; do not reach for a subgroup that reaches significance.

## Reproducibility

Every run records a manifest with the git sha, a config fingerprint, resolved
device and dtype, package versions, per-stage timings, and metrics. Seeds are
set across python, numpy, and torch. Known determinism limits are recorded in
the manifest rather than assumed away: MPS does not support
`torch.use_deterministic_algorithms`, so small numeric drift between runs is
expected and should not be read as an effect.

## Scale

The pilot profile is what actually runs on the target machine. The full profile
describes the intended scaled-up run. When reporting any result, state which
profile produced it; a pilot-scale null is weaker evidence than a full-scale
null and the writeup must not blur them.
