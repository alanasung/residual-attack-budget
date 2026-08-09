# When Zero Jailbreak ASR Still Leaves Room to Attack

**Target project.** Measuring Headroom in Adversarial Evaluations
**Research areas.** AI security; Evaluations; Misuse risk

## Summary

Climb a ladder of attacks from black-box prompting to white-box internal steering, and read the headroom off the gap between the rungs.

## Hypothesis

Measured attack success rate is substantially a function of attacker access rather than of the model's intrinsic willingness. Relaxing the discrete constraint should recover ASR that black-box search could not find. Crucially, small or absent headroom is an accepted and informative outcome: it is evidence for the motivating World 2, and the design must be able to report it as a finding rather than as a failed experiment.

A hypothesis worth testing has to be able to lose. This one loses if the
measurements below come back null, and the design is built so that a null is
reportable rather than a dead end.

## Research questions

1. How much ASR is recovered at each rung: direct request, black-box template search, discrete token optimization, continuous embedding PGD, refusal-direction steering, and forced-prefill as the ceiling?
2. Why does GCG break down as prompts lengthen and models harden? Measured through optimization diagnostics, gradient-to-token-swap agreement, loss trajectories, and plateau rates, not inferred from ASR alone.
3. What is lost when the discrete constraint is relaxed to continuous embeddings, and how much survives projection back to realizable tokens? This is the motivating second application question, made empirical.
4. Do eval-awareness cues shift the curve, and is the shift uniform across rungs or concentrated at the black-box end?

## Method

1. Use a HarmBench subset as the primary benchmark so the result speaks to the motivating motivating setting, with matched benign controls and no novel uplift content authored here.
2. Implement each rung of the ladder against a small safety-tuned open model.
3. Instrument GCG with optimization diagnostics and sweep prompt length and model hardening to characterize where and why it degrades.
4. Extract the refusal direction from contrastive pairs for the steering rung.
5. Project continuous PGD solutions back toward the token manifold and measure ASR retained, which converts the upper bound into a bound an actual attacker could approach.
6. Score with a refusal classifier plus a compliance-quality check, so that incoherent output is not counted as a successful attack.
7. Plot ASR against access level and report headroom between rungs.

## Measurements

- ASR per ladder rung
- headroom: ASR(ceiling) minus ASR(black-box)
- ASR retained after projection back to the token manifold
- GCG optimization diagnostics: plateau rate, gradient/swap agreement, loss trajectory shape as a function of prompt length
- attack cost in queries and wall-clock per rung
- false-success rate from the compliance-quality check

## Threats to validity

- Gibberish that merely fails to refuse is not a jailbreak. Success requires both non-refusal and on-topic, coherent compliance.
- Embedding-space PGD can leave the token manifold entirely, making the upper bound unachievable for any real attacker. Report manifold distance alongside ASR so the bound is interpreted honestly.

## How the result should be interpreted

Forced prefill demonstrates that the capability exists and is gated at the point of refusal. It does NOT by itself establish that PAIR or GCG failed for discrete-optimization reasons, and it is compatible with the motivating World 2. The World 1 versus World 2 inference rests on the combination of the prefill ceiling, the projected-back-to-tokens rung, and the GCG optimization diagnostics, and the writeup must say so.

## Feasibility

The pilot is written for an Apple M4 with 10 cores, unified memory, the PyTorch
MPS backend, no CUDA device, and no configured API keys. Model choices are
capped accordingly (meta-llama/Llama-3.2-1B-Instruct, Qwen/Qwen2.5-1.5B-Instruct). The
`full` profile documents the scaled-up version of the same experiment for when a
real GPU is available, so the reduction in scale is explicit rather than hidden.

## Relationship to the posting

This proposal was independent model before implementation began. That check, the drift it found,
and the revisions made in response are recorded in
[docs/ALIGNMENT.md](ALIGNMENT.md).
