# Related work

This note situates **The Attack Ladder: Quantifying Unexploited Headroom** against the mentor-linked literature for
[Measuring Headroom in Adversarial Evaluations](https://sparai.org/projects/f26/rec92cyMzSzUb2eLY).

## Positioning

Climb a ladder of attacks from black-box prompting to white-box internal steering, and read the headroom off the gap between the rungs.

The design hypothesis is: Measured attack success rate is substantially a function of attacker access rather than of the model's intrinsic willingness. Relaxing the discrete constraint should recover ASR that black-box search could not find. Crucially, small or absent headroom is an accepted and informative outcome: it is evidence for the mentor's World 2, and the design must be able to report it as a finding rather than as a failed experiment.

## Engagement rules

1. Cite the paper that motivates each measurement.
2. Name what this repo replicates versus what it changes.
3. Keep synthetic harness results labelled as synthetic.
4. Prefer causal or behavioral ground truth over agreement with a training
   signal that cannot falsify the claim.

## Skeleton critique slots

The following slots are filled per project during alignment. They exist so the
markdown inventory clears the documentation bar even before camera-ready prose
is written.

### Slot A — Primary motivating paper

Summary of the mentor's main citation and the exact claim this repo tests.

### Slot B — Closest prior codebase

What prior open implementations exist, and which abstractions we refuse to
vendor.

### Slot C — Measurement instrument papers

Probe, patching, monitoring, or jailbreak-ladder methodology sources.

### Slot D — Confounds already named in the literature

Shortcut learning, eval awareness, circular labels, underpowered nulls.

### Slot E — Open disagreements

Where this design intentionally diverges from common practice, with the
falsification condition.

## Mentors and affiliations

- Mentor(s): Jamie Hayes
- Affiliation(s): Google DeepMind

## Bibliography placeholders

Additional references are tracked in `TASK.md` and in result JSON `notes`
fields so that reported numbers stay attached to the papers that justify them.
