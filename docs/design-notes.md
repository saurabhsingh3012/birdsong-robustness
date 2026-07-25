# Design notes

Longer-form reasoning that does not belong in code docstrings. The load-bearing decisions and
their signal-processing justifications live in the module docstrings themselves; this file is the
cross-cutting narrative and the traps that took real effort to avoid.

## The central idea: validate the instrument before you point it at anything

A robustness benchmark degrades audio and measures a classifier's response. If the degradations
are miscalibrated, the "robustness" it reports is a measurement of its own bugs, and (this is the
insidious part) the result still looks like a plausible curve. There is no error message. A
"0 dB SNR" that is really 7 dB easier just makes the model look more robust than it is.

So this repository is built inside-out. Before any model exists, `verify.py` measures every
degradation back from its output waveform, using methods that share no implementation with the
degradation, and `validate.py` prints requested-versus-measured across the whole grid. That table
is the deliverable right now. It is small, it is honest, and it is verifiable by anyone who runs
`birdsong-validate`.

## Five traps, each avoided and each demonstrated numerically

These are the specific ways a plausible-looking implementation of each axis is quietly wrong. The
validation harness exists to prove this repository is on the right side of each.

1. **The SNR basis trap.** Bird recordings are mostly silence. Compute signal power over the whole
   file (the default in essentially every augmentation library) and the "signal" is diluted by
   the silence, so a requested 0 dB SNR is far higher *during the actual call*, which is the only
   place detection happens. On this repo's own test clip (16% active) the discrepancy is 7.96 dB,
   and it varies per recording, so it is not even a constant bias. Fixed by measuring signal power
   over the active region, and reporting both bases so the gap is visible.

2. **The resample-rejection trap.** `scipy.signal.resample_poly`'s default anti-alias filter
   reaches only ~20 dB of rejection just above the target Nyquist. A "16 kHz downsample" built on
   it leaves the top band audible, and the axis measures something far milder than its name. Fixed
   with explicit Kaiser anti-alias and anti-image stages; measured rejection is now > 100 dB.

3. **The full-scale-SQNR trap.** The textbook `6.02·bits + 1.76` dB assumes a full-scale signal.
   Bird recordings sit 15–30 dB below full scale, so the shortcut overstates SQNR by the crest
   factor. Fixed by predicting from `10·log10(P_x / (step²/12))` over the active region, and by
   noting the digital-silence caveat, where a per-clip average over mostly-zero samples appears to
   beat theory because a quantiser maps exact zero to zero with no error.

4. **The reverberation-confounds-attenuation trap.** Reverberation adds energy. Measure
   "attenuation with distance" on a reverberant clip and you measure the room, not the distance.
   Fixed by isolating spreading + absorption for the attenuation checks (`include_reverb=False`)
   and validating reverberation separately via RT60 and DRR.

5. **The filtfilt-doubles-the-response trap.** Zero-phase `filtfilt`/`sosfiltfilt` applies a
   filter's magnitude response twice, so a requested 8 kHz cutoff lands at neither 8 kHz nor a
   consistent offset. Real microphones and codecs are causal anyway. Fixed by applying band limits
   causally and compensating only the *linear-phase* FIR delay, so requested and measured cutoffs
   agree to under a hertz.

## Why one-factor-at-a-time, with named composites

A full factorial over six axes at five levels is 15,625 conditions per recording, computationally
silly and, worse, uninterpretable, because each interaction is estimated from a single observation.
OFAT from a clean baseline makes each axis's marginal sensitivity directly readable as a
degradation curve. It cannot see interactions, and interactions are real (clipping a distant quiet
recording differs from clipping a close one), so a small hand-picked set of `COMPOSITE_CONDITIONS`
covers the plausible field scenarios, reported separately and never averaged into the per-axis
curves.

## Why F0.5 and a frozen threshold

F0.5 weights precision twice as heavily as recall, which is the right asymmetry for most passive
acoustic monitoring: a false positive enters an occurrence database and needs a human to remove it,
while a missed call is usually one of many from the same individual over a survey. It is also the
metric the author's prior BirdNET work was scored on, so the numbers would be comparable.

The operating threshold is chosen once on the clean dev split and frozen across every degraded
condition. Re-tuning per condition would report each condition's best case and systematically
understate degradation, and a deployed detector does not get to re-tune per weather condition
either.

## Why synthesise instead of shipping audio

Two reasons. Licensing: Xeno-canto and the soundscape sets carry per-recording terms that vary by
recording, so redistributing clips in a Git repository is a licensing problem waiting to happen.
Reproducibility: a reader who clones the repo should be able to reproduce every number in the
README from a seed, not from a binary blob whose provenance is a URL that may rot. The synthetic
signals are DSP probes shaped like bird song (harmonically rich FM sweeps with silent gaps), which
is exactly what the degradation and verification code needs to be exercised honestly. They are
**not** birds and nothing should feed them to a classifier and treat them as such.
