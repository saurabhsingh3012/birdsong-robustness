# birdsong-robustness

A controlled-degradation robustness benchmark for bioacoustic classifiers.

Bird-song classifiers are usually reported with a single accuracy number on a clean-ish test
split. Field deployments do not see clean audio. They see distant birds behind traffic, calls
overlapping in a dawn chorus, cheap recorders clipping and band-limiting everything, and rain
landing directly on the frequencies the birds use. **This project measures how a classifier
degrades under each of those conditions, one physically-motivated axis at a time, with every
degradation calibrated so that "0 dB SNR" or "100 m away" means exactly what it says.**

> **Status — read this first.** No classifier has been evaluated yet. This repository is the
> *instrument*: a degradation pipeline plus an independent measurement layer that proves the
> pipeline is calibrated, and a benchmark protocol that a model plugs into. The honest,
> verifiable result today is the requested-versus-measured validation in
> [Results](#results-what-is-actually-measured). Classifier accuracy numbers go in the
> [Results](#results-what-is-actually-measured) section **when a model has actually been run**,
> and not before. See [What this can and cannot claim](#what-this-can-and-cannot-claim).

---

## Why this is worth building

The author previously built a bioacoustic classifier at Boston University (Google MixIT source
separation feeding Cornell's BirdNET) that reached F0.5 0.815 across 223 species against a 0.667
baseline. Numbers like that are earned on a benchmark, and a benchmark is only as honest as the
audio it degrades. The recurring failure in this space is a robustness study whose "0 dB SNR" is
quietly 7 dB easier than it claims, or whose "16 kHz downsample" leaves the top band 20 dB down
instead of gone — so the reported robustness is really a measurement of the benchmark's own bugs.

This project is built the other way round: **every degradation is required to produce exactly the
physical effect it advertises, and an independent measurement layer proves it does** before any
model is ever run. That inversion — validate the instrument first — is the point.

---

## What is here

| Module | What it does | Produces real numbers? |
|---|---|---|
| `degradations.py` | Five parameterised degradation axes (noise, overlap, distance, codec/bandwidth, clipping/gain) | — (it is the instrument) |
| `verify.py` | Independent measurement of what a degradation *actually did*, re-derived from the output waveform | yes |
| `validate.py` | Sweeps every degradation across its grid, measures the result, prints requested-vs-measured | **yes — this is the headline result** |
| `noise.py` | Coloured (pink/brown/white) and environmental (wind/rain) noise models | yes |
| `synth.py` | Synthetic bird-song-like test signals; the repo ships **no audio** | — |
| `protocol.py` | Benchmark protocol: manifest schema, degradation grid, metrics, model interface | not yet — awaiting a model |

The split is deliberate: the part that produces real numbers (`validate`) and the part that
cannot yet (`protocol`, which needs a model) never get confused with each other.

---

## Install and run

```bash
pip install -e ".[dev]"

# The headline result: sweep every degradation and check requested vs measured.
birdsong-validate                 # exits non-zero if any axis drifts out of tolerance
birdsong-validate --json out.json # machine-readable
birdsong-validate --suite distance --suite "call overlap"   # one axis at a time

# The tests that pin the degradation mathematics.
pytest -q
```

No audio downloads, no model weights, no network. Every waveform is synthesised from a seed.

---

## Results: what is actually measured

Running `birdsong-validate` sweeps all five degradation axes across their parameter grids and, for
each condition, measures the output waveform with code in `verify.py` that **does not share
implementation with the degradation it is checking** — a residual subtraction rather than a stored
gain, a Welch transfer function rather than the filter's own coefficients, a Schroeder integration
rather than the decay constant that generated the reverb tail.

**Latest run: 121 checks, 0 outside tolerance.** The numbers below are copied verbatim from the
run committed to [`docs/validation_results.json`](docs/validation_results.json). They are the
real, reproducible contribution of this repository as it stands.

### Additive noise — requested SNR vs measured (active basis)

| Requested | Measured | Error |
|---:|---:|---:|
| +20 dB | +20.000 dB | −0.000 |
| +10 dB | +10.000 dB | −0.000 |
| +5 dB | +5.000 dB | −0.000 |
| 0 dB | −0.000 dB | −0.000 |
| −5 dB | −5.000 dB | −0.000 |
| −10 dB | −10.000 dB | −0.000 |

Exact, because the mix is closed-form arithmetic — but only once you fix the *basis*. On the same
0 dB mixture, an SNR measured over the whole (mostly silent) file reads **−7.96 dB**: the active
region is only 16% of the clip, so a whole-file SNR is optimistic by **7.96 dB** here, and by a
different amount on every recording. The harness reports both so the gap is visible rather than
argued about. (Verified identically for white, pink, brown, wind, and rain maskers.)

### Distance — three separate physical effects, each checked

Spherical spreading (inverse-square-law level loss), measured back from the waveform:

| Distance | Requested loss | Measured | Error |
|---:|---:|---:|---:|
| 10 m | −20.000 dB | −20.000 dB | +0.000 |
| 25 m | −27.959 dB | −27.959 dB | +0.000 |
| 100 m | −40.000 dB | −40.000 dB | +0.000 |
| 200 m | −46.021 dB | −46.021 dB | +0.000 |

Atmospheric absorption (ISO 9613-1, 20 °C / 70 % RH), measured per octave against the standard —
this is the part that makes distance a *low-pass filter*, not a volume knob. At 100 m:

| Band | Predicted | Measured | Error |
|---:|---:|---:|---:|
| 1 kHz | −0.498 dB | −0.498 dB | −0.000 |
| 2 kHz | −0.904 dB | −0.904 dB | −0.000 |
| 4 kHz | −2.309 dB | −2.309 dB | −0.000 |
| 8 kHz | −7.763 dB | −7.763 dB | +0.000 |

8 kHz loses **15×** what 1 kHz loses over the same distance — exactly the upper-harmonic loss that
strips species-discriminative detail. Reverberation (RT60 via Schroeder backward integration) and
direct-to-reverberant ratio track their requests to under 1% (0.8 s RT60 → 0.792 s measured;
−26 dB DRR at 200 m → −26.021 dB measured).

### Codec and bandwidth loss

| Condition | Quantity | Requested | Measured | Error |
|---|---|---:|---:|---:|
| Butterworth low-pass 8 kHz | −3 dB point | 8000 Hz | 7999.0 Hz | −0.99 Hz |
| 16 kHz resample round trip | rejection above Nyquist | ≥ 80 dB | **103.6 dB** | +23.6 |
| 8 kHz resample round trip | rejection above Nyquist | ≥ 80 dB | **103.5 dB** | +23.5 |

The resample rejection is the load-bearing one. `scipy.signal.resample_poly`'s default anti-alias
filter reaches only about 20 dB just above the target Nyquist — a "16 kHz downsample" built on it
would leave the top band audible, and the axis would be measuring something far milder than its
name. With explicit Kaiser anti-alias and anti-image stages the measured rejection is **> 100 dB**:
the band is genuinely gone.

### Quantisation — measured SQNR vs theory, for the signal's actual level

| Bit depth | Predicted SQNR | Measured | Error |
|---:|---:|---:|---:|
| 16-bit | 88.15 dB | 88.15 dB | −0.000 |
| 12-bit | 64.07 dB | 64.15 dB | +0.08 |
| 8-bit | 39.99 dB | 40.11 dB | +0.12 |
| 4-bit | 15.90 dB | 15.93 dB | +0.03 |

Prediction uses `10·log10(P_x / (step²/12))` over the active region — **not** the textbook
`6.02·bits + 1.76` dB, which assumes a full-scale signal and would overstate SQNR by the crest
factor (~15 dB here). Broken down by amplitude, 8-bit linear PCM swings from **24.1 dB** SQNR on
the quietest quartile to **44.7 dB** on the loudest, while 8-bit µ-law holds **37–38 dB** flat
across the range — which is why quiet distant calls survive µ-law and not linear PCM, and why both
8-bit conditions are in the grid.

### Gain and clipping

Gain is exact to floating-point (−40 → +20 dB, all error +0.000), which is the assertion that
**no degradation anywhere quietly renormalises its output** — the property the level, distance, and
clipping axes all silently depend on. Clipping hits its requested clipped-sample fraction exactly
(0.1% → 10%, all error < 1e-4), parameterised by clipped fraction rather than a dBFS threshold so
the axis means the same thing on a hot recording and a quiet one.

### Noise colour — measured PSD slope vs analytic target

| Colour | Target slope | Measured | Bird-band (2–8 kHz) energy |
|---|---:|---:|---:|
| white | 0.00 dB/oct | −0.003 | 37% |
| pink | −3.01 dB/oct | −3.013 | 17% |
| brown | −6.02 dB/oct | −6.024 | 0.2% |

At equal RMS, white noise puts twice as much energy in the bird band as pink does, so "0 dB SNR"
is a different amount of *in-band masking* for each — the reason noise colour is a benchmark axis
and not an implementation detail (see below). Synthesised rain lands **57%** of its energy in the
2–8 kHz bird band; wind lands **~100%** below 500 Hz. They stress a classifier in different ways,
which is why both are in the grid.

---

## The signal-processing reasoning

A few of the modelling choices are the difference between a realistic benchmark and a misleading
one. Each is documented at length in the relevant module docstring; the short version:

**Pink and brown noise, not white.** White noise is flat per hertz, so in a 0–16 kHz recording
half its energy sits above 8 kHz where almost no passerine song lives. Mixing white noise at a
given SNR spends most of the noise budget outside the band the classifier listens to, making the
benchmark easier than it looks. Natural background noise is close to *pink* (−3 dB/octave, constant
power per octave), which concentrates masking energy in the 1–8 kHz range where song and the
classifier's discriminative features overlap. Brown noise (−6 dB/octave) is the low-frequency
extreme — distant traffic rumble, wind pressure — and is useful precisely because it *should not*
hurt a bird classifier much; a model that degrades sharply under it is telling you about its gain
staging, not its acoustic discrimination.

**Air absorption, because distance is a filter not a volume knob.** Air absorbs high frequencies
far more than low ones — roughly with frequency squared, moderated by molecular relaxation (hence
the humidity dependence). A bird at 100 m is not simply 40 dB quieter than at 1 m; it is 40 dB
quieter *and spectrally tilted*, with the upper harmonics that carry species-discriminative detail
selectively removed. Modelling distance as a gain change tests level sensitivity when the real
field failure is loss of high-frequency structure. The implementation is the full ISO 9613-1
expression, because the humidity dependence is strong and non-monotonic — at 4 kHz, dry air absorbs
more than twice what humid air does, and a fit tuned at 70% RH would mis-tilt every dry-climate
deployment.

**Codec and bandwidth loss, because field recorders are cheap.** Autonomous recorders run at
16 kHz, archives sit at 22.05 kHz, telephony-grade codecs cap at 8 kHz. Each caps bird song at its
Nyquist, removing upper harmonics and sometimes the entire fundamental of high-frequency species.
Resampling is not a neutral format conversion here — it is a destructive band-limiting operation,
and the benchmark only tests that honestly if the top band is genuinely emptied (> 100 dB down,
above), not merely dented. µ-law companding is modelled separately from linear PCM because the two
treat quiet distant calls — the ones a deployment is most likely to miss — very differently.

---

## What this can and cannot claim

**Can, and does — verifiable by re-running `birdsong-validate`:**

- Every degradation produces the physical effect it advertises, to the tolerances tabled above.
- The measurement layer is independent of the degradation code, so the agreement is evidence, not
  a tautology.
- The SNR basis problem, the resample-rejection trap, the full-scale-SQNR trap, and the
  reverberation-confounds-attenuation trap are each identified and avoided, with the size of each
  demonstrated numerically.

**Cannot, and does not claim:**

- **Any classifier accuracy number.** No model has been run. There are no BirdNET (or any other)
  weights in this repository or the environment it was built in. There is deliberately no line
  anywhere of the form "model X drops to F0.5 Y at Z dB SNR" — inventing one would be the exact
  dishonesty this project exists to prevent.
- **That the degradations are *realistic*.** Calibration is not realism. The noise models are
  parametric approximations, the reverberation is a statistical impulse response rather than a
  measured one, and the test signals are synthetic DSP probes shaped like bird song — not birds.
  Validating against real field recordings is the first item on the [roadmap](ROADMAP.md).

When a model *is* integrated (`protocol.py` defines exactly the interface it plugs into), the
results that appear here will carry the model version, the checkpoint hash, the dataset, and the
date they were produced. Until then, this section reports the instrument, not the subject.

---

## Project layout

```
src/birdsong_robustness/
  degradations.py   five parameterised degradation axes  (the instrument)
  verify.py         independent measurement of each effect (produces the real numbers)
  validate.py       the requested-vs-measured harness      (birdsong-validate)
  protocol.py       manifest schema, grid, metrics, model interface  (awaiting a model)
  noise.py          coloured + environmental noise models
  synth.py          synthetic bird-song-like test signals   (no audio shipped)
  _dsp.py           shared primitives, so degradation and measurement agree on definitions
tests/              244 tests pinning the degradation mathematics
docs/               validation_results.json — the committed run behind the tables above
```

See [ROADMAP.md](ROADMAP.md) for what stands between this and real classifier results, and
[`docs/`](docs/) for the design notes. Licensed MIT.
