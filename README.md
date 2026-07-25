# birdsong-robustness

A controlled-degradation robustness benchmark for bioacoustic classifiers.

Bird-song classifiers are usually reported with a single accuracy number on a clean-ish test
split. Field deployments do not see clean audio. They see distant birds behind traffic, calls
overlapping in a dawn chorus, cheap recorders clipping and band-limiting everything, and rain
landing directly on the frequencies the birds use. This project measures how a classifier
degrades under each of those conditions, one physically-motivated axis at a time, with every
degradation calibrated so that "0 dB SNR" or "100 m away" means exactly what it says.

Run on real audio, the benchmark gives a clear answer. BirdNET, swept across the full
degradation grid on real Xeno-canto recordings of six species, holds a clean-baseline F0.5 of
0.908 and shrugs off noise, distance, reverb, quantisation and gain. Overlapping calls are the
exception: the dawn-chorus cocktail-party problem drops it to 0.58. Those curves are in the
real-audio results below, and every number in them is a real BirdNET inference.

Underneath, the degradation pipeline is validated on its own: 121 requested-versus-measured
checks, 0 out of tolerance (the [calibration results](#results-what-is-actually-measured)). See
[What this can and cannot claim](#what-this-can-and-cannot-claim) for the limits.

---

## Why this is worth building

The author previously built a bioacoustic classifier at Boston University (Google MixIT source
separation feeding Cornell's BirdNET) that reached F0.5 0.815 across 223 species against a 0.667
baseline. Numbers like that are earned on a benchmark, and a benchmark is only as honest as the
audio it degrades. The recurring failure in this space is a robustness study whose "0 dB SNR" is
quietly 7 dB easier than it claims, or whose "16 kHz downsample" leaves the top band 20 dB down
instead of gone, so the reported robustness is really a measurement of the benchmark's own bugs.

This project is built the other way round. Every degradation is required to produce exactly the
physical effect it advertises, and an independent measurement layer proves it does before any
model is ever run. Validating the instrument first is the whole approach.

---

## What is here

| Module | What it does | Produces real numbers? |
|---|---|---|
| `degradations.py` | Five parameterised degradation axes (noise, overlap, distance, codec/bandwidth, clipping/gain) | — (it is the instrument) |
| `verify.py` | Independent measurement of what a degradation *actually did*, re-derived from the output waveform | yes |
| `validate.py` | Sweeps every degradation across its grid, measures the result, prints requested-vs-measured | **yes — this is the headline result** |
| `noise.py` | Coloured (pink/brown/white) and environmental (wind/rain) noise models | yes |
| `synth.py` | Synthetic bird-song-like test signals; the repo ships **no audio** | — |
| `protocol.py` | Benchmark protocol: manifest schema, degradation grid, metrics, model interface | **yes — run against BirdNET, see Results — real audio** |
| `birdnet_adapter.py` | BirdNET behind the model interface (the real classifier under test) | **yes — the real-audio results** |

The split is deliberate: the calibration numbers (`validate`, no classifier involved) and the
classifier numbers (`protocol` + BirdNET) are produced by different code, tested differently, and
reported in different sections, so the two kinds of number never get quoted for each other.

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

# The real-audio evaluation: BirdNET on real Xeno-canto recordings. Separate, optional, and
# never run in CI. Needs the BirdNET model, plus a network + Xeno-canto key to build the set.
pip install -e ".[birdnet]"
export XENO_CANTO_KEY=...            # your key; never committed
python scripts/build_dataset.py      # six species of q:A recordings into data/ (git-ignored)
python scripts/run_real_eval.py      # sweeps BirdNET across the grid; resumable
```

The `birdsong-validate` path (the calibration result) needs no audio, no model weights and no
network; every waveform is synthesised from a seed. The real-audio path is separate and optional,
and the audio it downloads is never committed (see [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md)
for attribution).

---

## Results: real audio, how BirdNET degrades

The degradation pipeline is validated separately (the calibration results below); this is what it was built *for*. I ran BirdNET (birdnetlib, TFLite) on 6 species of real Xeno-canto recordings (Erithacus, Fringilla, Parus, Phylloscopus, Sylvia, Turdus; 30 dev / 60 test clips, quality-A), swept every degradation axis, and measured macro F0.5 at each point. Every number below is a real BirdNET inference; reproduce with `python scripts/run_real_eval.py`.

Clean baseline: F0.5 = 0.908 (precision 0.888, recall 1.000). Every species is detected in its own clips; the missing 0.11 is cross-species false positives, real BirdNET confusion rather than a bug.

### What it shrugs off, and what breaks it

| axis | range of conditions | F0.5 range | verdict |
|---|---|---|---|
| Additive noise (SNR) | 6 conditions | 0.92 – 0.96 | **robust** — holds to −10 dB |
| Noise colour | 5 conditions | 0.86 – 0.94 | robust; wind/rain worst |
| **Call overlap** | 12 conditions | 0.58 – 0.78 | **the weakness** — falls to 0.58 |
| Distance (10–200 m) | 5 conditions | 0.85 – 0.90 | robust |
| Reverb (RT60 0.1–0.8 s) | 4 conditions | 0.86 – 0.92 | robust |
| Band-limiting | 8 conditions | 0.70 – 0.90 | robust |
| Quantisation (bit depth) | 7 conditions | 0.89 – 0.95 | robust even at 4-bit |
| Clipping | 5 conditions | 0.84 – 0.91 | robust until heavy |
| Gain | 5 conditions | 0.91 – 0.91 | **invariant** (BirdNET normalises) |
| Composite (realistic) | 5 conditions | 0.63 – 0.92 | hardest — down to 0.64 |

### The two findings that matter

**1. Overlapping calls are the failure mode, the cocktail-party problem.** As competing calls are mixed in at increasing density and decreasing signal-to-interference ratio, precision collapses (BirdNET fires on the interferers):

| overlap condition | F0.5 | precision | recall |
|---|---|---|---|
| d0.25_sir+0dB | 0.680 | 0.640 | 1.000 |
| d0.25_sir+10dB | 0.783 | 0.749 | 0.983 |
| d0.25_sir-5dB | 0.654 | 0.612 | 0.983 |
| d0.50_sir+0dB | 0.643 | 0.602 | 1.000 |
| d0.50_sir+10dB | 0.743 | 0.712 | 0.983 |
| d0.50_sir-5dB | 0.603 | 0.565 | 0.983 |
| d0.75_sir+0dB | 0.620 | 0.591 | 0.950 |
| d0.75_sir+10dB | 0.715 | 0.682 | 0.983 |
| d0.75_sir-5dB | 0.639 | 0.607 | 0.983 |
| d1.00_sir+0dB | 0.617 | 0.618 | 0.817 |
| d1.00_sir+10dB | 0.738 | 0.708 | 0.967 |
| d1.00_sir-5dB | 0.583 | 0.597 | 0.750 |

From 0.78 at light overlap to 0.58 at full density / −5 dB SIR, a 0.33 drop from the clean baseline and by far the largest of any axis. Recall stays high; it's precision that dies. This is the single most actionable result: for dawn-chorus deployment, source separation matters more than any amount of denoising.

**2. Aggressive downsampling and realistic composites hurt; isolated codec artefacts don't.** 8 kHz resampling drops F0.5 to ~0.70 (the high-frequency content that separates species is gone), and the realistic composite scenarios are the hardest of all:

| realistic scenario | F0.5 |
|---|---|
| archive telephony grade | 0.635 |
| cheap recorder windy morning | 0.916 |
| dawn chorus close | 0.674 |
| distant bird light rain | 0.742 |
| overdriven roadside | 0.906 |

Telephony-grade archival audio (0.64) and a close dawn chorus (0.67) are where BirdNET is weakest, which is exactly the audio conservation projects actually have.

---

## Results: what is actually measured

Running `birdsong-validate` sweeps all five degradation axes across their parameter grids and, for
each condition, measures the output waveform with code in `verify.py` that does not share
implementation with the degradation it is checking: a residual subtraction rather than a stored
gain, a Welch transfer function rather than the filter's own coefficients, a Schroeder integration
rather than the decay constant that generated the reverb tail.

Latest run: 121 checks, 0 outside tolerance. The numbers below are copied verbatim from the
run committed to [`docs/validation_results.json`](docs/validation_results.json), so anyone can
regenerate them.

### Additive noise: requested SNR vs measured (active basis)

| Requested | Measured | Error |
|---:|---:|---:|
| +20 dB | +20.000 dB | −0.000 |
| +10 dB | +10.000 dB | −0.000 |
| +5 dB | +5.000 dB | −0.000 |
| 0 dB | −0.000 dB | −0.000 |
| −5 dB | −5.000 dB | −0.000 |
| −10 dB | −10.000 dB | −0.000 |

Exact, because the mix is closed-form arithmetic, but only once you fix the *basis*. On the same
0 dB mixture, an SNR measured over the whole (mostly silent) file reads −7.96 dB: the active
region is only 16% of the clip, so a whole-file SNR is optimistic by 7.96 dB here, and by a
different amount on every recording. The harness reports both so the gap is visible rather than
argued about. (Verified identically for white, pink, brown, wind, and rain maskers.)

### Distance: three separate physical effects, each checked

Spherical spreading (inverse-square-law level loss), measured back from the waveform:

| Distance | Requested loss | Measured | Error |
|---:|---:|---:|---:|
| 10 m | −20.000 dB | −20.000 dB | +0.000 |
| 25 m | −27.959 dB | −27.959 dB | +0.000 |
| 100 m | −40.000 dB | −40.000 dB | +0.000 |
| 200 m | −46.021 dB | −46.021 dB | +0.000 |

Atmospheric absorption (ISO 9613-1, 20 °C / 70 % RH), measured per octave against the standard.
This is the part that makes distance a *low-pass filter*, not a volume knob. At 100 m:

| Band | Predicted | Measured | Error |
|---:|---:|---:|---:|
| 1 kHz | −0.498 dB | −0.498 dB | −0.000 |
| 2 kHz | −0.904 dB | −0.904 dB | −0.000 |
| 4 kHz | −2.309 dB | −2.309 dB | −0.000 |
| 8 kHz | −7.763 dB | −7.763 dB | +0.000 |

8 kHz loses 15× what 1 kHz loses over the same distance, exactly the upper-harmonic loss that
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
filter reaches only about 20 dB just above the target Nyquist, so a "16 kHz downsample" built on it
would leave the top band audible, and the axis would be measuring something far milder than its
name. With explicit Kaiser anti-alias and anti-image stages the measured rejection is > 100 dB:
the band is genuinely gone.

### Quantisation: measured SQNR vs theory, for the signal's actual level

| Bit depth | Predicted SQNR | Measured | Error |
|---:|---:|---:|---:|
| 16-bit | 88.15 dB | 88.15 dB | −0.000 |
| 12-bit | 64.07 dB | 64.15 dB | +0.08 |
| 8-bit | 39.99 dB | 40.11 dB | +0.12 |
| 4-bit | 15.90 dB | 15.93 dB | +0.03 |

Prediction uses `10·log10(P_x / (step²/12))` over the active region, not the textbook
`6.02·bits + 1.76` dB, which assumes a full-scale signal and would overstate SQNR by the crest
factor (~15 dB here). Broken down by amplitude, 8-bit linear PCM swings from 24.1 dB SQNR on
the quietest quartile to 44.7 dB on the loudest, while 8-bit µ-law holds 37–38 dB flat
across the range. That is why quiet distant calls survive µ-law and not linear PCM, and why both
8-bit conditions are in the grid.

### Gain and clipping

Gain is exact to floating-point (−40 → +20 dB, all error +0.000), which is the assertion that
no degradation anywhere quietly renormalises its output, the property the level, distance, and
clipping axes all silently depend on. Clipping hits its requested clipped-sample fraction exactly
(0.1% → 10%, all error < 1e-4), parameterised by clipped fraction rather than a dBFS threshold so
the axis means the same thing on a hot recording and a quiet one.

### Noise colour: measured PSD slope vs analytic target

| Colour | Target slope | Measured | Bird-band (2–8 kHz) energy |
|---|---:|---:|---:|
| white | 0.00 dB/oct | −0.003 | 37% |
| pink | −3.01 dB/oct | −3.013 | 17% |
| brown | −6.02 dB/oct | −6.024 | 0.2% |

At equal RMS, white noise puts twice as much energy in the bird band as pink does, so "0 dB SNR"
is a different amount of *in-band masking* for each, the reason noise colour is a benchmark axis
and not an implementation detail (see below). Synthesised rain lands 57% of its energy in the
2–8 kHz bird band; wind lands ~100% below 500 Hz. They stress a classifier in different ways,
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
extreme (distant traffic rumble, wind pressure) and is useful precisely because it *should not*
hurt a bird classifier much; a model that degrades sharply under it is telling you about its gain
staging, not its acoustic discrimination.

**Air absorption, because distance is a filter not a volume knob.** Air absorbs high frequencies
far more than low ones, roughly with frequency squared, moderated by molecular relaxation (hence
the humidity dependence). A bird at 100 m is not simply 40 dB quieter than at 1 m; it is 40 dB
quieter *and spectrally tilted*, with the upper harmonics that carry species-discriminative detail
selectively removed. Modelling distance as a gain change tests level sensitivity when the real
field failure is loss of high-frequency structure. The implementation is the full ISO 9613-1
expression, because the humidity dependence is strong and non-monotonic: at 4 kHz, dry air absorbs
more than twice what humid air does, and a fit tuned at 70% RH would mis-tilt every dry-climate
deployment.

**Codec and bandwidth loss, because field recorders are cheap.** Autonomous recorders run at
16 kHz, archives sit at 22.05 kHz, telephony-grade codecs cap at 8 kHz. Each caps bird song at its
Nyquist, removing upper harmonics and sometimes the entire fundamental of high-frequency species.
Resampling is not a neutral format conversion here; it is a destructive band-limiting operation,
and the benchmark only tests that honestly if the top band is genuinely emptied (> 100 dB down,
above), not merely dented. µ-law companding is modelled separately from linear PCM because the two
treat quiet distant calls (the ones a deployment is most likely to miss) very differently.

---

## What this can and cannot claim

**Can, and does:**

- Every degradation produces the physical effect it advertises, to the tolerances tabled above
  (verifiable by re-running `birdsong-validate`).
- The measurement layer is independent of the degradation code, so the agreement is evidence, not
  a tautology.
- The SNR basis problem, the resample-rejection trap, the full-scale-SQNR trap, and the
  reverberation-confounds-attenuation trap are each identified and avoided, with the size of each
  demonstrated numerically.
- **A real classifier has been run on real audio.** BirdNET, across the full grid, on six species
  of Xeno-canto recordings; the measured degradation curves are in the real-audio results
  section, carrying model id, dataset, and date (`docs/real_eval_results.json`). Every value came
  out of the model; none was chosen by hand.

**Cannot, and does not claim:**

- **That the six-species number is BirdNET's field accuracy.** The real-audio result is measured
  on six common, well-represented species and one 12-second clip per recording, a deliberately
  favourable set, with clip-level labels only. It measures the *shape* of degradation (where the
  model falls off), not BirdNET's accuracy on a 200-species dawn-chorus soundscape; the clean
  baseline is an upper bound, not a headline.
- **That the degradations are *realistic*.** Calibration is not realism. The noise models are
  parametric approximations, the reverberation is a statistical impulse response rather than a
  measured one, and the test signals are synthetic DSP probes shaped like bird song, not birds.
  Validating against real field recordings is the first item on the [roadmap](ROADMAP.md).

The real-audio results now here carry the model, the dataset, and the date they were produced, as
this project always insisted they must (`docs/real_eval_results.json`). The degradation-pipeline
calibration above is still reported separately and still involves no classifier, so the two kinds
of number never get quoted for each other.

---

## Project layout

```
src/birdsong_robustness/
  degradations.py   five parameterised degradation axes  (the instrument)
  verify.py         independent measurement of each effect (produces the real numbers)
  validate.py       the requested-vs-measured harness      (birdsong-validate)
  protocol.py       manifest schema, grid, metrics, model interface  (run against BirdNET)
  birdnet_adapter.py  BirdNET behind the model interface     (the real classifier under test)
  noise.py          coloured + environmental noise models
  synth.py          synthetic bird-song-like test signals   (no audio shipped)
  _dsp.py           shared primitives, so degradation and measurement agree on definitions
scripts/            build_dataset.py (Xeno-canto downloader), run_real_eval.py (the sweep)
tests/              the degradation mathematics, plus the BirdNET adapter's aggregation
docs/               validation_results.json (calibration), real_eval_results.json (real audio),
                    DATA_SOURCES.md (Xeno-canto attribution)
```

See [ROADMAP.md](ROADMAP.md) for what still stands between the six-species result and a field-scale
one, and [`docs/`](docs/) for the design notes. Licensed MIT.
