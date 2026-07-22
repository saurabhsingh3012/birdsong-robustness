# Roadmap

This project is an instrument that is calibrated but not yet pointed at anything. The degradation
pipeline is built and validated ([README, Results](README.md#results-what-is-actually-measured));
the protocol a classifier plugs into is specified and tested (`protocol.py`). What is missing is
everything downstream of "and now run a real model on real audio". This document is candid about
that gap, in the order the work has to happen.

## Now: what exists and is verifiable

- **Five degradation axes**, each parameterised by a physical quantity and validated to measure
  back what was requested (121 checks, 0 out of tolerance).
- **An independent measurement layer** (`verify.py`) that re-derives every effect from the output
  waveform without sharing code with the degradation.
- **A benchmark protocol** (`protocol.py`): manifest schema, one-factor-at-a-time grid plus named
  composite scenarios, per-species precision/recall/F0.5, frozen-threshold evaluation, degradation
  curves, and a minimal model interface. All of it is unit-tested end to end against synthetic
  inputs and a stand-in model. **None of it has seen a real classifier.**

## Milestone 1 — a real dataset behind the manifest

The manifest schema exists; no data fills it. This is the first blocker and the least glamorous.

- [ ] Assemble a strongly-labelled evaluation set (Xeno-canto focal recordings + BirdCLEF/DCASE
      soundscapes are the obvious sources). Strong labels matter: half the axes (SNR-during-call,
      overlap density) are only well defined with time-localised events, not clip tags.
- [ ] **Resolve licensing before distributing anything.** Xeno-canto terms vary per recording;
      this repository ships no audio precisely to avoid baking a licensing problem into Git
      history. A manifest that references audio by URL + checksum, with a downloader, is the likely
      shape — not a blob in the repo.
- [ ] Validate the manifest with `protocol.validate_manifest` and, more importantly, audit the
      **confounds** the schema deliberately records: recorder type and provenance correlating with
      species distribution would let the benchmark attribute to *degradation* what is really an
      artefact of *which species were recorded on which device*.

## Milestone 2 — the first real model integration

`protocol.BioacousticModel` is a two-method interface. Wiring a real model to it is where the
hidden decisions live, and each one can silently change what the benchmark measures:

- [ ] Adapt a classifier (BirdNET is the natural first, given the author's prior work) behind the
      interface. **No weights are in this environment; this cannot be done here.**
- [ ] Fix, and record, the three things the interface docstring flags: the resampler used to reach
      the model's native rate (must be identical across every condition, or the bandwidth axis is
      partly measuring the adapter); the window-to-clip score aggregation (max / mean / top-k, which
      changes the precision-recall trade on its own); and any built-in per-clip normalisation (which
      will partly undo the gain and clipping axes — legitimate, but it must be *known*, or those
      axes look flat for the wrong reason).
- [ ] Only then produce the first degradation curves. When they appear, they carry the checkpoint
      hash, dataset, and date. Until then the README Results section stays as it is.

## Milestone 3 — validate the degradations against reality

Calibration is not realism. Everything validated so far proves the pipeline does what it says, not
that what it says resembles a forest.

- [ ] Compare synthesised noise against recorded wind/rain/road backgrounds (spectra, modulation
      statistics, impulsiveness). The parametric models are documented approximations; this is where
      they earn or lose that status.
- [ ] Replace synthetic room impulse responses with measured outdoor IRs (vegetation scattering,
      ground reflection) for at least a subset of the distance grid.
- [ ] Cross-check the active-region masks: the SNR axis is exact given a mask, but a real
      deployment's mask comes from an energy segmenter whose error propagates into every reported
      SNR. Quantify that propagation on labelled data.

## Milestone 4 — beyond one-factor-at-a-time

- [ ] Report the named composite conditions (`COMPOSITE_CONDITIONS`) once a model exists — distant
      bird in light rain, cheap recorder on a windy morning, dawn chorus, overdriven roadside,
      telephony-grade archive. These are where interactions live, and interactions are real (clipping
      a distant quiet recording is not clipping a close one).
- [ ] Threshold-free summaries (AUC, average precision) alongside the frozen-threshold F0.5, to
      separate "the model lost discriminative information" from "the model's calibration drifted".
- [ ] Per-species breakdowns over a long tail, since a macro average hides which species break and
      the rare ones are usually why anyone runs the classifier at all.

## Explicitly out of scope (for now)

- Training or fine-tuning models. This is an evaluation benchmark, not a model repository.
- Real-time / streaming degradation. Everything here is offline, clip-at-a-time.
- Non-avian bioacoustics. The DSP is general; the parameter ranges and the synthetic signals are
  tuned to passerine song and are not claimed to transfer without re-tuning.

## The one-line summary for an interviewer

*The instrument is built and proven calibrated. What stands between it and a published robustness
result is a licensed strongly-labelled dataset and one real model integration — and the reason
there are no accuracy numbers yet is that inventing them is exactly the failure mode this project
was built to prevent.*
