# Roadmap

This project's instrument is calibrated **and now pointed at a real classifier**: BirdNET has
been swept across the full degradation grid on a real Xeno-canto set, and the measured degradation
curves are in the README's Results — real audio section (`docs/real_eval_results.json`). That
closes Milestones 1 and 2 below. What remains is making the degradations *realistic* (not just
calibrated) and scaling the evaluation past six easy species — the honest gaps, in order.

## Now: what exists and is verifiable

- **Five degradation axes**, each parameterised by a physical quantity and validated to measure
  back what was requested (121 checks, 0 out of tolerance).
- **An independent measurement layer** (`verify.py`) that re-derives every effect from the output
  waveform without sharing code with the degradation.
- **A benchmark protocol** (`protocol.py`): manifest schema, one-factor-at-a-time grid plus named
  composite scenarios, per-species precision/recall/F0.5, frozen-threshold evaluation, degradation
  curves, and a minimal model interface. All of it is unit-tested end to end against synthetic
  inputs and a stand-in model. **None of it has seen a real classifier.**

## Milestone 1 — a real dataset behind the manifest ✅ (partial)

- [x] Assembled a real evaluation set: `scripts/build_dataset.py` pulls quality-A focal recordings
      for six species from the Xeno-canto v3 API and writes a `protocol.Manifest`. **Clip-level
      labels only** — the strong (time-localised) labels that would make the SNR and overlap axes
      exact are still future work; those axes currently use the energy-based active-region estimate.
- [x] **Licensing handled by not redistributing audio.** `data/` is git-ignored; the set is rebuilt
      locally from the API, and per-recording attribution + licences are committed in
      `docs/DATA_SOURCES.md`.
- [x] Manifest validated with `protocol.validate_manifest` (clean).
- [ ] Still open: strong labels; BirdCLEF/DCASE soundscapes; and a proper audit of the confounds
      the schema records (recorder/provenance vs. species distribution) — the six-species focal set
      is favourable and does not stress these yet.

## Milestone 2 — the first real model integration ✅

- [x] Adapted BirdNET behind the interface (`src/birdsong_robustness/birdnet_adapter.py`), via
      `birdnetlib` + a TFLite backend. BirdNET was the natural first, given the author's prior work.
- [x] Fixed and recorded the three flagged decisions: **resampling** — the whole pipeline runs at
      BirdNET's 48 kHz, so there is no resampler in the path at all; **window→clip aggregation** —
      max confidence over the clip's 3-second windows, recorded in the results; **built-in
      normalisation** — BirdNET's internal per-window standardisation is left on (legitimate model
      behaviour) and the gain axis is read as "model + its own AGC", noted in the limitations.
- [x] Produced the first degradation curves (`scripts/run_real_eval.py` → `docs/real_eval_results.json`),
      carrying model id, dataset, and date. The README Results section now reports them.

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

*The instrument is built, proven calibrated, and now run: BirdNET, on real Xeno-canto audio,
across the full grid, producing real degradation curves. Every number carries its model, dataset
and date — because inventing them is exactly the failure mode this project was built to prevent.
What is left is realism over calibration, and scale over six easy species.*
