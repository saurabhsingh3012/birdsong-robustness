#!/usr/bin/env python3
"""End-to-end smoke demo of the benchmark *plumbing*, with no real model and no audio.

This runs the full evaluation path -- synthesise a manifest, sweep part of the degradation grid,
select an operating threshold on the dev split, accumulate per-species metrics -- using a trivial
stand-in "model" that is just a bandpass energy detector. It exists so that a reader can watch the
protocol machinery execute, not to produce a result.

EVERY NUMBER THIS PRINTS IS A FACT ABOUT A BANDPASS FILTER RUN ON SYNTHETIC DSP PROBES. None of it
is a classifier accuracy figure, and none of it should be read as one. When a real model is wired
in behind ``protocol.BioacousticModel`` (see protocol.py and ROADMAP.md), the same code path
produces real degradation curves -- carrying a checkpoint hash and a dataset, which this does not.

Run:  python scripts/demo_protocol_smoke.py
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from birdsong_robustness import protocol
from birdsong_robustness.protocol import DEFAULT_GRID, evaluate, synthetic_manifest


class BandpassEnergyDetector:
    """A stand-in that reports each species' share of clip energy in a fixed band. NOT a model.

    It satisfies ``protocol.BioacousticModel`` so that the protocol check and the whole evaluate()
    path are exercised. Any score it emits is a property of a Butterworth filter.
    """

    def __init__(self, species: tuple[str, ...], band_hz: tuple[float, float] = (2000.0, 6000.0)):
        self.species = species
        self._band = band_hz

    def predict(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        freqs, psd = sps.welch(audio, fs=sr, nperseg=2048, window="hann")
        total = float(np.sum(psd)) + 1e-12
        band = (freqs >= self._band[0]) & (freqs < self._band[1])
        score = float(np.sum(psd[band]) / total)
        # Same score for every species -- this detector cannot tell them apart, which is the point:
        # it demonstrates plumbing, not discrimination.
        return dict.fromkeys(self.species, score)


def main() -> None:
    print(__doc__)
    print("=" * 88)
    print(f"protocol status: {protocol.PROTOCOL_STATUS}")
    print("=" * 88)

    manifest, audio = synthetic_manifest(n_recordings=9, seed=0)
    model = BandpassEnergyDetector(manifest.species)

    # A handful of conditions is enough to show the path; the full grid has ~60.
    grid = [spec for spec in DEFAULT_GRID if spec.axis in ("clean", "snr", "distance")]

    result = evaluate(
        model,
        manifest,
        load_audio=lambda entry: audio[entry.recording_id],
        grid=grid,
        interferer_pool=list(audio.values()),
        model_id="bandpass-energy-detector (NOT A CLASSIFIER)",
    )

    print(f"\nfrozen operating threshold (chosen on the dev split): {result.threshold:.3f}")
    print(f"conditions evaluated: {len(result.macro)}\n")
    print(f"{'condition':<26} {'macro F0.5*':>12}   (* of a bandpass filter -- NOT accuracy)")
    print("-" * 70)
    for spec in grid:
        metrics = result.macro[spec.label]
        print(f"{spec.label:<26} {metrics.f_beta:>12.3f}")

    print(
        "\nThe numbers above describe a bandpass filter on synthetic probes. They are here to show\n"
        "the evaluation machinery runs end to end. Real classifier results require a real model\n"
        "and a licensed dataset -- see ROADMAP.md."
    )


if __name__ == "__main__":
    main()
