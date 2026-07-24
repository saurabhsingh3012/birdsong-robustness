"""Tests for the BirdNET adapter's import-safe, backend-free parts.

The adapter wraps BirdNET, whose TFLite weights are not present in CI, so the model itself is not
exercised here. What *is* tested is the pure logic that decides what score each species gets --
the window-to-clip aggregation and the fixed-rate passthrough -- because that logic is what turns
BirdNET's raw per-window detections into the benchmark's per-clip scores, and a bug in it would
silently corrupt every real-audio number.
"""

from __future__ import annotations

import numpy as np

from birdsong_robustness.birdnet_adapter import BIRDNET_SR, _to_48k, aggregate_detections

SPECIES = ("Turdus merula", "Erithacus rubecula", "Parus major")


def test_module_imports_without_birdnet_backend():
    """The package must import in CI, where neither birdnetlib nor a TFLite backend is installed."""
    import birdsong_robustness.birdnet_adapter as adapter

    assert hasattr(adapter, "BirdNETModel")


def test_aggregation_takes_max_confidence_across_windows():
    """A species detected in several windows gets its single highest confidence."""
    detections = [
        {"scientific_name": "Turdus merula", "confidence": 0.31},
        {"scientific_name": "Turdus merula", "confidence": 0.94},
        {"scientific_name": "Turdus merula", "confidence": 0.12},
    ]
    scores = aggregate_detections(detections, SPECIES)
    assert scores["Turdus merula"] == 0.94


def test_species_not_detected_score_zero():
    """Every vocabulary species is present in the output; undetected ones are exactly 0."""
    scores = aggregate_detections(
        [{"scientific_name": "Turdus merula", "confidence": 0.5}], SPECIES
    )
    assert set(scores) == set(SPECIES)
    assert scores["Erithacus rubecula"] == 0.0
    assert scores["Parus major"] == 0.0


def test_out_of_vocabulary_detections_are_ignored():
    """BirdNET considers its full 6522-label set; only the manifest's species are scored."""
    scores = aggregate_detections(
        [
            {"scientific_name": "Corvus corone", "confidence": 0.99},
            {"scientific_name": "Parus major", "confidence": 0.4},
        ],
        SPECIES,
    )
    assert "Corvus corone" not in scores
    assert scores["Parus major"] == 0.4


def test_to_48k_passthrough_is_identity_at_native_rate():
    """At BirdNET's native rate the adapter must not resample (and needs no librosa)."""
    x = np.linspace(-0.5, 0.5, 1000, dtype=np.float64)
    out = _to_48k(x, BIRDNET_SR)
    assert out.dtype == np.float32
    assert np.allclose(out, x.astype(np.float32))
