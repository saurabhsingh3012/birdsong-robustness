"""Tests for the benchmark protocol.

The protocol has no classifier behind it, so what is tested here is the machinery that would sit
around one: that the manifest schema rejects the inconsistencies that quietly corrupt a
benchmark, that the metrics behave the way robustness reporting needs them to (macro not micro,
frozen threshold, no free lunch for untested classes), and that every condition in the grid runs
end to end and preserves clip length. The one thing not tested is accuracy, because there is
nothing to be accurate.
"""

from __future__ import annotations

import numpy as np
import pytest

from birdsong_robustness import protocol, synth
from birdsong_robustness.protocol import (
    COMPOSITE_CONDITIONS,
    DEFAULT_GRID,
    BioacousticModel,
    DegradationSpec,
    Event,
    Manifest,
    RecordingEntry,
    apply_spec,
    evaluate,
    fbeta,
    macro_average,
    per_species_metrics,
    select_threshold,
    synthetic_manifest,
    validate_manifest,
)

SR = 32000


# --------------------------------------------------------------------------------------
# Manifest schema and validation
# --------------------------------------------------------------------------------------


def _good_entry(**overrides) -> RecordingEntry:
    base = {
        "recording_id": "r1",
        "path": "audio/r1.wav",
        "sr": SR,
        "duration_s": 5.0,
        "species": ("A",),
        "events": (Event(0.5, 1.5, "A"),),
        "split": "test",
    }
    base.update(overrides)
    return RecordingEntry(**base)


def test_valid_manifest_has_no_problems():
    manifest = Manifest(
        dataset_id="d",
        species=("A", "B"),
        entries=(_good_entry(split="dev"), _good_entry(recording_id="r2")),
    )
    assert validate_manifest(manifest) == []


def test_manifest_round_trips_through_json():
    """A manifest must survive serialisation, since that is how a dataset would be shipped."""
    manifest = Manifest(
        dataset_id="d",
        species=("A", "B"),
        entries=(_good_entry(split="dev"), _good_entry(recording_id="r2")),
    )
    restored = Manifest.from_dict(__import__("json").loads(manifest.to_json()))
    assert restored == manifest


def test_duplicate_recording_id_is_caught():
    """A duplicate silently doubles a recording's weight in the macro average."""
    manifest = Manifest(
        dataset_id="d",
        species=("A",),
        entries=(_good_entry(split="dev"), _good_entry()),
    )
    assert any("duplicate recording_id" in p for p in validate_manifest(manifest))


def test_event_outside_the_recording_is_caught():
    manifest = Manifest(
        dataset_id="d",
        species=("A",),
        entries=(_good_entry(split="dev", events=(Event(4.0, 6.0, "A"),)),),
    )
    assert any("outside" in p for p in validate_manifest(manifest))


def test_reversed_event_bounds_are_caught():
    manifest = Manifest(
        dataset_id="d",
        species=("A",),
        entries=(_good_entry(split="dev", events=(Event(2.0, 1.0, "A"),)),),
    )
    assert any("start_s" in p for p in validate_manifest(manifest))


def test_species_not_in_vocabulary_is_caught():
    manifest = Manifest(
        dataset_id="d",
        species=("A",),
        entries=(_good_entry(split="dev", species=("A", "Z"), events=(Event(0.5, 1.5, "A"),)),),
    )
    assert any("Z" in p and "vocabulary" in p for p in validate_manifest(manifest))


def test_event_species_absent_from_clip_labels_is_caught():
    """Strong and clip-level labels disagreeing is a real dataset bug, not a warning."""
    manifest = Manifest(
        dataset_id="d",
        species=("A", "B"),
        entries=(_good_entry(split="dev", species=("A",), events=(Event(0.5, 1.5, "B"),)),),
    )
    assert any("clip-level" in p for p in validate_manifest(manifest))


def test_missing_dev_split_is_caught():
    """Without a dev split the operating threshold would have to come from the test set."""
    manifest = Manifest(dataset_id="d", species=("A",), entries=(_good_entry(),))
    assert any("dev" in p for p in validate_manifest(manifest))


def test_unknown_split_is_caught():
    manifest = Manifest(dataset_id="d", species=("A",), entries=(_good_entry(split="train"),))
    assert any("split" in p for p in validate_manifest(manifest))


def test_active_mask_from_events():
    """The strong labels must translate to the sample-level mask the SNR axis needs."""
    entry = _good_entry(events=(Event(1.0, 2.0, "A"),))
    mask = entry.active_mask(int(5.0 * SR))
    assert mask[int(1.5 * SR)]
    assert not mask[int(0.5 * SR)]
    assert mask.sum() == pytest.approx(SR, rel=0.01)


def test_active_mask_is_empty_without_events():
    """No events must yield an all-False mask, forcing an explicit fallback decision upstream."""
    entry = _good_entry(events=())
    assert not entry.active_mask(int(5.0 * SR)).any()


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def test_fbeta_half_weights_precision_over_recall():
    """F0.5 must sit closer to precision than to recall when they differ."""
    high_precision = fbeta(0.9, 0.3, beta=0.5)
    high_recall = fbeta(0.3, 0.9, beta=0.5)
    assert high_precision > high_recall


def test_fbeta_reduces_to_f1_at_beta_one():
    assert fbeta(0.6, 0.4, beta=1.0) == pytest.approx(2 * 0.6 * 0.4 / (0.6 + 0.4))


def test_fbeta_is_zero_when_both_are_zero():
    assert fbeta(0.0, 0.0) == 0.0


def test_per_species_counts_are_correct():
    """A hand-worked multi-label example, so the counting itself is pinned down."""
    y_true = [["A"], ["A", "B"], ["B"], []]
    y_score = [
        {"A": 0.9, "B": 0.1},  # A present & detected (tp A); B absent & not detected
        {"A": 0.9, "B": 0.2},  # A tp; B present but below threshold (fn B)
        {"A": 0.8, "B": 0.9},  # A absent & detected (fp A); B tp
        {"A": 0.1, "B": 0.1},  # nothing present, nothing detected
    ]
    metrics = per_species_metrics(y_true, y_score, ("A", "B"), threshold=0.5)
    assert (metrics["A"].tp, metrics["A"].fp, metrics["A"].fn) == (2, 1, 0)
    assert (metrics["B"].tp, metrics["B"].fp, metrics["B"].fn) == (1, 0, 1)


def test_untested_species_scores_zero_not_one():
    """A class with no positives and no predictions must not be scored as perfect.

    Scoring absent classes as 1.0 is the single easiest way to manufacture a good-looking macro
    average over a long species tail.
    """
    metrics = per_species_metrics([["A"]], [{"A": 0.9}], ("A", "Z"), threshold=0.5)
    assert metrics["Z"].f_beta == 0.0
    assert metrics["Z"].support == 0


def test_macro_average_excludes_zero_support_species():
    """The macro average is over species that actually occur, but all are reported individually."""
    metrics = per_species_metrics([["A"]], [{"A": 0.9}], ("A", "Z"), threshold=0.5)
    macro = macro_average(metrics, min_support=1)
    assert macro.f_beta == pytest.approx(1.0)  # only A counts, and A is perfect here
    assert "Z" in metrics


def test_macro_is_not_micro_on_an_imbalanced_split():
    """Macro weights a rare species the same as a common one; micro does not. That is the point.

    One common species scored perfectly and one rare species scored poorly must drag the macro
    average well below what a count-weighted micro average would report.
    """
    y_true = [["common"]] * 100 + [["rare"]] * 2
    y_score = [{"common": 0.9, "rare": 0.0}] * 100 + [{"common": 0.0, "rare": 0.1}] * 2
    metrics = per_species_metrics(y_true, y_score, ("common", "rare"), threshold=0.5)
    macro = macro_average(metrics)
    assert metrics["common"].f_beta == pytest.approx(1.0)
    assert metrics["rare"].f_beta == pytest.approx(0.0)
    assert macro.f_beta == pytest.approx(0.5, abs=0.01)


def test_select_threshold_picks_a_separating_point():
    """On cleanly separable scores the chosen threshold must sit between the classes."""
    y_true = [["A"], [], ["A"], []]
    y_score = [{"A": 0.8}, {"A": 0.2}, {"A": 0.9}, {"A": 0.1}]
    threshold = select_threshold(y_true, y_score, ("A",))
    assert 0.2 <= threshold < 0.8


def test_per_species_length_mismatch_raises():
    with pytest.raises(ValueError, match="rows"):
        per_species_metrics([["A"]], [{"A": 0.5}, {"A": 0.5}], ("A",), 0.5)


# --------------------------------------------------------------------------------------
# The grid and apply_spec
# --------------------------------------------------------------------------------------


def test_default_grid_has_a_clean_baseline_and_covers_every_axis():
    """Every degradation axis must appear, and there must be exactly one clean reference."""
    axes = {spec.axis for spec in DEFAULT_GRID}
    for expected in (
        "clean",
        "snr",
        "noise_type",
        "overlap",
        "distance",
        "reverb",
        "bandwidth",
        "quantisation",
        "clipping",
        "gain",
    ):
        assert expected in axes
    assert sum(spec.axis == "clean" for spec in DEFAULT_GRID) == 1


def test_grid_labels_are_unique():
    """Labels key the results dictionary; a collision would silently overwrite a condition."""
    labels = [spec.label for spec in DEFAULT_GRID]
    assert len(labels) == len(set(labels))


def test_clean_spec_is_close_to_identity():
    """The clean baseline must barely touch the signal, or every degradation is measured from
    a moving reference."""
    clip = synth.bird_call(duration_s=5.0, sr=SR, rng=1)
    out = apply_spec(clip.audio, SR, DegradationSpec(), active=clip.active)
    assert len(out) == len(clip.audio)
    np.testing.assert_allclose(out, clip.audio, atol=1e-9)


@pytest.fixture(scope="module")
def grid_fixtures():
    """Target clip and interferer pool, built once and reused across every grid condition.

    Module-scoped on purpose: regenerating a soundscape for each of the ~100 parametrised
    conditions is what turns a fast test suite into a slow one, and the input signal does not
    depend on the condition.
    """
    clip = synth.bird_call(duration_s=5.0, sr=SR, rng=1, n_syllables=6)
    _, sources = synth.soundscape(duration_s=5.0, sr=SR, rng=2, n_species=3)
    return clip, [s.audio for s in sources]


@pytest.mark.parametrize("spec", DEFAULT_GRID, ids=lambda s: s.label)
def test_every_grid_condition_is_length_preserving_and_finite(spec, grid_fixtures):
    """The property most easily broken by adding a convolution or resampler to the chain."""
    clip, interferers = grid_fixtures
    out = apply_spec(clip.audio, SR, spec, active=clip.active, interferers=interferers, rng=7)
    assert len(out) == len(clip.audio)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("spec", COMPOSITE_CONDITIONS, ids=lambda s: s.label)
def test_every_composite_condition_runs(spec, grid_fixtures):
    """The named field scenarios stack several axes; they must compose without error."""
    clip, interferers = grid_fixtures
    out = apply_spec(clip.audio, SR, spec, active=clip.active, interferers=interferers, rng=7)
    assert len(out) == len(clip.audio)
    assert np.all(np.isfinite(out))


def test_apply_spec_overlap_without_interferers_raises():
    clip = synth.bird_call(duration_s=5.0, sr=SR, rng=1)
    with pytest.raises(ValueError, match="interferers"):
        apply_spec(clip.audio, SR, DegradationSpec(overlap_density=0.5), active=clip.active)


# --------------------------------------------------------------------------------------
# End-to-end plumbing with a stand-in model (no real classifier exists)
# --------------------------------------------------------------------------------------


class _OracleModel:
    """A trivial stand-in that reports energy in a species' band. NOT a classifier.

    It exists only to exercise :func:`evaluate` end to end -- threshold selection, the grid
    sweep, metric accumulation. Any number it produces is a fact about a bandpass filter, and it
    is never reported as accuracy. It deliberately satisfies the
    :class:`BioacousticModel` protocol so that the protocol check itself is tested.
    """

    def __init__(self, bands):
        self.species = tuple(bands)
        self._bands = bands

    def predict(self, audio, sr):
        from scipy import signal as sps

        freqs, psd = sps.welch(audio, fs=sr, nperseg=2048, window="hann")
        total = float(np.sum(psd)) + 1e-12
        out = {}
        for name, (low, high) in self._bands.items():
            band = (freqs >= low) & (freqs < high)
            out[name] = float(np.sum(psd[band]) / total)
        return out


def test_oracle_model_satisfies_the_protocol():
    model = _OracleModel({"A": (2000.0, 4000.0)})
    assert isinstance(model, BioacousticModel)


def test_evaluate_runs_end_to_end_on_synthetic_data():
    """The whole evaluation path must execute without a real model or any audio on disk.

    This is a plumbing test. It asserts the *shape* of the result -- every condition scored,
    thresholds frozen, metrics present -- and never asserts an accuracy value, because the model
    is a bandpass filter and the signals are not birds.
    """
    manifest, audio = synthetic_manifest(n_recordings=6, seed=0)
    model = _OracleModel(dict.fromkeys(manifest.species, (2000.0, 6000.0)))

    result = evaluate(
        model,
        manifest,
        load_audio=lambda entry: audio[entry.recording_id],
        grid=DEFAULT_GRID[:4],  # a few conditions are enough for a plumbing check
        interferer_pool=list(audio.values()),
        model_id="oracle-bandpass-not-a-real-model",
    )
    assert set(result.macro) == {spec.label for spec in DEFAULT_GRID[:4]}
    assert 0.0 <= result.threshold <= 1.0
    for label in result.macro:
        assert label in result.per_condition
    # The result must serialise, since that is how a real run would be archived.
    assert "oracle-bandpass-not-a-real-model" in result.to_json()


def test_evaluate_is_deterministic():
    """Same seed, same numbers -- paired conditions require it."""
    manifest, audio = synthetic_manifest(n_recordings=6, seed=0)
    model = _OracleModel(dict.fromkeys(manifest.species, (2000.0, 6000.0)))
    load = lambda entry: audio[entry.recording_id]  # noqa: E731
    a = evaluate(model, manifest, load, grid=DEFAULT_GRID[:3], interferer_pool=list(audio.values()))
    b = evaluate(model, manifest, load, grid=DEFAULT_GRID[:3], interferer_pool=list(audio.values()))
    for label in a.macro:
        assert a.macro[label].f_beta == b.macro[label].f_beta


def test_protocol_status_records_the_real_run():
    """A machine-readable guarantee that the real classifier run is recorded, with provenance.

    The banner used to promise 'no model yet'. A model has now been run (BirdNET, on real
    Xeno-canto audio), so the invariant flips: the status must name the classifier and point at
    the committed results, so no one can quote a number without its provenance.
    """
    status = protocol.PROTOCOL_STATUS.lower()
    assert "birdnet" in status
    assert "real_eval_results.json" in status
