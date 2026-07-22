"""Tests for the synthetic test signals.

These signals are the fixtures for everything else, so their properties have to hold or the
downstream measurements are exercising the wrong thing. The load-bearing property is that the
clip has genuine silent gaps and real high-frequency structure: without the gaps the active-basis
SNR test is meaningless, and without the harmonics the band-limiting test has nothing to remove.

None of this makes them birds. They are DSP test signals shaped like bird song, and the tests
assert exactly the structural properties the DSP code needs, nothing about biological realism.
"""

from __future__ import annotations

import numpy as np
import pytest

from birdsong_robustness import synth

SR = 32000


def test_clip_has_silent_gaps():
    """The active-basis SNR axis is meaningless without genuine silence between calls."""
    clip = synth.bird_call(duration_s=5.0, sr=SR, rng=1)
    assert 0.05 < clip.active_fraction < 0.5
    assert not clip.active.all()
    assert clip.active.any()


def test_clip_has_real_high_frequency_content():
    """Band-limiting tests need energy above the cutoffs, or they remove nothing.

    A meaningful fraction of the clip's energy must sit above 6 kHz, which only happens because
    the syllables are harmonically rich FM sweeps rather than pure tones.
    """
    from scipy import signal as sps

    clip = synth.bird_call(duration_s=5.0, sr=SR, rng=1)
    freqs, psd = sps.welch(clip.audio, fs=SR, nperseg=4096, window="hann")
    high = float(np.sum(psd[freqs > 6000.0]) / np.sum(psd))
    assert high > 0.01


def test_clip_is_reproducible():
    """The repository ships no audio, so the seed is the fixture and must be deterministic."""
    a = synth.bird_call(duration_s=2.0, sr=SR, rng=42)
    b = synth.bird_call(duration_s=2.0, sr=SR, rng=42)
    np.testing.assert_array_equal(a.audio, b.audio)
    np.testing.assert_array_equal(a.active, b.active)


def test_events_line_up_with_the_active_mask():
    """Each annotated event must correspond to active samples, or the ground truth is wrong."""
    clip = synth.bird_call(duration_s=5.0, sr=SR, rng=3)
    for start, end in clip.events:
        assert clip.active[start:end].any()


def test_peak_normalisation_is_honoured():
    clip = synth.bird_call(duration_s=3.0, sr=SR, rng=5, peak=0.5)
    assert np.max(np.abs(clip.audio)) == pytest.approx(0.5, abs=1e-9)


def test_no_aliasing_above_nyquist():
    """Harmonics that would alias are dropped; aliasing would masquerade as HF sensitivity later.

    Checked by the absence of a rising noise shelf near Nyquist, which is where aliased energy
    would fold back to.
    """
    from scipy import signal as sps

    clip = synth.bird_call(duration_s=5.0, sr=SR, rng=7, n_harmonics=6)
    freqs, psd = sps.welch(clip.audio, fs=SR, nperseg=4096, window="hann")
    near_nyquist = psd[freqs > 0.45 * SR]
    mid_band = psd[(freqs > 3000) & (freqs < 6000)]
    assert np.mean(near_nyquist) < 0.01 * np.mean(mid_band)


def test_soundscape_returns_sources_that_sum_to_the_mixture():
    """The overlap code needs the isolated sources; the mixture must actually be their sum."""
    mixture, sources = synth.soundscape(duration_s=5.0, sr=SR, rng=9, n_species=3)
    assert len(sources) == 3
    assert len(mixture.audio) == len(sources[0].audio)
    assert mixture.active.any()


def test_impulse_recovers_an_identity_response():
    """A unit impulse through an LTI system returns its impulse response -- the basis of probing."""
    x = synth.impulse(1000, position=100)
    assert x[100] == 1.0
    assert np.count_nonzero(x) == 1


def test_impulse_position_is_validated():
    with pytest.raises(ValueError, match="position"):
        synth.impulse(100, position=200)


def test_log_sweep_spans_the_band():
    """The exponential sweep is the deterministic probe for a fixed filter's response."""
    sweep = synth.log_sweep(2.0, SR, f_start=50.0, f_end=15000.0)
    assert len(sweep) == 2 * SR
    assert np.max(np.abs(sweep)) <= 0.5 + 1e-9
