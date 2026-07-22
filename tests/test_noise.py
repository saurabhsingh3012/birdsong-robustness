"""Tests for the noise models.

Coloured noise has an exact analytic target -- ``1/f**beta`` means a PSD slope of
``-3.0103 * beta`` dB per octave -- so these are comparisons against theory, not against another
implementation. Wind and rain are documented approximations with no closed form, so they are
tested for the structural properties that are the reason they are in the grid at all: where their
energy sits relative to the bird band, and whether they are impulsive or stationary.
"""

from __future__ import annotations

import numpy as np
import pytest

from birdsong_robustness.noise import (
    COLOR_BETA,
    color_noise,
    colored_noise,
    make_noise,
    rain_noise,
    wind_noise,
)
from birdsong_robustness.verify import measure_spectral_slope

SR = 32000
N = SR * 8


def band_fraction(x: np.ndarray, sr: int, low: float, high: float) -> float:
    """Share of a signal's power falling inside a frequency band."""
    from scipy import signal as sps

    freqs, psd = sps.welch(x, fs=sr, nperseg=4096, noverlap=2048, window="hann")
    total = float(np.sum(psd))
    band = (freqs >= low) & (freqs < high)
    return float(np.sum(psd[band]) / total) if total > 0 else 0.0


@pytest.mark.parametrize(
    ("color", "expected_slope"),
    [("white", 0.0), ("pink", -3.0103), ("brown", -6.0206), ("blue", 3.0103)],
)
def test_colored_noise_hits_its_analytic_slope(color, expected_slope):
    """Measured PSD slope must match ``-3.0103 * beta`` dB/octave.

    The tolerance is 0.25 dB/octave, which is loose relative to the roughly 0.01 dB/octave the
    frequency-domain synthesis actually achieves, but tight enough to reject the cascaded-IIR
    "pinking filter" approach whose slope drifts by around a decibel across the band.
    """
    x = color_noise(N, color, SR, rng=7)
    assert measure_spectral_slope(x, SR, f_min=200.0, f_max=8000.0) == pytest.approx(
        expected_slope, abs=0.25
    )


def test_beta_to_slope_mapping_is_consistent():
    """The registry's beta values must be the ones the slopes imply."""
    for beta in COLOR_BETA.values():
        x = colored_noise(N, beta=beta, sr=SR, rng=3)
        measured = measure_spectral_slope(x, SR, f_min=200.0, f_max=8000.0)
        assert measured == pytest.approx(-3.0103 * beta, abs=0.3)


@pytest.mark.parametrize("name", ["white", "pink", "brown", "wind", "rain"])
def test_every_noise_is_unit_rms_and_zero_mean(name):
    """All generators normalise to unit RMS, so SNR mixing has one less thing to get wrong.

    A DC offset would be a bias rather than noise, and would sail through an RMS check while
    corrupting every level measurement, so it is asserted separately.
    """
    x = make_noise(name, N, SR, rng=11)
    assert float(np.sqrt(np.mean(x**2))) == pytest.approx(1.0, abs=1e-9)
    assert abs(float(np.mean(x))) < 0.02


@pytest.mark.parametrize("name", ["white", "pink", "brown", "wind", "rain"])
def test_noise_is_reproducible_from_a_seed(name):
    """Same seed, same waveform. The repository ships no audio, so seeds are the fixtures."""
    a = make_noise(name, SR, SR, rng=42)
    b = make_noise(name, SR, SR, rng=42)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_noise():
    """Guards against a generator that ignores its seed and returns a constant realisation."""
    a = make_noise("pink", SR, SR, rng=1)
    b = make_noise("pink", SR, SR, rng=2)
    assert not np.allclose(a, b)


def test_pink_puts_less_energy_in_the_bird_band_than_white():
    """The reason the colour matters: at equal RMS the in-band masking is not equal.

    White noise spreads its power flat per hertz, so most of it lands above the 2-8 kHz band
    where passerine song lives. Pink concentrates towards the low end. Neither is "harder" in
    the abstract -- the point is that "0 dB SNR" means a different amount of in-band masking for
    each, which is exactly why noise type is a benchmark axis rather than an implementation
    detail.
    """
    white_band = band_fraction(make_noise("white", N, SR, rng=5), SR, 2000, 8000)
    pink_band = band_fraction(make_noise("pink", N, SR, rng=5), SR, 2000, 8000)
    brown_band = band_fraction(make_noise("brown", N, SR, rng=5), SR, 2000, 8000)
    assert white_band > pink_band > brown_band


def test_wind_energy_is_overwhelmingly_low_frequency():
    """Wind is turbulent pressure, not radiated sound: it should barely touch the bird band."""
    x = wind_noise(N, SR, rng=5)
    assert band_fraction(x, SR, 0, 500) > 0.9
    assert band_fraction(x, SR, 2000, 8000) < 0.01


def test_rain_energy_lands_on_the_bird_band():
    """Rain is the in-band case, and is the reason both it and wind are in the grid.

    If rain and wind masked the same part of the spectrum there would be no information in
    running both. They do not, and this test is what pins that down.
    """
    x = rain_noise(N, SR, rng=5)
    assert band_fraction(x, SR, 2000, 8000) > 0.3
    assert band_fraction(x, SR, 0, 500) < 0.1


def test_rain_is_more_impulsive_than_pink_noise():
    """Rain's damage comes partly from its impulsiveness, measured here as kurtosis.

    A Gaussian process has a kurtosis of 3. A shower of discrete droplet impacts is heavy
    tailed and should sit well above that -- which is what breaks any noise-floor estimate
    based on a percentile of the spectrogram.
    """

    def kurtosis(x: np.ndarray) -> float:
        centred = x - np.mean(x)
        return float(np.mean(centred**4) / np.mean(centred**2) ** 2)

    assert kurtosis(rain_noise(N, SR, rng=5, drops_per_second=200.0)) > kurtosis(
        color_noise(N, "pink", SR, rng=5)
    )


def test_heavy_rain_tends_towards_gaussian():
    """Many overlapping drops sum to something Gaussian, which is a real physical transition.

    Uses a short signal on purpose: at thousands of drops per second the synthesis cost is in
    the drop count, and a couple of seconds is more than enough to estimate kurtosis. The heavy
    condition is capped at a rate that still shows the trend without synthesising millions of
    impacts in a Python loop.
    """

    def kurtosis(x: np.ndarray) -> float:
        centred = x - np.mean(x)
        return float(np.mean(centred**4) / np.mean(centred**2) ** 2)

    short = SR * 2
    light = kurtosis(rain_noise(short, SR, rng=9, drops_per_second=150.0))
    heavy = kurtosis(rain_noise(short, SR, rng=9, drops_per_second=6000.0))
    assert heavy < light


def test_wind_gusts_modulate_the_envelope():
    """Gustiness is the operationally important part: it moves clip-level statistics.

    Compared by the standard deviation of the short-time RMS. A steady wind has an almost
    constant envelope; a gusty one does not.
    """

    def envelope_variation(x: np.ndarray) -> float:
        frames = x[: len(x) // 3200 * 3200].reshape(-1, 3200)
        levels = np.sqrt(np.mean(frames**2, axis=1))
        return float(np.std(levels) / np.mean(levels))

    steady = wind_noise(N, SR, rng=4, gust_depth=0.0)
    gusty = wind_noise(N, SR, rng=4, gust_depth=0.9)
    assert envelope_variation(gusty) > envelope_variation(steady)


def test_unknown_noise_names_raise():
    """A typo in a grid definition must fail immediately, not silently fall back to white."""
    with pytest.raises(ValueError, match="unknown noise"):
        make_noise("gaussian", 1000, SR)
    with pytest.raises(ValueError, match="unknown noise colour"):
        color_noise(1000, "beige", SR)


def test_gust_depth_is_validated():
    """A modulation depth of 1 or more would drive the envelope negative."""
    with pytest.raises(ValueError, match="gust_depth"):
        wind_noise(1000, SR, gust_depth=1.0)
