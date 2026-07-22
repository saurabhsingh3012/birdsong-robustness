"""Tests for the shared signal primitives.

These are the definitions the degradations and the verifier both build on. If ``power`` and
``rms`` disagreed about what they measure, or ``apply_fir_zero_delay`` left a group delay in
place, every requested-versus-measured number downstream would be wrong in a way that looks like
a passing test. So they get checked against closed-form answers rather than against each other.
"""

from __future__ import annotations

import numpy as np
import pytest

from birdsong_robustness._dsp import (
    apply_fir_zero_delay,
    as_float64,
    audible_mask,
    db,
    db_amplitude,
    dbfs,
    energy_active_mask,
    frame_signal,
    match_length,
    power,
    rms,
)


def test_power_and_rms_of_a_sine_are_analytic():
    """A unit-amplitude sine has mean square 1/2 and RMS 1/sqrt(2), exactly."""
    t = np.arange(48000) / 48000.0
    x = np.sin(2 * np.pi * 100.0 * t)  # whole number of cycles, so no partial-period error
    assert power(x) == pytest.approx(0.5, abs=1e-9)
    assert rms(x) == pytest.approx(1.0 / np.sqrt(2.0), abs=1e-9)


def test_db_conventions_differ_by_a_factor_of_two():
    """Power ratios use 10*log10, amplitude ratios 20*log10. Conflating them is a 2x error."""
    assert db(100.0) == pytest.approx(20.0)
    assert db_amplitude(100.0) == pytest.approx(40.0)
    assert db(1.0) == pytest.approx(0.0)


def test_dbfs_of_full_scale_square_wave_is_zero():
    """A signal at +/-1 everywhere has an RMS of 1, i.e. 0 dBFS."""
    x = np.tile([1.0, -1.0], 2048)
    assert dbfs(x) == pytest.approx(0.0, abs=1e-9)


def test_power_with_mask_ignores_the_silence():
    """Masked power must not be diluted by the zeros outside the mask.

    This is the arithmetic behind the SNR basis problem: the same signal measures 3 dB lower on
    a global basis than on an active basis when exactly half of it is silent.
    """
    x = np.concatenate([np.ones(1000), np.zeros(1000)])
    mask = np.concatenate([np.ones(1000, dtype=bool), np.zeros(1000, dtype=bool)])
    assert power(x) == pytest.approx(0.5)
    assert power(x, mask) == pytest.approx(1.0)
    assert db(power(x, mask) / power(x)) == pytest.approx(3.0103, abs=1e-3)


def test_power_rejects_an_empty_mask():
    """An all-False mask has no defined power and must raise rather than return a NaN."""
    with pytest.raises(ValueError, match="no samples"):
        power(np.ones(10), np.zeros(10, dtype=bool))


def test_as_float64_rejects_multichannel():
    """Everything in the package is mono; a stereo array must fail loudly, not be flattened."""
    with pytest.raises(ValueError, match="1-D mono"):
        as_float64(np.zeros((2, 100)))


def test_frame_signal_shape_and_content():
    """Framing must drop the trailing partial frame and keep the hop exact."""
    x = np.arange(100, dtype=float)
    frames = frame_signal(x, frame_len=10, hop=5)
    assert frames.shape == (19, 10)
    assert frames[0][0] == 0.0
    assert frames[1][0] == 5.0
    assert frames[-1][-1] == 99.0


def test_frame_signal_returns_typed_empty_when_too_short():
    """Callers rely on ``.shape[1]`` even when there are no frames."""
    frames = frame_signal(np.zeros(4), frame_len=10, hop=5)
    assert frames.shape == (0, 10)


def test_match_length_trims_and_pads():
    """Length preservation is what keeps a degraded clip aligned with its annotations."""
    assert len(match_length(np.ones(100), 50)) == 50
    padded = match_length(np.ones(50), 100)
    assert len(padded) == 100
    assert padded[75] == 0.0


def test_energy_active_mask_finds_the_burst():
    """A loud burst in a quiet clip must be marked active and the silence must not be."""
    sr = 16000
    x = np.zeros(sr)
    t = np.arange(sr // 4) / sr
    x[sr // 2 : sr // 2 + sr // 4] = np.sin(2 * np.pi * 3000 * t)
    mask = energy_active_mask(x, sr)
    assert mask[sr // 2 + 1000]
    assert not mask[1000]
    assert 0.2 < mask.mean() < 0.4


def test_energy_active_mask_on_silence_is_all_true():
    """Digital silence is degenerate; returning an all-False mask would break every caller."""
    assert energy_active_mask(np.zeros(8000), 16000).all()


def test_apply_fir_zero_delay_compensates_group_delay():
    """A symmetric FIR delays by (L-1)/2 samples; that delay must be removed.

    Checked against an identity filter, where the output must equal the input sample for
    sample. A residual slip of even a couple of samples would appear downstream as several
    decibels of phantom noise in the residual-based SNR measurement.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2048)
    taps = np.zeros(101)
    taps[50] = 1.0  # identity, centred: pure group delay, unity magnitude
    np.testing.assert_allclose(apply_fir_zero_delay(x, taps), x, atol=1e-12)


def test_apply_fir_zero_delay_rejects_even_length():
    """An even-length FIR has a half-sample delay that cannot be removed by an integer shift."""
    with pytest.raises(ValueError, match="odd-length"):
        apply_fir_zero_delay(np.zeros(100), np.ones(10))


def test_audible_mask_is_invariant_to_scaling():
    """The gate is relative to the signal's own RMS, so it must not move when the level does.

    This matters because the overlap mixer chooses the region *before* applying the SIR gain. A
    scale-dependent gate would make the region depend on the level it is used to set.
    """
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4096)
    quiet = audible_mask(x * 1e-6)
    loud = audible_mask(x * 1e6)
    np.testing.assert_array_equal(quiet, loud)


def test_audible_mask_excludes_digital_silence():
    """Exact zeros are never audible, which is the whole reason the gate exists."""
    x = np.concatenate([np.ones(100), np.zeros(100)])
    mask = audible_mask(x)
    assert mask[:100].all()
    assert not mask[100:].any()


def test_audible_mask_on_silence_is_all_false():
    """An identically zero signal has no audible samples and must not divide by its own RMS."""
    assert not audible_mask(np.zeros(100)).any()
