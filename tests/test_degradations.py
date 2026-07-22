"""Tests for the degradation mathematics.

Each degradation claims to produce an exact physical effect. These tests check that claim against
a closed-form answer wherever one exists -- the inverse square law, ISO 9613-1's absorption
coefficients, the uniform-quantiser noise power, an exponential decay's RT60 -- rather than
against the verifier, so that a shared misunderstanding between the two could not pass.

The end-to-end requested-versus-measured sweep lives in
:mod:`birdsong_robustness.validate`; this file tests the pieces.
"""

from __future__ import annotations

import numpy as np
import pytest

from birdsong_robustness import synth
from birdsong_robustness.degradations import (
    add_noise_at_snr,
    add_overlapping_calls,
    air_absorption_db_per_m,
    apply_air_absorption,
    apply_gain_db,
    apply_reverb,
    apply_spherical_spreading,
    band_limit,
    clip_to_fraction,
    design_brickwall_lowpass,
    gain_stage,
    hard_clip,
    mu_law_codec,
    quantize_pcm,
    resample_roundtrip,
    simulate_distance,
    soft_clip,
    synthesize_rir,
)
from birdsong_robustness.noise import make_noise
from birdsong_robustness.verify import (
    measure_clipped_fraction,
    measure_drr,
    measure_overlap_density,
    measure_rt60,
    measure_sir,
    measure_snr,
    measure_sqnr,
    predicted_sqnr_db,
)

SR = 32000


@pytest.fixture(scope="module")
def clip():
    """One synthetic bird-song-like clip with an exact activity mask, shared across tests."""
    return synth.bird_call(duration_s=5.0, sr=SR, rng=11, n_syllables=8)


# --------------------------------------------------------------------------------------
# Additive noise
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("snr_db", [20.0, 10.0, 0.0, -10.0])
@pytest.mark.parametrize("noise_name", ["white", "pink", "brown", "wind", "rain"])
def test_snr_is_exact_on_the_active_basis(clip, snr_db, noise_name):
    """The requested SNR must be recoverable from the output waveform, not from the gain.

    The verifier subtracts the clean signal back out, so this also catches a length mismatch or
    a sample slip, which a gain readback would not.
    """
    masker = make_noise(noise_name, len(clip.audio), SR, rng=3)
    mixture, _ = add_noise_at_snr(clip.audio, masker, snr_db, active_mask=clip.active)
    assert measure_snr(clip.audio, mixture, mask=clip.active) == pytest.approx(snr_db, abs=0.01)


def test_returned_gain_reconstructs_the_noise_component(clip):
    """``gain`` must let a caller recover the exact noise that was added."""
    masker = make_noise("pink", len(clip.audio), SR, rng=3)
    mixture, gain = add_noise_at_snr(clip.audio, masker, 0.0, active_mask=clip.active)
    np.testing.assert_allclose(mixture - clip.audio, gain * masker, atol=1e-12)


def test_global_basis_snr_is_optimistic_on_a_sparse_clip(clip):
    """The methodological point of the axis, as arithmetic.

    Requesting 0 dB on the global basis produces a clip whose SNR *during the call* is higher by
    ``-10*log10(active_fraction_weighted)``. For a clip that is mostly silence that is several
    decibels of unearned headroom, and it varies per recording, so it is not even a constant
    bias that could be corrected afterwards.
    """
    masker = make_noise("pink", len(clip.audio), SR, rng=3)
    mixture, _ = add_noise_at_snr(clip.audio, masker, 0.0, basis="global")
    during_the_call = measure_snr(clip.audio, mixture, mask=clip.active)
    over_the_file = measure_snr(clip.audio, mixture, basis="global")
    assert over_the_file == pytest.approx(0.0, abs=0.01)
    assert during_the_call > over_the_file + 3.0


def test_noise_is_length_matched(clip):
    """A short or long noise record must be padded or trimmed, never broadcast or refused."""
    short = make_noise("pink", len(clip.audio) // 2, SR, rng=3)
    mixture, _ = add_noise_at_snr(clip.audio, short, 0.0, active_mask=clip.active)
    assert len(mixture) == len(clip.audio)


def test_unknown_basis_raises(clip):
    with pytest.raises(ValueError, match="basis must be"):
        add_noise_at_snr(clip.audio, np.ones(len(clip.audio)), 0.0, basis="whole-file")


def test_active_basis_without_mask_or_sr_raises(clip):
    """Silently falling back to an energy segmenter would hide the error it introduces."""
    with pytest.raises(ValueError, match="requires either"):
        add_noise_at_snr(clip.audio, np.ones(len(clip.audio)), 0.0, basis="active")


# --------------------------------------------------------------------------------------
# Overlap
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def soundscape():
    """A multi-source scene, returned with its isolated sources for the overlap tests."""
    _, sources = synth.soundscape(duration_s=10.0, sr=SR, rng=21, n_species=4, calls_per_species=6)
    return sources


@pytest.mark.parametrize("sir_db", [10.0, 0.0, -5.0])
def test_sir_is_exact_over_the_overlap_region(soundscape, sir_db):
    """SIR is only meaningful where the two actually coincide, and must be exact there."""
    target, *interferers = soundscape
    result = add_overlapping_calls(
        target.audio,
        [s.audio for s in interferers],
        SR,
        sir_db=sir_db,
        density=0.5,
        target_active=target.active,
        rng=13,
    )
    measured = measure_sir(target.audio, result.interference, result.overlap_region)
    assert measured == pytest.approx(sir_db, abs=0.01)


@pytest.mark.parametrize("density", [0.25, 0.5, 0.75, 1.0])
def test_achieved_density_tracks_the_request(soundscape, density):
    """Coverage is quantised by syllable length, so the tolerance is a few percent, not exact.

    What must hold is that the achieved value is reported and is close. A mixer that silently
    returned 0.6 for a requested 0.25 would make the whole axis meaningless.
    """
    target, *interferers = soundscape
    result = add_overlapping_calls(
        target.audio,
        [s.audio for s in interferers],
        SR,
        density=density,
        target_active=target.active,
        rng=13,
    )
    assert result.achieved_density == pytest.approx(density, abs=0.05)


def test_density_measured_from_the_mixture_agrees_with_the_bookkeeping(soundscape):
    """An independent read of the summed waveform must not disagree materially.

    The verifier gates on the whole interference track's RMS while the mixer gates per placed
    segment, so a small difference is expected and is the reason this tolerance is 0.05 rather
    than zero.
    """
    target, *interferers = soundscape
    result = add_overlapping_calls(
        target.audio,
        [s.audio for s in interferers],
        SR,
        density=0.5,
        target_active=target.active,
        rng=13,
    )
    measured = measure_overlap_density(target.active, result.interference)
    assert measured == pytest.approx(result.achieved_density, abs=0.05)


def test_overlap_region_excludes_silence_inside_a_segment(soundscape):
    """The whole reason the audibility gate exists.

    Every sample credited as overlap must carry actual interference energy. If the region were
    span-based it would include the digital silence between an interferer's syllables, which
    both inflates the density and makes the SIR set over that region roughly a decibel off.
    """
    target, *interferers = soundscape
    result = add_overlapping_calls(
        target.audio,
        [s.audio for s in interferers],
        SR,
        density=0.5,
        target_active=target.active,
        rng=13,
    )
    assert np.all(np.abs(result.interference[result.overlap_region]) > 0.0)


def test_overlap_requires_interferers(soundscape):
    target = soundscape[0]
    with pytest.raises(ValueError, match="at least one interferer"):
        add_overlapping_calls(target.audio, [], SR, target_active=target.active)


def test_overlap_rejects_out_of_range_density(soundscape):
    target, *interferers = soundscape
    with pytest.raises(ValueError, match="density"):
        add_overlapping_calls(
            target.audio,
            [s.audio for s in interferers],
            SR,
            density=1.5,
            target_active=target.active,
        )


# --------------------------------------------------------------------------------------
# Distance
# --------------------------------------------------------------------------------------


def test_spherical_spreading_is_six_db_per_doubling(clip):
    """Pressure falls as 1/r, so the level falls 6 dB per doubling -- not 3."""
    near = apply_spherical_spreading(clip.audio, 10.0)
    far = apply_spherical_spreading(clip.audio, 20.0)
    ratio = float(np.sqrt(np.mean(far**2) / np.mean(near**2)))
    assert 20.0 * np.log10(ratio) == pytest.approx(-6.0206, abs=1e-6)


def test_air_absorption_matches_iso_9613_tabulated_values():
    """Spot values against ISO 9613-1 Table 1 at 20 C, 70% RH, one atmosphere.

    Tabulated in dB/km: 125 Hz 0.3, 500 Hz 2.8, 1 kHz 5.0, 2 kHz 9.0, 4 kHz 22.9, 8 kHz 76.6.
    A 3% tolerance covers the standard's own rounding. Getting this right is the difference
    between modelling distance as a low-pass filter and modelling it as a volume knob.
    """
    freqs = np.array([125.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    expected_db_per_km = np.array([0.3, 2.8, 5.0, 9.0, 22.9, 76.6])
    measured = air_absorption_db_per_m(freqs, 20.0, 70.0) * 1000.0
    np.testing.assert_allclose(measured, expected_db_per_km, rtol=0.03, atol=0.05)


def test_air_absorption_rises_with_frequency():
    """Absorption grows roughly with frequency squared; it must be monotonic across the band."""
    freqs = np.array([100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0])
    alpha = air_absorption_db_per_m(freqs)
    assert np.all(np.diff(alpha) > 0)


def test_air_absorption_depends_on_humidity_non_monotonically():
    """The humidity dependence is why the full ISO expression is used instead of a fit.

    At 4 kHz, dry air absorbs far more than humid air -- a fit tuned at 70% would be badly wrong
    at 20%, and distance simulation would be silently mis-tilted for any deployment in a dry
    climate.
    """
    dry = float(air_absorption_db_per_m(4000.0, 20.0, 20.0))
    humid = float(air_absorption_db_per_m(4000.0, 20.0, 70.0))
    assert dry > humid * 2.0


def test_air_absorption_tilts_the_spectrum_not_just_the_level(clip):
    """The point of the axis: 8 kHz must lose much more than 1 kHz over the same distance."""
    loss_1k = float(air_absorption_db_per_m(1000.0)) * 100.0
    loss_8k = float(air_absorption_db_per_m(8000.0)) * 100.0
    assert loss_8k > loss_1k * 10.0


def test_apply_air_absorption_preserves_length_and_is_a_no_op_at_zero(clip):
    filtered = apply_air_absorption(clip.audio, SR, 100.0)
    assert len(filtered) == len(clip.audio)
    np.testing.assert_array_equal(apply_air_absorption(clip.audio, SR, 0.0), clip.audio)


def test_apply_air_absorption_rejects_even_taps(clip):
    with pytest.raises(ValueError, match="odd"):
        apply_air_absorption(clip.audio, SR, 100.0, n_taps=1024)


@pytest.mark.parametrize("rt60", [0.1, 0.2, 0.4, 0.8])
def test_synthesised_rir_has_the_requested_rt60(rt60):
    """Schroeder backward integration must recover the decay constant that generated the tail.

    The direct path is skipped before integrating: with a high direct-to-reverberant ratio the
    Schroeder curve plunges by roughly the DRR in the first few samples, and fitting through
    that plunge measures the DRR rather than the RT60.
    """
    rir = synthesize_rir(SR, rt60_s=rt60, drr_db=6.0, rng=2)
    assert measure_rt60(rir, SR, skip_ms=8.0) == pytest.approx(rt60, rel=0.05)


@pytest.mark.parametrize("drr_db", [20.0, 6.0, 0.0, -10.0, -26.0])
def test_synthesised_rir_has_the_requested_drr(drr_db):
    """The DRR is what carries the distance information in a reverberant field.

    ``direct_index=0`` is supplied rather than located by peak magnitude: below about -15 dB the
    tail contains samples louder than the direct impulse, and peak-picking would silently
    flatten the measurement out.
    """
    rir = synthesize_rir(SR, rt60_s=0.4, drr_db=drr_db, rng=2)
    assert measure_drr(rir, SR, direct_index=0) == pytest.approx(drr_db, abs=0.01)


def test_reverb_preserves_length(clip):
    """A convolution grows a signal; the tail must be truncated to keep annotations aligned."""
    rir = synthesize_rir(SR, rt60_s=0.5, rng=2)
    assert len(apply_reverb(clip.audio, rir)) == len(clip.audio)


def test_simulate_distance_attenuates_and_low_passes(clip):
    """Distance must do both things at once, which a gain change cannot.

    Reverberation is disabled here because it adds energy back and would confound a pure
    attenuation reading -- the same reason the validation harness disables it for these rows.
    """
    from birdsong_robustness.verify import measure_octave_band_levels

    far = simulate_distance(clip.audio, SR, distance_m=100.0, include_reverb=False)
    assert float(np.sqrt(np.mean(far**2))) < float(np.sqrt(np.mean(clip.audio**2)))

    near_bands = measure_octave_band_levels(clip.audio, SR)
    far_bands = measure_octave_band_levels(far, SR)
    loss_1k = near_bands[1000.0] - far_bands[1000.0]
    loss_8k = near_bands[8000.0] - far_bands[8000.0]
    assert loss_8k > loss_1k + 5.0


def test_simulate_distance_preserves_length(clip):
    degraded = simulate_distance(clip.audio, SR, distance_m=50.0, rng=1)
    assert len(degraded) == len(clip.audio)


def test_spreading_rejects_non_positive_distance(clip):
    with pytest.raises(ValueError, match="distances must be positive"):
        apply_spherical_spreading(clip.audio, 0.0)


# --------------------------------------------------------------------------------------
# Bandwidth and codecs
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("cutoff", [2000.0, 4000.0, 8000.0])
def test_band_limit_puts_the_three_db_point_where_requested(clip, cutoff):
    """A Butterworth low-pass is -3 dB at its cutoff by definition, applied causally.

    ``sosfiltfilt`` would apply the magnitude response twice and move the measured -3 dB point,
    which is why the implementation uses ``sosfilt``. This test would catch that regression.
    """
    from birdsong_robustness.verify import measure_bandwidth

    measured = measure_bandwidth(lambda x: band_limit(x, SR, high_hz=cutoff), SR, rng=5)
    assert measured == pytest.approx(cutoff, rel=0.02)


def test_band_limit_high_pass_removes_low_frequencies():
    """The low cut matters too: windshields and cheap capsules roll off the bottom."""
    t = np.arange(SR) / SR
    low = np.sin(2 * np.pi * 100 * t)
    filtered = band_limit(low, SR, low_hz=1000.0)
    assert float(np.sqrt(np.mean(filtered[SR // 2 :] ** 2))) < 0.02


def test_band_limit_validates_its_edges():
    with pytest.raises(ValueError, match="require"):
        band_limit(np.zeros(1000), SR, low_hz=8000.0, high_hz=2000.0)
    with pytest.raises(ValueError, match="require"):
        band_limit(np.zeros(1000), SR, high_hz=SR)


def test_band_limit_without_edges_is_the_identity():
    x = np.arange(100, dtype=float)
    np.testing.assert_array_equal(band_limit(x, SR), x)


@pytest.mark.parametrize("target_sr", [8000, 16000, 22050])
def test_resample_roundtrip_empties_the_band_above_nyquist(clip, target_sr):
    """The claim that matters: the top band is gone, not merely dented.

    ``scipy.signal.resample_poly``'s default anti-alias filter reaches only about 20 dB of
    rejection just above the target Nyquist. This asserts 80 dB, which is what the explicit
    Kaiser design in ``design_brickwall_lowpass`` is specified for.
    """
    from birdsong_robustness.verify import measure_stopband_attenuation

    rejection = measure_stopband_attenuation(
        lambda x: resample_roundtrip(x, SR, target_sr), SR, target_sr / 2.0, rng=5
    )
    assert rejection >= 80.0


def test_resample_roundtrip_preserves_length_and_upward_is_a_no_op(clip):
    """A round trip cannot manufacture bandwidth, so an upward target must not change anything."""
    assert len(resample_roundtrip(clip.audio, SR, 16000)) == len(clip.audio)
    np.testing.assert_array_equal(resample_roundtrip(clip.audio, SR, SR * 2), clip.audio)


def test_brickwall_design_is_odd_length():
    """An even-length FIR would have a half-sample group delay that cannot be compensated."""
    assert len(design_brickwall_lowpass(SR, 8000.0)) % 2 == 1


def test_brickwall_design_rejects_edges_outside_the_band():
    with pytest.raises(ValueError, match="stopband edge"):
        design_brickwall_lowpass(SR, SR)


@pytest.mark.parametrize("bits", [16, 12, 10, 8])
def test_quantisation_sqnr_matches_theory(clip, bits):
    """Measured SQNR must match ``10*log10(P_x / (step**2/12))`` over the active region.

    Not the textbook ``6.02*bits + 1.76`` dB: that assumes a full-scale signal, and this clip
    sits well below full scale, so the shortcut would overstate the prediction by the crest
    factor and every row would look wrong in the same direction.
    """
    quantised = quantize_pcm(clip.audio, bits)
    predicted = predicted_sqnr_db(clip.audio, bits, clip.active)
    assert measure_sqnr(clip.audio, quantised, clip.active) == pytest.approx(predicted, abs=0.5)


def test_quantisation_output_is_on_the_lattice(clip):
    """Every output sample must be an exact multiple of the step, or it is not quantised."""
    bits = 8
    step = 2.0 / (2**bits)
    quantised = quantize_pcm(clip.audio, bits)
    np.testing.assert_allclose(quantised / step, np.round(quantised / step), atol=1e-9)


def test_dither_costs_sqnr_but_decorrelates_the_error(clip):
    """Triangular dither trades about 4.8 dB of SQNR for an error uncorrelated with the signal.

    The trade is the point: undithered quantisation produces harmonic distortion that tracks the
    signal, which a classifier can key on in ways that do not generalise.
    """
    plain = quantize_pcm(clip.audio, 8)
    dithered = quantize_pcm(clip.audio, 8, dither=True, rng=1)
    sqnr_plain = measure_sqnr(clip.audio, plain, clip.active)
    sqnr_dithered = measure_sqnr(clip.audio, dithered, clip.active)
    assert sqnr_dithered < sqnr_plain

    corr_plain = abs(float(np.corrcoef(clip.audio, plain - clip.audio)[0, 1]))
    corr_dithered = abs(float(np.corrcoef(clip.audio, dithered - clip.audio)[0, 1]))
    assert corr_dithered < corr_plain


def test_quantisation_validates_bit_depth():
    with pytest.raises(ValueError, match="bits must be"):
        quantize_pcm(np.zeros(10), 1)
    with pytest.raises(ValueError, match="bits must be"):
        quantize_pcm(np.zeros(10), 32)


def test_mu_law_holds_sqnr_across_the_dynamic_range(clip):
    """The reason 8-bit mu-law and 8-bit PCM are both in the grid rather than one of them.

    Logarithmic companding keeps SQNR roughly constant across amplitude; a uniform quantiser is
    better at the top and much worse at the bottom. Quiet distant calls -- the ones a field
    deployment is most likely to miss -- live at the bottom.
    """
    from birdsong_robustness.verify import measure_sqnr_by_amplitude_band

    pcm_bands = measure_sqnr_by_amplitude_band(clip.audio, quantize_pcm(clip.audio, 8), clip.active)
    mu_bands = measure_sqnr_by_amplitude_band(
        clip.audio, mu_law_codec(clip.audio, bits=8), clip.active
    )
    pcm_spread = max(pcm_bands.values()) - min(pcm_bands.values())
    mu_spread = max(mu_bands.values()) - min(mu_bands.values())
    assert mu_spread < pcm_spread / 4.0

    quietest = min(pcm_bands)
    assert mu_bands[quietest] > pcm_bands[quietest] + 5.0


def test_mu_law_roundtrip_is_monotonic_and_bounded():
    """Companding must preserve ordering, or it is distorting rather than quantising."""
    x = np.linspace(-1.0, 1.0, 2001)
    y = mu_law_codec(x)
    assert np.all(np.diff(y) >= -1e-12)
    assert np.max(np.abs(y)) <= 1.0 + 1e-12


# --------------------------------------------------------------------------------------
# Gain and clipping
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("gain_db", [-40.0, -6.0, 0.0, 6.0, 20.0])
def test_gain_is_exactly_the_requested_gain(clip, gain_db):
    """No hidden normalisation. This is the assertion that protects three other axes."""
    amplified = apply_gain_db(clip.audio, gain_db)
    ratio = float(np.sqrt(np.mean(amplified**2) / np.mean(clip.audio**2)))
    assert 20.0 * np.log10(ratio) == pytest.approx(gain_db, abs=1e-9)


@pytest.mark.parametrize("fraction", [0.001, 0.01, 0.05, 0.1])
def test_clip_to_fraction_hits_the_requested_fraction(clip, fraction):
    """Parameterising by clipped fraction rather than threshold is what makes the axis portable.

    A -3 dBFS threshold destroys a hot recording and leaves a quiet one untouched, so a grid
    over thresholds is confounded with recording level. A grid over clipped fraction is not.
    """
    clipped, threshold = clip_to_fraction(clip.audio, fraction)
    assert measure_clipped_fraction(clipped, threshold) == pytest.approx(fraction, abs=1e-4)


def test_clipping_reduces_crest_factor(clip):
    """Clipping is a peak-reduction, and the crest factor is the physical consequence."""
    from birdsong_robustness.verify import measure_crest_factor_db

    clipped, _ = clip_to_fraction(clip.audio, 0.05)
    assert measure_crest_factor_db(clipped) < measure_crest_factor_db(clip.audio) - 3.0


def test_zero_fraction_leaves_the_signal_alone(clip):
    clipped, _ = clip_to_fraction(clip.audio, 0.0)
    np.testing.assert_array_equal(clipped, clip.audio)


def test_clip_to_fraction_rejects_a_fraction_of_one(clip):
    with pytest.raises(ValueError, match="fraction must be"):
        clip_to_fraction(clip.audio, 1.0)


def test_hard_clip_bounds_the_signal(clip):
    clipped = hard_clip(clip.audio, 0.1)
    assert np.max(np.abs(clipped)) <= 0.1 + 1e-12


def test_soft_clip_has_a_smooth_transfer_curve_and_hard_clip_a_corner():
    """The defining difference between the two saturations, tested on the transfer curve itself.

    "Lower-order distortion" is exactly what a smooth transfer curve means: hard clipping has a
    derivative discontinuity at the threshold -- a corner -- and a corner is what generates the
    slowly-decaying high-order harmonics that splatter broadband energy across the spectrum.
    ``tanh`` has no corner; its curvature is bounded everywhere. Measured as the peak second
    difference of the input-output curve over a ramp, the hard corner is orders of magnitude
    sharper. Comparing harmonic spectra directly is avoided here because the result depends on
    the arbitrary drive setting; the transfer-curve geometry does not.
    """
    ramp = np.linspace(-2.0, 2.0, 20001)
    hard_curvature = np.max(np.abs(np.diff(hard_clip(ramp, 1.0), 2)))
    soft_curvature = np.max(np.abs(np.diff(soft_clip(ramp / 2.0, drive=3.0), 2)))
    assert hard_curvature > 100.0 * soft_curvature


def test_soft_clip_approaches_linear_at_low_drive():
    """As drive tends to zero the saturation must vanish, or the parameter means nothing."""
    ramp = np.linspace(-1.0, 1.0, 4001)
    gentle = soft_clip(ramp, drive=0.01)
    correlation = float(np.corrcoef(ramp, gentle)[0, 1])
    assert correlation > 0.9999


def test_gain_stage_orders_gain_then_clip_then_quantise(clip):
    """Ordering is physical: a converter clips its input, it does not clip its own output.

    An overdriven stage must show clipped samples at the limit and a quantised output lattice.
    """
    driven = gain_stage(clip.audio, gain_db=20.0, bits=12, clip_threshold=1.0)
    step = 2.0 / (2**12)
    assert np.max(np.abs(driven)) <= 1.0 + 1e-12
    assert measure_clipped_fraction(driven, 1.0 - step) > 0.0
    np.testing.assert_allclose(driven / step, np.round(driven / step), atol=1e-9)


def test_soft_clip_rejects_zero_drive():
    with pytest.raises(ValueError, match="drive must be positive"):
        soft_clip(np.zeros(10), drive=0.0)
