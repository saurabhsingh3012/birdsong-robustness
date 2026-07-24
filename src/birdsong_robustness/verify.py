"""Independent measurement of what a degradation actually did.

Nothing in :mod:`birdsong_robustness.degradations` is trusted here. Every function in this
module re-derives an effect from the *output waveform*, using a method that does not share code
with the function that produced it. Where a shared definition is unavoidable -- what "power"
means, say -- it lives in :mod:`birdsong_robustness._dsp` so both sides provably agree.

Why this module is what makes the real numbers quotable
-------------------------------------------------------
A real classifier has now been run against this benchmark -- BirdNET, on a Xeno-canto set (see
the README's "Results -- real audio"). A curve such as "BirdNET holds to about 5 dB SNR, then
falls off" is only worth quoting if 0 dB SNR really is 0 dB SNR and 100 m really is 100 m. This
module is what earns that: independently of the degradation code, it establishes that requesting
0 dB SNR produces a mixture measuring 0 dB, that requesting a 100 m propagation attenuates by the
amount ISO 9613-1 predicts for 100 m, that requesting an 8 kHz band limit produces a signal
whose measured -3 dB point is at 8 kHz, that requesting a 0.6 s RT60 produces an impulse
response whose Schroeder decay measures 0.6 s. A benchmark whose degradation axes have not been
verified is producing numbers about its own bugs. This module is what stops that.

Every measurement below returns a plain float in physical units, so the validation harness can
subtract it from the requested value and print the error.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import signal as sps

from ._dsp import (
    EPS,
    as_float64,
    audible_mask,
    db,
    db_amplitude,
    energy_active_mask,
    match_length,
    power,
    rms,
    trapezoid,
)

__all__ = [
    "measure_bandwidth",
    "measure_clipped_fraction",
    "measure_crest_factor_db",
    "measure_drr",
    "measure_level_dbfs",
    "measure_octave_band_levels",
    "measure_overlap_density",
    "measure_rt60",
    "measure_snr",
    "measure_spectral_slope",
    "measure_sqnr",
    "measure_transfer_function",
    "predicted_sqnr_db",
]


# --------------------------------------------------------------------------------------
# Level and ratio measurements
# --------------------------------------------------------------------------------------


def measure_snr(
    clean: np.ndarray,
    degraded: np.ndarray,
    mask: np.ndarray | None = None,
    sr: int | None = None,
    basis: str = "active",
) -> float:
    """Measure the SNR of ``degraded`` relative to ``clean`` by residual subtraction.

    The noise component is recovered as ``degraded - clean`` rather than read back from the
    mixer's internal scale factor. That is the point: it measures the waveform that was
    actually produced, so it catches length mismatches, sample misalignment, accidental
    renormalisation, and any filtering the degradation applied to the signal path -- all of
    which a scale-factor readback would miss.

    Args:
        clean: Reference signal.
        degraded: Signal after degradation. Must be sample-aligned with ``clean``.
        mask: Boolean mask for the signal-power basis. Estimated from ``clean`` if omitted and
            ``basis="active"``.
        sr: Sample rate, needed only to estimate a mask.
        basis: ``"active"`` or ``"global"``. See
            :func:`birdsong_robustness.degradations.add_noise_at_snr` for why this matters.

    Returns:
        SNR in dB.
    """
    x = as_float64(clean)
    y = match_length(as_float64(degraded), len(x))
    residual = y - x

    if basis == "active":
        if mask is None:
            if sr is None:
                raise ValueError("basis='active' requires either mask or sr")
            mask = energy_active_mask(x, sr)
        use_mask = np.asarray(mask, dtype=bool)
    elif basis == "global":
        use_mask = None
    else:
        raise ValueError(f"basis must be 'active' or 'global', got {basis!r}")

    return db(power(x, use_mask) / max(power(residual), EPS))


def measure_level_dbfs(x: np.ndarray, mask: np.ndarray | None = None) -> float:
    """RMS level in dB relative to full scale."""
    return db_amplitude(rms(x, mask))


def measure_crest_factor_db(x: np.ndarray) -> float:
    """Peak-to-RMS ratio in dB. Falls when a signal is clipped or compressed."""
    arr = as_float64(x)
    return db_amplitude(float(np.max(np.abs(arr))) / max(rms(arr), EPS))


def measure_sqnr(clean: np.ndarray, quantized: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Signal-to-quantisation-noise ratio in dB, from the residual, over an optional ``mask``.

    Supply the same mask to :func:`predicted_sqnr_db` or the two will not be comparable -- see
    the note there about digital silence.
    """
    x = as_float64(clean)
    y = match_length(as_float64(quantized), len(x))
    use_mask = None if mask is None else np.asarray(mask, dtype=bool)
    return db(power(x, use_mask) / max(power(y - x, use_mask), EPS))


def predicted_sqnr_db(clean: np.ndarray, bits: int, mask: np.ndarray | None = None) -> float:
    """Theoretical SQNR for uniform quantisation of ``clean`` at ``bits`` bits.

    Uses the exact form ``10 * log10(P_x / (step**2 / 12))`` rather than the textbook
    ``6.02 * bits + 1.76`` dB. The textbook version assumes a full-scale signal; bird
    recordings typically sit 15--30 dB below full scale, and using the shortcut would predict
    an SQNR that is too high by exactly that margin. Comparing a measurement against a
    prediction that is wrong by the crest factor would produce a confident, meaningless table.

    The digital-silence trap
    -----------------------
    ``step**2 / 12`` is the variance of a quantisation error assumed uniform over one step. That
    assumption fails wherever the input is *exactly* zero, because a quantiser maps zero to zero
    with no error at all. In a sparse recording -- and bird recordings are extremely sparse --
    most samples are in that state, so the measured error power over the whole clip is lower
    than ``step**2 / 12`` by roughly the active fraction, and the measurement appears to beat
    theory by several decibels.

    This is the same basis problem that
    :func:`birdsong_robustness.degradations.add_noise_at_snr` documents for SNR, reappearing on
    a different axis: a per-clip average over a signal that is mostly silence does not describe
    what happens where the signal is. Passing the active ``mask`` to both this function and
    :func:`measure_sqnr` makes them agree.

    A second caveat, which no mask fixes: the uniform-error model also requires the signal to be
    large compared with one quantiser step. At very coarse bit depths a quiet signal falls below
    ``step / 2`` and is quantised to zero outright, so the error equals the signal and the SQNR
    floors out. Predictions below roughly 8 bits should be read as an upper bound.

    Args:
        clean: Reference signal.
        bits: Bit depth.
        mask: Region over which to compute signal power. Strongly recommended.
    """
    step = 2.0 / (2**bits)
    use_mask = None if mask is None else np.asarray(mask, dtype=bool)
    return db(power(clean, use_mask) / (step**2 / 12.0))


def measure_sqnr_by_amplitude_band(
    clean: np.ndarray,
    coded: np.ndarray,
    mask: np.ndarray | None = None,
    percentiles: tuple[tuple[float, float], ...] = ((0, 25), (25, 50), (50, 75), (75, 100)),
) -> dict[tuple[float, float], float]:
    """SQNR broken down by the instantaneous amplitude of the reference signal.

    A single global SQNR number hides the difference between a uniform quantiser and a
    companding one, because the global figure is dominated by the loudest samples where the two
    behave similarly. Splitting by amplitude percentile exposes the actual trade: logarithmic
    companding holds SQNR roughly constant across the dynamic range, while a uniform quantiser
    is better at the top and much worse at the bottom.

    That distinction matters here because distant and quiet calls -- exactly the ones a field
    deployment is most likely to miss -- live in the bottom amplitude bands.

    Args:
        clean: Reference signal.
        coded: Signal after the codec round trip.
        mask: Restrict to this region (typically the active mask).
        percentiles: Amplitude percentile ranges to report.

    Returns:
        Mapping from percentile range to SQNR in dB.
    """
    x = as_float64(clean)
    y = match_length(as_float64(coded), len(x))
    idx = np.flatnonzero(np.ones(len(x), dtype=bool) if mask is None else np.asarray(mask, bool))
    if idx.size == 0:
        raise ValueError("mask selects no samples")

    envelope = np.abs(x[idx])
    out: dict[tuple[float, float], float] = {}
    for lo, hi in percentiles:
        t_lo, t_hi = np.percentile(envelope, [lo, hi])
        sel = idx[(envelope >= t_lo) & (envelope <= t_hi)]
        if sel.size < 8:
            continue
        sig = float(np.mean(x[sel] ** 2))
        err = float(np.mean((y[sel] - x[sel]) ** 2))
        out[(lo, hi)] = db(sig / max(err, EPS))
    return out


# --------------------------------------------------------------------------------------
# Spectral measurements
# --------------------------------------------------------------------------------------


def _welch(x: np.ndarray, sr: int, nperseg: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD estimate with a fixed configuration, so all spectra here are comparable."""
    arr = as_float64(x)
    nperseg = min(nperseg, len(arr))
    freqs, psd = sps.welch(arr, fs=sr, nperseg=nperseg, noverlap=nperseg // 2, window="hann")
    return freqs, psd


def measure_spectral_slope(
    x: np.ndarray,
    sr: int,
    f_min: float = 200.0,
    f_max: float = 8000.0,
) -> float:
    """Fit the PSD slope in dB per octave over ``[f_min, f_max]``.

    Used to confirm that the noise generators produce the colours they claim: white should
    measure 0 dB/octave, pink -3.01, brown -6.02. Those are exact analytic targets, so any
    deviation is a real defect in the generator rather than a matter of taste.

    The fit band deliberately excludes the lowest frequencies, where the ``f_min_hz`` knee in
    :func:`birdsong_robustness.noise.colored_noise` flattens the spectrum by design, and the
    region approaching Nyquist, where the Welch estimate is affected by the anti-alias edge.

    Returns:
        Slope in dB/octave (negative means falling with frequency).
    """
    freqs, psd = _welch(x, sr)
    band = (freqs >= f_min) & (freqs <= f_max) & (psd > 0)
    if band.sum() < 8:
        raise ValueError("not enough spectral bins in the fit band")
    log_f = np.log10(freqs[band])
    psd_db = 10.0 * np.log10(psd[band])
    slope_per_decade = float(np.polyfit(log_f, psd_db, 1)[0])
    return slope_per_decade * float(np.log10(2.0))


def measure_transfer_function(
    process: Callable[[np.ndarray], np.ndarray],
    sr: int,
    duration_s: float = 4.0,
    rng: np.random.Generator | int | None = None,
    nperseg: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the magnitude response of an arbitrary process by white-noise excitation.

    White noise rather than an impulse, because an impulse gives a single realisation with no
    averaging and is badly conditioned for responses with 80 dB of dynamic range. Welch
    averaging of the input and output PSDs and taking the ratio gives a smooth, low-variance
    estimate.

    ``process`` is treated as a black box -- it is handed a waveform and its output is measured.
    It does not have to be linear, though the result is only interpretable as a transfer
    function if it approximately is.

    Args:
        process: Callable mapping a waveform to a waveform of the same length.
        sr: Sample rate in Hz.
        duration_s: Length of the probe signal. Longer means less estimator variance.
        rng: Seed or Generator.
        nperseg: Welch segment length.

    Returns:
        ``(freqs_hz, magnitude_db)``.
    """
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    n = round(duration_s * sr)
    probe = generator.standard_normal(n)
    response = match_length(as_float64(process(probe)), n)

    freqs, psd_in = _welch(probe, sr, nperseg)
    _, psd_out = _welch(response, sr, nperseg)
    magnitude_db = 10.0 * np.log10((psd_out + EPS) / (psd_in + EPS))
    return freqs, magnitude_db


def measure_bandwidth(
    process: Callable[[np.ndarray], np.ndarray],
    sr: int,
    drop_db: float = -3.0,
    search_from_hz: float = 200.0,
    passband_tol_db: float = 1.0,
    rng: np.random.Generator | int | None = None,
) -> float:
    """Find the upper frequency at which a process' response falls ``drop_db`` below passband.

    The passband reference has to be found *adaptively*, and getting this wrong is an easy way
    to produce a confidently incorrect number. Averaging the response over a fixed window such
    as 500 Hz--8 kHz works for a 12 kHz cutoff and fails badly for a 2 kHz one, because most of
    that window is then stopband and the "passband reference" is dragged tens of decibels down,
    which pushes the apparent -3 dB crossing far above the true cutoff.

    So instead: anchor at ``search_from_hz``, walk upward while the response stays within
    ``passband_tol_db`` of the anchor level, and take the reference as the median over that
    contiguous flat run. This uses no prior knowledge of where the cutoff is, so the measurement
    stays independent of the value it is checking.

    The median rather than the maximum of the passband, because a Welch estimate ripples by a
    decibel or so and anchoring to the maximum biases every measured cutoff downward.

    Args:
        process: Callable mapping waveform to waveform.
        sr: Sample rate in Hz.
        drop_db: Threshold relative to the passband, e.g. -3.0 for the half-power point.
        search_from_hz: Frequency at which the passband is assumed to have begun.
        passband_tol_db: Ripple tolerance defining the end of the flat region.
        rng: Seed or Generator.

    Returns:
        Cutoff frequency in Hz, or Nyquist if the response never falls that far.
    """
    freqs, mag_db = measure_transfer_function(process, sr, rng=rng)
    candidates = np.flatnonzero(freqs >= search_from_hz)
    if candidates.size < 8:
        raise ValueError("search window is too narrow to establish a passband")
    start = int(candidates[0])

    # Anchor level: median of a short run at the start, so a single ripple peak cannot set it.
    anchor_end = min(start + 8, len(freqs))
    anchor = float(np.median(mag_db[start:anchor_end]))

    # Extend while the response stays flat within tolerance.
    end = anchor_end
    while end < len(freqs) and mag_db[end] >= anchor - passband_tol_db:
        end += 1
    reference = float(np.median(mag_db[start:end]))
    threshold = reference + drop_db

    below = np.flatnonzero(mag_db[start:] <= threshold)
    if below.size == 0:
        return float(sr / 2.0)
    idx = start + int(below[0])
    if idx == 0:
        return float(freqs[0])

    # Linear interpolation in dB between the bracketing bins.
    f_lo, f_hi = float(freqs[idx - 1]), float(freqs[idx])
    m_lo, m_hi = float(mag_db[idx - 1]), float(mag_db[idx])
    if m_lo == m_hi:
        return f_hi
    frac = (threshold - m_lo) / (m_hi - m_lo)
    return f_lo + frac * (f_hi - f_lo)


def measure_stopband_attenuation(
    process: Callable[[np.ndarray], np.ndarray],
    sr: int,
    stopband_from_hz: float,
    search_from_hz: float = 200.0,
    rng: np.random.Generator | int | None = None,
) -> float:
    """Worst-case (least) attenuation anywhere above ``stopband_from_hz``, in dB below passband.

    For a band-limiting degradation this is the claim that actually matters. The -3 dB point
    describes where the roll-off *starts*; what a downstream classifier experiences is whether
    the band above the cutoff has genuinely been emptied. A filter whose -3 dB point is in the
    right place but which only reaches -12 dB by Nyquist has not band-limited anything.

    Reported as a positive number of decibels of rejection. The minimum over the whole stopband
    is used rather than the mean, because a single leaky region is what would let content
    through.

    Args:
        process: Callable mapping waveform to waveform.
        sr: Sample rate in Hz.
        stopband_from_hz: Lower edge of the region that should be suppressed.
        search_from_hz: Frequency at which the passband reference is taken.
        rng: Seed or Generator.

    Returns:
        Rejection in dB (positive = attenuated).
    """
    freqs, mag_db = measure_transfer_function(process, sr, rng=rng)
    ref_band = (freqs >= search_from_hz) & (freqs <= min(0.8 * stopband_from_hz, 0.4 * sr))
    if ref_band.sum() < 4:
        raise ValueError("cannot establish a passband reference below the stopband edge")
    reference = float(np.median(mag_db[ref_band]))

    stop = freqs >= stopband_from_hz
    if not stop.any():
        raise ValueError("stopband lies above Nyquist")
    return float(reference - np.max(mag_db[stop]))


def measure_octave_band_levels(
    x: np.ndarray,
    sr: int,
    centers_hz: tuple[float, ...] = (125, 250, 500, 1000, 2000, 4000, 8000),
) -> dict[float, float]:
    """Integrate the PSD over octave bands and return band levels in dB.

    Octave bands rather than raw bins because that is the resolution at which atmospheric
    absorption is specified in ISO 9613, so a per-band comparison against the standard's
    predicted attenuation is like-for-like.

    Bands whose upper edge exceeds Nyquist are skipped rather than silently truncated, since a
    truncated band would report an attenuation that is an artefact of the sample rate.
    """
    freqs, psd = _welch(x, sr)
    levels: dict[float, float] = {}
    for fc in centers_hz:
        lo, hi = fc / np.sqrt(2.0), fc * np.sqrt(2.0)
        if hi >= sr / 2.0:
            continue
        band = (freqs >= lo) & (freqs < hi)
        if not band.any():
            continue
        levels[float(fc)] = db(float(trapezoid(psd[band], freqs[band])))
    return levels


# --------------------------------------------------------------------------------------
# Time-domain and structural measurements
# --------------------------------------------------------------------------------------


def measure_rt60(
    rir: np.ndarray,
    sr: int,
    fit_db: tuple[float, float] = (-5.0, -25.0),
    skip_ms: float = 0.0,
) -> float:
    """Measure reverberation time from an impulse response by Schroeder backward integration.

    Schroeder's method integrates the squared impulse response backwards in time, which turns
    the noisy decay of a single response into the smooth ensemble-average decay curve. A
    straight line is then fitted over ``fit_db`` and extrapolated to a 60 dB drop -- the T20 or
    T30 procedure of ISO 3382. Fitting over the top 20 dB and extrapolating is standard because
    the bottom of a real decay curve runs into the noise floor.

    ``skip_ms`` exists because of the direct sound. In a response with a high
    direct-to-reverberant ratio, the direct impulse alone accounts for most of the total energy,
    so the Schroeder curve plunges by roughly the DRR within the first few samples before the
    tail decay begins. Fitting through that plunge measures the DRR, not the RT60. Skipping past
    the direct path and early reflections before normalising isolates the late field, which is
    what RT60 is defined on.

    Args:
        rir: Impulse response.
        sr: Sample rate in Hz.
        fit_db: Decibel range over which to fit, relative to the curve's start.
        skip_ms: Milliseconds to discard from the front before integrating.

    Returns:
        RT60 in seconds.
    """
    arr = as_float64(rir)
    start = round(skip_ms * 1e-3 * sr)
    if start >= len(arr) - 8:
        raise ValueError("skip_ms discards essentially the whole impulse response")
    tail = arr[start:]

    energy = tail**2
    schroeder = np.cumsum(energy[::-1])[::-1]
    if schroeder[0] <= 0:
        raise ValueError("impulse response has zero energy")
    curve_db = 10.0 * np.log10(schroeder / schroeder[0] + EPS)

    hi_db, lo_db = max(fit_db), min(fit_db)
    band = np.flatnonzero((curve_db <= hi_db) & (curve_db >= lo_db))
    if band.size < 8:
        raise ValueError(
            f"decay curve does not span {lo_db} to {hi_db} dB; "
            "the response is too short or too noisy to fit"
        )

    t = band / sr
    slope_db_per_s = float(np.polyfit(t, curve_db[band], 1)[0])
    if slope_db_per_s >= 0:
        raise ValueError("decay curve is not decaying")
    return float(-60.0 / slope_db_per_s)


def measure_drr(
    rir: np.ndarray,
    sr: int,
    direct_window_ms: float = 0.0,
    direct_index: int | None = None,
) -> float:
    """Direct-to-reverberant energy ratio of an impulse response, in dB.

    DRR is the quantity that actually tracks distance in a reverberant field. The direct sound
    obeys the inverse square law while the reverberant field is roughly position-independent, so
    the ratio falls about 6 dB per doubling of distance -- which is why a distant recording sounds
    washed out rather than merely quiet, and why distance fills in the inter-syllable silences
    that a bird detector keys on.

    ``direct_window_ms`` defaults to zero, meaning the direct component is the peak sample alone.
    That matches how :func:`birdsong_robustness.degradations.synthesize_rir` defines the ratio it
    is asked for, so requested and measured describe the same quantity.

    The common alternative -- a +/-2.5 ms window around the peak, as used in much of the speech
    dereverberation literature -- deliberately folds the earliest reflections into the "direct"
    term, because they arrive too soon to be perceived separately. That is a defensible and
    different definition, and mixing the two silently is worth several decibels. It is available
    here by passing the window explicitly, but it is not the default, because the point of this
    module is to measure back what was requested.

    Where peak-picking breaks, and why the index is an argument
    ----------------------------------------------------------
    Locating the direct path as the loudest sample is the obvious thing to do and it fails
    exactly where the measurement gets interesting. Once the reverberant field carries far more
    energy than the direct path -- below roughly -15 dB DRR on this repository's synthesised
    responses -- some sample of the reverberant tail is louder than the direct impulse, and
    ``argmax`` locks onto noise. The measurement then reports a DRR several decibels higher than
    the truth and, worse, it does so quietly: the returned number stops falling with distance and
    just flattens out, which looks like a physical result rather than an instrument failure.

    That regime is not exotic. A direct-to-reverberant ratio of -26 dB is what this package's own
    distance model produces at 200 m, which is an ordinary distance for a passive recorder.

    So ``direct_index`` is exposed. Supplying it is not cheating: the arrival time of the direct
    path is a fact about the geometry, not about the synthesis, and in a measured response it is
    read off the recording once. The energy ratio is still re-derived from the waveform.

    Args:
        rir: Impulse response.
        sr: Sample rate in Hz.
        direct_window_ms: Half-width of the window around the direct path counted as direct
            sound.
        direct_index: Sample index of the direct arrival. Located by peak magnitude when None,
            which is only reliable while the direct path is still the loudest sample.

    Returns:
        DRR in dB.
    """
    arr = as_float64(rir)
    if len(arr) < 2:
        raise ValueError("impulse response is too short")
    if direct_index is None:
        peak = int(np.argmax(np.abs(arr)))
    else:
        peak = int(direct_index)
        if not 0 <= peak < len(arr):
            raise ValueError(f"direct_index {peak} lies outside the impulse response")
    half = round(direct_window_ms * 1e-3 * sr)
    start, end = max(0, peak - half), min(len(arr), peak + half + 1)

    direct_energy = float(np.sum(arr[start:end] ** 2))
    total_energy = float(np.sum(arr**2))
    reverberant_energy = total_energy - direct_energy
    if direct_energy <= 0:
        raise ValueError("impulse response has no direct component")
    return db(direct_energy / max(reverberant_energy, EPS))


def measure_clipped_fraction(
    x: np.ndarray, threshold: float | None = None, tol: float = 1e-9
) -> float:
    """Fraction of samples sitting at the clipping threshold.

    If ``threshold`` is omitted it is taken as the signal's peak magnitude, which is what an
    analyst inspecting an unfamiliar recording would do. ``tol`` absorbs float round-off; a
    genuinely clipped sample sits exactly at the limit, so the tolerance can be tiny.
    """
    arr = as_float64(x)
    limit = float(np.max(np.abs(arr))) if threshold is None else float(threshold)
    if limit <= 0:
        return 0.0
    return float(np.mean(np.abs(arr) >= limit - tol))


def measure_overlap_density(
    target_active: np.ndarray,
    interference: np.ndarray,
    floor_db: float = -40.0,
) -> float:
    """Fraction of the target's active samples that carry audible interference.

    ``floor_db`` is relative to the interference's own RMS, via the shared
    :func:`birdsong_robustness._dsp.audible_mask`. Interference below that is treated as absent,
    which prevents a filter's or convolution's ringing tail from being counted as coverage and
    inflating the measured density.

    Note that the gate here is applied to the *whole* interference track, whereas
    :func:`birdsong_robustness.degradations.add_overlapping_calls` applies it per placed segment.
    That is deliberate -- this function has only the summed waveform to work from, which is
    exactly the situation of someone auditing a mixture they did not create. The two therefore
    disagree slightly when segments differ a lot in level, and the size of that disagreement is
    reported by the validation harness rather than hidden by making both sides use the same
    bookkeeping.
    """
    active = np.asarray(target_active, dtype=bool)
    interf = as_float64(interference)
    if active.shape != interf.shape:
        raise ValueError("target_active and interference must have the same shape")
    if not active.any():
        raise ValueError("target has no active region")

    if rms(interf) <= 0:
        return 0.0
    return float(np.mean(audible_mask(interf, floor_db)[active]))


def measure_sir(
    target: np.ndarray,
    interference: np.ndarray,
    region: np.ndarray | None = None,
) -> float:
    """Signal-to-interference ratio in dB over ``region`` (default: the whole clip)."""
    t = as_float64(target)
    i = match_length(as_float64(interference), len(t))
    mask = None if region is None else np.asarray(region, dtype=bool)
    return db(power(t, mask) / max(power(i, mask), EPS))
