"""Controlled, parameterised degradations of bioacoustic audio.

This is the core contribution of the repository. Every function here takes an explicit physical
or signal-processing parameter -- an SNR in decibels, a distance in metres, a sample rate in
hertz, a clipped-sample fraction -- and is required to produce *exactly* that effect, verifiable
by measuring the output. :mod:`birdsong_robustness.verify` does the measuring and
:mod:`birdsong_robustness.validate` reports requested against measured across the whole grid.

Design rules that apply to everything in this module
----------------------------------------------------
1. **Length-preserving.** A degraded clip must stay sample-aligned with its annotations, so
   resampling and convolution results are trimmed or padded back to the input length.
2. **No hidden normalisation.** A degradation that quietly renormalises the output to unit peak
   destroys the very thing several of these axes are trying to measure (level, clipping,
   distance attenuation). Where normalisation happens it is an explicit argument.
3. **Deterministic given a seed.** Anything stochastic takes an ``rng``.
4. **Composable in a documented order.** See :func:`birdsong_robustness.protocol.apply_spec`
   for the canonical ordering and the reasoning behind it.

The axes
--------
The five axes were chosen because each isolates a different failure mode and each corresponds
to something a field deployment actually does to audio:

* **Additive noise at controlled SNR** -- the base case, and the one where the choice of noise
  colour matters more than practitioners usually assume (see :mod:`birdsong_robustness.noise`).
* **Call overlap** -- the cocktail-party axis. This is the one a source-separation front end
  such as MixIT is supposed to fix, so it is the axis where a separation stage has to justify
  itself.
* **Distance** -- attenuation, air absorption and reverberation together. Distance is not a
  volume knob: it is a low-pass filter and a reverberation increase as well, and treating it as
  a gain change is the single most common oversimplification in bioacoustic augmentation.
* **Codec and bandwidth loss** -- what cheap recorders, lossy storage and resampling do.
* **Clipping and gain staging** -- what a badly set input gain does. Cheap autonomous recorders
  clip constantly and it is rarely modelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import signal as sps

from ._dsp import (
    apply_fir_zero_delay,
    as_float64,
    audible_mask,
    db,
    energy_active_mask,
    match_length,
    power,
)

SPEED_OF_SOUND_M_S = 343.0

__all__ = [
    "SPEED_OF_SOUND_M_S",
    "add_noise_at_snr",
    "add_overlapping_calls",
    "air_absorption_db_per_m",
    "apply_air_absorption",
    "apply_gain_db",
    "apply_reverb",
    "apply_spherical_spreading",
    "band_limit",
    "clip_to_fraction",
    "hard_clip",
    "mu_law_codec",
    "quantize_pcm",
    "resample_roundtrip",
    "simulate_distance",
    "soft_clip",
    "synthesize_rir",
]


# --------------------------------------------------------------------------------------
# Axis 1: additive noise at a controlled SNR
# --------------------------------------------------------------------------------------


def add_noise_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    active_mask: np.ndarray | None = None,
    basis: str = "active",
    sr: int | None = None,
) -> tuple[np.ndarray, float]:
    """Add ``noise`` to ``clean`` scaled so the signal-to-noise ratio is exactly ``snr_db``.

    The SNR basis problem
    ---------------------
    This is the methodological point that makes the whole axis worth building carefully.

    A bird recording is mostly silence. In the synthetic clips used by this repository's
    validation harness the call is present about 18% of the time; real soundscape recordings
    are often sparser than that. If you compute signal power over the entire file -- the
    default in essentially every augmentation library -- then the "signal" power you divide by
    is diluted by all the silence. Requesting "0 dB SNR" on a global basis produces a clip
    whose SNR *during the actual call*, which is the only place the classifier has anything to
    detect, is far higher. The benchmark is then quietly easier than it claims, and by an
    amount that varies with how sparse each recording happens to be, so it is not even a
    consistent bias.

    Computing the signal power over the active region instead makes the requested SNR mean
    what a reader assumes it means: the local SNR the detector faces. The validation harness
    reports both numbers for every mix so the size of this discrepancy is visible rather than
    argued about.

    The cost is that the active mask has to come from somewhere. With strong labels it is
    exact. With :func:`~birdsong_robustness._dsp.energy_active_mask` it is an estimate whose
    error propagates into the SNR. That is a real limitation and it is why ``basis`` is an
    explicit argument rather than a hardcoded choice.

    Args:
        clean: Target signal.
        noise: Noise signal. Trimmed or zero-padded to the length of ``clean``. Its incoming
            level is irrelevant -- it is rescaled.
        snr_db: Requested SNR in dB, on the chosen ``basis``.
        active_mask: Boolean mask marking where the target is present. Required for
            ``basis="active"`` unless ``sr`` is given, in which case an energy-based mask is
            estimated.
        basis: ``"active"`` (signal power measured only where the target is present) or
            ``"global"`` (signal power measured over the whole clip).
        sr: Sample rate, used only to estimate an active mask when one is not supplied.

    Returns:
        ``(mixture, noise_gain)`` where ``noise_gain`` is the linear scale factor applied to
        ``noise``. Returning the gain lets callers reconstruct the exact noise component.

    Raises:
        ValueError: If ``basis`` is unknown, or ``basis="active"`` with no mask and no ``sr``.
    """
    x = as_float64(clean)
    n = match_length(as_float64(noise), len(x))

    if basis not in {"active", "global"}:
        raise ValueError(f"basis must be 'active' or 'global', got {basis!r}")

    if basis == "active":
        if active_mask is None:
            if sr is None:
                raise ValueError("basis='active' requires either active_mask or sr")
            active_mask = energy_active_mask(x, sr)
        mask = np.asarray(active_mask, dtype=bool)
    else:
        mask = None

    signal_power = power(x, mask)
    # Noise power is always measured over the whole clip: the noise is stationary background
    # and is present everywhere, so restricting it to the target's active region would be
    # measuring the same quantity with more variance and no change in expectation.
    noise_power = power(n)

    if noise_power <= 0:
        raise ValueError("noise signal has zero power")
    if signal_power <= 0:
        raise ValueError("clean signal has zero power in the chosen basis")

    gain = float(np.sqrt(signal_power / (noise_power * 10.0 ** (snr_db / 10.0))))
    return x + gain * n, gain


# --------------------------------------------------------------------------------------
# Axis 2: call overlap / cocktail-party mixing
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OverlapResult:
    """Outcome of an overlap mix, including what was actually achieved.

    ``requested_density`` and ``achieved_density`` differ when the interferer material runs
    out before the requested fraction of the target's active region is covered. Reporting the
    achieved value rather than silently missing the target is the entire point.

    ``overlap_region`` is the exact set of samples over which the SIR was set. It is returned
    so that a verifier can measure the SIR back on the same footing instead of guessing at the
    region from the waveform, which is the sort of quiet convention mismatch that turns a
    validation table into a comparison of two different definitions.
    """

    audio: np.ndarray
    interference: np.ndarray
    overlap_region: np.ndarray
    requested_density: float
    achieved_density: float
    requested_sir_db: float
    n_placed: int


def add_overlapping_calls(
    target: np.ndarray,
    interferers: list[np.ndarray],
    sr: int,
    sir_db: float = 0.0,
    density: float = 0.5,
    target_active: np.ndarray | None = None,
    rng: np.random.Generator | int | None = None,
    max_attempts: int = 400,
    audible_floor_db: float = -40.0,
    candidates_per_step: int = 32,
) -> OverlapResult:
    """Overlay competing calls onto ``target`` at a controlled density and SIR.

    Two knobs, because they are genuinely independent and conflating them is a common flaw in
    overlap benchmarks:

    * **Density** -- what fraction of the target's *active* time has at least one interferer
      on top of it. This is the temporal axis. Density 0 means the interferers exist but never
      coincide with the target; density 1 means every syllable is contested.
    * **SIR** -- how loud the interference is relative to the target, measured over the region
      where they actually overlap. This is the level axis.

    Measuring density against the target's active region rather than against the whole clip is
    deliberate. An interferer sitting in a silent gap is not an overlap problem, it is just
    another sound in the recording; counting it as "overlap" would inflate the density and make
    the axis mean nothing.

    Interferer segments are drawn from the active regions of the supplied clips, so what gets
    pasted in is a call rather than a stretch of that clip's silence. Coverage is credited only
    where the pasted material is actually above ``audible_floor_db`` -- see
    :func:`birdsong_robustness._dsp.audible_mask` for why the span-based alternative both
    inflates density and quietly misses the requested SIR.

    Args:
        target: The signal whose detection is being tested.
        interferers: Candidate competing recordings. Segments are extracted from these.
        sr: Sample rate in Hz.
        sir_db: Target-to-interference ratio in dB, measured over the overlap region.
        density: Requested fraction of the target's active samples that should be overlapped,
            in [0, 1].
        target_active: Ground truth activity mask for ``target``. Estimated if omitted.
        rng: Seed or Generator.
        max_attempts: Placement attempts before giving up and reporting the achieved density.
        audible_floor_db: Level, relative to a segment's own RMS, below which pasted material is
            not counted as overlap. Must match the floor the verifier uses.
        candidates_per_step: Placements drawn per step, of which the best-fitting one is taken.
            Larger values track the requested density more closely at linear cost.

    Returns:
        An :class:`OverlapResult`.
    """
    x = as_float64(target)
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be in [0, 1]")
    if not interferers:
        raise ValueError("at least one interferer is required")

    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    active = (
        np.asarray(target_active, dtype=bool)
        if target_active is not None
        else energy_active_mask(x, sr)
    )
    n_active = int(active.sum())
    if n_active == 0:
        raise ValueError("target has no active region to overlap")

    interference = np.zeros_like(x)
    covered = np.zeros(len(x), dtype=bool)
    target_covered = 0
    goal = round(density * n_active)
    n_placed = 0

    # Pre-extract candidate segments: contiguous active runs from each interferer clip.
    segments: list[np.ndarray] = []
    for clip in interferers:
        clip = as_float64(clip)
        clip_active = energy_active_mask(clip, sr)
        edges = np.diff(clip_active.astype(np.int8))
        starts = list(np.flatnonzero(edges == 1) + 1)
        ends = list(np.flatnonzero(edges == -1) + 1)
        if clip_active[0]:
            starts.insert(0, 0)
        if clip_active[-1]:
            ends.append(len(clip))
        for s, e in zip(starts, ends):
            if e - s >= int(0.02 * sr):  # ignore sub-20 ms fragments
                segments.append(clip[s:e])
    if not segments:
        raise ValueError("no usable active segments found in the interferer clips")

    active_idx = np.flatnonzero(active)

    def propose() -> tuple[int, int, np.ndarray, np.ndarray, int] | None:
        """Draw one candidate placement and report how much new coverage it would buy."""
        seg = segments[int(generator.integers(0, len(segments)))]
        if len(seg) >= len(x):
            seg = seg[: len(x) - 1]
        if len(seg) < 2:
            return None
        # Anchor the segment so it lands on the target's active region: pick an active sample
        # and place the segment starting somewhere that will cover it.
        anchor = int(active_idx[int(generator.integers(0, len(active_idx)))])
        start = int(np.clip(anchor - generator.integers(0, len(seg)), 0, len(x) - len(seg)))
        end = start + len(seg)

        # Credit coverage only where this segment is audible. A frame-based segmenter routinely
        # returns runs that contain exact digital silence between syllables; counting those as
        # overlap is what makes span-based density both too high and internally inconsistent
        # with the SIR that gets set over the same region.
        seg_audible = audible_mask(seg, audible_floor_db)
        gained = int((active[start:end] & seg_audible & ~covered[start:end]).sum())
        if gained == 0:
            return None
        return start, end, seg, seg_audible, gained

    # Coverage is quantised by syllable length, so a purely greedy loop -- take any placement
    # that gains something, stop once past the goal -- systematically overshoots. One long
    # syllable can carry a requested density of 0.50 straight past 0.60, and the overshoot is
    # worst exactly where the axis is most informative, at low densities.
    #
    # Each step therefore draws ``candidates_per_step`` placements and takes the one whose gain
    # is closest to what is still needed, which is a cheap approximation to a subset-sum fit. It
    # is not optimal and does not need to be, and it cannot be perfect: when every available
    # syllable is longer than the remaining need, the request is simply unreachable without
    # truncating a syllable mid-way, which would splatter broadband click energy across the
    # spectrum and corrupt every other axis measured on the same clip. The achieved value is
    # reported either way.
    attempts = 0
    while target_covered < goal and attempts < max_attempts:
        attempts += 1
        remaining = goal - target_covered

        best: tuple[int, int, np.ndarray, np.ndarray, int] | None = None
        best_score = np.inf
        for _ in range(candidates_per_step):
            candidate = propose()
            if candidate is None:
                continue
            score = abs(candidate[4] - remaining)
            if score < best_score:
                best_score = score
                best = candidate
        if best is None:
            continue

        start, end, seg, seg_audible, gained = best
        interference[start:end] += seg
        covered[start:end] |= seg_audible
        target_covered += gained
        n_placed += 1

    overlap_region = active & covered
    achieved = float(target_covered / n_active)

    if overlap_region.any() and power(interference, overlap_region) > 0:
        # Set the level over the overlap region only: that is where the SIR is meaningful.
        t_power = power(x, overlap_region)
        i_power = power(interference, overlap_region)
        gain = float(np.sqrt(t_power / (i_power * 10.0 ** (sir_db / 10.0))))
        interference = interference * gain

    return OverlapResult(
        audio=x + interference,
        interference=interference,
        overlap_region=overlap_region,
        requested_density=float(density),
        achieved_density=achieved,
        requested_sir_db=float(sir_db),
        n_placed=n_placed,
    )


# --------------------------------------------------------------------------------------
# Axis 3: distance (spreading + atmospheric absorption + reverberation)
# --------------------------------------------------------------------------------------


def air_absorption_db_per_m(
    freq_hz: np.ndarray | float,
    temperature_c: float = 20.0,
    relative_humidity_pct: float = 70.0,
    pressure_kpa: float = 101.325,
) -> np.ndarray:
    """Atmospheric absorption coefficient in dB/m, per ISO 9613-1.

    Why this matters and why a plain gain change is not enough
    ---------------------------------------------------------
    Air absorbs sound, and it absorbs high frequencies far more than low ones -- absorption
    rises roughly with the square of frequency, moderated by molecular relaxation of oxygen and
    nitrogen, which is where the humidity dependence comes from. At 20 C and 70% humidity the
    coefficient is a small fraction of a decibel per hundred metres at 500 Hz but of order
    10 dB per hundred metres at 8 kHz.

    Bird song lives at 2--8 kHz. So a bird at 100 m is not simply 40 dB quieter than the same
    bird at 1 m; it is 40 dB quieter *and* spectrally tilted, with the upper harmonics that
    carry much of the species-discriminative detail selectively removed. Modelling distance as
    a gain change therefore tests the wrong thing entirely: it tests level sensitivity when the
    real field failure is loss of high-frequency structure.

    The implementation is the full ISO 9613-1 expression, including the oxygen and nitrogen
    relaxation frequencies, rather than a fitted approximation, because the humidity dependence
    is strong and non-monotonic and a fit that is good at 70% can be poor at 20%.

    Args:
        freq_hz: Frequency or array of frequencies in Hz.
        temperature_c: Air temperature in degrees Celsius.
        relative_humidity_pct: Relative humidity, percent.
        pressure_kpa: Atmospheric pressure in kilopascals.

    Returns:
        Absorption coefficient in dB per metre, same shape as ``freq_hz``.
    """
    f = np.asarray(freq_hz, dtype=np.float64)
    t_kelvin = temperature_c + 273.15
    t_ref = 293.15  # 20 C reference
    t_triple = 273.16  # triple point of water
    p_rel = pressure_kpa / 101.325

    if t_kelvin <= 0:
        raise ValueError("temperature below absolute zero")
    if p_rel <= 0:
        raise ValueError("pressure must be positive")

    # Saturation vapour pressure, relative to reference pressure (ISO 9613-1 Annex B).
    psat_rel = 10.0 ** (-6.8346 * (t_triple / t_kelvin) ** 1.261 + 4.6151)
    # Molar concentration of water vapour, as a percentage.
    h = relative_humidity_pct * psat_rel / p_rel

    # Relaxation frequencies of oxygen and nitrogen.
    f_ro = p_rel * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    f_rn = (
        p_rel
        * (t_kelvin / t_ref) ** -0.5
        * (9.0 + 280.0 * h * np.exp(-4.170 * ((t_kelvin / t_ref) ** (-1.0 / 3.0) - 1.0)))
    )

    f2 = f**2
    classical = 1.84e-11 * (1.0 / p_rel) * (t_kelvin / t_ref) ** 0.5
    oxygen = 0.01275 * np.exp(-2239.1 / t_kelvin) / (f_ro + f2 / f_ro)
    nitrogen = 0.1068 * np.exp(-3352.0 / t_kelvin) / (f_rn + f2 / f_rn)
    alpha = 8.686 * f2 * (classical + (t_kelvin / t_ref) ** -2.5 * (oxygen + nitrogen))
    return alpha


def apply_air_absorption(
    x: np.ndarray,
    sr: int,
    distance_m: float,
    temperature_c: float = 20.0,
    relative_humidity_pct: float = 70.0,
    pressure_kpa: float = 101.325,
    n_taps: int = 1023,
    min_gain_db: float = -80.0,
) -> np.ndarray:
    """Apply the frequency-dependent attenuation of ``distance_m`` metres of air.

    The target magnitude response is ``10 ** (-alpha(f) * distance / 20)``, realised as a
    linear-phase FIR designed by frequency sampling. Linear phase, and delay-compensated, so
    the output stays sample-aligned with the input: the residual-based measurements in
    :mod:`birdsong_robustness.verify` would otherwise report alignment error as attenuation.

    Args:
        x: Input signal.
        sr: Sample rate in Hz.
        distance_m: Propagation distance in metres. Zero or negative returns the input.
        temperature_c: Air temperature, Celsius.
        relative_humidity_pct: Relative humidity, percent.
        pressure_kpa: Atmospheric pressure, kPa.
        n_taps: FIR length. Must be odd. 1023 taps resolves this smooth, monotonic response to
            well under a decibel across the band.
        min_gain_db: Floor on the target response. At long distances the true response at
            16 kHz can be -200 dB, which is unrepresentable and would make the FIR design
            ill-conditioned; -80 dB is already far below any recorder's noise floor.

    Returns:
        Filtered signal, same length as ``x``.
    """
    arr = as_float64(x)
    if distance_m <= 0:
        return arr
    if n_taps % 2 == 0:
        raise ValueError("n_taps must be odd for a symmetric linear-phase FIR")

    nyq = sr / 2.0
    design_f = np.linspace(0.0, nyq, 512)
    alpha = air_absorption_db_per_m(design_f, temperature_c, relative_humidity_pct, pressure_kpa)
    gain_db = np.maximum(-alpha * distance_m, min_gain_db)
    gain = 10.0 ** (gain_db / 20.0)
    gain[0] = min(gain[0], 1.0)  # DC: absorption is zero, gain is 1

    taps = sps.firwin2(n_taps, design_f / nyq, gain, window="hamming")
    return apply_fir_zero_delay(arr, taps)


def apply_spherical_spreading(
    x: np.ndarray, distance_m: float, ref_distance_m: float = 1.0
) -> np.ndarray:
    """Apply inverse-square-law amplitude loss for a point source in a free field.

    Sound intensity from a point source falls as ``1 / r**2``, so pressure -- which is what a
    microphone measures and what a waveform represents -- falls as ``1 / r``. The level change
    is therefore ``20 * log10(ref / r)`` dB: 6 dB per doubling of distance, not 3.

    Args:
        x: Input signal.
        distance_m: Distance to the receiver.
        ref_distance_m: Distance at which ``x`` was notionally recorded.

    Returns:
        Scaled signal.
    """
    if distance_m <= 0 or ref_distance_m <= 0:
        raise ValueError("distances must be positive")
    return as_float64(x) * (ref_distance_m / distance_m)


def synthesize_rir(
    sr: int,
    rt60_s: float = 0.6,
    drr_db: float = 6.0,
    n_early: int = 14,
    early_window_s: float = 0.06,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Synthesise a room impulse response with a specified RT60 and direct-to-reverberant ratio.

    Structure: a unit direct impulse at t = 0, a sparse set of early reflections within
    ``early_window_s``, and an exponentially decaying Gaussian noise tail. This is the standard
    statistical model for the late field, and it is used here rather than a geometric room
    simulation because the benchmark needs a *parameter that can be dialled and measured back*,
    not architectural realism. RT60 is recoverable from the synthesised response by Schroeder
    backward integration, which is exactly what :func:`birdsong_robustness.verify.measure_rt60`
    does.

    Outdoor "reverberation" is really scattering from vegetation and ground reflection rather
    than room modes, so RT60 values here are short (0.1--0.8 s) compared with indoor acoustics.
    The relevant effect for a classifier is the same either way: temporal smearing that fills in
    the silent gaps between syllables, which is precisely the structure a bird detector uses.

    The tail envelope is ``exp(-t / tau)`` with ``tau = rt60 / (3 * ln 10)``, the value that
    puts the level exactly 60 dB down at ``t = rt60``.

    Args:
        sr: Sample rate in Hz.
        rt60_s: Requested reverberation time (60 dB decay) in seconds.
        drr_db: Direct-to-reverberant energy ratio in dB. Large positive values mean a dry,
            close recording.
        n_early: Number of discrete early reflections.
        early_window_s: Time window over which early reflections are scattered.
        rng: Seed or Generator.

    Returns:
        Impulse response, normalised so the direct path has amplitude 1. Length is
        ``1.5 * rt60`` seconds, past which the tail is more than 90 dB down.
    """
    if rt60_s <= 0:
        raise ValueError("rt60_s must be positive")
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    n = max(round(1.5 * rt60_s * sr), 16)
    t = np.arange(n) / sr
    tau = rt60_s / (3.0 * np.log(10.0))  # 60 dB down exactly at t = rt60_s
    envelope = np.exp(-t / tau)

    # Late field: Gaussian noise under the decay envelope.
    tail = generator.standard_normal(n) * envelope
    tail[0] = 0.0  # keep the direct path exclusively in the direct component

    # Early reflections: discrete, sparse, following the same envelope, random polarity.
    early = np.zeros(n)
    n_early_samples = max(1, round(early_window_s * sr))
    if n_early > 0 and n_early_samples > 1:
        positions = generator.integers(1, min(n_early_samples, n), size=n_early)
        signs = generator.choice([-1.0, 1.0], size=n_early)
        for pos, sign in zip(positions, signs):
            early[pos] += sign * float(envelope[pos]) * float(generator.uniform(0.3, 1.0))

    reverberant = tail + early
    rev_energy = float(np.sum(reverberant**2))
    if rev_energy <= 0:
        raise ValueError("degenerate reverberant component")

    # Direct energy is 1 (unit impulse), so scale the reverberant part to hit the requested DRR.
    scale = float(np.sqrt(1.0 / (rev_energy * 10.0 ** (drr_db / 10.0))))
    rir = np.zeros(n)
    rir[0] = 1.0
    rir += scale * reverberant
    return rir


def apply_reverb(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve with a room impulse response, preserving length and direct-path level.

    Truncating the convolution tail rather than letting the signal grow is deliberate: a
    degraded clip must remain the same duration as its annotations.
    """
    arr = as_float64(x)
    y = sps.fftconvolve(arr, as_float64(rir), mode="full")
    return match_length(y, len(arr))


def simulate_distance(
    x: np.ndarray,
    sr: int,
    distance_m: float,
    ref_distance_m: float = 1.0,
    temperature_c: float = 20.0,
    relative_humidity_pct: float = 70.0,
    rt60_s: float = 0.4,
    drr_at_ref_db: float = 20.0,
    include_reverb: bool = True,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Full distance simulation: spreading, then air absorption, then reverberation.

    The three effects are composed in propagation order, which is also the order in which they
    are physically applied to the wavefront.

    Direct-to-reverberant ratio falls with distance as ``20 * log10(d / d_ref)``. This is the
    standard relation for a diffuse reverberant field: the reverberant energy is roughly
    position-independent while the direct energy obeys the inverse square law, so the ratio
    degrades by 6 dB per doubling. It is why a distant recording sounds *washed out* rather
    than merely quiet, and it is the mechanism by which distance destroys the inter-syllable
    silence that detectors rely on.

    Args:
        x: Input signal, notionally recorded at ``ref_distance_m``.
        sr: Sample rate in Hz.
        distance_m: Source-to-receiver distance in metres.
        ref_distance_m: Reference distance of the input recording.
        temperature_c: Air temperature, Celsius.
        relative_humidity_pct: Relative humidity, percent.
        rt60_s: Reverberation time of the environment.
        drr_at_ref_db: Direct-to-reverberant ratio at the reference distance.
        include_reverb: Set False to isolate spreading and absorption, which is what the
            validation harness does when checking the attenuation numbers -- reverberation adds
            energy back and would confound a pure attenuation measurement.
        rng: Seed or Generator for the impulse response.

    Returns:
        Degraded signal, same length as ``x``.
    """
    y = apply_spherical_spreading(x, distance_m, ref_distance_m)
    y = apply_air_absorption(
        y,
        sr,
        distance_m - ref_distance_m if distance_m > ref_distance_m else 0.0,
        temperature_c=temperature_c,
        relative_humidity_pct=relative_humidity_pct,
    )
    if include_reverb and distance_m > 0:
        drr = drr_at_ref_db - 20.0 * np.log10(max(distance_m / ref_distance_m, 1e-9))
        rir = synthesize_rir(sr, rt60_s=rt60_s, drr_db=float(drr), rng=rng)
        y = apply_reverb(y, rir)
    return y


# --------------------------------------------------------------------------------------
# Axis 4: codec, bandwidth and bit-depth loss
# --------------------------------------------------------------------------------------


def design_brickwall_lowpass(
    sr: int,
    stopband_edge_hz: float,
    stopband_atten_db: float = 80.0,
    transition_hz: float | None = None,
) -> np.ndarray:
    """Design a Kaiser-window FIR low-pass with a specified stopband edge and rejection.

    ``scipy.signal.kaiserord`` gives the tap count and beta needed to hit a required attenuation
    over a required transition width, so both the stopband edge and the depth of the stopband
    are *design inputs* rather than whatever a library default happened to produce. That is what
    makes the resulting degradation checkable: the harness can assert the measured rejection
    against the number that was asked for.

    Args:
        sr: Sample rate in Hz.
        stopband_edge_hz: Frequency above which the requested rejection must hold.
        stopband_atten_db: Required rejection in dB.
        transition_hz: Width of the transition band. Defaults to 8% of the stopband edge, which
            keeps the filter a few hundred taps at typical audio rates.

    Returns:
        Odd-length symmetric FIR coefficients.
    """
    nyq = sr / 2.0
    if not 0 < stopband_edge_hz < nyq:
        raise ValueError(f"stopband edge must lie in (0, {nyq})")
    width = transition_hz if transition_hz is not None else 0.08 * stopband_edge_hz
    width = float(min(width, stopband_edge_hz * 0.5))

    n_taps, beta = sps.kaiserord(stopband_atten_db, width / nyq)
    n_taps = int(n_taps) | 1  # force odd so the group delay is an integer
    # firwin's cutoff is the -6 dB point, which sits mid-transition; placing it half a
    # transition width below the stopband edge puts the stopband exactly where requested.
    cutoff = max(stopband_edge_hz - width / 2.0, width)
    return sps.firwin(n_taps, cutoff / nyq, window=("kaiser", beta))


def resample_roundtrip(
    x: np.ndarray,
    sr: int,
    target_sr: int,
    stopband_atten_db: float = 80.0,
) -> np.ndarray:
    """Resample down to ``target_sr`` and back, losing everything above ``target_sr / 2``.

    This models the very common field situation of audio that has passed through a lower sample
    rate at some point -- a recorder configured at 16 kHz, an archive stored at 22.05 kHz, a
    streaming codec's internal rate -- and is then presented to a classifier expecting its
    native rate. The classifier receives a signal at the right rate with an empty upper band.

    It matters more for bird song than for speech. A 16 kHz round trip caps the signal at 8 kHz,
    which removes the upper harmonics of most passerine song and the entire fundamental of some
    high-frequency species. Resampling is not a neutral format conversion here; it is a
    destructive band-limiting operation.

    Why the explicit filters
    ------------------------
    ``scipy.signal.resample_poly`` applies its own Kaiser anti-alias FIR, but the default is
    short and its stopband rejection measures only about 20 dB just above the target Nyquist.
    Twenty decibels is not "the band was removed" -- a classifier would still see the residue,
    and the axis would be measuring something much milder than its name claims. Under Rule 4 of
    this module, a degradation has to deliver what it advertises.

    So the round trip is bracketed with an explicit ``stopband_atten_db`` design: an anti-alias
    filter before decimation, and an anti-imaging filter after interpolation to suppress the
    spectral replicas the upsampling stage creates. The rejection is then a stated design
    parameter that the validation harness checks.

    Args:
        x: Input signal.
        sr: Original sample rate in Hz.
        target_sr: Intermediate sample rate. Values at or above ``sr`` are a no-op, since a
            round trip cannot manufacture bandwidth.
        stopband_atten_db: Required rejection above ``target_sr / 2``.

    Returns:
        Signal at rate ``sr``, band-limited to ``target_sr / 2``, same length as ``x``.
    """
    arr = as_float64(x)
    if target_sr <= 0:
        raise ValueError("target_sr must be positive")
    if target_sr >= sr:
        return arr

    taps = design_brickwall_lowpass(sr, target_sr / 2.0, stopband_atten_db)
    filtered = apply_fir_zero_delay(arr, taps)

    ratio = Fraction(target_sr, sr).limit_denominator(1000)
    down = sps.resample_poly(filtered, ratio.numerator, ratio.denominator)
    up = sps.resample_poly(down, ratio.denominator, ratio.numerator)
    up = match_length(up, len(arr))

    # Anti-imaging: the interpolation stage replicates the spectrum at multiples of target_sr,
    # and resample_poly's internal filter alone leaves those replicas only ~20 dB down.
    return apply_fir_zero_delay(up, taps)


def band_limit(
    x: np.ndarray,
    sr: int,
    low_hz: float | None = None,
    high_hz: float | None = None,
    order: int = 8,
) -> np.ndarray:
    """Butterworth band-limiting, modelling a microphone or codec's usable band.

    Applied causally with ``sosfilt`` rather than zero-phase ``sosfiltfilt`` on purpose. Real
    microphones and codecs are causal and do impose phase distortion, and ``filtfilt`` would
    square the magnitude response, putting the measured -3 dB point somewhere other than the
    requested cutoff and making the requested-versus-measured table meaningless.

    The low cut is as important as the high cut for field audio: cheap recorders and windshields
    roll off the bottom, and a high-pass is also what a deployment would apply to suppress wind.

    Args:
        x: Input signal.
        sr: Sample rate in Hz.
        low_hz: High-pass cutoff. None to skip.
        high_hz: Low-pass cutoff. None to skip.
        order: Butterworth order. 8 gives 48 dB/octave, similar to a codec's band edge.

    Returns:
        Filtered signal, same length as ``x``.
    """
    arr = as_float64(x)
    nyq = sr / 2.0
    if low_hz is None and high_hz is None:
        return arr

    if low_hz is not None and high_hz is not None:
        if not 0 < low_hz < high_hz < nyq:
            raise ValueError(f"require 0 < low_hz < high_hz < {nyq}")
        sos = sps.butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")
    elif high_hz is not None:
        if not 0 < high_hz < nyq:
            raise ValueError(f"require 0 < high_hz < {nyq}")
        sos = sps.butter(order, high_hz / nyq, btype="low", output="sos")
    else:
        assert low_hz is not None
        if not 0 < low_hz < nyq:
            raise ValueError(f"require 0 < low_hz < {nyq}")
        sos = sps.butter(order, low_hz / nyq, btype="high", output="sos")

    return sps.sosfilt(sos, arr)


def quantize_pcm(x: np.ndarray, bits: int, dither: bool = False, rng=None) -> np.ndarray:
    """Uniform mid-tread PCM quantisation to ``bits`` bits over the range [-1, 1).

    Models a cheap recorder's converter, or aggressive lossy storage. The theoretical
    signal-to-quantisation-noise ratio is ``10 * log10(P_x / (step**2 / 12))``, where
    ``step = 2 / 2**bits`` and ``step**2 / 12`` is the variance of a uniform error over one
    step. For a full-scale signal that reduces to the familiar ``6.02 * bits + 1.76`` dB, but
    the general form is used here because bird recordings are nowhere near full scale and the
    familiar version would overstate the SQNR by the crest factor.

    That prediction is exact enough to be a genuine test, which is why this function exists as
    a separate axis rather than being folded into a generic "add noise".

    Args:
        x: Input signal, expected in [-1, 1].
        bits: Bit depth, 2 to 24.
        dither: Add triangular-PDF dither before quantising. Dither removes the correlation
            between quantisation error and signal -- it converts harmonic distortion into
            noise -- at the cost of about 4.8 dB of SQNR.
        rng: Seed or Generator, used only when ``dither`` is True.

    Returns:
        Quantised signal.
    """
    if not 2 <= bits <= 24:
        raise ValueError("bits must be between 2 and 24")
    arr = as_float64(x)
    step = 2.0 / (2**bits)

    if dither:
        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        # Triangular PDF dither, peak-to-peak two steps: the standard choice.
        tpdf = generator.uniform(-0.5, 0.5, size=arr.shape) + generator.uniform(
            -0.5, 0.5, size=arr.shape
        )
        arr = arr + tpdf * step

    quantised = np.round(arr / step) * step
    return np.clip(quantised, -1.0, 1.0 - step)


def mu_law_codec(x: np.ndarray, mu: float = 255.0, bits: int = 8) -> np.ndarray:
    """G.711-style mu-law companding round trip.

    Mu-law compresses amplitude logarithmically before quantising, so quantisation steps are
    fine near silence and coarse near full scale. The result is roughly constant SQNR across a
    wide dynamic range instead of the linear quantiser's SQNR that collapses for quiet signals.

    This is the right model for "the audio went through a telephony-grade or low-cost codec",
    and it interacts with bird song in a specific way: quiet distant calls survive mu-law far
    better than they survive 8-bit linear PCM, so the two 8-bit conditions in the grid are not
    redundant.

    Args:
        x: Input signal in [-1, 1].
        mu: Companding parameter. 255 is the North American G.711 standard.
        bits: Bit depth of the companded domain.

    Returns:
        Signal after the encode/decode round trip.
    """
    arr = np.clip(as_float64(x), -1.0, 1.0)
    # Encode: compress, then quantise uniformly in the compressed domain.
    compressed = np.sign(arr) * np.log1p(mu * np.abs(arr)) / np.log1p(mu)
    step = 2.0 / (2**bits)
    quantised = np.round(compressed / step) * step
    quantised = np.clip(quantised, -1.0, 1.0)
    # Decode: expand.
    return np.sign(quantised) * ((1.0 + mu) ** np.abs(quantised) - 1.0) / mu


# --------------------------------------------------------------------------------------
# Axis 5: gain staging and clipping
# --------------------------------------------------------------------------------------


def apply_gain_db(x: np.ndarray, gain_db: float) -> np.ndarray:
    """Scale by ``gain_db`` decibels. No clipping, no normalisation -- explicitly linear."""
    return as_float64(x) * 10.0 ** (gain_db / 20.0)


def hard_clip(x: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    """Symmetric hard clipping at ``threshold``.

    What a converter does when the input exceeds full scale. It generates odd-order harmonic
    distortion that spreads broadband energy across the spectrum -- which for bird audio means
    a loud nearby call can splatter energy into the bands where quieter species live.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    return np.clip(as_float64(x), -threshold, threshold)


def clip_to_fraction(x: np.ndarray, fraction: float) -> tuple[np.ndarray, float]:
    """Hard-clip so that exactly ``fraction`` of samples are at the limit.

    Parameterising by the *fraction of samples clipped* rather than by a threshold in dBFS is
    what makes this axis comparable across recordings. A threshold of -3 dBFS clips a hot
    recording severely and a quiet one not at all, so a grid over thresholds would be
    confounded with recording level. A grid over clipped fraction is not.

    The threshold is the ``1 - fraction`` quantile of ``|x|``, so the achieved fraction is exact
    up to ties at the threshold value.

    Args:
        x: Input signal.
        fraction: Target fraction of clipped samples, in [0, 1).

    Returns:
        ``(clipped_signal, threshold)``.
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1)")
    arr = as_float64(x)
    if fraction == 0.0:
        return arr.copy(), float(np.max(np.abs(arr)))
    threshold = float(np.quantile(np.abs(arr), 1.0 - fraction))
    if threshold <= 0:
        raise ValueError("signal is too sparse to clip at the requested fraction")
    return np.clip(arr, -threshold, threshold), threshold


def soft_clip(x: np.ndarray, drive: float = 3.0) -> np.ndarray:
    """Smooth saturating overload via ``tanh``, normalised to preserve the clipping ceiling.

    Analogue front ends and many preamps saturate gradually rather than at a hard corner. Soft
    clipping produces lower-order distortion products than hard clipping at the same
    peak-reduction, so including both separates "the model dislikes distortion" from "the model
    dislikes the broadband splatter of hard clipping specifically".

    Args:
        x: Input signal.
        drive: Saturation amount. Approaches linear as drive tends to 0.
    """
    if drive <= 0:
        raise ValueError("drive must be positive")
    return np.tanh(drive * as_float64(x)) / np.tanh(drive)


def gain_stage(
    x: np.ndarray,
    gain_db: float,
    bits: int = 16,
    clip_threshold: float = 1.0,
) -> np.ndarray:
    """Model a recorder's input gain: analogue gain, then hard clipping, then quantisation.

    This ordering is the physical one and it is what makes badly staged gain a two-sided
    problem. Too much gain clips at the converter; too little gain leaves the signal down among
    the quantisation steps, losing effective bit depth. Both are common in deployed autonomous
    recorders and neither is captured by simply scaling a waveform.
    """
    y = apply_gain_db(x, gain_db)
    y = hard_clip(y, clip_threshold)
    return quantize_pcm(y, bits)


def snr_from_components(clean: np.ndarray, noise_component: np.ndarray) -> float:
    """Convenience: SNR in dB of a clean signal against a known noise component."""
    return db(power(clean) / max(power(noise_component), 1e-30))
