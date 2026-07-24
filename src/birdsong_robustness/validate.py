"""Sweep every degradation across its grid and report requested against measured.

This is the part of the repository that currently produces real numbers, and it is the only part
that does. It answers one question -- *does this pipeline do what it says it does?* -- and it
answers it by measurement rather than by assertion.

The method, for every axis, is the same three steps:

1. Ask a degradation for a specific physical effect: 0 dB SNR, 100 m of propagation, an 8 kHz
   band edge, 1% of samples clipped.
2. Measure the *output waveform* with :mod:`birdsong_robustness.verify`, which does not share
   code with the degradation and in most cases uses a different method entirely -- a residual
   subtraction rather than a stored gain, a Welch transfer function rather than the filter
   coefficients, a Schroeder integration rather than the decay constant that generated the tail.
3. Print requested, measured, error, and a tolerance the check has to meet.

Why bother, when the code is short enough to read
-------------------------------------------------
Because every one of these axes has a plausible-looking implementation that is quietly wrong, and
the wrongness only shows up as a classifier result that is hard to argue with:

* An SNR computed over a whole file rather than over the call is roughly 7 dB optimistic on this
  repository's own test signals, and by a different amount on every recording.
* ``filtfilt`` applies a filter's magnitude response twice, so a requested 8 kHz cutoff measures
  at neither 8 kHz nor a consistent offset from it.
* ``resample_poly``'s default anti-alias filter reaches about 20 dB of rejection just above the
  target Nyquist. A "16 kHz round trip" built on it does not remove the top band, it dents it.
* A quantiser's textbook SQNR of ``6.02 * bits + 1.76`` dB assumes a full-scale signal. Bird
  recordings sit 15--30 dB below full scale, so the textbook figure is wrong by the crest factor.
* Reverberation adds energy, so measuring "attenuation with distance" on a reverberant signal
  measures the room, not the distance.

A benchmark built on any of those produces numbers about its own bugs. The table below is the
evidence that this one does not.

What this harness does *not* establish
--------------------------------------
That the degradations are *realistic*. Calibration is not realism. The noise models are
parametric approximations, the reverberation is a statistical impulse response rather than a
measured one, and the test signals are synthetic. Those are limitations of the fixtures, not of
the measurements, and validating against real field recordings is the first item on the roadmap.

Run it with ``python -m birdsong_robustness.validate`` or the ``birdsong-validate`` console
script. ``--json`` writes machine-readable results; the exit status is non-zero if any check
misses its tolerance, which is what makes this useful in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from . import degradations, noise, protocol, synth, verify
from ._dsp import EPS, trapezoid

__all__ = ["Check", "ValidationReport", "main", "run_validation"]

SR = 32000
CLIP_SECONDS = 5.0
NOISE_SECONDS = 10.0
SEED = 20240517


@dataclass(frozen=True)
class Check:
    """One requested-versus-measured comparison.

    ``tolerance`` is stated per check rather than globally because the axes are not equally
    determined. An SNR mix is exact arithmetic and should agree to a thousandth of a decibel; a
    Welch-estimated -3 dB point on a finite noise record is good to a percent or so and pretending
    otherwise would produce a table that fails for reasons unrelated to the pipeline.

    ``direction`` exists because not every requirement is two-sided, and forcing one to be is a
    way to write a check that cannot fail. A stopband specified as "at least 80 dB of rejection"
    is satisfied by 102 dB and violated by 5 dB; scoring it as ``|measured - 80| <= 80`` would
    pass both. ``"at-least"`` and ``"at-most"`` say what is actually required.
    """

    axis: str
    condition: str
    quantity: str
    unit: str
    requested: float
    measured: float
    tolerance: float
    direction: str = "two-sided"
    note: str = ""

    @property
    def error(self) -> float:
        """Measured minus requested, in the check's own unit."""
        return self.measured - self.requested

    @property
    def comparator(self) -> str:
        """Symbol describing the requirement, for the report table."""
        return {"two-sided": "==", "at-least": ">=", "at-most": "<="}[self.direction]

    @property
    def passed(self) -> bool:
        """True when the measurement satisfies the requirement, allowing for ``tolerance``."""
        if self.direction == "at-least":
            return self.measured >= self.requested - self.tolerance
        if self.direction == "at-most":
            return self.measured <= self.requested + self.tolerance
        if self.direction != "two-sided":
            raise ValueError(f"unknown direction {self.direction!r}")
        return abs(self.error) <= self.tolerance


@dataclass(frozen=True)
class Observation:
    """A measured quantity with no requested counterpart.

    Some things worth reporting are descriptive rather than targeted -- the spectral slope of
    synthesised wind, the share of a masker's energy that lands in the bird band. They belong in
    the report because they are what justifies the modelling choices, but they are not pass/fail
    and are kept in a separate structure so they can never be counted as if they were.
    """

    axis: str
    condition: str
    quantity: str
    unit: str
    measured: float
    note: str = ""


@dataclass(frozen=True)
class ValidationReport:
    """Everything one validation run produced."""

    checks: tuple[Check, ...]
    observations: tuple[Observation, ...]

    @property
    def n_failed(self) -> int:
        """Number of checks outside tolerance."""
        return sum(1 for c in self.checks if not c.passed)

    @property
    def worst(self) -> Check | None:
        """The two-sided check with the largest error relative to its tolerance.

        One-sided checks are excluded because "error" has no comparable meaning for them: a
        stopband 22 dB deeper than required is not a 22 dB error, it is headroom.
        """
        two_sided = [c for c in self.checks if c.direction == "two-sided"]
        if not two_sided:
            return None
        return max(two_sided, key=lambda c: abs(c.error) / max(c.tolerance, EPS))

    def to_json(self, indent: int = 2) -> str:
        """Serialise checks and observations for downstream tooling."""
        return json.dumps(
            {
                "n_checks": len(self.checks),
                "n_failed": self.n_failed,
                "checks": [
                    {
                        "axis": c.axis,
                        "condition": c.condition,
                        "quantity": c.quantity,
                        "unit": c.unit,
                        "requested": c.requested,
                        "measured": c.measured,
                        "error": c.error,
                        "tolerance": c.tolerance,
                        "direction": c.direction,
                        "passed": c.passed,
                        "note": c.note,
                    }
                    for c in self.checks
                ],
                "observations": [
                    {
                        "axis": o.axis,
                        "condition": o.condition,
                        "quantity": o.quantity,
                        "unit": o.unit,
                        "measured": o.measured,
                        "note": o.note,
                    }
                    for o in self.observations
                ],
            },
            indent=indent,
        )


# --------------------------------------------------------------------------------------
# Small measurement helpers used by more than one axis
# --------------------------------------------------------------------------------------


def _response_at(
    freqs: np.ndarray, magnitude_db: np.ndarray, centre_hz: float, width_frac: float = 0.03
) -> float:
    """Median response in a narrow band around ``centre_hz``.

    A median over a proportional band rather than the single nearest bin: a Welch estimate
    ripples by a few tenths of a decibel bin to bin, and reading one bin would put that ripple
    straight into the reported error.
    """
    band = (freqs >= centre_hz * (1.0 - width_frac)) & (freqs <= centre_hz * (1.0 + width_frac))
    if not band.any():
        raise ValueError(f"no spectral bins near {centre_hz} Hz")
    return float(np.median(magnitude_db[band]))


def _band_energy_fraction(x: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    """Fraction of a signal's total power falling in ``[low_hz, high_hz)``."""
    freqs, psd = sps.welch(x, fs=sr, nperseg=4096, noverlap=2048, window="hann")
    total = float(trapezoid(psd, freqs))
    band = (freqs >= low_hz) & (freqs < high_hz)
    if total <= 0 or not band.any():
        return 0.0
    return float(trapezoid(psd[band], freqs[band]) / total)


def _test_clip(seed: int = SEED) -> synth.SyntheticClip:
    """The standard target clip. One definition so every axis is measured on the same signal."""
    return synth.bird_call(duration_s=CLIP_SECONDS, sr=SR, rng=seed, n_syllables=8)


# --------------------------------------------------------------------------------------
# Axis validations
# --------------------------------------------------------------------------------------


def check_noise_models(rng_seed: int = SEED) -> tuple[list[Check], list[Observation]]:
    """Coloured noise must have the spectral slope its name claims.

    ``1/f**beta`` has an exact analytic slope of ``-3.0103 * beta`` dB per octave, so this is a
    comparison against theory rather than against another implementation. Wind and rain have no
    analytic target -- they are parametric approximations -- so they are reported as observations
    instead, with the quantity that actually justifies including both: where their energy lands
    relative to the 2--8 kHz band that carries passerine song.
    """
    checks: list[Check] = []
    observations: list[Observation] = []
    n = int(NOISE_SECONDS * SR)

    for name, beta in (("white", 0.0), ("pink", 1.0), ("brown", 2.0)):
        signal = noise.color_noise(n, name, SR, rng=rng_seed)
        measured = verify.measure_spectral_slope(signal, SR, f_min=200.0, f_max=8000.0)
        checks.append(
            Check(
                axis="noise colour",
                condition=name,
                quantity="PSD slope",
                unit="dB/octave",
                requested=-3.0103 * beta,
                measured=measured,
                tolerance=0.25,
                note="analytic target for 1/f**beta",
            )
        )
        checks.append(
            Check(
                axis="noise colour",
                condition=name,
                quantity="RMS",
                unit="linear",
                requested=1.0,
                measured=float(np.sqrt(np.mean(signal**2))),
                tolerance=1e-9,
                note="generators normalise to unit RMS",
            )
        )

    for name in ("white", "pink", "brown", "wind", "rain"):
        signal = noise.make_noise(name, n, SR, rng=rng_seed)
        observations.append(
            Observation(
                axis="noise colour",
                condition=name,
                quantity="energy in 2-8 kHz bird band",
                unit="fraction",
                measured=_band_energy_fraction(signal, SR, 2000.0, 8000.0),
            )
        )
        observations.append(
            Observation(
                axis="noise colour",
                condition=name,
                quantity="energy below 500 Hz",
                unit="fraction",
                measured=_band_energy_fraction(signal, SR, 0.0, 500.0),
            )
        )
    return checks, observations


def check_snr(rng_seed: int = SEED) -> tuple[list[Check], list[Observation]]:
    """A requested SNR must be the SNR measured on the same basis, and the basis must be stated.

    The measurement recovers the noise as ``degraded - clean`` rather than reading back the gain
    the mixer applied, so a length mismatch, a sample slip or a stray renormalisation would show
    up here as decibels of error.

    The global-basis figure is recorded as an observation for every mix. It is not a target and
    it is not an error -- it is the size of the discrepancy between what "0 dB SNR" means when
    computed over a whole sparse recording and what it means during the call.
    """
    checks: list[Check] = []
    observations: list[Observation] = []
    clip = _test_clip(rng_seed)

    for noise_name in protocol.NOISE_TYPES:
        for snr_db in protocol.SNR_LEVELS_DB:
            masker = noise.make_noise(noise_name, len(clip.audio), SR, rng=rng_seed)
            mixture, _ = degradations.add_noise_at_snr(
                clip.audio, masker, snr_db, active_mask=clip.active
            )
            measured = verify.measure_snr(clip.audio, mixture, mask=clip.active)
            checks.append(
                Check(
                    axis="SNR",
                    condition=f"{noise_name} @ {snr_db:+.0f} dB",
                    quantity="SNR (active basis)",
                    unit="dB",
                    requested=snr_db,
                    measured=measured,
                    tolerance=0.05,
                )
            )
            if noise_name == "pink":
                global_basis = verify.measure_snr(clip.audio, mixture, basis="global")
                observations.append(
                    Observation(
                        axis="SNR",
                        condition=f"pink @ {snr_db:+.0f} dB requested",
                        quantity="same mixture read on the global basis",
                        unit="dB",
                        measured=global_basis,
                        note="what a whole-file SNR would have reported",
                    )
                )
                observations.append(
                    Observation(
                        axis="SNR",
                        condition=f"pink @ {snr_db:+.0f} dB requested",
                        quantity="active minus global basis",
                        unit="dB",
                        measured=measured - global_basis,
                        note="the silence-dilution offset; constant for a given clip",
                    )
                )

    observations.append(
        Observation(
            axis="SNR",
            condition="test clip",
            quantity="active fraction",
            unit="fraction",
            measured=clip.active_fraction,
            note="drives the size of the active-vs-global gap",
        )
    )
    return checks, observations


def check_overlap(rng_seed: int = SEED) -> tuple[list[Check], list[Observation]]:
    """Overlap density and SIR must both land where they were asked to.

    Two knobs are checked because they are independent and conflating them is the usual flaw in
    overlap benchmarks. Density is checked twice: against the mixer's own bookkeeping, and again
    against :func:`birdsong_robustness.verify.measure_overlap_density`, which re-derives it from
    the summed waveform with a global rather than per-segment audibility gate -- the position
    someone auditing a mixture they did not create would be in.
    """
    checks: list[Check] = []
    observations: list[Observation] = []
    _, sources = synth.soundscape(
        duration_s=10.0, sr=SR, rng=rng_seed, n_species=4, calls_per_species=6
    )
    target = sources[0]
    interferers = [s.audio for s in sources[1:]]

    for density in protocol.OVERLAP_DENSITIES[1:]:
        for sir_db in protocol.OVERLAP_SIR_DB:
            result = degradations.add_overlapping_calls(
                target.audio,
                interferers,
                SR,
                sir_db=sir_db,
                density=density,
                target_active=target.active,
                rng=rng_seed,
            )
            condition = f"density {density:.2f}, SIR {sir_db:+.0f} dB"
            checks.append(
                Check(
                    axis="overlap",
                    condition=condition,
                    quantity="overlap density",
                    unit="fraction",
                    requested=density,
                    measured=result.achieved_density,
                    tolerance=0.05,
                    note="coverage is quantised by syllable length",
                )
            )
            checks.append(
                Check(
                    axis="overlap",
                    condition=condition,
                    quantity="SIR over overlap region",
                    unit="dB",
                    requested=sir_db,
                    measured=verify.measure_sir(
                        target.audio, result.interference, result.overlap_region
                    ),
                    tolerance=0.05,
                )
            )
            observations.append(
                Observation(
                    axis="overlap",
                    condition=condition,
                    quantity="density re-measured from the mixture",
                    unit="fraction",
                    measured=verify.measure_overlap_density(target.active, result.interference),
                    note="global audibility gate rather than per-segment",
                )
            )
    return checks, observations


def check_distance(rng_seed: int = SEED) -> tuple[list[Check], list[Observation]]:
    """Distance is spreading *and* absorption *and* reverberation, and each is checked separately.

    Reverberation is switched off for the attenuation checks on purpose: it adds energy back into
    the signal, so a broadband level measured on a reverberant clip would understate the
    attenuation by an amount that depends on the room rather than on the distance.

    Air absorption is checked per frequency against ISO 9613-1, not broadband. That is the whole
    point of the axis -- distance is a low-pass filter, and a broadband number would average away
    exactly the spectral tilt that removes a species' discriminative harmonics.
    """
    checks: list[Check] = []
    observations: list[Observation] = []
    clip = _test_clip(rng_seed)
    reference_level = verify.measure_level_dbfs(clip.audio, clip.active)

    for distance in protocol.DISTANCES_M:
        attenuated = degradations.apply_spherical_spreading(clip.audio, distance, 1.0)
        checks.append(
            Check(
                axis="distance",
                condition=f"{distance:.0f} m",
                quantity="spreading loss",
                unit="dB",
                requested=float(-20.0 * np.log10(distance)),
                measured=verify.measure_level_dbfs(attenuated, clip.active) - reference_level,
                tolerance=0.01,
                note="inverse square law: 6 dB per doubling",
            )
        )

    for distance in protocol.DISTANCES_M[1:]:
        freqs, magnitude = verify.measure_transfer_function(
            lambda x, d=distance: degradations.apply_air_absorption(x, SR, d),
            SR,
            rng=rng_seed,
        )
        for centre in (1000.0, 2000.0, 4000.0, 8000.0):
            predicted = float(-degradations.air_absorption_db_per_m(centre) * distance)
            checks.append(
                Check(
                    axis="distance",
                    condition=f"{distance:.0f} m @ {centre / 1000:.0f} kHz",
                    quantity="air absorption",
                    unit="dB",
                    requested=predicted,
                    measured=_response_at(freqs, magnitude, centre),
                    tolerance=0.30,
                    note="ISO 9613-1, 20 C, 70% RH",
                )
            )

    for rt60 in protocol.RT60_S:
        rir = degradations.synthesize_rir(SR, rt60_s=rt60, drr_db=6.0, rng=rng_seed)
        checks.append(
            Check(
                axis="distance",
                condition=f"RT60 {rt60:.1f} s",
                quantity="RT60 (Schroeder T20)",
                unit="s",
                requested=rt60,
                measured=verify.measure_rt60(rir, SR, skip_ms=8.0),
                tolerance=0.05 * rt60,
                note="5% tolerance; direct path skipped before integrating",
            )
        )

    # The direct-to-reverberant ratio is the third component of distance and the one that fills
    # in the inter-syllable silences a detector keys on. simulate_distance derives it as
    # 20*log10(d/d_ref) below the reference DRR; the impulse response it builds is measured here
    # against that derived target, so both the derivation and the synthesis are checked.
    reference_drr_db = 20.0
    peak_picking_valid_to_db = 0.0
    for distance in protocol.DISTANCES_M[1:]:
        requested_drr = reference_drr_db - 20.0 * np.log10(distance)
        rir = degradations.synthesize_rir(SR, rt60_s=0.4, drr_db=requested_drr, rng=rng_seed)
        checks.append(
            Check(
                axis="distance",
                condition=f"{distance:.0f} m",
                quantity="direct-to-reverberant ratio",
                unit="dB",
                requested=float(requested_drr),
                measured=verify.measure_drr(rir, SR, direct_index=0),
                tolerance=0.05,
                note="6 dB per doubling, from 20 dB at the 1 m reference",
            )
        )
        # Record where locating the direct path by peak magnitude stops working. This is a
        # property of the measurement, not of the degradation, and it is reported rather than
        # hidden because a peak-picking DRR silently flattens out below it.
        if int(np.argmax(np.abs(rir))) == 0:
            peak_picking_valid_to_db = min(peak_picking_valid_to_db, float(requested_drr))

    observations.append(
        Observation(
            axis="distance",
            condition="DRR measurement",
            quantity="lowest DRR where the direct path is still the loudest sample",
            unit="dB",
            measured=peak_picking_valid_to_db,
            note="below this, measure_drr needs direct_index; argmax locks onto the tail",
        )
    )
    return checks, observations


def check_bandwidth(rng_seed: int = SEED) -> tuple[list[Check], list[Observation]]:
    """Band edges must be where they were requested, and stopbands must actually be empty.

    Two different claims, checked separately, because a filter can satisfy one and fail the other
    badly. The -3 dB point says where roll-off begins; the stopband rejection says whether the
    content above it is gone. For the resampling axis the second is the one that matters, and it
    is the one ``scipy.signal.resample_poly``'s defaults do not deliver.

    The resample round trip's measured -3 dB point sits about 7% below the nominal target Nyquist.
    That is by design, not an error: the anti-alias and anti-image filters are specified to reach
    full rejection *at* the target Nyquist, which forces their transition band to sit entirely
    below it. It is checked against that stated design rather than against the nominal rate.
    """
    checks: list[Check] = []
    observations: list[Observation] = []

    for cutoff in protocol.BAND_LIMIT_HZ:
        measured = verify.measure_bandwidth(
            lambda x, c=cutoff: degradations.band_limit(x, SR, high_hz=c), SR, rng=rng_seed
        )
        checks.append(
            Check(
                axis="bandwidth",
                condition=f"Butterworth low-pass {cutoff / 1000:.0f} kHz",
                quantity="-3 dB point",
                unit="Hz",
                requested=cutoff,
                measured=measured,
                tolerance=0.02 * cutoff,
                note="causal sosfilt; filtfilt would square the response",
            )
        )

    for target_sr in protocol.RESAMPLE_HZ:

        def process(x: np.ndarray, rate: int = target_sr) -> np.ndarray:
            return degradations.resample_roundtrip(x, SR, rate)

        nyquist = target_sr / 2.0
        rejection = verify.measure_stopband_attenuation(process, SR, nyquist, rng=rng_seed)
        checks.append(
            Check(
                axis="bandwidth",
                condition=f"round trip via {target_sr} Hz",
                quantity="rejection above target Nyquist",
                unit="dB",
                requested=80.0,
                measured=rejection,
                tolerance=0.5,
                direction="at-least",
                note="the claim that matters: the band above Nyquist is gone, not dented",
            )
        )
        cutoff_hz = verify.measure_bandwidth(process, SR, rng=rng_seed)
        observations.append(
            Observation(
                axis="bandwidth",
                condition=f"round trip via {target_sr} Hz",
                quantity="-3 dB point",
                unit="Hz",
                measured=cutoff_hz,
                note=f"nominal Nyquist {nyquist:.0f} Hz",
            )
        )
        observations.append(
            Observation(
                axis="bandwidth",
                condition=f"round trip via {target_sr} Hz",
                quantity="-3 dB point / target Nyquist",
                unit="ratio",
                measured=cutoff_hz / nyquist,
                note="consistently ~0.93: the transition band sits below the edge by design",
            )
        )
    return checks, observations


def check_quantisation(rng_seed: int = SEED) -> tuple[list[Check], list[Observation]]:
    """Measured SQNR must match the theoretical prediction for the signal's actual level.

    The prediction is ``10 * log10(P_x / (step**2 / 12))`` computed over the active region, not
    the textbook ``6.02 * bits + 1.76`` dB. The textbook figure assumes a full-scale signal; the
    test clip sits well below that, and using the shortcut would overstate the prediction by the
    crest factor and make every row of this table look wrong in the same direction.

    Mu-law is not checked against a target -- it has no simple closed form -- but it is reported
    against uniform PCM at the same bit depth, broken down by amplitude, because that comparison
    is the reason both are in the grid.
    """
    checks: list[Check] = []
    observations: list[Observation] = []
    clip = _test_clip(rng_seed)

    for bits in protocol.BIT_DEPTHS:
        quantised = degradations.quantize_pcm(clip.audio, bits)
        predicted = verify.predicted_sqnr_db(clip.audio, bits, clip.active)
        measured = verify.measure_sqnr(clip.audio, quantised, clip.active)
        checks.append(
            Check(
                axis="quantisation",
                condition=f"{bits}-bit PCM",
                quantity="SQNR",
                unit="dB",
                requested=predicted,
                measured=measured,
                tolerance=0.5,
                note="uniform-error model; an upper bound below about 8 bits",
            )
        )

    pcm8 = degradations.quantize_pcm(clip.audio, 8)
    mulaw8 = degradations.mu_law_codec(clip.audio, bits=8)
    for name, coded in (("8-bit PCM", pcm8), ("8-bit mu-law", mulaw8)):
        bands = verify.measure_sqnr_by_amplitude_band(clip.audio, coded, clip.active)
        for (low, high), sqnr in bands.items():
            observations.append(
                Observation(
                    axis="quantisation",
                    condition=f"{name}, amplitude {low:.0f}-{high:.0f} percentile",
                    quantity="SQNR",
                    unit="dB",
                    measured=sqnr,
                )
            )
    return checks, observations


def check_gain_and_clipping(rng_seed: int = SEED) -> tuple[list[Check], list[Observation]]:
    """Gain must be exactly the gain requested, and clipping must hit the requested fraction.

    The gain check looks trivial and is included precisely because of that: it is the assertion
    that no degradation quietly renormalises its output. A hidden peak normalisation would make
    the level, distance and clipping axes all measure something other than their name, and it
    would be invisible in every other row of this report.

    Clipping is parameterised by the fraction of samples driven into the limit rather than by a
    threshold in dBFS, so that the axis means the same thing on a hot recording and a quiet one.
    Crest factor is reported alongside as the physical consequence.
    """
    checks: list[Check] = []
    observations: list[Observation] = []
    clip = _test_clip(rng_seed)
    reference_level = verify.measure_level_dbfs(clip.audio, clip.active)

    for gain_db in protocol.GAIN_DB:
        amplified = degradations.apply_gain_db(clip.audio, gain_db)
        checks.append(
            Check(
                axis="gain",
                condition=f"{gain_db:+.0f} dB",
                quantity="level change",
                unit="dB",
                requested=gain_db,
                measured=verify.measure_level_dbfs(amplified, clip.active) - reference_level,
                tolerance=1e-6,
                note="verifies no hidden normalisation anywhere in the chain",
            )
        )

    observations.append(
        Observation(
            axis="clipping",
            condition="unclipped",
            quantity="crest factor",
            unit="dB",
            measured=verify.measure_crest_factor_db(clip.audio),
        )
    )
    for fraction in protocol.CLIP_FRACTIONS[1:]:
        clipped, threshold = degradations.clip_to_fraction(clip.audio, fraction)
        checks.append(
            Check(
                axis="clipping",
                condition=f"{fraction * 100:.1f}% of samples",
                quantity="clipped fraction",
                unit="fraction",
                requested=fraction,
                measured=verify.measure_clipped_fraction(clipped, threshold),
                tolerance=1e-4,
            )
        )
        observations.append(
            Observation(
                axis="clipping",
                condition=f"{fraction * 100:.1f}% of samples",
                quantity="crest factor",
                unit="dB",
                measured=verify.measure_crest_factor_db(clipped),
                note=f"clip threshold {threshold:.4f}",
            )
        )
    return checks, observations


def check_protocol_wiring(rng_seed: int = SEED) -> tuple[list[Check], list[Observation]]:
    """Every condition in the protocol grid must run end to end and preserve clip length.

    Not a DSP measurement -- a wiring check. Length preservation is what keeps a degraded clip
    aligned with its annotations, and it is the property most easily broken by adding a
    convolution or a resampler to the chain. Running the whole grid also confirms that no
    condition raises, which is otherwise only discovered halfway through an evaluation.
    """
    checks: list[Check] = []
    clip = _test_clip(rng_seed)
    _, sources = synth.soundscape(duration_s=CLIP_SECONDS, sr=SR, rng=rng_seed, n_species=3)
    interferers = [s.audio for s in sources]

    grid = list(protocol.DEFAULT_GRID) + list(protocol.COMPOSITE_CONDITIONS)
    n_finite = 0
    for spec in grid:
        degraded = protocol.apply_spec(
            clip.audio,
            SR,
            spec,
            active=clip.active,
            interferers=interferers,
            rng=rng_seed,
        )
        if len(degraded) == len(clip.audio) and np.all(np.isfinite(degraded)):
            n_finite += 1

    checks.append(
        Check(
            axis="protocol",
            condition=f"{len(grid)} grid conditions",
            quantity="length-preserving and finite",
            unit="conditions",
            requested=float(len(grid)),
            measured=float(n_finite),
            tolerance=0.0,
            note="every condition in DEFAULT_GRID plus the named composites",
        )
    )
    return checks, []


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

CHECK_SUITES: tuple[tuple[str, Callable[[int], tuple[list[Check], list[Observation]]]], ...] = (
    ("noise models", check_noise_models),
    ("additive noise SNR", check_snr),
    ("call overlap", check_overlap),
    ("distance", check_distance),
    ("bandwidth and codec", check_bandwidth),
    ("quantisation", check_quantisation),
    ("gain and clipping", check_gain_and_clipping),
    ("protocol wiring", check_protocol_wiring),
)


def run_validation(seed: int = SEED, suites: Sequence[str] | None = None) -> ValidationReport:
    """Run the requested suites and collect every check and observation."""
    checks: list[Check] = []
    observations: list[Observation] = []
    for name, suite in CHECK_SUITES:
        if suites is not None and name not in suites:
            continue
        suite_checks, suite_observations = suite(seed)
        checks.extend(suite_checks)
        observations.extend(suite_observations)
    return ValidationReport(checks=tuple(checks), observations=tuple(observations))


def _format_check_table(checks: Sequence[Check]) -> str:
    """Render checks as a fixed-width requested/measured/error table grouped by axis."""
    header = (
        f"{'condition':<34} {'quantity':<32} {'unit':<9} {'cmp':<4}"
        f"{'requested':>12} {'measured':>12} {'error':>10} {'tol':>9}  ok"
    )
    lines: list[str] = []
    for axis in dict.fromkeys(c.axis for c in checks):
        rows = [c for c in checks if c.axis == axis]
        lines.append("")
        lines.append(f"[{axis}]")
        lines.append(header)
        lines.append("-" * len(header))
        for check in rows:
            lines.append(
                f"{check.condition:<34} {check.quantity:<32} {check.unit:<9} "
                f"{check.comparator:<4}{check.requested:>12.4f} {check.measured:>12.4f} "
                f"{check.error:>+10.4f} {check.tolerance:>9.4f}  "
                f"{'PASS' if check.passed else 'FAIL'}"
            )
    return "\n".join(lines)


def _format_observation_table(observations: Sequence[Observation]) -> str:
    """Render observations, which have no requested value and no pass/fail."""
    header = f"{'condition':<44} {'quantity':<38} {'unit':<9} {'measured':>12}"
    lines: list[str] = []
    for axis in dict.fromkeys(o.axis for o in observations):
        rows = [o for o in observations if o.axis == axis]
        lines.append("")
        lines.append(f"[{axis}]")
        lines.append(header)
        lines.append("-" * len(header))
        for observation in rows:
            lines.append(
                f"{observation.condition:<44} {observation.quantity:<38} "
                f"{observation.unit:<9} {observation.measured:>12.4f}"
            )
    return "\n".join(lines)


def format_report(report: ValidationReport, show_observations: bool = True) -> str:
    """Render the whole report, banner included."""
    lines = [
        "=" * 118,
        "birdsong-robustness :: degradation pipeline validation",
        "=" * 118,
        "",
        "Requested parameters against parameters measured back from the output waveform.",
        "Measurements are independent of the code being measured; see verify.py.",
        "",
        "This validation report involves no classifier: it checks the degradation pipeline only",
        "and contains no accuracy numbers, and none should be inferred from it. The real-audio",
        "classifier results (BirdNET on a Xeno-canto set) are produced separately by",
        "scripts/run_real_eval.py -- see docs/real_eval_results.json and the README.",
    ]
    lines.append(_format_check_table(report.checks))
    if show_observations and report.observations:
        lines.append("")
        lines.append("=" * 118)
        lines.append("Observations (measured, no requested target -- descriptive only)")
        lines.append("=" * 118)
        lines.append(_format_observation_table(report.observations))

    passed = len(report.checks) - report.n_failed
    lines.append("")
    lines.append("=" * 118)
    lines.append(f"{passed}/{len(report.checks)} checks within tolerance.")
    worst = report.worst
    if worst is not None:
        lines.append(
            f"Largest error relative to tolerance: {worst.axis} / {worst.condition} / "
            f"{worst.quantity} -- {worst.error:+.4f} {worst.unit} "
            f"against a tolerance of {worst.tolerance:.4f}."
        )
    lines.append("=" * 118)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns 0 when every check is within tolerance, 1 otherwise."""
    parser = argparse.ArgumentParser(
        prog="birdsong-validate",
        description=(
            "Sweep every degradation across its parameter grid and report requested against "
            "measured. Validates the degradation pipeline only; no classifier is involved."
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED, help="master seed (default: %(default)s)")
    parser.add_argument("--json", type=str, default=None, help="also write results to this path")
    parser.add_argument(
        "--no-observations", action="store_true", help="print only the pass/fail checks"
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=[name for name, _ in CHECK_SUITES],
        help="run only this suite; repeatable",
    )
    args = parser.parse_args(argv)

    report = run_validation(seed=args.seed, suites=args.suite)
    print(format_report(report, show_observations=not args.no_observations))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            handle.write(report.to_json())
        print(f"\nWrote {args.json}")

    return 1 if report.n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
