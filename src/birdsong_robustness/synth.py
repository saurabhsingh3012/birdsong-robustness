"""Synthetic bird-song-like test signals and DSP probe signals.

Why synthesise instead of shipping audio
----------------------------------------
This repository deliberately contains no audio files. Xeno-canto and the DCASE/BirdCLEF
soundscape sets carry per-recording Creative Commons terms that vary by recording, so
redistributing clips inside a Git repository is a licensing problem waiting to happen. It is
also a reproducibility problem: a reader who clones the repo should be able to reproduce every
number in the README from a seed, not from a binary blob whose provenance is a URL that may
rot.

Everything here is therefore generated from ``numpy.random.Generator`` with an explicit seed.

What these signals are and are not
----------------------------------
:func:`bird_call` produces a frequency-modulated, harmonically structured, syllabically
repeated signal with the gross time-frequency morphology of passerine song: syllables of
tens to a few hundred milliseconds, fundamentals in the 2--6 kHz range, sweeps rather than
steady tones, and silent inter-syllable gaps.

That morphology is what the degradation and verification code needs to be exercised
honestly -- a band-limiting test on a steady 1 kHz sine tells you much less than the same test
on a signal with real high-frequency structure, and an "SNR during the active region" test is
meaningless without genuine silent gaps.

It is emphatically **not** a species model, and nothing here should be fed to a classifier and
treated as a bird. It is a DSP test signal shaped like bird song.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


@dataclass(frozen=True)
class SyntheticClip:
    """A synthetic clip together with the ground truth a real dataset would have to annotate.

    Attributes:
        audio: Float64 waveform, peak-normalised to ``peak``.
        sr: Sample rate in Hz.
        active: Boolean mask, True where a call is present. Known exactly here; in real data
            this would come from strong labels or an energy-based segmenter, and its error
            would propagate into any active-basis SNR figure.
        events: ``(start_sample, end_sample)`` pairs for each syllable.
    """

    audio: np.ndarray
    sr: int
    active: np.ndarray
    events: list[tuple[int, int]] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sr

    @property
    def active_fraction(self) -> float:
        """Fraction of the clip containing signal. Typically 0.1--0.3 for real soundscapes."""
        return float(np.mean(self.active))


def _syllable(
    duration_s: float,
    sr: int,
    f_start: float,
    f_end: float,
    n_harmonics: int,
    harmonic_rolloff_db: float,
    fm_depth: float,
    fm_rate_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One syllable: a harmonically rich FM sweep under a smooth amplitude envelope.

    The instantaneous frequency glides from ``f_start`` to ``f_end`` (log-linear, because
    birds sweep in roughly musical rather than linear intervals) with an additional sinusoidal
    vibrato of depth ``fm_depth``. Harmonics fall off at ``harmonic_rolloff_db`` per harmonic,
    which is what puts real energy above 8 kHz and makes band-limiting tests meaningful.
    """
    n = max(2, round(duration_s * sr))
    t = np.arange(n) / sr

    # Log-linear glide. Guard against non-positive frequencies.
    f_start = max(f_start, 1.0)
    f_end = max(f_end, 1.0)
    glide = f_start * (f_end / f_start) ** (t / max(t[-1], 1e-9))
    vibrato = 1.0 + fm_depth * np.sin(2 * np.pi * fm_rate_hz * t + rng.uniform(0, 2 * np.pi))
    f_inst = glide * vibrato

    # Phase is the integral of instantaneous frequency; cumulative-sum it rather than using
    # sin(2*pi*f*t), which is a classic and badly wrong shortcut for FM signals.
    dt = 1.0 / sr
    phase = 2 * np.pi * np.cumsum(f_inst) * dt

    nyq = sr / 2.0
    out = np.zeros(n)
    for h in range(1, n_harmonics + 1):
        amp = 10.0 ** (-harmonic_rolloff_db * (h - 1) / 20.0)
        # Drop harmonics that would alias. Aliasing here would be a synthesis artefact that
        # later shows up as "the classifier is sensitive to high frequencies".
        if np.max(f_inst) * h >= nyq:
            break
        out += amp * np.sin(h * phase + rng.uniform(0, 2 * np.pi))

    # Raised-cosine attack/decay. Hard edges would splatter broadband energy and corrupt every
    # spectral measurement downstream.
    ramp_len = max(1, int(0.15 * n))
    env = np.ones(n)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp_len)))
    env[:ramp_len] *= ramp
    env[-ramp_len:] *= ramp[::-1]
    return out * env


def bird_call(
    duration_s: float = 5.0,
    sr: int = 32000,
    rng: np.random.Generator | int | None = None,
    n_syllables: int = 8,
    syllable_ms: tuple[float, float] = (60.0, 180.0),
    f0_hz: tuple[float, float] = (2200.0, 5200.0),
    n_harmonics: int = 4,
    harmonic_rolloff_db: float = 9.0,
    peak: float = 0.7,
) -> SyntheticClip:
    """Generate a bird-song-like clip with silent gaps and an exact activity mask.

    Args:
        duration_s: Total clip duration.
        sr: Sample rate in Hz. 32 kHz is a reasonable default; note that BirdNET operates at
            48 kHz internally and resamples, which is itself a degradation worth being aware of.
        rng: Seed or Generator.
        n_syllables: Number of syllables placed in the clip.
        syllable_ms: Inclusive range of syllable durations, milliseconds.
        f0_hz: Inclusive range from which each syllable's start and end fundamental is drawn.
        n_harmonics: Harmonics per syllable, including the fundamental.
        harmonic_rolloff_db: Level drop per harmonic, dB.
        peak: Peak amplitude of the returned waveform.

    Returns:
        A :class:`SyntheticClip`.
    """
    generator = _as_rng(rng)
    n = round(duration_s * sr)
    audio = np.zeros(n, dtype=np.float64)
    active = np.zeros(n, dtype=bool)
    events: list[tuple[int, int]] = []

    for _ in range(n_syllables):
        dur = generator.uniform(*syllable_ms) / 1000.0
        length = round(dur * sr)
        if length >= n:
            continue
        start = int(generator.integers(0, n - length))
        f_a = generator.uniform(*f0_hz)
        # Sweep up or down by up to a factor of 1.8 -- a realistic glide range.
        f_b = f_a * generator.uniform(1 / 1.8, 1.8)
        syl = _syllable(
            duration_s=dur,
            sr=sr,
            f_start=f_a,
            f_end=f_b,
            n_harmonics=n_harmonics,
            harmonic_rolloff_db=harmonic_rolloff_db,
            fm_depth=generator.uniform(0.0, 0.06),
            fm_rate_hz=generator.uniform(20.0, 90.0),
            rng=generator,
        )
        end = start + len(syl)
        audio[start:end] += syl * generator.uniform(0.6, 1.0)
        active[start:end] = True
        events.append((start, end))

    peak_abs = float(np.max(np.abs(audio)))
    if peak_abs > 0:
        audio = audio * (peak / peak_abs)

    events.sort()
    return SyntheticClip(audio=audio, sr=sr, active=active, events=events)


def soundscape(
    duration_s: float = 10.0,
    sr: int = 32000,
    rng: np.random.Generator | int | None = None,
    n_species: int = 3,
    calls_per_species: int = 4,
) -> tuple[SyntheticClip, list[SyntheticClip]]:
    """Generate a multi-source clip plus the individual per-source clips that compose it.

    Returned separately because the overlap and SIR verification code needs access to the
    isolated sources; once summed, separating them again is the hard problem this whole field
    is about.

    Args:
        duration_s: Clip duration.
        sr: Sample rate.
        rng: Seed or Generator.
        n_species: Number of distinct synthetic sources.
        calls_per_species: Syllables per source.

    Returns:
        ``(mixture, sources)``.
    """
    generator = _as_rng(rng)
    sources: list[SyntheticClip] = []
    for _ in range(n_species):
        sources.append(
            bird_call(
                duration_s=duration_s,
                sr=sr,
                rng=generator,
                n_syllables=calls_per_species,
                f0_hz=(generator.uniform(1500, 3000), generator.uniform(3500, 7000)),
            )
        )

    mix = np.sum([s.audio for s in sources], axis=0)
    active = np.any([s.active for s in sources], axis=0)
    events = sorted(e for s in sources for e in s.events)

    peak_abs = float(np.max(np.abs(mix)))
    if peak_abs > 0:
        mix = mix * (0.7 / peak_abs)

    return SyntheticClip(audio=mix, sr=sr, active=active, events=events), sources


def impulse(n_samples: int, position: int = 0, amplitude: float = 1.0) -> np.ndarray:
    """A unit impulse. Feeding this through an LTI degradation recovers its impulse response."""
    x = np.zeros(n_samples, dtype=np.float64)
    if not 0 <= position < n_samples:
        raise ValueError("position must lie inside the signal")
    x[position] = amplitude
    return x


def log_sweep(
    duration_s: float,
    sr: int,
    f_start: float = 20.0,
    f_end: float | None = None,
    amplitude: float = 0.5,
) -> np.ndarray:
    """An exponential sine sweep, the standard probe for measuring a magnitude response.

    Preferred over white noise when measuring a fixed filter because it puts deterministic
    energy at every frequency and needs no averaging, so the measured transfer function has no
    estimator variance to argue about.
    """
    if f_end is None:
        f_end = 0.45 * sr
    n = round(duration_s * sr)
    t = np.arange(n) / sr
    k = np.log(f_end / f_start)
    phase = 2 * np.pi * f_start * duration_s / k * (np.exp(k * t / duration_s) - 1.0)
    return amplitude * np.sin(phase)
