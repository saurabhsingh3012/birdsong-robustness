"""Noise models used as maskers in the robustness benchmark.

Why not just use white noise
----------------------------
White noise is the default choice in most augmentation libraries and it is the wrong one for
bioacoustics. Its power is flat per hertz, so in a 0--16 kHz recording roughly half its energy
sits above 8 kHz where almost no passerine song lives. Mixing white noise at "0 dB SNR" therefore
spends most of the noise budget outside the band the classifier actually listens to, and the
resulting benchmark is easier than it looks.

Natural background noise is close to *pink* -- power falling at about 3 dB per octave, i.e.
constant power per octave rather than per hertz. That distributes masking energy the way a
real forest, road or river does, and it concentrates it in the 1--8 kHz range where bird song
and the classifier's discriminative features overlap. Pink noise at a given SNR is a
substantially harder and more realistic masker than white noise at the same SNR.

Brown noise (6 dB/octave) is included as the low-frequency-dominated extreme: it is a
reasonable stand-in for distant traffic rumble and for the pressure-fluctuation component of
wind, and it is useful precisely because it should *not* hurt a bird classifier much. A model
that degrades sharply under brown noise is telling you something about its front-end gain
staging or normalisation, not about its acoustic discrimination.

Wind and rain
-------------
Wind and rain are modelled parametrically rather than sampled, because this repository ships
no audio. Both models are documented approximations, not recordings, and the README says so.
They are built to have the right *structure*, which is what stresses a classifier:

* **Wind** at a microphone is dominated by turbulent pressure fluctuations ("pseudo-sound"),
  not by radiated sound. Its spectrum is steeply low-frequency weighted and it is strongly
  amplitude-modulated by gusts on a timescale of seconds. It masks bird song less by direct
  spectral overlap than by driving the front end -- level normalisation, AGC, and per-clip
  spectral mean subtraction all react to it.
* **Rain** is the opposite case. It is a dense superposition of impulsive droplet impacts with
  energy concentrated in roughly 1--10 kHz, which lands directly on top of the bird band, plus
  a broadband hiss floor. It should be the more damaging of the two for in-band detection.

Keeping both in the grid separates "the model is confused by in-band masking" from "the model
is confused by loud low-frequency energy", which are different failure modes with different
fixes.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import signal as sps

NoiseColor = Literal["white", "pink", "brown", "blue", "violet"]

#: Spectral exponent ``beta`` for PSD proportional to ``1 / f**beta``.
#: The PSD slope in dB/octave is ``-3.0103 * beta``.
COLOR_BETA: dict[str, float] = {
    "violet": -2.0,  # +6 dB/octave
    "blue": -1.0,  # +3 dB/octave
    "white": 0.0,  # flat
    "pink": 1.0,  # -3 dB/octave
    "brown": 2.0,  # -6 dB/octave
}


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    """Accept a Generator, an int seed, or None and always return a Generator."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def colored_noise(
    n_samples: int,
    beta: float = 1.0,
    sr: int = 32000,
    rng: np.random.Generator | int | None = None,
    f_min_hz: float = 10.0,
) -> np.ndarray:
    """Generate noise whose power spectral density follows ``1 / f**beta``.

    The synthesis is done in the frequency domain: draw complex Gaussian coefficients with a
    flat expected magnitude, multiply by ``f ** (-beta / 2)`` (amplitude, hence the halved
    exponent), and invert. This gives an exact spectral shape in expectation, unlike the
    cascaded-IIR "pinking filter" approach whose slope drifts by a decibel or so across the
    band. Since the whole point of this package is that requested parameters must be
    recoverable by measurement, the exact method is the right one.

    Args:
        n_samples: Length of the output in samples.
        beta: Spectral exponent. 0 = white, 1 = pink, 2 = brown, -1 = blue, -2 = violet.
        sr: Sample rate in Hz, used only to place the ``f_min_hz`` knee.
        rng: Seed or Generator. Seeded for reproducibility.
        f_min_hz: Frequencies below this are held flat instead of continuing to rise. Without
            this the DC-adjacent bins of a brown-noise spectrum carry effectively unbounded
            power and the output is dominated by a slow random walk rather than by audible
            noise. 10 Hz is below anything a bird recording cares about.

    Returns:
        Float64 array of length ``n_samples``, zero-mean, normalised to unit RMS.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    generator = _as_rng(rng)

    # Work at the next power of two for FFT efficiency, then trim. Trimming does not disturb
    # the spectral slope because the process is stationary.
    n_fft = int(2 ** np.ceil(np.log2(max(n_samples, 16))))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    scale = np.ones_like(freqs)
    knee = max(f_min_hz, freqs[1])
    active = freqs >= knee
    scale[active] = (freqs[active] / knee) ** (-beta / 2.0)
    scale[0] = 0.0  # remove DC; a DC offset is not noise, it is a bias

    phases = generator.uniform(0.0, 2.0 * np.pi, size=freqs.shape)
    magnitudes = generator.rayleigh(scale=1.0 / np.sqrt(2.0), size=freqs.shape)
    spectrum = magnitudes * scale * np.exp(1j * phases)

    # Real-signal constraints: DC and (for even n_fft) Nyquist must be real.
    spectrum[0] = 0.0
    if n_fft % 2 == 0:
        spectrum[-1] = np.abs(spectrum[-1])

    x = np.fft.irfft(spectrum, n=n_fft)[:n_samples]
    x = x - float(np.mean(x))
    rms = float(np.sqrt(np.mean(x**2)))
    if rms > 0:
        x = x / rms
    return x


def color_noise(
    n_samples: int,
    color: NoiseColor = "pink",
    sr: int = 32000,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Convenience wrapper around :func:`colored_noise` taking a named colour."""
    if color not in COLOR_BETA:
        raise ValueError(f"unknown noise colour {color!r}; expected one of {sorted(COLOR_BETA)}")
    return colored_noise(n_samples, beta=COLOR_BETA[color], sr=sr, rng=rng)


def wind_noise(
    n_samples: int,
    sr: int = 32000,
    rng: np.random.Generator | int | None = None,
    corner_hz: float = 300.0,
    beta: float = 2.5,
    gust_rate_hz: float = 0.2,
    gust_depth: float = 0.7,
) -> np.ndarray:
    """Synthesise wind-like microphone noise.

    Structure, in order of importance to a classifier:

    1. A steep low-frequency tilt (``beta`` ~ 2.5, i.e. about -7.5 dB/octave) because
       microphone wind noise is turbulent pressure fluctuation, which is not acoustic
       radiation and does not obey the flatter spectra of distant sound sources.
    2. A first-order roll-off above ``corner_hz`` on top of that tilt, representing the fact
       that turbulent eddies large enough to produce pressure at the diaphragm are slow.
    3. Slow amplitude modulation by gusts. This is the part that matters operationally: a
       gust changes the clip-level statistics that per-clip normalisation depends on, so a
       model can lose a detection during a gust even though the in-band masking barely moved.

    Args:
        n_samples: Output length in samples.
        sr: Sample rate in Hz.
        rng: Seed or Generator.
        corner_hz: -3 dB corner of the additional low-pass, in Hz.
        beta: Spectral exponent of the underlying coloured noise.
        gust_rate_hz: Approximate gust modulation rate. 0.2 Hz is a gust every five seconds.
        gust_depth: Modulation depth in [0, 1). 0 is steady wind, 0.9 is very gusty.

    Returns:
        Unit-RMS float64 array of length ``n_samples``.
    """
    if not 0.0 <= gust_depth < 1.0:
        raise ValueError("gust_depth must be in [0, 1)")
    generator = _as_rng(rng)

    base = colored_noise(n_samples, beta=beta, sr=sr, rng=generator)

    # Additional single-pole low-pass. Butterworth order 1 keeps the composite slope smooth
    # instead of putting a hard shelf into the spectrum.
    nyq = sr / 2.0
    wn = min(corner_hz / nyq, 0.99)
    b, a = sps.butter(1, wn, btype="low")
    base = sps.lfilter(b, a, base)

    # Gust envelope: low-pass filtered noise, rectified into a positive envelope.
    env_noise = generator.standard_normal(n_samples)
    env_wn = min(max(gust_rate_hz / nyq, 1e-6), 0.99)
    be, ae = sps.butter(2, env_wn, btype="low")
    env = sps.lfilter(be, ae, env_noise)
    env_std = float(np.std(env))
    if env_std > 0:
        env = env / env_std
    envelope = 1.0 + gust_depth * np.tanh(env)  # bounded, strictly positive for depth < 1
    base = base * envelope

    base = base - float(np.mean(base))
    rms = float(np.sqrt(np.mean(base**2)))
    if rms > 0:
        base = base / rms
    return base


def rain_noise(
    n_samples: int,
    sr: int = 32000,
    rng: np.random.Generator | int | None = None,
    drops_per_second: float = 900.0,
    drop_freq_hz: tuple[float, float] = (1200.0, 8000.0),
    drop_decay_ms: float = 1.2,
    hiss_fraction: float = 0.35,
) -> np.ndarray:
    """Synthesise rain-like noise as a Poisson shower of damped impacts plus a hiss floor.

    Each droplet is a short exponentially-damped sinusoid with a randomly drawn centre
    frequency -- the classic model for a small impulsive impact on a surface with a lightly
    damped resonance. Superposing many of them at a Poisson rate reproduces the two features
    that make rain hard for a detector:

    * the energy sits in roughly 1--10 kHz, directly on top of bird song, unlike wind; and
    * it is *impulsive*, so short-time spectral statistics fluctuate violently. Anything that
      estimates a noise floor from a percentile of the spectrogram will get it wrong.

    ``drops_per_second`` is the intensity knob. A few hundred is light rain on foliage; several
    thousand approaches the continuous roar of heavy rain, at which point the sum tends to a
    Gaussian by the central limit theorem and the impulsive character disappears -- which is
    itself a realistic transition.

    Args:
        n_samples: Output length in samples.
        sr: Sample rate in Hz.
        rng: Seed or Generator.
        drops_per_second: Poisson rate of droplet impacts.
        drop_freq_hz: Inclusive range from which each droplet's resonance is drawn (log-uniform).
        drop_decay_ms: Exponential decay time constant of a droplet, in milliseconds.
        hiss_fraction: Fraction of total power assigned to a broadband blue-ish hiss floor,
            representing the unresolved far-field of many distant drops.

    Returns:
        Unit-RMS float64 array of length ``n_samples``.
    """
    if not 0.0 <= hiss_fraction <= 1.0:
        raise ValueError("hiss_fraction must be in [0, 1]")
    generator = _as_rng(rng)

    duration_s = n_samples / sr
    n_drops = int(generator.poisson(drops_per_second * duration_s))
    out = np.zeros(n_samples, dtype=np.float64)

    if n_drops > 0:
        decay_tau = drop_decay_ms * 1e-3
        # Truncate each droplet at 5 time constants (-43 dB); beyond that it is inaudible and
        # the cost of synthesising it is not.
        drop_len = max(4, int(5.0 * decay_tau * sr))
        t = np.arange(drop_len) / sr
        envelope = np.exp(-t / decay_tau)

        starts = generator.integers(0, n_samples, size=n_drops)
        f_lo, f_hi = drop_freq_hz
        freqs = np.exp(generator.uniform(np.log(f_lo), np.log(f_hi), size=n_drops))
        phases = generator.uniform(0, 2 * np.pi, size=n_drops)
        # Impact amplitude is heavy-tailed: most drops are small, a few are close and loud.
        amps = generator.lognormal(mean=0.0, sigma=0.5, size=n_drops)

        for start, f0, phase, amp in zip(starts, freqs, phases, amps):
            end = min(start + drop_len, n_samples)
            length = end - start
            if length <= 0:
                continue
            out[start:end] += amp * envelope[:length] * np.sin(2 * np.pi * f0 * t[:length] + phase)

    impacts_rms = float(np.sqrt(np.mean(out**2)))
    if impacts_rms > 0:
        out = out / impacts_rms
    else:  # pathologically low rate: fall back to pure hiss rather than returning silence
        hiss_fraction = 1.0

    if hiss_fraction > 0:
        # Blue-ish (beta = -0.5) hiss: slightly rising, matching the high-frequency emphasis of
        # many small impacts heard at a distance through foliage.
        hiss = colored_noise(n_samples, beta=-0.5, sr=sr, rng=generator)
        out = np.sqrt(1.0 - hiss_fraction) * out + np.sqrt(hiss_fraction) * hiss

    out = out - float(np.mean(out))
    rms = float(np.sqrt(np.mean(out**2)))
    if rms > 0:
        out = out / rms
    return out


#: Registry mapping a noise name to a callable ``(n_samples, sr, rng) -> np.ndarray``.
#: Used by the degradation grid so that noise type is a first-class, enumerable axis.
NOISE_FACTORIES: dict[str, object] = {
    "white": lambda n, sr, rng: color_noise(n, "white", sr, rng),
    "pink": lambda n, sr, rng: color_noise(n, "pink", sr, rng),
    "brown": lambda n, sr, rng: color_noise(n, "brown", sr, rng),
    "wind": lambda n, sr, rng: wind_noise(n, sr, rng),
    "rain": lambda n, sr, rng: rain_noise(n, sr, rng),
}


def make_noise(
    name: str,
    n_samples: int,
    sr: int = 32000,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Build a named noise signal from :data:`NOISE_FACTORIES`, normalised to unit RMS."""
    if name not in NOISE_FACTORIES:
        raise ValueError(f"unknown noise {name!r}; expected one of {sorted(NOISE_FACTORIES)}")
    factory = NOISE_FACTORIES[name]
    return factory(n_samples, sr, _as_rng(rng))  # type: ignore[operator]
