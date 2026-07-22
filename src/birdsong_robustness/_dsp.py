"""Low-level signal primitives shared by :mod:`degradations` and :mod:`verify`.

Kept in one place so that the degradation code and the code that checks it agree on what
"power", "RMS" and "the active region" mean. If those definitions drifted apart, the validation
table would be measuring the difference between two conventions rather than the behaviour of
the degradation, and it would look like a passing test.
"""

from __future__ import annotations

import numpy as np

#: Floor used when converting to decibels, to keep ``log10(0)`` out of the code paths.
#: -400 dB is far below float64 resolution for any realistic signal.
EPS = 1e-20

#: ``np.trapz`` was renamed ``np.trapezoid`` in NumPy 2.0 and removed under the old name. Bind it
#: once here so the package works across the 1.x/2.x boundary without pinning a NumPy version,
#: which a benchmark meant to be re-run years from now should avoid doing.
trapezoid = getattr(np, "trapezoid", None) or np.trapz


def as_float64(x: np.ndarray) -> np.ndarray:
    """Return a contiguous float64 view/copy. All DSP here is done in double precision."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D mono signal, got shape {arr.shape}")
    return arr


def power(x: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean square power of ``x``, optionally restricted to a boolean ``mask``."""
    arr = as_float64(x)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != arr.shape:
            raise ValueError(f"mask shape {mask.shape} does not match signal shape {arr.shape}")
        if not mask.any():
            raise ValueError("mask selects no samples; cannot compute power")
        arr = arr[mask]
    return float(np.mean(arr**2))


def rms(x: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Root-mean-square level of ``x``, optionally restricted to a boolean ``mask``."""
    return float(np.sqrt(power(x, mask)))


def db(value: float) -> float:
    """Convert a *power* ratio to decibels."""
    return float(10.0 * np.log10(max(float(value), EPS)))


def db_amplitude(value: float) -> float:
    """Convert an *amplitude* ratio to decibels."""
    return float(20.0 * np.log10(max(abs(float(value)), EPS)))


def dbfs(x: np.ndarray, mask: np.ndarray | None = None) -> float:
    """RMS level of ``x`` in dB relative to full scale (1.0 amplitude)."""
    return db_amplitude(rms(x, mask))


def frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Split into overlapping frames, dropping any trailing partial frame.

    Returns:
        Array of shape ``(n_frames, frame_len)``. Empty with the right second dimension if the
        signal is shorter than one frame, so callers can rely on ``.shape[1]``.
    """
    arr = as_float64(x)
    if frame_len <= 0 or hop <= 0:
        raise ValueError("frame_len and hop must be positive")
    n_frames = 1 + (len(arr) - frame_len) // hop if len(arr) >= frame_len else 0
    if n_frames <= 0:
        return np.empty((0, frame_len), dtype=np.float64)
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return arr[idx]


def energy_active_mask(
    x: np.ndarray,
    sr: int,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    threshold_db: float = -25.0,
) -> np.ndarray:
    """Estimate which samples contain signal, by frame energy relative to the loudest frame.

    This is a crude segmenter and it is only used when the caller has not supplied a ground
    truth mask. It matters because of the SNR-basis problem documented in
    :func:`birdsong_robustness.degradations.add_noise_at_snr`: bird recordings are mostly
    silence, so an SNR computed over the whole file is not the SNR the classifier experiences
    during a call.

    In real use the mask should come from strong labels. When it comes from an energy
    threshold instead, its error propagates directly into the reported SNR, which is why the
    validation harness reports both bases rather than picking one.

    Args:
        x: Input signal.
        sr: Sample rate in Hz.
        frame_ms: Analysis frame length.
        hop_ms: Analysis hop.
        threshold_db: Frames whose energy is more than this far below the loudest frame are
            treated as inactive. -25 dB is permissive enough to keep syllable tails.

    Returns:
        Boolean mask the same length as ``x``.
    """
    arr = as_float64(x)
    frame_len = max(1, round(frame_ms * 1e-3 * sr))
    hop = max(1, round(hop_ms * 1e-3 * sr))
    frames = frame_signal(arr, frame_len, hop)
    mask = np.zeros(len(arr), dtype=bool)
    if frames.shape[0] == 0:
        return np.ones(len(arr), dtype=bool)

    frame_db = 10.0 * np.log10(np.mean(frames**2, axis=1) + EPS)
    keep = frame_db >= (float(np.max(frame_db)) + threshold_db)
    for i, is_active in enumerate(keep):
        if is_active:
            start = i * hop
            mask[start : min(start + frame_len, len(arr))] = True
    if not mask.any():  # degenerate (e.g. digital silence): treat everything as active
        mask[:] = True
    return mask


def audible_mask(x: np.ndarray, floor_db: float = -40.0) -> np.ndarray:
    """Samples whose magnitude exceeds ``floor_db`` relative to the signal's own RMS.

    The overlap axis needs a single answer to "is there competing sound here?", and both the
    code that *places* interference and the code that *measures* it must use it, or the
    requested-versus-measured table for that axis would be comparing two different definitions
    of the word "overlap".

    The specific failure this avoids: a segment cut out of a real recording by a frame-based
    segmenter usually contains exact digital silence between syllables. Marking the whole
    segment span as "overlapped" counts that silence as competing sound, which both inflates
    the reported density and -- because the signal-to-interference ratio is then set over a
    region that is partly silent -- makes the interference roughly a decibel hotter than
    requested wherever it is actually present. Measured on this repository's synthetic
    soundscapes, the span-based convention overstated density by about 18% and undershot the
    requested SIR by 0.9 dB at low densities.

    The gate is relative to the signal's own RMS, so it is invariant to any later scaling. That
    matters because the SIR gain is applied *after* the region is chosen.

    Args:
        x: Signal to gate.
        floor_db: Threshold in dB relative to the RMS of ``x``. -40 dB is far below anything a
            classifier's front end would resolve against its own noise floor.

    Returns:
        Boolean mask the same length as ``x``. All-False if ``x`` is identically zero.
    """
    arr = as_float64(x)
    level = rms(arr)
    if level <= 0:
        return np.zeros(len(arr), dtype=bool)
    return np.abs(arr) > level * 10.0 ** (floor_db / 20.0)


def match_length(x: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad ``x`` to exactly ``n`` samples.

    Resamplers and convolutions change length by a sample or two. Every degradation in this
    package is length-preserving so that a degraded clip stays aligned with its annotations.
    """
    arr = as_float64(x)
    if len(arr) == n:
        return arr
    if len(arr) > n:
        return arr[:n]
    return np.concatenate([arr, np.zeros(n - len(arr), dtype=np.float64)])


def apply_fir_zero_delay(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Convolve with a linear-phase FIR and compensate its group delay.

    A symmetric FIR of length ``L`` delays by ``(L - 1) / 2`` samples. Compensating keeps the
    degraded signal sample-aligned with the clean one, which the residual-based SNR
    measurement in :mod:`verify` depends on -- a few samples of slip there would show up as
    several decibels of phantom "noise".

    Uses ``scipy.signal.fftconvolve`` rather than ``filtfilt`` because ``filtfilt`` applies the
    magnitude response twice, which would make requested and measured attenuation differ by a
    factor of two.
    """
    from scipy import signal as sps

    arr = as_float64(x)
    h = as_float64(taps)
    if len(h) % 2 == 0:
        raise ValueError("expected an odd-length symmetric FIR so the delay is an integer")
    delay = (len(h) - 1) // 2
    y = sps.fftconvolve(arr, h, mode="full")
    return y[delay : delay + len(arr)]
