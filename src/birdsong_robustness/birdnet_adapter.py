"""A real classifier behind the :class:`~birdsong_robustness.protocol.BioacousticModel` interface.

This is the adapter the protocol was written around (see the ``BioacousticModel`` docstring in
``protocol.py``): it wraps `BirdNET <https://github.com/kahst/BirdNET-Analyzer>`_ -- Cornell's
open bioacoustic classifier -- so it can be swept across the degradation grid. BirdNET is the
natural first model given the author's earlier MixIT + BirdNET work.

The three model-specific decisions the interface docstring insists must be *fixed and recorded*
are made explicit here, because each one silently changes what the benchmark measures:

* **Resampling.** BirdNET runs at 48 kHz. The whole pipeline in this project also runs at
  48 kHz, so in the normal path no resampling happens inside the adapter at all. If a caller
  hands audio at another rate, :func:`_to_48k` uses one fixed resampler (``librosa.resample``,
  ``kaiser_best``) for *every* condition, so the bandwidth axis never measures the adapter.
* **Window-to-clip aggregation.** BirdNET scores 3-second windows. This adapter takes the
  **maximum** confidence per species across a clip's windows -- appropriate for a detection
  task where a species is "present" if it is confidently detected anywhere in the clip. The
  choice is fixed here and recorded in :attr:`aggregation`.
* **Built-in normalisation.** BirdNET applies its own per-window standardisation internally.
  That will partly flatten the gain axis; that is legitimate model behaviour and is reported,
  not disabled. It is noted in the README's real-audio limitations.

This module is import-safe without BirdNET installed: ``birdnetlib`` and ``librosa`` are imported
lazily inside :meth:`BirdNETModel.__init__`, so CI (which installs neither) can still import the
package. Install the real dependencies with ``pip install -e ".[birdnet]"``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

__all__ = ["BirdNETModel", "aggregate_detections"]

BIRDNET_SR = 48000


def _to_48k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample to BirdNET's 48 kHz with one fixed resampler, or pass through when already 48 kHz."""
    x = np.asarray(audio, dtype=np.float32)
    if sr == BIRDNET_SR:
        return x
    import librosa

    return librosa.resample(x, orig_sr=sr, target_sr=BIRDNET_SR, res_type="kaiser_best")


def aggregate_detections(
    detections: Iterable[Mapping[str, object]], species: Sequence[str]
) -> dict[str, float]:
    """Reduce BirdNET's per-window detections to one score per species in ``species``.

    The score is the maximum confidence for that species across the clip's 3-second windows;
    species absent from ``detections`` (BirdNET did not surface them above its floor) score 0.
    Kept as a free function so the aggregation is unit-testable without a TFLite backend.
    """
    wanted = set(species)
    scores: dict[str, float] = dict.fromkeys(species, 0.0)
    for detection in detections:
        name = detection.get("scientific_name")
        if name in wanted:
            scores[name] = max(scores[name], float(detection["confidence"]))  # type: ignore[arg-type]
    return scores


class BirdNETModel:
    """BirdNET wrapped as a :class:`~birdsong_robustness.protocol.BioacousticModel`.

    Args:
        species: The scientific names scored per clip. These are the manifest's vocabulary and
            must be BirdNET labels (e.g. ``"Turdus merula"``). ``predict`` returns a score for
            each; species BirdNET does not surface above ``min_conf`` score 0.
        min_conf: BirdNET's detection floor. Set low (0.01) so the score sweep in
            :func:`~birdsong_robustness.protocol.select_threshold` sees graded confidences rather
            than a hard-thresholded set.
        model_version: Recorded verbatim into the result for provenance.

    The constructor loads the BirdNET TFLite model once and reuses it, so ``predict`` is cheap.
    Nothing about the model's global (all-species) behaviour is restricted: BirdNET considers its
    full label set on every clip and the target species' score reflects that real competition.
    """

    def __init__(
        self,
        species: Sequence[str],
        min_conf: float = 0.01,
        model_version: str = "birdnetlib-Analyzer",
    ) -> None:
        from birdnetlib.analyzer import Analyzer  # lazy: keeps the package import-safe in CI

        self.species: tuple[str, ...] = tuple(species)
        self.min_conf = float(min_conf)
        self.aggregation = "max-over-3s-windows"
        self._analyzer = Analyzer()
        self.model_id = f"{model_version}/{self._analyzer_version()}"

    def _analyzer_version(self) -> str:
        version = getattr(self._analyzer, "version", None)
        return str(version) if version else "v2.4"

    def predict(self, audio: np.ndarray, sr: int) -> Mapping[str, float]:
        """Return BirdNET's per-species score in [0, 1] for each species in :attr:`species`.

        The score is the maximum BirdNET confidence for that species across the clip's 3-second
        windows. Species not detected above :attr:`min_conf` in any window score 0.
        """
        from birdnetlib import RecordingBuffer

        buffer = _to_48k(audio, sr)
        recording = RecordingBuffer(self._analyzer, buffer, rate=BIRDNET_SR, min_conf=self.min_conf)
        recording.analyze()
        return aggregate_detections(recording.detections, self.species)
