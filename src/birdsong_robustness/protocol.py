"""The benchmark protocol: manifest schema, degradation grid, metrics, and the model interface.

STATUS: AWAITING MODEL INTEGRATION
==================================
Everything in this module is implemented and unit-tested against synthetic inputs, and **no
classifier has been run against it**. There are no model weights in this repository and none in
the environment it was developed in. Consequently there are no accuracy numbers here, in the
README, or anywhere else in the project, and any that appear in future must carry the model
version, the checkpoint hash, the dataset, and the date they were produced.

What is real today is the degradation pipeline and its calibration -- see
:mod:`birdsong_robustness.validate` and the Results section of the README. What is specified but
not yet exercised is the evaluation on the next page down: the manifest a real dataset would have
to supply, the grid a model would be swept over, the metrics that would be computed, and the
adapter a model would implement. Writing that down first is deliberate; a robustness protocol
decided *after* seeing which conditions a model happens to survive is not a protocol.

Why the grid is one-factor-at-a-time, plus a few named composites
----------------------------------------------------------------
A full factorial over six axes at five levels each is 15,625 conditions per recording. That is
computationally silly and, more importantly, uninterpretable: with that many cells the
interesting question ("which axis breaks this model?") is buried under interactions estimated
from one observation apiece.

So the grid is one-factor-at-a-time from a clean baseline. Each condition varies exactly one axis,
which makes the marginal sensitivity of the model to that axis directly readable as a degradation
curve. OFAT cannot see interactions, and interactions here are real -- clipping after a distant,
quiet recording behaves differently from clipping a close one. That is what
:data:`COMPOSITE_CONDITIONS` is for: a small, hand-specified set of plausible field scenarios,
each a named combination, reported separately and never averaged into the per-axis curves.

Why the operating threshold is frozen on clean data
---------------------------------------------------
Detection metrics depend on a score threshold. If the threshold is re-tuned per condition, the
benchmark reports each condition's *best case* and systematically understates degradation -- a
model whose scores all collapse by 0.3 but keep their ordering will look almost unharmed. A
deployed detector does not get to re-tune per weather condition either.

So: choose the threshold once, on the clean split, by maximising macro F0.5, and hold it fixed
across every degraded condition. :func:`select_threshold` does the first part and
:func:`evaluate` enforces the second. Threshold-free summaries (AUC, average precision) are
worth reporting alongside, because they separate "the model lost discriminative information"
from "the model's calibration drifted" -- two different problems with two different fixes.

Why F0.5
--------
F0.5 weights precision twice as heavily as recall. For most passive acoustic monitoring
deployments that is the right asymmetry: a false positive enters an occurrence database and
needs a human to remove it, whereas a missed call is usually one of many from the same
individual over a survey period. It is also the metric the author's earlier BirdNET work was
scored on, so the numbers would be comparable. It is not universally right -- rare-species
detection wants the opposite weighting -- which is why beta is a parameter everywhere rather
than a constant.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from . import degradations, noise, synth
from ._dsp import energy_active_mask

__all__ = [
    "BAND_LIMIT_HZ",
    "BIT_DEPTHS",
    "CLIP_FRACTIONS",
    "COMPOSITE_CONDITIONS",
    "DEFAULT_GRID",
    "DISTANCES_M",
    "GAIN_DB",
    "NOISE_TYPES",
    "OVERLAP_DENSITIES",
    "OVERLAP_SIR_DB",
    "PROTOCOL_STATUS",
    "RESAMPLE_HZ",
    "RT60_S",
    "SNR_LEVELS_DB",
    "BenchmarkResult",
    "BioacousticModel",
    "DegradationSpec",
    "Event",
    "Manifest",
    "RecordingEntry",
    "SpeciesMetrics",
    "apply_spec",
    "degradation_curve",
    "evaluate",
    "fbeta",
    "macro_average",
    "per_species_metrics",
    "select_threshold",
    "validate_manifest",
]

#: Single source of truth for the claim the README, the CLI banner and the docs all repeat.
PROTOCOL_STATUS = (
    "awaiting model integration -- the protocol is specified and tested, "
    "but no classifier has been evaluated and no accuracy numbers exist"
)


# --------------------------------------------------------------------------------------
# Dataset manifest schema
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """One annotated vocalisation.

    Strong (time-localised) labels rather than clip-level tags, because half the axes in this
    benchmark are only well defined with them. "SNR during the call" needs to know when the call
    is; overlap density needs to know what fraction of the call is contested. A clip-level tag
    reduces both to a guess, and :func:`birdsong_robustness._dsp.energy_active_mask` is the
    fallback, not the design.

    Attributes:
        start_s: Onset in seconds from the start of the recording.
        end_s: Offset in seconds. Must be greater than ``start_s``.
        species: Species code. Must appear in the manifest's species list.
        quality: Annotator confidence in [0, 1]. Used to exclude uncertain events from the
            active-region mask without discarding them from the label set.
    """

    start_s: float
    end_s: float
    species: str
    quality: float = 1.0


@dataclass(frozen=True)
class RecordingEntry:
    """One recording in the manifest.

    ``recorder`` and ``source`` are carried because they are confounders, not metadata trivia.
    A benchmark whose "clean" split is mostly close-range directional-microphone focal recordings
    and whose species distribution correlates with recorder type will attribute to *degradation*
    what is really an effect of *which species were recorded on which device*. Keeping the fields
    means the analysis can at least check.

    Attributes:
        recording_id: Unique key.
        path: Location of the audio, relative to the manifest. This repository ships no audio.
        sr: Native sample rate in Hz. Kept because resampling to a model's rate is itself a
            degradation, and it must be applied identically in every condition.
        duration_s: Duration in seconds.
        species: Every species present. Multi-label; soundscapes routinely contain several.
        events: Strong labels. May be empty for clip-level-only recordings, which then cannot
            take part in the SNR or overlap axes.
        split: ``"dev"`` (threshold selection) or ``"test"`` (reporting). Never both.
        source: Provenance, e.g. ``"xeno-canto"`` or ``"birdclef-soundscape"``.
        license_id: Per-recording licence. Varies recording by recording on Xeno-canto, which is
            why this repository redistributes none of it.
        recorder: Device class, e.g. ``"audiomoth"``, ``"shotgun"``, ``"unknown"``.
        latitude: Decimal degrees, or None.
        longitude: Decimal degrees, or None.
        recorded_at: ISO 8601 date or datetime string, or None.
    """

    recording_id: str
    path: str
    sr: int
    duration_s: float
    species: tuple[str, ...]
    events: tuple[Event, ...] = ()
    split: str = "test"
    source: str = "unknown"
    license_id: str = "unknown"
    recorder: str = "unknown"
    latitude: float | None = None
    longitude: float | None = None
    recorded_at: str | None = None

    def active_mask(self, n_samples: int, min_quality: float = 0.5) -> np.ndarray:
        """Build a sample-level activity mask from the strong labels.

        Returns all-False when there are no events, which callers must treat as "this recording
        cannot be used on the active-basis axes" rather than silently falling back to an energy
        segmenter. The fallback exists (:func:`birdsong_robustness._dsp.energy_active_mask`) but
        choosing it has to be an explicit decision, because its error propagates straight into
        every SNR figure computed from it.
        """
        mask = np.zeros(n_samples, dtype=bool)
        for event in self.events:
            if event.quality < min_quality:
                continue
            start = max(0, round(event.start_s * self.sr))
            end = min(n_samples, round(event.end_s * self.sr))
            if end > start:
                mask[start:end] = True
        return mask


@dataclass(frozen=True)
class Manifest:
    """A dataset description: the species vocabulary and the recordings.

    The vocabulary is explicit rather than inferred from the events, so that a species with zero
    positives in the test split still appears in the per-species table. Silently dropping it
    would inflate every macro average by removing exactly the hardest classes.
    """

    dataset_id: str
    species: tuple[str, ...]
    entries: tuple[RecordingEntry, ...]

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON. Tuples become lists; this round-trips through :meth:`from_dict`."""
        return json.dumps(asdict(self), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        """Rebuild from a parsed JSON object, restoring the nested dataclasses."""
        entries = []
        for raw in data["entries"]:
            raw = dict(raw)
            raw["species"] = tuple(raw.get("species", ()))
            raw["events"] = tuple(Event(**e) for e in raw.get("events", ()))
            entries.append(RecordingEntry(**raw))
        return cls(
            dataset_id=str(data["dataset_id"]),
            species=tuple(data["species"]),
            entries=tuple(entries),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> Manifest:
        """Load from a JSON file on disk."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_manifest(manifest: Manifest) -> list[str]:
    """Check a manifest for the errors that quietly corrupt a benchmark.

    Returns a list of problems rather than raising on the first one, so that a dataset can be
    fixed in a single pass. An empty list means the manifest is internally consistent -- it says
    nothing about whether the audio exists or the labels are correct.

    The checks, and why each is here:

    * duplicate ``recording_id`` -- a duplicate silently doubles a recording's weight in a macro
      average, and the duplicate is usually the same clip under two provenance records.
    * unknown species in ``species`` or in an event -- means the vocabulary and the labels were
      built from different snapshots, and every per-species number will be off by the difference.
    * an event outside ``[0, duration_s]`` or with ``end_s <= start_s`` -- produces an empty or
      out-of-range mask, which turns an active-basis SNR into a division by a power of zero.
    * a species in an event that is not in the entry's ``species`` set -- the clip-level and
      strong labels disagree, and whichever one the harness happens to read wins.
    * ``split`` outside {dev, test} -- a recording that belongs to neither is silently dropped;
      one that belongs to both leaks the threshold selection into the reported numbers.
    * empty ``dev`` split -- the operating threshold has to come from somewhere other than the
      test set.
    """
    problems: list[str] = []
    vocabulary = set(manifest.species)
    if not manifest.species:
        problems.append("manifest declares no species vocabulary")
    if len(vocabulary) != len(manifest.species):
        problems.append("species vocabulary contains duplicates")

    seen: set[str] = set()
    n_dev = 0
    for entry in manifest.entries:
        tag = f"entry {entry.recording_id!r}"
        if entry.recording_id in seen:
            problems.append(f"{tag}: duplicate recording_id")
        seen.add(entry.recording_id)

        if entry.split not in {"dev", "test"}:
            problems.append(f"{tag}: split {entry.split!r} is not 'dev' or 'test'")
        n_dev += entry.split == "dev"

        if entry.sr <= 0:
            problems.append(f"{tag}: sample rate {entry.sr} is not positive")
        if entry.duration_s <= 0:
            problems.append(f"{tag}: duration {entry.duration_s} is not positive")

        for species in entry.species:
            if species not in vocabulary:
                problems.append(f"{tag}: species {species!r} is not in the manifest vocabulary")

        for i, event in enumerate(entry.events):
            where = f"{tag} event {i}"
            if event.end_s <= event.start_s:
                problems.append(f"{where}: end_s {event.end_s} <= start_s {event.start_s}")
            if event.start_s < 0 or event.end_s > entry.duration_s:
                problems.append(
                    f"{where}: [{event.start_s}, {event.end_s}] falls outside "
                    f"[0, {entry.duration_s}]"
                )
            if event.species not in vocabulary:
                problems.append(f"{where}: species {event.species!r} is not in the vocabulary")
            elif event.species not in entry.species:
                problems.append(
                    f"{where}: species {event.species!r} is not in the entry's clip-level labels"
                )
            if not 0.0 <= event.quality <= 1.0:
                problems.append(f"{where}: quality {event.quality} is outside [0, 1]")

    if manifest.entries and n_dev == 0:
        problems.append("no recordings in the 'dev' split; the threshold cannot be selected")
    return problems


# --------------------------------------------------------------------------------------
# The degradation grid
# --------------------------------------------------------------------------------------

#: Additive-noise SNRs, in dB, on the active basis. The range brackets what a passive recorder
#: sees between a close call on a still morning (>20 dB) and a distant call in rain (<-10 dB).
SNR_LEVELS_DB: tuple[float, ...] = (20.0, 10.0, 5.0, 0.0, -5.0, -10.0)

#: Noise types swept at fixed SNR. See :mod:`birdsong_robustness.noise` for why the colour is not
#: a cosmetic choice: the same nominal SNR is a very different amount of in-band masking.
NOISE_TYPES: tuple[str, ...] = ("pink", "brown", "white", "wind", "rain")

#: Fraction of the target's active time carrying audible competing sound.
OVERLAP_DENSITIES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Target-to-interferer ratio over the overlap region, in dB.
OVERLAP_SIR_DB: tuple[float, ...] = (10.0, 0.0, -5.0)

#: Source-to-receiver distances in metres. 1 m is the reference (no-op); 200 m is beyond the
#: detection range of most passerines on most recorders, and is included as the floor case.
DISTANCES_M: tuple[float, ...] = (1.0, 10.0, 25.0, 50.0, 100.0, 200.0)

#: Reverberation times in seconds. Short, because outdoor "reverberation" is vegetation
#: scattering and ground reflection, not room modes.
RT60_S: tuple[float, ...] = (0.1, 0.2, 0.4, 0.8)

#: Low-pass cutoffs in Hz, modelling a microphone or codec band edge. 2 kHz is below the
#: fundamental of most passerine song and is the destructive end of the axis.
BAND_LIMIT_HZ: tuple[float, ...] = (2000.0, 4000.0, 8000.0, 12000.0)

#: Intermediate sample rates for the resample round trip. Each caps the signal at its Nyquist.
RESAMPLE_HZ: tuple[int, ...] = (8000, 16000, 22050, 24000)

#: PCM bit depths. 16 is a competent recorder; 8 and below are the cheap-hardware regime where
#: quiet distant calls fall into the quantiser's dead zone.
BIT_DEPTHS: tuple[int, ...] = (16, 12, 10, 8, 6, 4)

#: Fraction of samples driven into the clipping limit.
CLIP_FRACTIONS: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.05, 0.10)

#: Input-gain errors in dB. Negative is an under-driven recorder losing effective bit depth;
#: positive is an over-driven one clipping. Both are common and both are two different failures.
GAIN_DB: tuple[float, ...] = (-40.0, -20.0, -6.0, 0.0, 6.0, 20.0)


@dataclass(frozen=True)
class DegradationSpec:
    """One point in the grid: a complete, reproducible description of a condition.

    Every field defaults to the identity, so a spec is written by naming only what it changes.
    Frozen and hashable, so a spec can key a results dictionary and a run can be resumed without
    re-deriving which conditions were already done.
    """

    label: str = "clean"
    axis: str = "clean"
    noise_type: str | None = None
    snr_db: float | None = None
    overlap_density: float | None = None
    overlap_sir_db: float = 0.0
    distance_m: float | None = None
    rt60_s: float = 0.4
    include_reverb: bool = True
    band_limit_hz: float | None = None
    resample_hz: int | None = None
    bit_depth: int | None = None
    codec: str | None = None
    clip_fraction: float | None = None
    gain_db: float = 0.0

    def with_seed_label(self, suffix: str) -> DegradationSpec:
        """Return a copy with ``suffix`` appended to the label, for composite naming."""
        return replace(self, label=f"{self.label}+{suffix}")


def apply_spec(
    audio: np.ndarray,
    sr: int,
    spec: DegradationSpec,
    active: np.ndarray | None = None,
    interferers: Sequence[np.ndarray] | None = None,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Apply a :class:`DegradationSpec` to a waveform in the canonical order.

    The order is the physical signal chain, and it is not interchangeable:

    1. **Overlap.** Competing birds are other sources in the same scene, so their sound is
       present before the target's own propagation is modelled. (A fuller model would propagate
       each interferer from its own distance; this one does not, and the README says so.)
    2. **Distance** -- spreading, air absorption, reverberation.
    3. **Additive background noise at the requested SNR.** After propagation, not before. The
       SNR that matters is the one at the microphone; adding noise first and then attenuating
       the sum would leave the requested SNR unchanged in name but would attenuate the noise
       along with the signal, which is not what a distant recording sounds like.
    4. **Band limiting** -- the microphone and preamp response.
    5. **Gain, then clipping** -- the analogue input stage, where too much gain clips.
    6. **Resampling, then quantisation or codec** -- the converter and the storage format, last,
       because everything upstream of it is analogue.

    Steps 5 and 6 in particular must stay in this order: quantising and then clipping measures
    something that no recorder does.

    Args:
        audio: Clean input waveform.
        sr: Sample rate in Hz.
        spec: The condition to apply.
        active: Ground truth activity mask for the target. Estimated if omitted, with the
            accuracy cost documented in :func:`birdsong_robustness._dsp.energy_active_mask`.
        interferers: Competing recordings, required when ``spec.overlap_density`` is set.
        rng: Seed or Generator, so a condition is reproducible.

    Returns:
        The degraded waveform, the same length as ``audio``.
    """
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    y = np.asarray(audio, dtype=np.float64)
    mask = energy_active_mask(y, sr) if active is None else np.asarray(active, dtype=bool)

    if spec.overlap_density is not None and spec.overlap_density > 0:
        if not interferers:
            raise ValueError("overlap_density requires interferers")
        result = degradations.add_overlapping_calls(
            y,
            list(interferers),
            sr,
            sir_db=spec.overlap_sir_db,
            density=spec.overlap_density,
            target_active=mask,
            rng=generator,
        )
        y = result.audio

    if spec.distance_m is not None and spec.distance_m > 1.0:
        y = degradations.simulate_distance(
            y,
            sr,
            distance_m=spec.distance_m,
            rt60_s=spec.rt60_s,
            include_reverb=spec.include_reverb,
            rng=generator,
        )

    if spec.snr_db is not None:
        noise_name = spec.noise_type or "pink"
        masker = noise.make_noise(noise_name, len(y), sr, rng=generator)
        y, _ = degradations.add_noise_at_snr(y, masker, spec.snr_db, active_mask=mask)

    if spec.band_limit_hz is not None:
        y = degradations.band_limit(y, sr, high_hz=spec.band_limit_hz)

    if spec.gain_db:
        y = degradations.apply_gain_db(y, spec.gain_db)

    if spec.clip_fraction is not None and spec.clip_fraction > 0:
        y, _ = degradations.clip_to_fraction(y, spec.clip_fraction)

    if spec.resample_hz is not None:
        y = degradations.resample_roundtrip(y, sr, spec.resample_hz)

    if spec.codec == "mu-law":
        y = degradations.mu_law_codec(y, bits=spec.bit_depth or 8)
    elif spec.bit_depth is not None:
        y = degradations.quantize_pcm(y, spec.bit_depth)

    return y


def _build_default_grid() -> tuple[DegradationSpec, ...]:
    """Assemble the one-factor-at-a-time grid from the level constants above."""
    grid: list[DegradationSpec] = [DegradationSpec()]

    for snr in SNR_LEVELS_DB:
        grid.append(
            DegradationSpec(
                label=f"snr_pink_{snr:+.0f}dB", axis="snr", noise_type="pink", snr_db=snr
            )
        )
    for name in NOISE_TYPES:
        grid.append(
            DegradationSpec(
                label=f"noise_{name}_0dB", axis="noise_type", noise_type=name, snr_db=0.0
            )
        )
    for density in OVERLAP_DENSITIES[1:]:
        for sir in OVERLAP_SIR_DB:
            grid.append(
                DegradationSpec(
                    label=f"overlap_d{density:.2f}_sir{sir:+.0f}dB",
                    axis="overlap",
                    overlap_density=density,
                    overlap_sir_db=sir,
                )
            )
    for distance in DISTANCES_M[1:]:
        grid.append(
            DegradationSpec(label=f"distance_{distance:.0f}m", axis="distance", distance_m=distance)
        )
    for rt60 in RT60_S:
        grid.append(
            DegradationSpec(
                label=f"reverb_rt60_{rt60:.1f}s",
                axis="reverb",
                distance_m=10.0,
                rt60_s=rt60,
            )
        )
    for cutoff in BAND_LIMIT_HZ:
        grid.append(
            DegradationSpec(
                label=f"bandlimit_{cutoff:.0f}Hz", axis="bandwidth", band_limit_hz=cutoff
            )
        )
    for rate in RESAMPLE_HZ:
        grid.append(DegradationSpec(label=f"resample_{rate}Hz", axis="bandwidth", resample_hz=rate))
    for bits in BIT_DEPTHS:
        grid.append(DegradationSpec(label=f"pcm_{bits}bit", axis="quantisation", bit_depth=bits))
    grid.append(
        DegradationSpec(label="mulaw_8bit", axis="quantisation", codec="mu-law", bit_depth=8)
    )
    for fraction in CLIP_FRACTIONS[1:]:
        grid.append(
            DegradationSpec(label=f"clip_{fraction:.3f}", axis="clipping", clip_fraction=fraction)
        )
    for gain in GAIN_DB:
        if gain == 0.0:
            continue
        grid.append(DegradationSpec(label=f"gain_{gain:+.0f}dB", axis="gain", gain_db=gain))
    return tuple(grid)


#: The one-factor-at-a-time grid. Each entry varies exactly one axis from clean.
DEFAULT_GRID: tuple[DegradationSpec, ...] = _build_default_grid()


#: Named multi-axis scenarios, reported separately and never folded into the per-axis curves.
#: Each is a plausible field situation rather than a corner of a factorial design.
COMPOSITE_CONDITIONS: tuple[DegradationSpec, ...] = (
    DegradationSpec(
        label="distant_bird_light_rain",
        axis="composite",
        distance_m=50.0,
        noise_type="rain",
        snr_db=0.0,
    ),
    DegradationSpec(
        label="cheap_recorder_windy_morning",
        axis="composite",
        noise_type="wind",
        snr_db=5.0,
        band_limit_hz=8000.0,
        resample_hz=16000,
        bit_depth=12,
        gain_db=-12.0,
    ),
    DegradationSpec(
        label="dawn_chorus_close",
        axis="composite",
        overlap_density=0.75,
        overlap_sir_db=0.0,
        noise_type="pink",
        snr_db=10.0,
    ),
    DegradationSpec(
        label="overdriven_roadside",
        axis="composite",
        noise_type="brown",
        snr_db=0.0,
        gain_db=12.0,
        clip_fraction=0.01,
        bit_depth=16,
    ),
    DegradationSpec(
        label="archive_telephony_grade",
        axis="composite",
        distance_m=25.0,
        resample_hz=8000,
        codec="mu-law",
        bit_depth=8,
    ),
)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    """F-beta from precision and recall. Zero when both are zero.

    ``beta < 1`` weights precision more heavily. The identity ``F1 = 2PR / (P + R)`` is the
    ``beta = 1`` case of the expression used here.
    """
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    b2 = beta * beta
    denominator = b2 * precision + recall
    if denominator <= 0.0:
        return 0.0
    return float((1.0 + b2) * precision * recall / denominator)


@dataclass(frozen=True)
class SpeciesMetrics:
    """Detection metrics for one species (or a macro average over species).

    ``support`` is the number of true positives available. It is carried through every
    aggregation because a per-species F0.5 computed from three positives is not evidence, and a
    macro average that does not report the support distribution hides that.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    support: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f_beta: float = 0.0
    beta: float = 0.5

    @classmethod
    def from_counts(cls, tp: int, fp: int, fn: int, beta: float = 0.5) -> SpeciesMetrics:
        """Build from raw counts.

        A species with no predictions and no positives gets precision and recall of 0 rather
        than being reported as perfect. Scoring an untested class as 1.0 is the single easiest
        way to manufacture a good-looking macro average.
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return cls(
            tp=int(tp),
            fp=int(fp),
            fn=int(fn),
            support=int(tp + fn),
            precision=float(precision),
            recall=float(recall),
            f_beta=fbeta(precision, recall, beta),
            beta=float(beta),
        )


def per_species_metrics(
    y_true: Sequence[Iterable[str]],
    y_score: Sequence[Mapping[str, float]],
    species: Sequence[str],
    threshold: float,
    beta: float = 0.5,
) -> dict[str, SpeciesMetrics]:
    """Per-species detection metrics at a fixed score threshold.

    Multi-label throughout: each clip carries a set of present species and a score per species,
    and each species is scored as its own independent binary detection problem. Treating a
    soundscape as single-label -- taking the argmax -- would make every clip containing two birds
    an automatic error and would measure the dataset's polyphony, not the model.

    Args:
        y_true: Per-clip iterables of species actually present.
        y_score: Per-clip mappings from species to score. Missing species score 0.
        species: The full vocabulary, including species with no positives in this split.
        threshold: Scores strictly above this count as detections.
        beta: F-beta weighting.

    Returns:
        Mapping from species to :class:`SpeciesMetrics`.
    """
    if len(y_true) != len(y_score):
        raise ValueError(f"y_true has {len(y_true)} rows but y_score has {len(y_score)}")

    truth = [set(row) for row in y_true]
    out: dict[str, SpeciesMetrics] = {}
    for name in species:
        tp = fp = fn = 0
        for present, scores in zip(truth, y_score):
            detected = float(scores.get(name, 0.0)) > threshold
            is_present = name in present
            if detected and is_present:
                tp += 1
            elif detected:
                fp += 1
            elif is_present:
                fn += 1
        out[name] = SpeciesMetrics.from_counts(tp, fp, fn, beta)
    return out


def macro_average(metrics: Mapping[str, SpeciesMetrics], min_support: int = 1) -> SpeciesMetrics:
    """Unweighted mean over species with at least ``min_support`` positives.

    Macro rather than micro, because a micro average over a long-tailed species distribution is
    dominated by the handful of common species and says almost nothing about the 200 rare ones --
    which are usually the reason anyone is running the classifier.

    ``min_support`` excludes species that never occur in the split. Their metrics are still
    reported individually; they are simply not averaged, because an F0.5 of 0 from zero
    opportunities is not a measurement.
    """
    eligible = [m for m in metrics.values() if m.support >= min_support]
    if not eligible:
        return SpeciesMetrics()
    n = len(eligible)
    beta = eligible[0].beta
    return SpeciesMetrics(
        tp=sum(m.tp for m in eligible),
        fp=sum(m.fp for m in eligible),
        fn=sum(m.fn for m in eligible),
        support=sum(m.support for m in eligible),
        precision=float(sum(m.precision for m in eligible) / n),
        recall=float(sum(m.recall for m in eligible) / n),
        f_beta=float(sum(m.f_beta for m in eligible) / n),
        beta=beta,
    )


def select_threshold(
    y_true: Sequence[Iterable[str]],
    y_score: Sequence[Mapping[str, float]],
    species: Sequence[str],
    beta: float = 0.5,
    candidates: Sequence[float] | None = None,
) -> float:
    """Pick the operating threshold that maximises macro F-beta on clean development data.

    Selected once, on the dev split, and then frozen -- see the module docstring. Ties are broken
    towards the lower threshold, which is the conservative choice: it makes the reported
    degradation larger rather than smaller.
    """
    if candidates is None:
        candidates = tuple(float(t) for t in np.linspace(0.01, 0.99, 99))
    best_threshold = float(candidates[0])
    best_score = -1.0
    for threshold in candidates:
        score = macro_average(per_species_metrics(y_true, y_score, species, threshold, beta)).f_beta
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def degradation_curve(
    results: Mapping[str, SpeciesMetrics],
    grid: Sequence[DegradationSpec],
    axis: str,
    level_of: Callable[[DegradationSpec], float],
) -> list[tuple[float, float]]:
    """Extract ``(level, macro F-beta)`` points for one axis, sorted by level.

    The curve, not the single worst-case number, is the deliverable. "F0.5 falls from x to y at
    0 dB SNR" is one point; the shape tells you whether the model degrades gracefully or has a
    cliff, and a cliff is an operational fact -- it means a deployment either sits above the knee
    or produces nothing usable.
    """
    points = [
        (float(level_of(spec)), float(results[spec.label].f_beta))
        for spec in grid
        if spec.axis == axis and spec.label in results
    ]
    return sorted(points)


# --------------------------------------------------------------------------------------
# The model interface
# --------------------------------------------------------------------------------------


@runtime_checkable
class BioacousticModel(Protocol):
    """What a classifier must provide to be benchmarked. Nothing implements this yet.

    Deliberately minimal: a species vocabulary and a function from waveform to per-species
    scores. Everything model-specific belongs inside the adapter, and three things in particular:

    * **Resampling.** BirdNET runs at 48 kHz internally; other models want 32 kHz or 16 kHz. The
      adapter owns that conversion, and it must use the *same* resampler in every condition.
      Otherwise the benchmark's bandwidth axis is partly measuring the adapter.
    * **Windowing and aggregation.** BirdNET scores 3-second windows. Turning a window sequence
      into one score per clip is a choice -- max, mean, top-k mean -- and it changes the
      precision/recall trade-off on its own. It has to be fixed before any condition is run and
      recorded alongside the results.
    * **Any built-in normalisation.** Per-clip gain normalisation will partly undo the gain axis
      and interact with the clipping axis. That is legitimate model behaviour and should be
      reported, not disabled -- but it has to be known about, or the gain axis looks flat for
      the wrong reason.

    Scores are expected in [0, 1] and comparable across species; if a model's outputs are not,
    the adapter must say so, because :func:`select_threshold` picks a single global threshold.
    """

    species: Sequence[str]

    def predict(self, audio: np.ndarray, sr: int) -> Mapping[str, float]:
        """Return a score in [0, 1] for each species in :attr:`species`."""
        ...


@dataclass(frozen=True)
class BenchmarkResult:
    """Everything one benchmark run produced.

    ``model_id`` and ``threshold`` are stored with the numbers because a robustness result
    without the checkpoint it came from and the operating point it was scored at is not
    reproducible and should not be quoted.
    """

    model_id: str
    dataset_id: str
    threshold: float
    beta: float
    per_condition: dict[str, dict[str, SpeciesMetrics]] = field(default_factory=dict)
    macro: dict[str, SpeciesMetrics] = field(default_factory=dict)

    def curve(
        self,
        axis: str,
        level_of: Callable[[DegradationSpec], float],
        grid: Sequence[DegradationSpec] = DEFAULT_GRID,
    ) -> list[tuple[float, float]]:
        """Degradation curve for one axis, from the macro metrics of this run."""
        return degradation_curve(self.macro, grid, axis, level_of)

    def to_json(self, indent: int = 2) -> str:
        """Serialise, including the status banner so results can never be quoted context-free."""
        payload = {
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "threshold": self.threshold,
            "beta": self.beta,
            "macro": {k: asdict(v) for k, v in self.macro.items()},
            "per_condition": {
                condition: {species: asdict(m) for species, m in species_metrics.items()}
                for condition, species_metrics in self.per_condition.items()
            },
        }
        return json.dumps(payload, indent=indent, sort_keys=True)


def evaluate(
    model: BioacousticModel,
    manifest: Manifest,
    load_audio: Callable[[RecordingEntry], np.ndarray],
    grid: Sequence[DegradationSpec] = DEFAULT_GRID,
    beta: float = 0.5,
    threshold: float | None = None,
    interferer_pool: Sequence[np.ndarray] | None = None,
    model_id: str = "unspecified",
    seed: int = 0,
) -> BenchmarkResult:
    """Run a model across the grid and return per-condition metrics. Never yet executed for real.

    The control flow is the protocol:

    1. Score the clean dev split and, unless one is supplied, choose the operating threshold on
       it by maximising macro F-beta.
    2. For each condition in the grid, degrade every test recording with the *same* seed offset
       so that condition-to-condition differences are the condition and not the noise draw.
    3. Score, accumulate per-species counts, macro-average.

    Step 2's seeding deserves the emphasis. If every condition draws fresh randomness, the
    difference between two adjacent SNR levels contains a noise-realisation term that can be
    larger than the effect. Deriving each condition's generator from ``seed`` and the recording
    id makes the comparison paired.

    Args:
        model: Anything satisfying :class:`BioacousticModel`.
        manifest: The dataset description.
        load_audio: Callable turning an entry into a waveform at ``entry.sr``. Supplied by the
            caller because this repository ships no audio and no loader.
        grid: Conditions to sweep.
        beta: F-beta weighting.
        threshold: Operating point. Selected on the dev split when None.
        interferer_pool: Waveforms used for the overlap axis.
        model_id: Recorded verbatim in the result. Should identify a checkpoint, not a family.
        seed: Base seed for the paired condition draws.

    Returns:
        A :class:`BenchmarkResult`.
    """
    dev = [e for e in manifest.entries if e.split == "dev"]
    test = [e for e in manifest.entries if e.split == "test"]
    if not test:
        raise ValueError("manifest has no test recordings")

    if threshold is None:
        if not dev:
            raise ValueError("no dev split and no threshold supplied")
        dev_true = [list(e.species) for e in dev]
        dev_scores = [dict(model.predict(load_audio(e), e.sr)) for e in dev]
        threshold = select_threshold(dev_true, dev_scores, manifest.species, beta)

    per_condition: dict[str, dict[str, SpeciesMetrics]] = {}
    macro: dict[str, SpeciesMetrics] = {}

    for condition_index, spec in enumerate(grid):
        y_true: list[list[str]] = []
        y_score: list[dict[str, float]] = []
        for entry_index, entry in enumerate(test):
            audio = load_audio(entry)
            mask = entry.active_mask(len(audio))
            rng = np.random.default_rng([seed, condition_index, entry_index])
            degraded = apply_spec(
                audio,
                entry.sr,
                spec,
                active=mask if mask.any() else None,
                interferers=interferer_pool,
                rng=rng,
            )
            y_true.append(list(entry.species))
            y_score.append(dict(model.predict(degraded, entry.sr)))

        metrics = per_species_metrics(y_true, y_score, manifest.species, threshold, beta)
        per_condition[spec.label] = metrics
        macro[spec.label] = macro_average(metrics)

    return BenchmarkResult(
        model_id=model_id,
        dataset_id=manifest.dataset_id,
        threshold=float(threshold),
        beta=float(beta),
        per_condition=per_condition,
        macro=macro,
    )


def synthetic_manifest(
    n_recordings: int = 6,
    species: Sequence[str] = ("synthA", "synthB", "synthC"),
    duration_s: float = 5.0,
    sr: int = 32000,
    seed: int = 0,
) -> tuple[Manifest, dict[str, np.ndarray]]:
    """Build a manifest and matching audio from :mod:`birdsong_robustness.synth`.

    Exists so that the protocol code paths -- manifest validation, grid application, metric
    accumulation -- can be executed end to end in CI without any audio in the repository and
    without a model. The waveforms are DSP test signals, not birds, and any metric computed over
    them describes the plumbing, not a classifier's ability to identify anything.

    Returns:
        ``(manifest, audio_by_recording_id)``.
    """
    generator = np.random.default_rng(seed)
    entries: list[RecordingEntry] = []
    audio: dict[str, np.ndarray] = {}

    for i in range(n_recordings):
        name = species[i % len(species)]
        clip = synth.bird_call(duration_s=duration_s, sr=sr, rng=generator, n_syllables=6)
        recording_id = f"synth-{i:03d}"
        audio[recording_id] = clip.audio
        entries.append(
            RecordingEntry(
                recording_id=recording_id,
                path=f"audio/{recording_id}.wav",
                sr=sr,
                duration_s=duration_s,
                species=(name,),
                events=tuple(
                    Event(start_s=s / sr, end_s=e / sr, species=name) for s, e in clip.events
                ),
                split="dev" if i < max(1, n_recordings // 3) else "test",
                source="synthetic",
                license_id="MIT",
                recorder="synthetic",
            )
        )

    return Manifest(
        dataset_id="synthetic-smoke", species=tuple(species), entries=tuple(entries)
    ), audio
