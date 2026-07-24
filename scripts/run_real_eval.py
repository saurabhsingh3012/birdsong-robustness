"""Run BirdNET across the degradation grid on the real Xeno-canto set, for real.

This is the step ``ROADMAP.md`` called Milestone 2 and the README called "when a model has
actually been run". It loads the manifest built by ``scripts/build_dataset.py``, wraps BirdNET
behind :class:`~birdsong_robustness.protocol.BioacousticModel`, selects the operating threshold
once on the clean dev split, sweeps every degradation condition on the test split, and writes the
measured degradation curves to ``docs/real_eval_results.json`` (committed).

The sweep is **resumable**: every condition's metrics are written to a checkpoint
(``data/eval_checkpoint.json``, git-ignored) as soon as it finishes, so a run that is interrupted
picks up exactly where it stopped rather than starting over. The sweep logic mirrors
:func:`birdsong_robustness.protocol.evaluate` exactly -- same frozen threshold, same paired
per-condition seeding ``[seed, condition_index, entry_index]`` -- but is inlined here so each
condition can be checkpointed independently.

NETWORK-FREE but MODEL-REQUIRED (BirdNET weights, downloaded once by ``birdnetlib``). Nothing
here runs in CI. Every number it writes is produced by BirdNET on real audio -- none is invented.

Usage::

    pip install -e ".[birdnet]"
    python scripts/build_dataset.py     # once, needs XENO_CANTO_KEY
    python scripts/run_real_eval.py     # resumable; re-run to continue an interrupted sweep
"""

from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from birdsong_robustness.birdnet_adapter import BirdNETModel
from birdsong_robustness.protocol import (
    COMPOSITE_CONDITIONS,
    DEFAULT_GRID,
    DegradationSpec,
    Manifest,
    apply_spec,
    macro_average,
    per_species_metrics,
    select_threshold,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHECKPOINT = DATA / "eval_checkpoint.json"
RESULTS = ROOT / "docs" / "real_eval_results.json"
BETA = 0.5
SEED = 0

# One representative clean dev clip per species is used as the overlap-axis interferer pool, so
# competing calls in the overlap conditions are *real birds*, not synthetic probes.
N_INTERFERERS_PER_SPECIES = 1


def _round(x: float, n: int = 4) -> float:
    return round(float(x), n)


def _metrics_dict(m) -> dict:
    return {
        "f_beta": _round(m.f_beta),
        "precision": _round(m.precision),
        "recall": _round(m.recall),
        "tp": m.tp,
        "fp": m.fp,
        "fn": m.fn,
        "support": m.support,
    }


def _level_of(spec: DegradationSpec) -> float | None:
    """The single varied quantity for a per-axis curve, or None for categorical axes."""
    return {
        "snr": spec.snr_db,
        "distance": spec.distance_m,
        "reverb": spec.rt60_s,
        "clipping": spec.clip_fraction,
        "gain": spec.gain_db,
        "quantisation": float(spec.bit_depth) if spec.bit_depth is not None else None,
    }.get(spec.axis)


def _load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {"threshold": None, "conditions": {}}


def _save_checkpoint(state: dict) -> None:
    CHECKPOINT.write_text(json.dumps(state), encoding="utf-8")


def main() -> int:
    manifest = Manifest.from_json_file(DATA / "manifest.json")
    problems = validate_manifest(manifest)
    if problems:
        print("MANIFEST PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 2

    # Pre-load every clip once (float32 to keep the resident set small), keyed by recording id.
    cache: dict[str, np.ndarray] = {}
    for entry in manifest.entries:
        audio, _ = sf.read(str(DATA / entry.path), dtype="float32")
        cache[entry.recording_id] = np.ascontiguousarray(audio)

    def load_audio(entry):
        return cache[entry.recording_id]

    dev = [e for e in manifest.entries if e.split == "dev"]
    test = [e for e in manifest.entries if e.split == "test"]
    species = manifest.species
    print(f"dataset: {len(species)} species, {len(dev)} dev / {len(test)} test clips", flush=True)

    pool: list[np.ndarray] = []
    seen: set[str] = set()
    for entry in dev:
        sp = entry.species[0]
        if sp not in seen:
            pool.append(cache[entry.recording_id])
            seen.add(sp)
        if len(seen) >= len(species) * N_INTERFERERS_PER_SPECIES:
            break

    print("loading BirdNET...", flush=True)
    model = BirdNETModel(species=species)
    print(f"model: {model.model_id}, aggregation={model.aggregation}", flush=True)

    grid = tuple(DEFAULT_GRID) + tuple(COMPOSITE_CONDITIONS)
    state = _load_checkpoint()

    # Frozen operating threshold: selected once on the clean dev split, then held fixed.
    if state["threshold"] is None:
        print("selecting threshold on clean dev split...", flush=True)
        dev_true = [list(e.species) for e in dev]
        dev_scores = [dict(model.predict(load_audio(e), e.sr)) for e in dev]
        state["threshold"] = float(select_threshold(dev_true, dev_scores, species, BETA))
        _save_checkpoint(state)
    threshold = state["threshold"]
    print(f"threshold = {threshold:.3f}", flush=True)

    done = state["conditions"]
    remaining = [(i, s) for i, s in enumerate(grid) if s.label not in done]
    print(
        f"{len(done)} conditions already done; {len(remaining)} remaining "
        f"of {len(grid)} on {len(test)} test clips",
        flush=True,
    )

    for condition_index, spec in enumerate(grid):
        if spec.label in done:
            continue
        y_true: list[list[str]] = []
        y_score: list[dict[str, float]] = []
        for entry_index, entry in enumerate(test):
            audio = load_audio(entry)
            mask = entry.active_mask(len(audio))
            rng = np.random.default_rng([SEED, condition_index, entry_index])
            degraded = apply_spec(
                audio,
                entry.sr,
                spec,
                active=mask if mask.any() else None,
                interferers=pool,
                rng=rng,
            )
            y_true.append(list(entry.species))
            y_score.append(dict(model.predict(degraded, entry.sr)))

        metrics = per_species_metrics(y_true, y_score, species, threshold, BETA)
        macro = macro_average(metrics)
        done[spec.label] = {
            "macro": _metrics_dict(macro),
            "per_species": {sp: _metrics_dict(m) for sp, m in metrics.items()},
        }
        _save_checkpoint(state)
        print(
            f"  [{len(done)}/{len(grid)}] {spec.label:<26} "
            f"F{BETA}={macro.f_beta:.3f} P={macro.precision:.3f} R={macro.recall:.3f}",
            flush=True,
        )

    _write_results(manifest, grid, dev, test, threshold, done)
    return 0


def _write_results(manifest, grid, dev, test, threshold, done: dict) -> None:
    species = manifest.species
    clean = done["clean"]["macro"]

    axes: dict[str, list[dict]] = defaultdict(list)
    for spec in grid:
        cond = done[spec.label]
        row = {"label": spec.label, **cond["macro"]}
        level = _level_of(spec)
        if level is not None:
            row["level"] = _round(level)
        params = {}
        for f in (
            "noise_type",
            "snr_db",
            "overlap_density",
            "overlap_sir_db",
            "distance_m",
            "rt60_s",
            "band_limit_hz",
            "resample_hz",
            "bit_depth",
            "codec",
            "clip_fraction",
            "gain_db",
        ):
            v = getattr(spec, f)
            if v != getattr(DegradationSpec(), f):
                params[f] = v
        row["params"] = params
        axes[spec.axis].append(row)
    for axis in axes:
        axes[axis].sort(key=lambda r: (r.get("level", 0.0), r["label"]))

    payload = {
        "meta": {
            "classifier": "BirdNET (birdnetlib, TFLite)",
            "aggregation": "max-over-3s-windows",
            "dataset_id": manifest.dataset_id,
            "species": list(species),
            "n_species": len(species),
            "n_dev_clips": len(dev),
            "n_test_clips": len(test),
            "clip_seconds": 12.0,
            "sample_rate_hz": 48000,
            "beta": BETA,
            "threshold": _round(threshold),
            "seed": SEED,
            "date": dt.date.today().isoformat(),
            "note": (
                "Every number here was produced by BirdNET on real Xeno-canto audio via "
                "scripts/run_real_eval.py; none is invented. Clip-level labels only (no strong "
                "labels): the SNR and overlap axes use the energy-based active-region estimate "
                "documented in degradations.add_noise_at_snr, whose error propagates into them."
            ),
        },
        "clean_baseline": {
            "macro": clean,
            "per_species": done["clean"]["per_species"],
        },
        "axes": dict(axes),
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS.relative_to(ROOT)}", flush=True)
    print(
        f"CLEAN baseline macro F{BETA}: {clean['f_beta']:.3f} "
        f"(P={clean['precision']:.3f} R={clean['recall']:.3f}) at threshold {threshold:.3f}"
    )
    for axis, title in [
        ("snr", "SNR sweep (pink noise)"),
        ("distance", "Distance (m)"),
        ("reverb", "Reverb RT60 (s)"),
        ("noise_type", "Noise colour at 0 dB"),
        ("bandwidth", "Bandwidth / resample"),
    ]:
        print(f"\n{title}:")
        for r in axes.get(axis, []):
            lvl = r.get("level")
            tag = r["label"] if lvl is None else f"{lvl:g}"
            print(
                f"  {tag:<22} F{BETA}={r['f_beta']:.3f}  P={r['precision']:.3f} R={r['recall']:.3f}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
