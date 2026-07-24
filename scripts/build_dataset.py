"""Build a small, real, labelled multi-species evaluation set from Xeno-canto.

This is the loader the benchmark always specified but never had (see ``ROADMAP.md``,
Milestone 1). It downloads high-quality (``q:A``) focal recordings for a handful of common
European passerines, extracts one energy-centred clip per recording, and writes a
``protocol.Manifest`` plus a provenance record.

NETWORK + KEY REQUIRED. Nothing here runs in CI. The Xeno-canto API key is read from the
environment variable ``XENO_CANTO_KEY`` and is never written to disk or committed. No audio is
committed either: ``data/`` is git-ignored and the recordings carry per-recording Creative
Commons terms. What *is* committed is the attribution in ``docs/DATA_SOURCES.md`` (written here)
and, from the evaluation step, ``docs/real_eval_results.json``.

Usage::

    export XENO_CANTO_KEY=...            # never commit this
    python scripts/build_dataset.py     # writes data/ and docs/DATA_SOURCES.md
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# The vocabulary. Scientific names are the label keys because that is what BirdNET emits
# (``detection["scientific_name"]``). All six are common, acoustically distinct European
# passerines with thousands of q:A recordings on Xeno-canto and are in BirdNET's 6522-label set.
SPECIES: tuple[tuple[str, str, str], ...] = (
    ("Turdus", "merula", "Eurasian Blackbird"),
    ("Erithacus", "rubecula", "European Robin"),
    ("Fringilla", "coelebs", "Common Chaffinch"),
    ("Parus", "major", "Great Tit"),
    ("Sylvia", "atricapilla", "Eurasian Blackcap"),
    ("Phylloscopus", "collybita", "Common Chiffchaff"),
)

SR = 48000  # BirdNET's native rate; the degradation pipeline runs at this rate too.
CLIP_S = 12.0  # four BirdNET 3-second frames per clip.
BAND_HZ = (1500.0, 10000.0)  # passerine band, for energy-centred clip selection.
N_PER_SPECIES = 15
N_DEV = 5  # first five (by ascending XC id) go to the dev/threshold split.
MIN_LEN_S, MAX_LEN_S = 6.0, 150.0
API = "https://xeno-canto.org/api/3/recordings"
UA = {"User-Agent": "birdsong-robustness-research/0.1 (academic robustness benchmark)"}

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CLIPS = DATA / "clips"
CACHE = DATA / "xc_cache"


def _length_seconds(raw: str) -> float:
    """Parse Xeno-canto ``length`` ('m:ss' or 'h:mm:ss') into seconds."""
    parts = [float(p) for p in str(raw).split(":")]
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60.0 + p
    return seconds


def _api(query: str, key: str, page: int = 1) -> dict:
    url = f"{API}?" + urllib.parse.urlencode({"query": query, "key": key, "page": page})
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1024:
        return
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as r:
        data = r.read()
    dest.write_bytes(data)


def _best_window(y: np.ndarray, sr: int, clip_s: float, band: tuple[float, float]) -> np.ndarray:
    """Return the ``clip_s``-second window with the most energy in the passerine band."""
    win = round(clip_s * sr)
    if len(y) <= win:
        out = np.zeros(win, dtype=np.float64)
        out[: len(y)] = y
        return out
    # Band-energy envelope from an STFT, then a sliding sum the length of the clip.
    hop = 512
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    frame_energy = stft[band_mask].sum(axis=0)
    frames_per_win = max(1, win // hop)
    csum = np.concatenate([[0.0], np.cumsum(frame_energy)])
    if len(csum) <= frames_per_win:
        start_frame = 0
    else:
        window_sums = csum[frames_per_win:] - csum[:-frames_per_win]
        start_frame = int(np.argmax(window_sums))
    start = min(start_frame * hop, len(y) - win)
    return y[start : start + win].astype(np.float64)


def _fetch_species(gen: str, sp: str, key: str, forbidden: set[str]) -> list[dict]:
    """Page through q:A recordings for one species and return usable candidate metadata."""
    query = f"gen:{gen} sp:{sp} q:A"
    first = _api(query, key, page=1)
    n_pages = int(first.get("numPages", 1))
    candidates: list[dict] = []
    for page in range(1, n_pages + 1):
        payload = first if page == 1 else _api(query, key, page=page)
        for rec in payload.get("recordings", []):
            if rec.get("q") != "A" or not rec.get("file"):
                continue
            also = {a.strip() for a in rec.get("also", []) if a and a.strip()}
            if also & forbidden:  # keep clips clean of other target species
                continue
            try:
                length = _length_seconds(rec.get("length", "0"))
            except ValueError:
                continue
            if not (MIN_LEN_S <= length <= MAX_LEN_S):
                continue
            candidates.append(rec)
        time.sleep(1.1)  # respect the ~1 req/sec rate limit
        if len(candidates) >= N_PER_SPECIES * 3:
            break
    candidates.sort(key=lambda r: int(r["id"]))
    return candidates


def main() -> int:
    key = os.environ.get("XENO_CANTO_KEY")
    if not key:
        print("XENO_CANTO_KEY not set in the environment.", file=sys.stderr)
        return 2

    CLIPS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    vocab_sci = {f"{g} {s}" for g, s, _ in SPECIES}
    entries: list[dict] = []
    provenance: list[dict] = []

    for gen, sp, common in SPECIES:
        sci = f"{gen} {sp}"
        forbidden = vocab_sci - {sci}
        print(f"\n=== {sci} ({common}) ===", flush=True)
        candidates = _fetch_species(gen, sp, key, forbidden)
        print(f"  {len(candidates)} candidate q:A recordings", flush=True)

        kept = 0
        for rec in candidates:
            if kept >= N_PER_SPECIES:
                break
            xc_id = rec["id"]
            mp3 = CACHE / f"XC{xc_id}.mp3"
            try:
                _download(rec["file"], mp3)
                time.sleep(1.1)
                y, _ = librosa.load(str(mp3), sr=SR, mono=True)
            except Exception as exc:
                print(f"  skip XC{xc_id}: {type(exc).__name__}", flush=True)
                continue
            if len(y) < int(MIN_LEN_S * SR):
                continue
            clip = _best_window(y, SR, CLIP_S, BAND_HZ)
            peak = float(np.max(np.abs(clip)))
            if peak < 0.02:
                continue
            clip = clip * (0.7 / peak)  # common preprocessing level for a fair baseline

            code = f"{gen[:3].lower()}{sp[:3].lower()}"
            clip_path = CLIPS / f"{code}_XC{xc_id}.wav"
            sf.write(str(clip_path), clip.astype(np.float32), SR)

            split = "dev" if kept < N_DEV else "test"
            entries.append(
                {
                    "recording_id": f"XC{xc_id}",
                    "path": f"clips/{clip_path.name}",
                    "sr": SR,
                    "duration_s": CLIP_S,
                    "species": [sci],
                    "events": [],
                    "split": split,
                    "source": "xeno-canto",
                    "license_id": rec.get("lic", "unknown"),
                    "recorder": rec.get("dvc") or rec.get("mic") or "unknown",
                    "latitude": float(rec["lat"]) if rec.get("lat") else None,
                    "longitude": float(rec["lon"]) if rec.get("lon") else None,
                    "recorded_at": rec.get("date") or None,
                }
            )
            provenance.append(
                {
                    "xc_id": xc_id,
                    "scientific_name": sci,
                    "common_name": common,
                    "recordist": rec.get("rec", "unknown"),
                    "country": rec.get("cnt", ""),
                    "license": rec.get("lic", "unknown"),
                    "type": rec.get("type", ""),
                    "url": rec.get("url", f"https://xeno-canto.org/{xc_id}"),
                    "split": split,
                }
            )
            kept += 1
            print(f"  kept XC{xc_id} ({split}) peak={peak:.2f}", flush=True)

        print(f"  -> {kept} recordings kept for {sci}", flush=True)

    manifest = {
        "dataset_id": "xeno-canto-6sp-v1",
        "species": sorted(vocab_sci),
        "entries": entries,
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (DATA / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    _write_data_sources(provenance)
    n_dev = sum(e["split"] == "dev" for e in entries)
    print(
        f"\nWrote {len(entries)} recordings "
        f"({n_dev} dev / {len(entries) - n_dev} test) across {len(SPECIES)} species.",
        flush=True,
    )
    return 0


def _write_data_sources(provenance: list[dict]) -> None:
    """Write the committed attribution file (Xeno-canto requires per-recording attribution)."""
    lines = [
        "# Data sources and attribution",
        "",
        "The real-audio results in the README are computed on focal recordings from",
        "[Xeno-canto](https://xeno-canto.org), rebuilt locally by `scripts/build_dataset.py`.",
        "**No audio is redistributed in this repository** — Xeno-canto recordings carry",
        "per-recording Creative Commons licences (listed below), so `data/` is git-ignored and",
        "only this attribution and the measured results are committed. Every recording is",
        "quality grade **A** (Xeno-canto's highest). Each clip used is a 12-second,",
        "energy-centred window of the source recording, resampled to 48 kHz.",
        "",
        "Xeno-canto asks that recordings be attributed to the recordist under their licence.",
        "The recordists below generously released these recordings; the benchmark is indebted",
        "to them and to the Xeno-canto Foundation.",
        "",
    ]
    by_species: dict[str, list[dict]] = {}
    for p in provenance:
        by_species.setdefault(f"{p['scientific_name']} ({p['common_name']})", []).append(p)
    for species in sorted(by_species):
        recs = sorted(by_species[species], key=lambda r: int(r["xc_id"]))
        lines.append(f"## {species}")
        lines.append("")
        lines.append("| Xeno-canto ID | Recordist | Country | Type | Licence | Split |")
        lines.append("|---|---|---|---|---|---|")
        for r in recs:
            lic = str(r["license"]).replace("https://creativecommons.org/licenses/", "CC ")
            lic = lic.replace("//creativecommons.org/licenses/", "CC ").rstrip("/")
            lines.append(
                f"| [XC{r['xc_id']}]({r['url']}) | {r['recordist']} | {r['country']} | "
                f"{r['type']} | {lic} | {r['split']} |"
            )
        lines.append("")
    out = ROOT / "docs" / "DATA_SOURCES.md"
    out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
