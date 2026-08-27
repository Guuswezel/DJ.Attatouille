from __future__ import annotations

import json
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .canonical import (
    MusicClock,
    bar_synchronous_audio,
    canonicalize_file,
    load_feature_audio,
    music_clock,
    save_sample,
    stable_id,
    transition_evidence,
    features_from_bars,
)
from .config import (
    CONTEXT_BEFORE_BARS,
    FEATURE_SAMPLE_RATE,
    FEATURE_VERSION,
    SAMPLES_PER_BAR,
    TOTAL_BARS,
    TRANSITION_BARS,
)
from .tracklist import read_tracklist


MARKDOWN_LINK = re.compile(r"^\[[^\]]+\]\((https?://[^)]+)\)$")
VERSION_NUMBER = re.compile(r"(\d+)")


def normalize_public_url(value: str) -> str:
    """Accept a plain URL plus common shell/Markdown copy artifacts."""
    url = value.strip()
    markdown = MARKDOWN_LINK.match(url)
    if markdown:
        url = markdown.group(1)
    for escaped, literal in ((r"\?", "?"), (r"\=", "="), (r"\&", "&")):
        url = url.replace(escaped, literal)
    if not url.startswith(("https://", "http://")):
        raise ValueError("Mix URL must be a plain http or https URL")
    return url


def javascript_runtime_args() -> list[str]:
    """Select a current runtime for yt-dlp's YouTube challenge solver."""
    candidates = (("deno", 2), ("node", 22))
    detected: list[str] = []
    for runtime, minimum_major in candidates:
        executable = shutil.which(runtime)
        if executable is None:
            continue
        version = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        version_text = version.stdout or version.stderr
        match = VERSION_NUMBER.search(version_text)
        if version.returncode == 0 and match and int(match.group(1)) >= minimum_major:
            return ["--js-runtimes", f"{runtime}:{executable}"]
        detected.append(f"{runtime} ({version_text.strip() or 'unknown version'})")
    detail = f" Found: {', '.join(detected)}." if detected else ""
    raise RuntimeError(
        "Downloading from YouTube requires Deno 2.3+ or Node.js 22+."
        f"{detail} Install a supported runtime and retry."
    )


def yt_dlp_command() -> list[str]:
    """Resolve the project-managed binary or the active Python package."""
    configured = os.environ.get("DJ_YT_DLP_BIN")
    if configured:
        executable = Path(configured).expanduser()
        if executable.is_file() and os.access(executable, os.X_OK):
            return [str(executable)]
        raise RuntimeError(
            f"The configured yt-dlp executable is missing or not executable: {executable}. "
            "Run ./scripts/setup-macos-transition-trainer.sh once."
        )
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    executable = shutil.which("yt-dlp")
    if executable:
        return [executable]
    raise RuntimeError(
        "The active trainer environment does not contain yt-dlp. "
        "Run ./scripts/setup-macos-transition-trainer.sh once."
    )


def read_manifest(root: Path) -> list[dict[str, Any]]:
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_manifest_rows(root: Path, rows: Iterable[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    existing = {row["id"]: row for row in read_manifest(root)}
    for row in rows:
        existing[row["id"]] = row
    temporary = manifest.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in existing.values()),
        encoding="utf-8",
    )
    temporary.replace(manifest)


def download_public_mix(url: str, destination: Path) -> Path:
    """Fetch public audio without cookies, API keys, or DRM bypasses."""
    url = normalize_public_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    template = str(destination.with_suffix(".%(ext)s"))
    command = [
        *yt_dlp_command(),
        *javascript_runtime_args(),
        "--no-playlist", "--no-write-info-json", "--no-write-thumbnail", "--force-overwrites",
        "--extract-audio", "--audio-format", "wav", "--audio-quality", "0",
        "--output", template, url,
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "yt-dlp could not download this public mix. Run the macOS trainer setup "
            "again to refresh its YouTube extractor and challenge solver, then retry."
        ) from error
    downloaded = destination.with_suffix(".wav")
    if not downloaded.exists():
        raise RuntimeError("yt-dlp finished without producing an audio file")
    return downloaded


def _localize_weak_timestamp(
    audio: np.ndarray,
    sample_rate: int,
    timestamp: float,
    clock: MusicClock,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    seed_bars, _, _ = bar_synchronous_audio(
        audio, sample_rate, center_seconds=timestamp, clock=clock,
    )
    _, seed_mir = features_from_bars(seed_bars)
    evidence = transition_evidence(seed_mir)
    search_start, search_end = 8, TOTAL_BARS - 8
    localized_bar = search_start + int(np.argmax(evidence[search_start:search_end]))
    bar_seconds = 4 * 60.0 / clock.bpm
    localized_seconds = float(np.clip(
        timestamp + (localized_bar - TOTAL_BARS // 2) * bar_seconds,
        0.0,
        len(audio) / sample_rate,
    ))
    bars, mask, start_bar = bar_synchronous_audio(
        audio, sample_rate, center_seconds=localized_seconds, clock=clock,
    )
    weak_bar = int(np.clip(TOTAL_BARS // 2 + round((timestamp - localized_seconds) / bar_seconds), 0, TOTAL_BARS - 1))
    weak_positive = np.zeros(TOTAL_BARS, dtype=np.float32)
    radius = max(2, int(round(45.0 / max(bar_seconds, 0.25))))
    weak_positive[max(0, weak_bar - radius):min(TOTAL_BARS, weak_bar + radius + 1)] = 1.0
    details = {
        "bpm": round(clock.bpm, 4),
        "clockConfidence": round(clock.confidence, 4),
        "weakTimestampSeconds": timestamp,
        "localizedCenterSeconds": round(localized_seconds, 4),
        "sourceStartBar": start_bar,
        "localizer": "mir-seed-v1",
    }
    return localized_seconds, bars, mask, {**details, "weakPositiveMask": weak_positive}


def ingest_professional_mix(
    *,
    root: Path,
    name: str,
    tracklist_path: Path,
    url: str | None = None,
    audio_path: Path | None = None,
) -> list[dict[str, Any]]:
    entries = read_tracklist(tracklist_path)
    if url:
        url = normalize_public_url(url)
    source_id = stable_id(name, url or str(audio_path), FEATURE_VERSION)
    raw_dir = root / "sources" / source_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    if audio_path is not None:
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        downloaded = raw_dir / f"input{audio_path.suffix.lower()}"
        if audio_path.resolve() != downloaded.resolve():
            shutil.copyfile(audio_path, downloaded)
    elif url:
        cached_download = next(
            (candidate for candidate in sorted(raw_dir.glob("input.*"))
             if candidate.is_file() and not candidate.name.endswith((".part", ".ytdl"))),
            None,
        )
        if cached_download is not None:
            downloaded = cached_download
            print(f"Reusing downloaded mix: {downloaded}", file=sys.stderr, flush=True)
        else:
            downloaded = download_public_mix(url, raw_dir / "input")
    else:
        raise ValueError("Provide either a public URL or a local audio file")
    print("Canonicalizing reference audio...", file=sys.stderr, flush=True)
    canonical = canonicalize_file(downloaded, raw_dir / "canonical.wav")
    audio, sample_rate = load_feature_audio(canonical)
    duration = len(audio) / sample_rate
    print("Detecting the full-mix beat and bar clock once...", file=sys.stderr, flush=True)
    clock = music_clock(audio, sample_rate)
    rows: list[dict[str, Any]] = []
    transition_times = [entry.seconds for entry in entries[1:] if entry.seconds < duration]
    for index, entry in enumerate(entries[1:], start=1):
        if entry.seconds >= duration:
            continue
        print(
            f"Localizing transition {index}/{len(transition_times)}: {entry.title}",
            file=sys.stderr,
            flush=True,
        )
        _, bars, mask, details = _localize_weak_timestamp(
            audio, sample_rate, entry.seconds, clock,
        )
        weak_positive = details.pop("weakPositiveMask")
        sample_id = stable_id(source_id, "professional", str(index), str(entry.seconds))
        relative = Path("samples") / "professional" / f"{sample_id}.npz"
        metadata = {
            "id": sample_id,
            "domain": "professional",
            "sourceId": source_id,
            "mixName": name,
            "url": url,
            "trackBefore": asdict(entries[index - 1]),
            "trackAfter": asdict(entry),
            "transitionStartBar": CONTEXT_BEFORE_BARS,
            "transitionEndBar": CONTEXT_BEFORE_BARS + TRANSITION_BARS,
            **details,
        }
        save_sample(
            root / relative,
            bars=bars,
            mask=mask,
            metadata=metadata,
            arrays={"weak_positive_mask": weak_positive.astype(np.float16)},
        )
        rows.append({"id": sample_id, "domain": "professional", "sample": str(relative), **metadata})

    # Localizer negatives are drawn midway between weak timestamps and never
    # from the first/last context margin of a mix.
    bar_seconds = 4 * 60.0 / clock.bpm
    for index, (left, right) in enumerate(zip(transition_times, transition_times[1:])):
        if right - left < 2 * 45:
            continue
        center = (left + right) / 2
        if min(abs(center - point) for point in transition_times) < 45:
            continue
        bars, mask, start_bar = bar_synchronous_audio(audio, sample_rate, center_seconds=center, clock=clock)
        sample_id = stable_id(source_id, "negative", str(index), str(center))
        relative = Path("samples") / "negative" / f"{sample_id}.npz"
        metadata = {
            "id": sample_id, "domain": "negative", "sourceId": source_id,
            "mixName": name, "centerSeconds": center, "sourceStartBar": start_bar,
            "bpm": clock.bpm, "clockConfidence": clock.confidence,
        }
        save_sample(
            root / relative, bars=bars, mask=mask, metadata=metadata,
            arrays={"weak_positive_mask": np.zeros(TOTAL_BARS, dtype=np.float16)},
        )
        rows.append({"id": sample_id, "domain": "negative", "sample": str(relative), **metadata})
    write_manifest_rows(root, rows)
    return rows


def _fft_band_gain(bar: np.ndarray, low_db: float, mid_db: float, high_db: float) -> np.ndarray:
    spectrum = np.fft.rfft(bar)
    bins = np.fft.rfftfreq(len(bar), d=1.0 / FEATURE_SAMPLE_RATE)
    gains = np.where(
        bins < 250,
        10 ** (low_db / 20),
        np.where(bins < 4_000, 10 ** (mid_db / 20), 10 ** (high_db / 20)),
    )
    return np.fft.irfft(spectrum * gains, n=len(bar)).astype(np.float32)


def baseline_mix(outgoing: np.ndarray, incoming: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    if outgoing.shape != incoming.shape or outgoing.shape != (TOTAL_BARS, SAMPLES_PER_BAR):
        raise ValueError("Synthetic decks must use the canonical bar grid")
    rendered = np.zeros_like(outgoing)
    transition_start = CONTEXT_BEFORE_BARS
    transition_end = transition_start + TRANSITION_BARS
    swap = 0.56
    for bar_index in range(TOTAL_BARS):
        if bar_index < transition_start:
            progress = 0.0
        elif bar_index >= transition_end:
            progress = 1.0
        else:
            progress = (bar_index - transition_start + 0.5) / TRANSITION_BARS
        out_gain = float(np.cos(progress * np.pi / 2))
        in_gain = float(np.sin(progress * np.pi / 2))
        bass_phase = float(np.clip((progress - swap + 0.08) / 0.16, 0.0, 1.0))
        out_bar = _fft_band_gain(outgoing[bar_index], -18 * bass_phase, -3 * progress, -5 * progress)
        in_bar = _fft_band_gain(incoming[bar_index], -18 * (1 - bass_phase), -3 * (1 - progress), -5 * (1 - progress))
        rendered[bar_index] = out_bar * out_gain + in_bar * in_gain
    peak = float(np.max(np.abs(rendered)))
    if peak > 0.98:
        rendered *= 0.98 / peak
    return rendered, {
        "overlapPhrases": 4.0,
        "bassSwapProgress": swap,
        "fadeOutCurve": 1.0,
        "fadeInCurve": 1.0,
        "outgoingLowDb": -18.0,
        "incomingLowDb": -18.0,
        "outgoingMidDb": -3.0,
        "incomingMidDb": -3.0,
        "outgoingHighDb": -5.0,
        "incomingHighDb": -5.0,
        "loopProbability": 0.0,
    }


def policy_features_from_decks(
    outgoing: np.ndarray,
    incoming: np.ndarray,
    *,
    outgoing_bpm: float,
    incoming_bpm: float,
    harmonic_compatibility: float = 0.5,
    outgoing_elapsed_fraction: float = 0.70,
    incoming_entry_fraction: float = 0.18,
    genre_compatibility: float = 0.5,
) -> np.ndarray:
    """Build the same 24-value semantic state consumed by the live policy."""
    _, outgoing_mir = features_from_bars(outgoing)
    _, incoming_mir = features_from_bars(incoming)
    out = np.mean(outgoing_mir[max(0, CONTEXT_BEFORE_BARS - 8):CONTEXT_BEFORE_BARS + 8], axis=0)
    inc = np.mean(incoming_mir[CONTEXT_BEFORE_BARS:CONTEXT_BEFORE_BARS + 16], axis=0)
    out_energy, in_energy = float(out[0]), float(inc[0])
    out_slope, in_slope = float(out[1]), float(inc[1])
    values = [
        incoming_bpm / max(outgoing_bpm, 1.0),
        harmonic_compatibility,
        out_energy,
        in_energy,
        abs(out_energy - in_energy),
        out_slope,
        in_slope,
        abs(out_slope - in_slope),
        float(out[2]),
        float(inc[2]),
        float(out[2] * inc[2]),
        float(out[10]),
        float(inc[10]),
        float(out[11]),
        float(inc[11]),
        float(out[11] * inc[11]),
        float(out[6]),
        float(inc[6]),
        float(np.mean(out[12:24])),
        float(np.mean(inc[12:24])),
        float(out[5]),
        float(inc[5]),
        float(np.clip(1.0 - out[5], 0.0, 1.0)),
        float(np.clip(1.0 - inc[5] * 0.5, 0.0, 1.0)),
        float(np.clip(outgoing_elapsed_fraction, 0.0, 1.0)),
        float(np.clip(1.0 - outgoing_elapsed_fraction, 0.0, 1.0)),
        float(np.clip(incoming_entry_fraction, 0.0, 1.0)),
        float(np.clip(genre_compatibility, 0.0, 1.0)),
        float(np.clip(1.0 - out[5] * 0.5, 0.0, 1.0)),
    ]
    return np.asarray(values, dtype=np.float32)


def _canonical_track(root: Path, path: Path) -> tuple[np.ndarray, int, Any, Path]:
    identity = stable_id(str(path.resolve()), str(path.stat().st_mtime_ns), FEATURE_VERSION)
    destination = root / "cache" / "tracks" / f"{identity}.wav"
    if not destination.exists():
        canonicalize_file(path, destination)
    audio, sample_rate = load_feature_audio(destination)
    return audio, sample_rate, music_clock(audio, sample_rate), destination


def synthesize_transitions(root: Path, track_paths: list[Path]) -> list[dict[str, Any]]:
    if len(track_paths) < 2:
        raise ValueError("At least two clean local tracks are required")
    tracks = [_canonical_track(root, path) for path in track_paths]
    rows: list[dict[str, Any]] = []
    for index in range(len(tracks) - 1):
        out_audio, out_rate, out_clock, _ = tracks[index]
        in_audio, in_rate, in_clock, _ = tracks[index + 1]
        out_duration, in_duration = len(out_audio) / out_rate, len(in_audio) / in_rate
        out_phrases = out_clock.downbeats[::8]
        in_phrases = in_clock.downbeats[::8]
        out_exit = float(min(out_phrases, key=lambda point: abs(point - out_duration * 0.70))) if out_phrases.size else out_duration * 0.70
        in_entry = float(min(in_phrases, key=lambda point: abs(point - in_duration * 0.18))) if in_phrases.size else in_duration * 0.18
        out_bar_seconds = 4 * 60.0 / out_clock.bpm
        in_bar_seconds = 4 * 60.0 / in_clock.bpm
        # Place the selected eight-bar phrase cues at the canonical transition
        # edges: outgoing exits at bar 48; incoming enters at bar 16.
        out_center = max(1.0, out_exit - 16 * out_bar_seconds)
        in_center = min(in_duration - 1.0, in_entry + 16 * in_bar_seconds)
        out_bars, out_mask, _ = bar_synchronous_audio(
            out_audio, out_rate, center_seconds=out_center, clock=out_clock,
        )
        in_bars, in_mask, _ = bar_synchronous_audio(
            in_audio, in_rate, center_seconds=in_center, clock=in_clock,
        )
        outgoing = out_bars.copy()
        incoming = in_bars.copy()
        outgoing[CONTEXT_BEFORE_BARS + TRANSITION_BARS:] = 0
        incoming[:CONTEXT_BEFORE_BARS] = 0
        mixed, controls = baseline_mix(outgoing, incoming)
        policy_features = policy_features_from_decks(
            outgoing,
            incoming,
            outgoing_bpm=out_clock.bpm,
            incoming_bpm=in_clock.bpm,
            outgoing_elapsed_fraction=out_exit / max(out_duration, 1.0),
            incoming_entry_fraction=in_entry / max(in_duration, 1.0),
        )
        mask = np.minimum(1.0, out_mask + in_mask)
        sample_id = stable_id(str(track_paths[index]), str(track_paths[index + 1]), "synthetic")
        relative = Path("samples") / "synthetic" / f"{sample_id}.npz"
        metadata = {
            "id": sample_id,
            "domain": "synthetic",
            "generator": "deterministic-baseline-v1",
            "outgoingTrack": str(track_paths[index]),
            "incomingTrack": str(track_paths[index + 1]),
            "outgoingBpm": out_clock.bpm,
            "incomingBpm": in_clock.bpm,
            "outgoingExitSeconds": round(out_exit, 4),
            "incomingEntrySeconds": round(in_entry, 4),
            "beatMatched": True,
            "barMatched": True,
            "phraseMatched": True,
            "transitionStartBar": CONTEXT_BEFORE_BARS,
            "transitionEndBar": CONTEXT_BEFORE_BARS + TRANSITION_BARS,
            "controls": controls,
        }
        weak_positive = np.zeros(TOTAL_BARS, dtype=np.float16)
        weak_positive[CONTEXT_BEFORE_BARS:CONTEXT_BEFORE_BARS + TRANSITION_BARS] = 1
        save_sample(
            root / relative,
            bars=mixed,
            mask=mask,
            metadata=metadata,
            outgoing=outgoing,
            incoming=incoming,
            arrays={"weak_positive_mask": weak_positive, "policy_features": policy_features},
        )
        rows.append({"id": sample_id, "domain": "synthetic", "sample": str(relative), **metadata})
    write_manifest_rows(root, rows)
    return rows


def domain_augmentation(bars: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply the same mastering/codec nuisance distribution to both domains."""
    result = bars.astype(np.float32).copy()
    gain = 10 ** (float(rng.uniform(-2.0, 2.0)) / 20)
    result *= gain
    # Random broad EQ tilt and bandwidth restriction.
    low_db, high_db = float(rng.uniform(-2.5, 2.5)), float(rng.uniform(-2.5, 2.5))
    bandwidth_high = float(rng.choice([5_500, 6_500, 7_900]))
    for index, bar in enumerate(result):
        spectrum = np.fft.rfft(bar)
        frequencies = np.fft.rfftfreq(len(bar), d=1.0 / FEATURE_SAMPLE_RATE)
        tilt = np.interp(
            frequencies, [0, FEATURE_SAMPLE_RATE / 2],
            [10 ** (low_db / 20), 10 ** (high_db / 20)],
        )
        spectrum *= tilt * (frequencies <= bandwidth_high)
        result[index] = np.fft.irfft(spectrum, n=len(bar))
    # Quantization is a cheap stochastic codec proxy; low noise prevents the
    # critic from learning silence/bit-depth fingerprints.
    bits = int(rng.choice([12, 14, 16]))
    levels = float(2 ** (bits - 1) - 1)
    result = np.round(np.clip(result, -1, 1) * levels) / levels
    result += rng.normal(0, float(rng.uniform(0, 3e-4)), size=result.shape).astype(np.float32)
    peak = float(np.max(np.abs(result)))
    if peak > 0.99:
        result *= 0.99 / peak
    return result.astype(np.float32)


def feature_augmentation(
    mel: np.ndarray,
    mir: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Cheap mastering/codec nuisance augmentation for training epochs.

    Canonical extraction already captured the expensive per-bar spectrum. The
    trainer only needs plausible stochastic gain, EQ, bandwidth, and codec
    variation; applying those transforms to mel/MIR avoids re-running librosa
    over millions of audio samples on every epoch.
    """
    augmented_mel = mel.astype(np.float32).copy()
    augmented_mir = mir.astype(np.float32).copy()
    gain_db = float(rng.uniform(-2.0, 2.0))
    low_db, high_db = float(rng.uniform(-2.5, 2.5)), float(rng.uniform(-2.5, 2.5))
    tilt_db = np.linspace(low_db, high_db, augmented_mel.shape[-1], dtype=np.float32)
    augmented_mel += (gain_db + tilt_db[None, :]) / 80.0

    bandwidth_fraction = float(rng.choice([0.70, 0.82, 1.0]))
    cutoff = max(1, int(round(augmented_mel.shape[-1] * bandwidth_fraction)))
    if cutoff < augmented_mel.shape[-1]:
        rolloff = np.linspace(0.0, 0.14, augmented_mel.shape[-1] - cutoff, dtype=np.float32)
        augmented_mel[:, cutoff:] -= rolloff[None, :]
    augmented_mel += rng.normal(0.0, 0.003, size=augmented_mel.shape).astype(np.float32)
    augmented_mel = np.clip(augmented_mel, 0.0, 1.0)

    gain = 10 ** (gain_db / 20.0)
    augmented_mir[:, 0] = np.clip(augmented_mir[:, 0] * gain, 0.0, 1.0)
    augmented_mir[:, 1] = np.clip(augmented_mir[:, 1] * gain, -1.0, 1.0)
    band_gains = 10 ** (np.asarray([low_db, (low_db + high_db) / 2, high_db]) / 20.0)
    bands = np.maximum(augmented_mir[:, 2:5] * band_gains[None, :], 0.0)
    augmented_mir[:, 2:5] = bands / np.maximum(np.sum(bands, axis=1, keepdims=True), 1e-6)
    augmented_mir[:, 5:] += rng.normal(0.0, 0.004, size=augmented_mir[:, 5:].shape).astype(np.float32)
    augmented_mir[:, 5:] = np.clip(augmented_mir[:, 5:], 0.0, 1.0)
    return augmented_mel.astype(np.float32), augmented_mir.astype(np.float32)


class TransitionDataset:
    """Small torch-compatible dataset without importing torch at module load."""

    def __init__(self, root: Path, domains: set[str] | None = None, augment: bool = True):
        self.root = root
        self.rows = [row for row in read_manifest(root) if domains is None or row["domain"] in domains]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with np.load(self.root / row["sample"], allow_pickle=False) as stored:
            audio = stored["audio"].astype(np.float32)
            mel = stored["mel"].astype(np.float32)
            mir = stored["mir"].astype(np.float32)
            if self.augment:
                mel, mir = feature_augmentation(mel, mir, np.random.default_rng())
            item: dict[str, Any] = {
                "id": row["id"], "domain": row["domain"], "audio": audio,
                "mel": mel, "mir": mir,
                "bar_mask": stored["bar_mask"].astype(np.float32),
                "weak_positive_mask": stored["weak_positive_mask"].astype(np.float32),
            }
            if "localized_transition" in stored:
                localized = stored["localized_transition"].astype(np.float32)
                target = np.zeros(TOTAL_BARS, dtype=np.float32)
                target[int(localized[0]):int(localized[2]) + 1] = 1.0
                item["transition_target"] = target
            else:
                item["transition_target"] = item["weak_positive_mask"]
            for key in ("outgoing_audio", "incoming_audio", "policy_features"):
                if key in stored:
                    item[key] = stored[key].astype(np.float32)
            return item


def dataset_summary(root: Path) -> dict[str, Any]:
    rows = read_manifest(root)
    domains: dict[str, int] = {}
    for row in rows:
        domains[row["domain"]] = domains.get(row["domain"], 0) + 1
    return {"featureVersion": FEATURE_VERSION, "samples": len(rows), "domains": domains}
