from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from .config import (
    CHANNELS,
    FEATURE_SAMPLE_RATE,
    FEATURE_VERSION,
    MEL_BINS,
    MIR_DIM,
    SAMPLE_RATE,
    SAMPLES_PER_BAR,
    TARGET_LUFS,
    TOTAL_BARS,
    TRUE_PEAK_DB,
)


@dataclass(frozen=True)
class MusicClock:
    bpm: float
    beats: np.ndarray
    downbeats: np.ndarray
    confidence: float


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:24]


def canonicalize_file(source: Path, destination: Path) -> Path:
    """Decode every domain through exactly the same mastering/codec path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-map_metadata", "-1", "-vn", "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE),
            "-af",
            (
                f"highpass=f=25,lowpass=f=20000,"
                f"loudnorm=I={TARGET_LUFS}:TP={TRUE_PEAK_DB}:LRA=11"
            ),
            "-c:a", "pcm_s16le", str(destination),
        ],
        check=True,
    )
    return destination


def load_feature_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, rate = librosa.load(path, sr=FEATURE_SAMPLE_RATE, mono=True)
    audio = np.nan_to_num(audio.astype(np.float32))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio /= peak
    return audio, int(rate)


def music_clock(audio: np.ndarray, sample_rate: int) -> MusicClock:
    hop = 512
    onset = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop)
    tempo_value, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset, sr=sample_rate, hop_length=hop, units="frames",
    )
    bpm = float(np.asarray(tempo_value).reshape(-1)[0]) if np.size(tempo_value) else 120.0
    bpm = float(np.clip(bpm if math.isfinite(bpm) and bpm > 0 else 120.0, 60.0, 190.0))
    beats = librosa.frames_to_time(np.asarray(beat_frames), sr=sample_rate, hop_length=hop)
    if beats.size < 16:
        beat_seconds = 60.0 / bpm
        beats = np.arange(0.0, len(audio) / sample_rate + beat_seconds, beat_seconds)
        confidence = 0.15
    else:
        intervals = np.diff(beats)
        confidence = float(np.clip(1.0 - np.std(intervals) / max(np.mean(intervals), 1e-6), 0.0, 1.0))

    # Select the four-beat phase with the strongest mean onset.  The learned
    # localizer can correct weak labels later; this only establishes a stable
    # bar-synchronous representation for both domains.
    beat_onsets = np.interp(beats, librosa.frames_to_time(np.arange(len(onset)), sr=sample_rate, hop_length=hop), onset)
    phase_scores = [float(np.mean(beat_onsets[phase::4])) if beat_onsets[phase::4].size else 0.0 for phase in range(4)]
    phase = int(np.argmax(phase_scores))
    return MusicClock(bpm=bpm, beats=beats.astype(np.float64), downbeats=beats[phase::4].astype(np.float64), confidence=confidence)


def _fixed_bar_edges(clock: MusicClock, duration: float) -> np.ndarray:
    downbeats = clock.downbeats
    if downbeats.size >= 3:
        median_bar = float(np.median(np.diff(downbeats)))
        left = list(downbeats)
        while left and left[0] > 0.25:
            left.insert(0, left[0] - median_bar)
        edges = np.asarray(left, dtype=np.float64)
        while edges[-1] < duration + median_bar:
            edges = np.append(edges, edges[-1] + median_bar)
        return edges
    bar_seconds = 4 * 60.0 / clock.bpm
    return np.arange(0.0, duration + bar_seconds * 2, bar_seconds)


def bar_synchronous_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    center_seconds: float,
    clock: MusicClock | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Warp a 64-bar context to a fixed sample grid without changing pitch.

    This representation is used for model input only.  The production renderer
    continues to use FFmpeg's pitch-preserving time stretcher.
    """
    clock = clock or music_clock(audio, sample_rate)
    edges = _fixed_bar_edges(clock, len(audio) / sample_rate)
    center_bar = int(np.clip(np.searchsorted(edges, center_seconds) - 1, 0, max(0, len(edges) - 2)))
    start_bar = center_bar - TOTAL_BARS // 2
    bars = np.zeros((TOTAL_BARS, SAMPLES_PER_BAR), dtype=np.float32)
    mask = np.zeros(TOTAL_BARS, dtype=np.float32)
    for output_bar in range(TOTAL_BARS):
        source_bar = start_bar + output_bar
        if source_bar < 0 or source_bar + 1 >= len(edges):
            continue
        start = max(0, int(round(edges[source_bar] * sample_rate)))
        end = min(len(audio), int(round(edges[source_bar + 1] * sample_rate)))
        if end - start < 32:
            continue
        segment = audio[start:end]
        rate = len(segment) / SAMPLES_PER_BAR
        # Phase-vocoder stretching changes time while retaining pitch. Linear
        # resampling would align beats but move keys and spectral bands, giving
        # the critic invalid musical evidence.
        stretched = librosa.effects.time_stretch(
            segment, rate=float(np.clip(rate, 0.5, 2.0)), n_fft=1024, hop_length=256,
        )
        bars[output_bar] = librosa.util.fix_length(stretched, size=SAMPLES_PER_BAR)
        mask[output_bar] = 1.0
    return bars, mask, start_bar


def features_from_bars(bars: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if bars.shape != (TOTAL_BARS, SAMPLES_PER_BAR):
        raise ValueError(f"Expected {(TOTAL_BARS, SAMPLES_PER_BAR)} beat-warped audio, got {bars.shape}")
    mel_rows: list[np.ndarray] = []
    mir_rows: list[np.ndarray] = []
    previous_spectrum: np.ndarray | None = None
    previous_energy = 0.0
    for bar in bars:
        n_fft, hop_length = 1024, 256
        spectrum = np.abs(librosa.stft(bar, n_fft=n_fft, hop_length=hop_length, center=False))
        power = spectrum ** 2
        mel = librosa.feature.melspectrogram(
            S=power, sr=FEATURE_SAMPLE_RATE, n_mels=MEL_BINS, fmax=FEATURE_SAMPLE_RATE / 2,
        )
        mel_db = librosa.power_to_db(mel + 1e-10, ref=1.0)
        mel_rows.append(np.clip((np.mean(mel_db, axis=1) + 80.0) / 80.0, 0.0, 1.0))

        frequencies = librosa.fft_frequencies(sr=FEATURE_SAMPLE_RATE, n_fft=n_fft)
        total = float(np.sum(power) + 1e-9)
        low = float(np.sum(power[frequencies < 250]) / total)
        mid = float(np.sum(power[(frequencies >= 250) & (frequencies < 4_000)]) / total)
        high = float(np.sum(power[frequencies >= 4_000]) / total)
        energy = float(np.sqrt(np.mean(bar ** 2) + 1e-10))
        mean_spectrum = np.mean(spectrum, axis=1)
        flux = 0.0 if previous_spectrum is None else float(
            np.linalg.norm(np.maximum(0.0, mean_spectrum - previous_spectrum))
            / (np.linalg.norm(previous_spectrum) + 1e-6)
        )
        nyquist = FEATURE_SAMPLE_RATE / 2
        centroid = float(np.mean(librosa.feature.spectral_centroid(S=spectrum, sr=FEATURE_SAMPLE_RATE))) / nyquist
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=spectrum, sr=FEATURE_SAMPLE_RATE))) / nyquist
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=spectrum, sr=FEATURE_SAMPLE_RATE))) / nyquist
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(bar, frame_length=1024, hop_length=hop_length)))
        onset = float(np.mean(librosa.onset.onset_strength(y=bar, sr=FEATURE_SAMPLE_RATE, hop_length=hop_length))) / 10.0
        chroma = np.mean(librosa.feature.chroma_stft(S=power, sr=FEATURE_SAMPLE_RATE), axis=1)
        # A deliberately cheap voice-presence proxy.  Harmonix embeddings can
        # be added as a separate version without silently changing this schema.
        vocal_proxy = float(np.clip(mid * (1.0 - low) * 1.7, 0.0, 1.0))
        row = np.asarray(
            [
                min(1.0, energy * 4.0),
                float(np.clip((energy - previous_energy) * 8.0, -1.0, 1.0)),
                low, mid, high, min(1.0, flux),
                np.clip(centroid, 0, 1), np.clip(bandwidth, 0, 1), np.clip(rolloff, 0, 1),
                np.clip(zcr * 4.0, 0, 1), np.clip(onset, 0, 1), vocal_proxy,
                *np.clip(chroma, 0, 1).tolist(),
            ],
            dtype=np.float32,
        )
        if row.size != MIR_DIM:
            raise RuntimeError(f"MIR schema drift: expected {MIR_DIM}, got {row.size}")
        mir_rows.append(row)
        previous_spectrum = mean_spectrum
        previous_energy = energy
    return np.asarray(mel_rows, dtype=np.float32), np.asarray(mir_rows, dtype=np.float32)


def normalize_canonical_bars(bars: np.ndarray) -> np.ndarray:
    """Give professional and synthetic samples the identical final signal path."""
    if bars.shape != (TOTAL_BARS, SAMPLES_PER_BAR):
        raise ValueError("Canonical normalization received a non-canonical bar tensor")
    with tempfile.TemporaryDirectory(prefix="dj-training-canonical-") as directory:
        raw = Path(directory) / "raw.wav"
        mastered = Path(directory) / "mastered.wav"
        sf.write(raw, bars.reshape(-1), FEATURE_SAMPLE_RATE, subtype="PCM_16")
        canonicalize_file(raw, mastered)
        audio, _ = librosa.load(mastered, sr=FEATURE_SAMPLE_RATE, mono=True)
    expected = TOTAL_BARS * SAMPLES_PER_BAR
    audio = librosa.util.fix_length(audio.astype(np.float32), size=expected)
    return audio.reshape(TOTAL_BARS, SAMPLES_PER_BAR)


def transition_evidence(mir: np.ndarray) -> np.ndarray:
    """Unsupervised seed used only before a learned localizer is available."""
    energy_change = np.abs(mir[:, 1])
    flux = mir[:, 5]
    bass_change = np.abs(np.gradient(mir[:, 2]))
    timbre_change = np.linalg.norm(np.gradient(mir[:, 6:9], axis=0), axis=1)
    evidence = 0.28 * energy_change + 0.30 * flux + 0.27 * bass_change + 0.15 * timbre_change
    if np.max(evidence) > 0:
        evidence = evidence / np.max(evidence)
    return evidence.astype(np.float32)


def save_sample(
    destination: Path,
    *,
    bars: np.ndarray,
    mask: np.ndarray,
    metadata: dict[str, Any],
    outgoing: np.ndarray | None = None,
    incoming: np.ndarray | None = None,
    arrays: dict[str, np.ndarray] | None = None,
) -> Path:
    bars = normalize_canonical_bars(bars)
    mel, mir = features_from_bars(bars)
    payload: dict[str, Any] = {
        "audio": bars.astype(np.float16),
        "mel": mel.astype(np.float16),
        "mir": mir.astype(np.float16),
        "bar_mask": mask.astype(np.float16),
        "transition_evidence": transition_evidence(mir).astype(np.float16),
        "metadata": np.asarray(json.dumps({"featureVersion": FEATURE_VERSION, **metadata})),
    }
    if outgoing is not None:
        payload["outgoing_audio"] = outgoing.astype(np.float16)
    if incoming is not None:
        payload["incoming_audio"] = incoming.astype(np.float16)
    if arrays:
        payload.update({name: value for name, value in arrays.items()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return destination


def inspect_sample(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as sample:
        metadata = json.loads(str(sample["metadata"]))
        result = {
            "featureVersion": metadata["featureVersion"],
            "audioShape": list(sample["audio"].shape),
            "melShape": list(sample["mel"].shape),
            "mirShape": list(sample["mir"].shape),
            "validBars": int(np.sum(sample["bar_mask"] > 0)),
            "domain": metadata.get("domain"),
        }
        if "policy_features" in sample:
            result["policyFeatureShape"] = list(sample["policy_features"].shape)
        return result
