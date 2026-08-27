from __future__ import annotations

import base64
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from fastapi import FastAPI, HTTPException
from mutagen import File as MutagenFile
from pydub import AudioSegment
from pydub.effects import high_pass_filter, low_pass_filter

app = FastAPI(title="DJ.Attatouille local analysis worker", version="0.1.0")

MUSIC_ROOT = Path(os.environ.get("MUSIC_ROOT", "/music")).resolve()
# The Rust API always sends paths from its container namespace (/music). When
# the worker is run natively on macOS for Metal/MPS acceleration, MUSIC_ROOT is
# the real host folder and this retains the same safe relative-path contract.
REQUEST_MUSIC_ROOT = Path(os.environ.get("REQUEST_MUSIC_ROOT", "/music")).resolve()
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data")).resolve()
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
BACKEND_URL = os.environ.get("BACKEND_URL", "").rstrip("/")
ANALYSIS_MODEL = os.environ.get("ANALYSIS_MODEL", "auto").lower()
ANALYSIS_DEVICE = os.environ.get("ANALYSIS_DEVICE", "auto").lower()
HARMONIX_MODEL = os.environ.get("HARMONIX_MODEL", "").strip()
# Full HTDemucs separation is excellent when producing stems, but it is by
# far the slowest part of preparing a party library.  The default fast profile
# preserves the spectral trends Harmonix needs using lightweight pseudo-stems;
# the full profile remains available for a deliberate highest-accuracy pass.
HARMONIX_PROFILE = os.environ.get("HARMONIX_PROFILE", "fast").lower()
ANALYSIS_SAMPLE_RATE = int(os.environ.get("ANALYSIS_SAMPLE_RATE", "22050"))
HARMONIX_PROXY_SAMPLE_RATE = min(
    ANALYSIS_SAMPLE_RATE,
    max(8_000, int(os.environ.get("HARMONIX_PROXY_SAMPLE_RATE", "22050"))),
)
HARMONIX_MODEL_SAMPLE_RATE = 44_100
VECTOR_COLLECTION = "track_embeddings"
VECTOR_SIZE = 512
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus"}
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
DJ_TARGET_LUFS = float(os.environ.get("DJ_TARGET_LUFS", "-14"))
DJ_MAX_TRIM_DB = float(os.environ.get("DJ_MAX_TRIM_DB", "8"))
MASTER_TARGET_DBFS = float(os.environ.get("MASTER_TARGET_DBFS", "-14"))
MASTER_CEILING_DBFS = float(os.environ.get("MASTER_CEILING_DBFS", "-1"))
CANCELLED_JOBS: set[tuple[str, str]] = set()
CANCELLED_JOBS_LOCK = threading.Lock()
# all-in-one-infer normally reconstructs its checkpoint on every API call.
# The worker is deliberately long lived, so retain one model per device/profile
# and avoid repeatedly moving the same Harmonix weights into unified memory.
HARMONIX_MODELS: dict[tuple[str, str], Any] = {}
HARMONIX_MODELS_LOCK = threading.Lock()
HARMONIX_INFERENCE_LOCK = threading.Lock()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def job_is_cancelled(job_type: str, job_id: str) -> bool:
    with CANCELLED_JOBS_LOCK:
        return (job_type, job_id) in CANCELLED_JOBS


def ensure_job_active(job_type: str, job_id: str) -> None:
    if job_is_cancelled(job_type, job_id):
        raise HTTPException(409, "Job cancelled")


def resolve_music_source(source_path: str) -> Path:
    """Map an API /music path into this worker's safely mounted/local library."""
    requested = Path(source_path).resolve()
    try:
        relative = requested.relative_to(REQUEST_MUSIC_ROOT)
    except ValueError as problem:
        raise HTTPException(400, "sourcePath must be inside /music") from problem
    source = (MUSIC_ROOT / relative).resolve()
    if source != MUSIC_ROOT and MUSIC_ROOT not in source.parents:
        raise HTTPException(400, "sourcePath must be inside the configured music library")
    return source


def report_preparation_progress(
    preparation_id: str,
    *,
    progress: int,
    discovered_track_count: int,
    analysed_track_count: int,
    failed_track_count: int,
    current_track: str | None,
    message: str,
) -> None:
    """Publish best-effort, local progress without delaying audio analysis.

    The analysis endpoint intentionally remains a single request so the Rust
    API can keep its existing completion/error handling. These callbacks make
    an otherwise long Harmonix pass observable in the frontend in real time.
    """
    if not BACKEND_URL:
        return
    payload = {
        "progress": max(0, min(99, int(progress))),
        "discoveredTrackCount": max(0, int(discovered_track_count)),
        "analysedTrackCount": max(0, int(analysed_track_count)),
        "failedTrackCount": max(0, int(failed_track_count)),
        "currentTrack": current_track,
        "message": message,
    }
    encoded_id = urllib.parse.quote(preparation_id, safe="")
    request = urllib.request.Request(
        f"{BACKEND_URL}/api/preparations/{encoded_id}/progress",
        data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=2):
            pass
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        # The callback is informational. The final response still decides job
        # success, including when a developer runs this worker on its own.
        return


def qdrant_request(method: str, suffix: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
    data = json.dumps(jsonable(payload)).encode() if payload is not None else None
    request = urllib.request.Request(
        f"{QDRANT_URL}{suffix}", data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode() or "{}")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None


def ensure_vector_collection() -> bool:
    existing = qdrant_request("GET", f"/collections/{VECTOR_COLLECTION}")
    if existing:
        return True
    result = qdrant_request("PUT", f"/collections/{VECTOR_COLLECTION}", {
        "vectors": {"size": VECTOR_SIZE, "distance": "Cosine"},
    })
    return result is not None


def index_vector(track_id: str, vector: np.ndarray, payload: dict[str, Any]) -> bool:
    if not ensure_vector_collection():
        return False
    result = qdrant_request("PUT", f"/collections/{VECTOR_COLLECTION}/points?wait=true", {
        "points": [{"id": track_id, "vector": vector.astype(float).tolist(), "payload": payload}],
    })
    return result is not None


def get_vector(track_id: str) -> np.ndarray | None:
    result = qdrant_request("POST", f"/collections/{VECTOR_COLLECTION}/points", {
        "ids": [track_id], "with_vector": True, "with_payload": False,
    })
    points = (result or {}).get("result", [])
    if not points:
        return None
    vector = points[0].get("vector")
    return np.array(vector, dtype=np.float32) if vector else None


def normalise_genre(value: str) -> str:
    cleaned = value.strip().lower().replace("_", " ")
    aliases = {
        "hiphop": "hip-hop", "hip hop": "hip-hop", "electronic": "electronic",
        "dance": "dance", "rnb": "r&b", "r&b/soul": "r&b", "rap": "hip-hop",
        "house music": "house", "edm": "electronic",
    }
    return aliases.get(cleaned, cleaned)


def tags_for(path: Path, track_id: str) -> tuple[dict[str, str], list[str], str | None]:
    metadata: dict[str, str] = {"title": path.stem, "artist": "Unknown artist", "album": "Unknown album"}
    genres: list[str] = []
    artwork_url: str | None = None
    try:
        audio = MutagenFile(path, easy=True)
        if audio:
            for field in ("title", "artist", "album", "genre"):
                values = audio.get(field, [])
                if values:
                    if field == "genre":
                        genres.extend(normalise_genre(part) for value in values for part in value.split("/"))
                    else:
                        metadata[field] = str(values[0])
        raw = MutagenFile(path)
        image: bytes | None = None
        mime = "image/jpeg"
        if raw and getattr(raw, "tags", None):
            tags = raw.tags
            apics = [value for key, value in tags.items() if key.startswith("APIC")]
            if apics:
                image, mime = apics[0].data, getattr(apics[0], "mime", mime)
            elif "covr" in tags and tags["covr"]:
                image = bytes(tags["covr"][0])
                mime = "image/png" if image.startswith(b"\x89PNG") else mime
        if image:
            suffix = ".png" if "png" in mime else ".jpg"
            artwork_dir = DATA_ROOT / "artwork"
            artwork_dir.mkdir(parents=True, exist_ok=True)
            (artwork_dir / f"{track_id}{suffix}").write_bytes(image)
            artwork_url = f"/media/artwork/{track_id}{suffix}"
    except Exception:
        pass
    return metadata, sorted({genre for genre in genres if genre}), artwork_url


def estimate_key(chroma: np.ndarray) -> str:
    profile = chroma.mean(axis=1)
    if not np.any(profile):
        return "Unknown"
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    scores: list[tuple[float, str]] = []
    for shift, key in enumerate(KEYS):
        scores.append((float(np.corrcoef(profile, np.roll(major, shift))[0, 1]), key))
        scores.append((float(np.corrcoef(profile, np.roll(minor, shift))[0, 1]), f"{key}m"))
    return max(scores, key=lambda pair: -100 if math.isnan(pair[0]) else pair[0])[1]


def heuristic_genres(bpm: float, centroid: float, energy: float) -> list[str]:
    if bpm >= 118 and centroid > 2100:
        return ["electronic", "dance"]
    if bpm < 95 and energy < 0.18:
        return ["r&b"]
    if bpm < 112:
        return ["hip-hop"]
    if bpm >= 122:
        return ["house"]
    return ["pop"]


def project_embedding(raw: np.ndarray | None, mfcc: np.ndarray, chroma: np.ndarray, spectral: np.ndarray) -> np.ndarray:
    if raw is not None and raw.size:
        vector = np.asarray(raw, dtype=np.float32)
        if vector.ndim > 1:
            # Harmonix can return embeddings with ensemble and frame axes.
            # Preserve the final feature axis and pool every leading axis into
            # one stable track vector for the 512D Qdrant representation.
            vector = vector.reshape(-1, vector.shape[-1]).mean(axis=0)
    else:
        vector = np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1), chroma.mean(axis=1), chroma.std(axis=1),
            np.array([spectral.mean(), spectral.std()], dtype=np.float32),
        ]).astype(np.float32)
    if vector.size == 0:
        vector = np.zeros(1, dtype=np.float32)
    if vector.size != VECTOR_SIZE:
        source = np.linspace(0, 1, vector.size)
        target = np.linspace(0, 1, VECTOR_SIZE)
        vector = np.interp(target, source, vector).astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def harmonix_device() -> str:
    """Choose CUDA first while retaining a portable local CPU default.

    Docker Desktop for macOS does not make the Apple GPU available to Linux
    containers. A requested CUDA device therefore fails with a direct setup
    message instead of silently making a very slow CPU analysis look broken.
    """
    if ANALYSIS_DEVICE not in {"auto", "cpu", "cuda", "mps"}:
        raise RuntimeError("ANALYSIS_DEVICE must be auto, cpu, cuda, or mps")
    if ANALYSIS_DEVICE == "cpu":
        return "cpu"
    try:
        import torch  # type: ignore
        has_cuda = bool(torch.cuda.is_available())
        mps_backend = getattr(torch.backends, "mps", None)
        has_mps = bool(mps_backend and mps_backend.is_available())
    except Exception:
        has_cuda = False
        has_mps = False
    if has_cuda:
        return "cuda"
    if has_mps:
        return "mps"
    if ANALYSIS_DEVICE == "cuda":
        raise RuntimeError(
            "ANALYSIS_DEVICE=cuda was requested, but CUDA is unavailable. "
            "Use the NVIDIA Compose override on a Linux host with the NVIDIA Container Toolkit."
        )
    if ANALYSIS_DEVICE == "mps":
        raise RuntimeError(
            "ANALYSIS_DEVICE=mps was requested, but Metal/MPS is unavailable. "
            "Run the worker natively on an Apple Silicon Mac, not inside Docker."
        )
    return "cpu"


def harmonix_model_name(device: str) -> str:
    if HARMONIX_MODEL:
        return HARMONIX_MODEL
    # The eight-fold ensemble can exceed M2 unified-memory headroom on Metal.
    # One Harmonix fold still provides its learned beat/section/embedding
    # features, while reliably leaving room for the audio pipeline.
    return "harmonix-fold0" if device == "mps" else "harmonix-all"


def harmonix_proxy_stems(samples: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Create inexpensive, frequency-aware pseudo-stems for Harmonix.

    Instead of spending several minutes running HTDemucs on the entire file,
    this pass uses harmonic/percussive separation and two light filters.  It
    retains the transient, bass, harmonic, and vocal-band trends required for
    beat grids, phrase boundaries, section energy, and embeddings, but does
    not pretend to be a source-separation result suitable for export.

    The model was trained at 44.1 kHz, so the reduced-rate analysis signal is
    resampled back to that clock only for its 100 fps spectrogram input. This
    keeps timing exact while intentionally discarding detail above the proxy
    Nyquist frequency.
    """
    if samples.size == 0 or sr <= 0:
        raise ValueError("Cannot build a Harmonix proxy from empty audio")
    from scipy.signal import butter, sosfilt  # type: ignore

    source = np.asarray(samples, dtype=np.float32)
    if sr != HARMONIX_PROXY_SAMPLE_RATE:
        source = librosa.resample(
            source, orig_sr=sr, target_sr=HARMONIX_PROXY_SAMPLE_RATE, res_type="soxr_hq",
        ).astype(np.float32)
    proxy_sr = HARMONIX_PROXY_SAMPLE_RATE
    harmonic, percussive = librosa.effects.hpss(source, kernel_size=31, margin=1.0)
    nyquist = proxy_sr / 2
    bass_filter = butter(4, min(260.0, nyquist * 0.92), btype="lowpass", fs=proxy_sr, output="sos")
    vocal_high = min(5_200.0, nyquist * 0.92)
    vocal_filter = butter(3, [max(120.0, nyquist * 0.02), vocal_high], btype="bandpass", fs=proxy_sr, output="sos")
    bass = sosfilt(bass_filter, harmonic).astype(np.float32)
    vocals = sosfilt(vocal_filter, harmonic).astype(np.float32)
    other = (harmonic - bass).astype(np.float32)

    def to_model_pcm(stem: np.ndarray) -> np.ndarray:
        if proxy_sr != HARMONIX_MODEL_SAMPLE_RATE:
            stem = librosa.resample(
                stem, orig_sr=proxy_sr, target_sr=HARMONIX_MODEL_SAMPLE_RATE, res_type="soxr_hq",
            )
        # all-in-one's normal Demucs path feeds signed 16-bit mono data to
        # madmom. Match that representation without writing four temporary
        # multi-minute wav files to disk.
        return np.rint(np.clip(stem, -1.0, 1.0) * 32767).astype(np.int16)

    return {
        "bass": to_model_pcm(bass),
        "drums": to_model_pcm(percussive),
        "other": to_model_pcm(other),
        "vocals": to_model_pcm(vocals),
    }


def cached_harmonix_model(device: str) -> Any:
    """Load each Harmonix checkpoint once for the lifetime of this worker."""
    model_name = harmonix_model_name(device)
    cache_key = (model_name, device)
    with HARMONIX_MODELS_LOCK:
        model = HARMONIX_MODELS.get(cache_key)
        if model is None:
            from allin1_infer.models import load_pretrained_model  # type: ignore
            model = load_pretrained_model(model_name=model_name, device=device)
            HARMONIX_MODELS[cache_key] = model
    return model


def harmonix_fast(path: Path, samples: np.ndarray, sr: int, device: str) -> dict[str, Any]:
    """Run the learned Harmonix heads over fast proxy stems, not HTDemucs."""
    from allin1_infer.helpers import run_inference  # type: ignore
    from allin1_infer.spectrogram import compute_spectrogram_from_stem_arrays  # type: ignore
    import torch  # type: ignore

    proxy_stems = harmonix_proxy_stems(samples, sr)
    # run_inference accepts a numpy file so it can keep all package
    # postprocessing semantics. The spectrogram is tiny compared with the
    # original audio and is discarded as soon as this track finishes.
    with tempfile.TemporaryDirectory(prefix="dj-harmonix-proxy-") as directory:
        spec_path = Path(directory) / "proxy.npy"
        spectrogram = compute_spectrogram_from_stem_arrays(proxy_stems, HARMONIX_MODEL_SAMPLE_RATE)
        np.save(spec_path, spectrogram)
        del proxy_stems, spectrogram
        model = cached_harmonix_model(device)
        # One worker can receive a cancellation request while an analysis is
        # running. Serialising actual model forwards also keeps Metal's unified
        # memory stable if an external caller opens another worker request.
        with HARMONIX_INFERENCE_LOCK:
            # allin1_infer.analyze() normally owns this inference-mode guard.
            # We call its lower-level runner to keep the checkpoint cached.
            with torch.inference_mode():
                result = run_inference(
                    path=path, spec_path=spec_path, model=model, device=device,
                    include_activations=False, include_embeddings=True,
                )
        # Metal's caching allocator otherwise keeps a previous track's
        # spectrogram-sized temporary buffers around, which can turn a long
        # preparation into avoidable unified-memory pressure on M-series Macs.
        if device == "mps":
            torch.mps.empty_cache()
    return {
        "bpm": float(getattr(result, "bpm", 0.0) or 0.0),
        "segments": [
            {"start": float(segment.start), "end": float(segment.end), "label": str(segment.label)}
            for segment in getattr(result, "segments", [])
        ],
        "beats": [float(item) for item in getattr(result, "beats", [])],
        "downbeats": [float(item) for item in getattr(result, "downbeats", [])],
        "embedding": np.asarray(getattr(result, "embeddings", [])),
    }


def harmonix(path: Path, samples: np.ndarray, sr: int) -> dict[str, Any] | None:
    if ANALYSIS_MODEL == "librosa":
        return None
    try:
        import allin1_infer as allin1  # type: ignore
        device = harmonix_device()
        if HARMONIX_PROFILE not in {"fast", "full"}:
            raise RuntimeError("HARMONIX_PROFILE must be fast or full")
        if HARMONIX_PROFILE == "fast":
            return harmonix_fast(path, samples, sr, device)
        # The API worker already processes tracks one at a time. Disable the
        # model package's optional multiprocessing pool: forking it from a
        # long-lived Uvicorn process can cause the server itself to exit after
        # a track finishes (or fails) its child work.
        options: dict[str, Any] = {
            "model": harmonix_model_name(device),
            "device": device,
            "include_embeddings": True,
            "multiprocess": False,
        }
        if device == "mps":
            # HTDemucs has a large Conv1d operation that current PyTorch MPS
            # cannot execute. Keep source separation on CPU (with the full
            # four-stem result) and place the actual Harmonix structure and
            # embedding ensemble on Metal. This is materially faster than the
            # old all-CPU worker without sacrificing model input quality.
            from allin1_infer.stems import DemucsProvider  # type: ignore
            options["stem_provider"] = DemucsProvider(device="cpu")
        result = allin1.analyze(
            str(path), **options,
        )
        segments = [
            {"start": float(segment.start), "end": float(segment.end), "label": str(segment.label)}
            for segment in getattr(result, "segments", [])
        ]
        return {
            "bpm": float(getattr(result, "bpm", 0.0) or 0.0),
            "segments": segments,
            "beats": [float(item) for item in getattr(result, "beats", [])],
            "downbeats": [float(item) for item in getattr(result, "downbeats", [])],
            "embedding": np.asarray(getattr(result, "embeddings", [])),
        }
    except Exception as problem:
        if ANALYSIS_MODEL in {"harmonix", "allin1"}:
            raise RuntimeError(f"Harmonix analysis failed: {problem}") from problem
        return None


def fallback_segments(
    duration: float, rms: np.ndarray, hop_length: int, sr: int, beats: np.ndarray,
) -> list[dict[str, Any]]:
    """Create musically useful coarse sections when the structure model is unavailable.

    Eight bars is long enough not to mistake a drum fill for a section, while
    still giving the transition planner several phrase boundaries per track.
    """
    beat_length = float(np.median(np.diff(beats))) if beats.size > 2 else 60.0 / 120.0
    step = max(8.0, min(32.0, beat_length * 32))  # eight bars / 32 beats
    raw_segments = []
    for index, start in enumerate(np.arange(0, max(duration, 1), step)):
        end = min(duration, float(start + step))
        frame_start = int(start * sr / hop_length)
        frame_end = max(frame_start + 1, int(end * sr / hop_length))
        energy = float(np.mean(rms[frame_start:frame_end])) if rms.size else 0.0
        raw_segments.append({"start": float(start), "end": end, "label": "section", "energy": energy})
    energies = np.asarray([segment["energy"] for segment in raw_segments], dtype=float)
    low, high = (np.quantile(energies, [0.30, 0.72]) if energies.size else (0.0, 0.0))
    for index, segment in enumerate(raw_segments):
        if index == 0:
            segment["label"] = "intro"
        elif index == len(raw_segments) - 1:
            segment["label"] = "outro"
        elif segment["energy"] <= low:
            segment["label"] = "breakdown"
        elif segment["energy"] >= high:
            segment["label"] = "chorus"
        else:
            segment["label"] = "verse"
    return raw_segments


def with_segment_energy(segments: list[dict[str, Any]], rms: np.ndarray, hop_length: int, sr: int) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for segment in segments:
        start, end = float(segment["start"]), float(segment["end"])
        frame_start = int(start * sr / hop_length)
        frame_end = max(frame_start + 1, int(end * sr / hop_length))
        item = dict(segment)
        item["energy"] = float(np.mean(rms[frame_start:frame_end])) if rms.size else 0.0
        annotated.append(item)
    reference = float(np.percentile([item["energy"] for item in annotated], 90)) if annotated else 0.0
    if reference > 0:
        for item in annotated:
            item["energy"] = round(float(np.clip(item["energy"] / reference, 0, 1)), 3)
    return annotated


def approximate_downbeats(beats: np.ndarray) -> np.ndarray:
    """Fallback to a four-beat grid when the model did not expose downbeats."""
    return beats[::4] if beats.size else beats


def snap_to_grid(point: float, grid: np.ndarray) -> float:
    return float(grid[np.argmin(np.abs(grid - point))]) if grid.size else point


def beat_safe_points(
    beats: np.ndarray, downbeats: np.ndarray, duration: float, segments: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    """Return phrase boundaries on downbeats, never arbitrary waveform points."""
    boundaries = [float(segment["start"]) for segment in segments[1:]] + [float(segment["end"]) for segment in segments[:-1]]
    phrase_grid = downbeats if downbeats.size else (beats if beats.size else np.asarray(boundaries))
    snapped = [snap_to_grid(point, phrase_grid) for point in boundaries]
    entries = sorted({round(point, 2) for point in snapped if 3 < point < duration - 15})
    exits = sorted({round(point, 2) for point in snapped if 15 < point < duration - 3})
    if not entries:
        entries = [min(round(duration * 0.12, 2), 12.0)]
    if not exits:
        exits = [max(round(duration * 0.75, 2), 5.0)]
    return entries, exits


def waveform_points(rms: np.ndarray, count: int = 512) -> list[float]:
    """Create the compact overview waveform for the player UI."""
    if not rms.size:
        return [0.0] * count
    chunks = np.array_split(rms, count)
    points = np.asarray([float(np.max(chunk)) if chunk.size else 0.0 for chunk in chunks])
    peak = float(points.max())
    return [round(float(point / peak), 3) if peak else 0.0 for point in points]


def detailed_waveform(rms: np.ndarray, duration: float, bpm: float, detected_beats: int) -> str:
    """Encode a full-resolution signal with at least 16 samples per beat.

    A byte per sample keeps the preparation document small enough for large
    crates, unlike storing thousands of BSON doubles for every track.
    """
    estimated_beats = duration * max(bpm, 60.0) / 60.0
    point_count = max(512, math.ceil(max(float(detected_beats), estimated_beats) * 16))
    values = np.asarray(waveform_points(rms, point_count), dtype=np.float32)
    encoded = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8).tobytes()
    return base64.b64encode(encoded).decode("ascii")


def integrated_loudness(path: Path, samples: np.ndarray) -> float:
    """Measure programme loudness locally, with a deterministic audio fallback.

    FFmpeg's EBU R128 meter is the same style of whole-track measurement used
    for broadcast/streaming loudness workflows. The fallback keeps preparation
    working with unusual decoders while still giving the DJ gain stage a useful
    RMS-derived estimate.
    """
    try:
        measured = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-filter:a", "ebur128=peak=true", "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False,
        )
        matches = re.findall(r"\bI:\s*(-?(?:\d+(?:\.\d+)?|inf))\s*LUFS", measured.stderr)
        if matches:
            value = float(matches[-1])
            if math.isfinite(value):
                return round(value, 2)
    except (OSError, ValueError):
        pass
    mean_square = float(np.mean(np.square(samples))) if samples.size else 0.0
    return round(-0.691 + 10 * math.log10(max(mean_square, 1e-12)), 2)


def gain_trim_db(track: dict[str, Any]) -> float:
    """The virtual gain knob: match a track to the deck target, never wildly."""
    try:
        loudness = float(track.get("loudnessLufs"))
    except (TypeError, ValueError):
        return 0.0  # Preparations created before loudness analysis stay playable.
    if not math.isfinite(loudness):
        return 0.0
    return round(float(np.clip(DJ_TARGET_LUFS - loudness, -DJ_MAX_TRIM_DB, DJ_MAX_TRIM_DB)), 2)


def ensure_track_loudness(track: dict[str, Any]) -> None:
    """Backfill loudness when a mix uses a preparation made before this feature."""
    try:
        if math.isfinite(float(track.get("loudnessLufs"))):
            return
    except (TypeError, ValueError):
        pass
    source = (MUSIC_ROOT / str(track.get("relativePath", ""))).resolve()
    if MUSIC_ROOT not in source.parents or not source.exists():
        return
    track["loudnessLufs"] = integrated_loudness(source, np.empty(0, dtype=np.float32))


def analyse_track(path: Path, preparation_id: str) -> dict[str, Any]:
    track_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{preparation_id}:{path.relative_to(MUSIC_ROOT)}"))
    metadata, tag_genres, artwork_url = tags_for(path, track_id)
    y, sr = librosa.load(path, sr=ANALYSIS_SAMPLE_RATE, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    hop = 512
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, trim=False)
    bpm = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 0.0
    if bpm < 55 and bpm > 0:
        bpm *= 2
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24, hop_length=hop)
    spectral = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    deep = harmonix(path, y, sr)
    if deep and deep["bpm"] > 0:
        bpm = deep["bpm"]
    if deep and deep.get("beats"):
        beat_times = np.asarray(deep["beats"], dtype=float)
    downbeats = np.asarray(deep.get("downbeats", []) if deep else [], dtype=float)
    if not downbeats.size:
        downbeats = approximate_downbeats(beat_times)
    raw_segments = deep["segments"] if deep and deep["segments"] else fallback_segments(duration, rms, hop, sr, beat_times)
    segments = with_segment_energy(raw_segments, rms, hop, sr)
    entries, exits = beat_safe_points(beat_times, downbeats, duration, segments)
    energy = float(np.clip(np.percentile(rms, 75) * 4.0, 0, 1)) if rms.size else 0.0
    loudness_lufs = integrated_loudness(path, y)
    genres = tag_genres or heuristic_genres(bpm, float(spectral.mean()) if spectral.size else 0, energy)
    vector = project_embedding(deep.get("embedding") if deep else None, mfcc, chroma, spectral)
    indexed = index_vector(track_id, vector, {
        "preparationId": preparation_id, "title": metadata["title"], "artist": metadata["artist"],
        "genres": genres, "bpm": bpm, "key": estimate_key(chroma), "energy": energy, "loudnessLufs": loudness_lufs,
    })
    first_drop = next((segment["start"] for segment in segments if segment["label"] in {"chorus", "drop"}), None)
    return {
        "id": track_id,
        "relativePath": path.relative_to(MUSIC_ROOT).as_posix(),
        "title": metadata["title"], "artist": metadata["artist"], "album": metadata["album"],
        "artworkUrl": artwork_url, "durationSeconds": round(duration, 2), "bpm": round(bpm, 2),
        "key": estimate_key(chroma), "energy": round(energy, 3), "loudnessLufs": loudness_lufs,
        "waveform": waveform_points(rms), "waveformDetail": detailed_waveform(rms, duration, bpm, len(beat_times)),
        "beatGrid": [round(float(point), 3) for point in beat_times], "downbeats": [round(float(point), 3) for point in downbeats],
        "genres": genres, "segments": segments,
        "cues": {"introEnd": entries[0], "firstDrop": first_drop, "safeEntries": entries, "safeExits": exits},
        "embeddingIndexed": indexed,
    }


@app.get("/health")
def health() -> dict[str, str]:
    try:
        device = harmonix_device() if ANALYSIS_MODEL != "librosa" else "cpu"
    except RuntimeError:
        device = "unavailable"
    return {
        "status": "ok",
        "engine": "local",
        "analysisDevice": device,
        "analysisProfile": HARMONIX_PROFILE if ANALYSIS_MODEL != "librosa" else "librosa",
    }


@app.post("/analyze")
def analyze(request: dict[str, Any]) -> dict[str, Any]:
    preparation_id = str(request["preparationId"])
    ensure_job_active("preparation", preparation_id)
    source = resolve_music_source(str(request.get("sourcePath", "/music")))
    if not source.exists():
        raise HTTPException(400, "The mounted music folder does not exist")
    paths = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)
    if not paths:
        raise HTTPException(422, "No supported music files were found in the selected folder")
    tracks, failures = [], []
    total = len(paths)
    report_preparation_progress(
        preparation_id, progress=8, discovered_track_count=total,
        analysed_track_count=0, failed_track_count=0, current_track=None,
        message=f"Found {total} track{'s' if total != 1 else ''}. Starting local analysis.",
    )
    for path in paths:
        processed = len(tracks) + len(failures)
        report_preparation_progress(
            preparation_id, progress=8 + round(processed / total * 87),
            discovered_track_count=total, analysed_track_count=len(tracks),
            failed_track_count=len(failures), current_track=path.name,
            message=f"Analysing track {processed + 1} of {total}",
        )
        try:
            ensure_job_active("preparation", preparation_id)
            tracks.append(analyse_track(path, preparation_id))
        except HTTPException:
            raise
        except Exception as problem:
            failures.append(f"{path.name}: {problem}")
        processed = len(tracks) + len(failures)
        report_preparation_progress(
            preparation_id, progress=8 + round(processed / total * 87),
            discovered_track_count=total, analysed_track_count=len(tracks),
            failed_track_count=len(failures), current_track=path.name,
            message=(f"Analysed {processed} of {total}" if len(failures) == 0 else f"Analysed {len(tracks)} of {total}; {len(failures)} skipped"),
        )
    ensure_job_active("preparation", preparation_id)
    if not tracks:
        raise HTTPException(422, "No audio files could be decoded. Ensure FFmpeg supports the library format.")
    genres = sorted({genre for track in tracks for genre in track["genres"]})
    report_preparation_progress(
        preparation_id, progress=97, discovered_track_count=total,
        analysed_track_count=len(tracks), failed_track_count=len(failures), current_track=None,
        message="Finalising music features and mix data",
    )
    return {
        "tracks": tracks, "genres": genres,
        "modelReport": {
            "structureModel": (
                "Harmonix fast spectral proxy via all-in-one-infer (with librosa fallback)"
                if ANALYSIS_MODEL != "librosa" and HARMONIX_PROFILE == "fast"
                else "Harmonix full source-separation pass via all-in-one-infer (with librosa fallback)"
                if ANALYSIS_MODEL != "librosa"
                else "librosa beat/structure fallback"
            ),
            "embeddingModel": "Harmonix embeddings projected to 512D (MFCC fallback)",
            "genreSource": "embedded metadata plus local acoustic classifier fallback",
            "qdrantCollection": VECTOR_COLLECTION,
        },
        "failures": failures,
    }


@app.post("/cancel-job")
def cancel_job(request: dict[str, Any]) -> dict[str, str]:
    job_type = str(request.get("jobType", ""))
    job_id = str(request.get("jobId", ""))
    if job_type not in {"preparation", "mix"} or not job_id:
        raise HTTPException(400, "jobType and jobId are required")
    with CANCELLED_JOBS_LOCK:
        CANCELLED_JOBS.add((job_type, job_id))
    return {"status": "cancelling", "jobType": job_type, "jobId": job_id}


@app.post("/cleanup-preparation")
def cleanup_preparation(request: dict[str, Any]) -> dict[str, Any]:
    """Remove only derived, local data for one previously analysed library."""
    preparation_id = str(request.get("preparationId", ""))
    if not preparation_id:
        raise HTTPException(400, "preparationId is required")

    qdrant_request("POST", f"/collections/{VECTOR_COLLECTION}/points/delete?wait=true", {
        "filter": {"must": [{"key": "preparationId", "match": {"value": preparation_id}}]},
    })

    deleted_artwork = 0
    artwork_dir = DATA_ROOT / "artwork"
    for track_id in request.get("trackIds", []):
        try:
            safe_id = str(uuid.UUID(str(track_id)))
        except (TypeError, ValueError, AttributeError):
            continue
        for suffix in (".jpg", ".png"):
            artwork = artwork_dir / f"{safe_id}{suffix}"
            try:
                artwork.unlink()
                deleted_artwork += 1
            except FileNotFoundError:
                pass
    return {"status": "ok", "deletedArtwork": deleted_artwork}


def key_score(first: str, second: str) -> float:
    if "Unknown" in {first, second}:
        return 0.55
    if first == second:
        return 1.0
    first_root, second_root = first.rstrip("m"), second.rstrip("m")
    try:
        distance = min((KEYS.index(first_root) - KEYS.index(second_root)) % 12, (KEYS.index(second_root) - KEYS.index(first_root)) % 12)
    except ValueError:
        return 0.45
    if first_root == second_root:
        return 0.88
    if distance in {5, 7}:
        return 0.86
    if distance in {2, 10}:
        return 0.67
    return 0.28


def vector_score(first_id: str, second_id: str) -> float:
    first, second = get_vector(first_id), get_vector(second_id)
    if first is None or second is None:
        return 0.5
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator else 0.5


PROTECTED_SECTIONS = {"chorus", "drop", "hook", "solo", "peak"}
PREFERRED_ENTRY_SECTIONS = {"intro", "breakdown", "verse", "start"}
BUILD_SECTION_LABELS = {"build", "build-up", "buildup", "pre-chorus", "prechorus", "rise"}
SENSITIVE_HANDOFF_SECTIONS = BUILD_SECTION_LABELS | {"breakdown", "bridge"}


def beat_duration(track: dict[str, Any], fallback_bpm: float | None = None) -> float:
    beats = np.asarray(track.get("beatGrid", []), dtype=float)
    if beats.size > 2:
        return float(np.median(np.diff(beats)))
    bpm = fallback_bpm or float(track.get("bpm", 120.0))
    return 60.0 / max(bpm, 1.0)


def segment_at(track: dict[str, Any], point: float) -> dict[str, Any] | None:
    for segment in track.get("segments", []):
        if float(segment["start"]) <= point < float(segment["end"]):
            return segment
    return track.get("segments", [])[-1] if track.get("segments") else None


def section_energy(track: dict[str, Any], point: float) -> float:
    segment = segment_at(track, point)
    return float(segment.get("energy", track.get("energy", 0.5))) if segment else float(track.get("energy", 0.5))


def section_label(segment: dict[str, Any] | None) -> str:
    return str((segment or {}).get("label", "")).lower().replace("_", " ").strip()


def is_build_section(segment: dict[str, Any] | None) -> bool:
    return section_label(segment) in BUILD_SECTION_LABELS


def is_protected_section(segment: dict[str, Any] | None) -> bool:
    if not segment:
        return False
    return str(segment.get("label", "")).lower() in PROTECTED_SECTIONS or float(segment.get("energy", 0.0)) >= 0.84


def safe_exit_for(track: dict[str, Any], source_start: float, minimum: float, maximum: float) -> float | None:
    """Choose a downbeat at a phrase end, protecting the middle of high-energy sections."""
    natural_end = float(track["durationSeconds"]) - 0.25
    remaining = natural_end - source_start
    if remaining <= 0.5:
        return None
    # The deck limit belongs to the playable part of this track, not an
    # assumed duration. This matters when a selected max is longer than the
    # song, or when an incoming phrase starts a few seconds into it.
    effective_minimum = min(minimum, remaining)
    lower = source_start + effective_minimum
    upper = min(natural_end, source_start + maximum)
    if upper < lower:
        return None
    safe_exits = [float(point) for point in track["cues"].get("safeExits", []) if lower <= float(point) <= upper]
    downbeats = [float(point) for point in track.get("downbeats", []) if lower <= float(point) <= upper]
    candidates = sorted({round(point, 3) for point in safe_exits + downbeats})
    target = (lower + upper) / 2
    beat = beat_duration(track)

    def penalty(point: float) -> float:
        before = segment_at(track, max(source_start, point - beat / 2))
        after = segment_at(track, min(track["durationSeconds"] - 0.01, point + beat / 2))
        value = abs(point - target)
        # Never abandon a chorus/drop halfway through. Its final phrase boundary
        # is allowed, but gets a small penalty so a breakdown/outro wins first.
        if is_protected_section(before):
            if point < float(before["end"]) - beat * 1.1:
                return 10_000.0
            value += 9.0
        if is_protected_section(after):
            value += 2.5
        if is_build_section(before):
            if point < float(before["end"]) - beat * 1.1:
                return 10_000.0
            value += 15.0
        if section_label(before) in {"outro", "breakdown", "bridge"}:
            value -= 5.0
        return value

    viable = [point for point in candidates if penalty(point) < 10_000]
    if viable:
        return min(viable, key=penalty)
    # Existing preparations may predate beat-grid storage; retain their safe
    # cues instead of choosing an arbitrary point in a high-energy section.
    if safe_exits:
        return min(safe_exits, key=lambda point: abs(point - target))
    # A feature-poor/old preparation may not have a usable beat grid. If the
    # requested maximum reaches beyond the file, completing the track is more
    # natural than failing the entire mix or cutting an arbitrary middle point.
    if upper >= natural_end - 0.5:
        return round(natural_end, 3)
    return None


def safe_entry_for(first: dict[str, Any], second: dict[str, Any], outgoing_exit: float, minimum_remaining: float) -> float | None:
    """Pair the exit with the incoming phrase whose energy and role fit it best."""
    latest_entry = float(second["durationSeconds"]) - minimum_remaining - 0.25
    entries = [float(point) for point in second["cues"].get("safeEntries", []) if 0 < float(point) <= latest_entry]
    entries += [float(point) for point in second.get("downbeats", []) if 0 < float(point) <= latest_entry]
    candidates = sorted({round(point, 3) for point in entries})
    # Starting on the first sample is a valid phrase boundary for shorter
    # songs. Prefer detected entries, but retain this option when they would
    # leave too little music to meet the requested minimum deck time.
    if not candidates and latest_entry >= 0:
        candidates = [0.0]
    if not candidates:
        return None
    outgoing_energy = section_energy(first, max(0.0, outgoing_exit - beat_duration(first)))
    outgoing_section = segment_at(first, max(0.0, outgoing_exit - beat_duration(first)))

    def penalty(point: float) -> float:
        section = segment_at(second, point + beat_duration(second) / 2)
        incoming_energy = section_energy(second, point + beat_duration(second) / 2)
        label = section_label(section)
        # High energy exits should be allowed to enter a later chorus/drop;
        # low-energy exits still favour the traditional intro/breakdown route.
        value = abs(incoming_energy - outgoing_energy) * 44
        if label in PREFERRED_ENTRY_SECTIONS and outgoing_energy < 0.58:
            value -= 6
        elif label in {"intro", "start"} and outgoing_energy >= 0.72:
            value += 8
        if is_protected_section(section) and outgoing_energy < 0.72:
            value += 20
        if is_build_section(section):
            value += 16
        if is_build_section(outgoing_section) and incoming_energy >= 0.70:
            value += 4
        return value + point / 3200  # prefer an earlier equivalent phase only

    return min(candidates, key=penalty)


def transition_between(
    first: dict[str, Any],
    second: dict[str, Any],
    source_start: float = 0.0,
    minimum: float = 2.0,
    maximum: float = 14.0,
    running_bpm: float | None = None,
) -> dict[str, Any] | None:
    exit_at = safe_exit_for(first, source_start, minimum, maximum)
    if exit_at is None:
        return None
    entry_at = safe_entry_for(first, second, exit_at, minimum)
    if entry_at is None:
        return None
    source_bpm = float(running_bpm or first["bpm"] or 120.0)
    tempo_ratio = second["bpm"] / source_bpm if source_bpm else 1.0
    requested_factor = source_bpm / max(float(second["bpm"]), 1.0)
    # Exact beat overlays are only used within a DJ-safe ±8% time-stretch. At
    # larger gaps we use a short filtered phrase hand-off rather than lay two
    # drifting beat grids on top of each other.
    beat_matched = 0.92 <= requested_factor <= 1.08
    tempo_factor = requested_factor if beat_matched else 1.0
    tempo_score = max(0.0, 1 - min(abs(tempo_ratio - 1), 0.25) / 0.25)
    harmonic = key_score(first["key"], second["key"])
    outgoing_energy = section_energy(first, max(source_start, exit_at - beat_duration(first, source_bpm)))
    incoming_energy = section_energy(second, entry_at + beat_duration(second) / 2)
    outgoing_section = segment_at(first, max(source_start, exit_at - beat_duration(first, source_bpm)))
    incoming_section = segment_at(second, entry_at + beat_duration(second) / 2)
    energy_delta = abs(outgoing_energy - incoming_energy)
    energy_score = max(0.0, 1 - energy_delta / 0.8)
    shared_genre = set(first["genres"]) & set(second["genres"])
    genre_score = 1.0 if shared_genre else 0.55
    similarity = vector_score(first["id"], second["id"])
    phase_score = 1.0 if beat_matched else 0.25
    score = 100 * (0.24 * harmonic + 0.20 * tempo_score + 0.23 * energy_score + 0.12 * genre_score + 0.13 * similarity + 0.08 * phase_score)
    beat = 60.0 / source_bpm
    desired_bars = 8 if score >= 79 and energy_delta < 0.18 else 4
    guard_outgoing = section_label(outgoing_section) in SENSITIVE_HANDOFF_SECTIONS and incoming_energy >= 0.70
    if guard_outgoing:
        desired_bars = 2
        fade_shape = "fast-outgoing"
    elif incoming_energy > outgoing_energy + 0.18:
        fade_shape = "incoming-lift"
    elif outgoing_energy > incoming_energy + 0.22:
        fade_shape = "long-release"
    else:
        fade_shape = "equal-power"
    overlap_bars = 0
    overlap = min(1.25, max(0.55, beat * 2))
    if beat_matched:
        for bars in (desired_bars, 4, 2):
            candidate = bars * 4 * beat
            if candidate <= exit_at - source_start - beat * 2 and entry_at + candidate * tempo_factor < second["durationSeconds"] - beat:
                overlap_bars, overlap = bars, candidate
                break
        if not overlap_bars:
            beat_matched = False
            tempo_factor = 1.0
    protected_exit = is_protected_section(segment_at(first, exit_at - beat / 2))
    loop_bars = 1 if beat_matched and score < 66 and not protected_exit else 0
    loop_seconds = loop_bars * 4 * beat
    if beat_matched:
        style = "beat-aligned phrase blend"
    elif score >= 55:
        style = "filtered phrase hand-off"
    else:
        style = "echo phrase cut"
    notes = [
        f"harmonic {harmonic:.0%}", f"section-energy match {energy_score:.0%}",
        f"embedding affinity {similarity:.0%}", "high-energy sections protected", f"{fade_shape} fade curve",
    ]
    if beat_matched:
        notes.extend([f"pitch-preserving tempo {(tempo_factor - 1):+.1%}", f"{overlap_bars}-bar downbeat overlay"])
    else:
        notes.append("tempo gap too large for an overlapping beat grid")
    if loop_bars:
        notes.append(f"{loop_bars}-bar phrase loop before the hand-off")
    if guard_outgoing:
        notes.append("outgoing build/breakdown is removed early before the incoming high-energy phrase")
    return {
        "fromTrackId": first["id"], "toTrackId": second["id"], "exitAtSeconds": round(exit_at, 2),
        "enterAtSeconds": round(entry_at, 2), "overlapSeconds": overlap, "bpmRatio": round(tempo_ratio, 4),
        "tempoFactor": round(tempo_factor, 4), "loopSeconds": round(loop_seconds, 3), "beatMatched": beat_matched,
        "overlapBars": overlap_bars, "fadeShape": fade_shape, "spectrumPlan": "progressive low-pass outgoing / bass-release incoming", "style": style,
        "qualityScore": round(score, 1), "renderQualityScore": None, "notes": notes,
    }


def ordered_tracks(tracks: list[dict[str, Any]], genre_order: list[str]) -> list[dict[str, Any]]:
    order = {normalise_genre(genre): index for index, genre in enumerate(genre_order)}
    return sorted(
        tracks,
        key=lambda track: (min((order.get(normalise_genre(genre), len(order)) for genre in track["genres"]), default=len(order)), track["energy"], track["title"].lower()),
    )


def clip_for(track: dict[str, Any], source_start: float, source_end: float) -> AudioSegment:
    source = (MUSIC_ROOT / track["relativePath"]).resolve()
    if MUSIC_ROOT not in source.parents or not source.exists():
        raise ValueError("track is outside the mounted music library")
    audio = AudioSegment.from_file(source)
    start_ms, end_ms = int(source_start * 1000), int(source_end * 1000)
    if end_ms <= start_ms:
        raise ValueError("planned clip has no playable duration")
    return audio[start_ms:end_ms].fade_in(180).fade_out(220)


def pitch_preserving_time_stretch(audio: AudioSegment, factor: float) -> AudioSegment:
    """Change tempo with FFmpeg's time-scale filter without moving musical pitch.

    Altering AudioSegment.frame_rate changes both BPM and pitch. `atempo`
    performs time-domain stretching instead, so a tempo correction leaves
    voices, keys and spectral content at their original frequencies.
    """
    if abs(factor - 1.0) < 0.002:
        return audio
    if not 0.5 <= factor <= 2.0:
        raise ValueError("pitch-preserving tempo factor is outside FFmpeg's safe atempo range")
    with tempfile.TemporaryDirectory(prefix="dj-tempo-") as directory:
        source = Path(directory) / "source.wav"
        output = Path(directory) / "stretched.wav"
        audio.export(source, format="wav")
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
             "-filter:a", f"atempo={factor:.8f}", str(output)],
            check=True,
        )
        return AudioSegment.from_file(output, format="wav")


def finite_dbfs(audio: AudioSegment) -> float:
    value = float(audio.dBFS)
    return value if math.isfinite(value) else -80.0


def master_bus(audio: AudioSegment) -> AudioSegment:
    """Apply a conservative final gain and true-peak ceiling to the whole mix."""
    if not len(audio):
        return audio
    # Track trims get every deck close to the same perceived level. The master
    # bus then establishes a predictable playback level without undoing that
    # relationship. Bound its move so quiet passages are not aggressively lifted.
    master_gain = float(np.clip(MASTER_TARGET_DBFS - finite_dbfs(audio), -4.0, 4.0))
    mastered = audio.apply_gain(master_gain)
    with tempfile.TemporaryDirectory(prefix="dj-master-") as directory:
        source = Path(directory) / "mix.wav"
        output = Path(directory) / "mastered.wav"
        mastered.export(source, format="wav")
        try:
            # A transparent look-ahead limiter catches only summed overlap
            # peaks. It does not replace the per-track DJ gain staging.
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
                 "-filter:a", "alimiter=limit=0.891:attack=5:release=50", str(output)],
                check=True,
            )
            mastered = AudioSegment.from_file(output, format="wav")
        except (OSError, subprocess.CalledProcessError):
            pass
    peak = float(mastered.max_dBFS)
    if math.isfinite(peak) and peak > MASTER_CEILING_DBFS:
        mastered = mastered.apply_gain(MASTER_CEILING_DBFS - peak)
    return mastered


def monitor_transition(outgoing: AudioSegment, incoming: AudioSegment, blended: AudioSegment, planned_score: float) -> float:
    """Post-render quality guard: balance and clipping are checked on real audio."""
    level_gap = abs(finite_dbfs(outgoing) - finite_dbfs(incoming))
    peak_penalty = max(0.0, float(blended.max_dBFS) + 0.3) * 35
    acoustic_score = max(0.0, 100 - level_gap * 5 - peak_penalty)
    return round(0.78 * planned_score + 0.22 * acoustic_score, 1)


def fade_gains(progress: float, fade_shape: str) -> tuple[float, float]:
    """Return non-linear equal-power gains for a phrase-aware crossfade."""
    if fade_shape == "fast-outgoing":
        outgoing_progress, incoming_progress = progress ** 0.55, progress ** 1.35
    elif fade_shape == "incoming-lift":
        outgoing_progress, incoming_progress = progress ** 1.18, progress ** 0.72
    elif fade_shape == "long-release":
        outgoing_progress, incoming_progress = progress ** 1.40, progress ** 1.10
    else:
        outgoing_progress = incoming_progress = progress
    outgoing_amplitude = max(0.0003, math.cos(outgoing_progress * math.pi / 2))
    incoming_amplitude = max(0.0003, math.sin(incoming_progress * math.pi / 2))
    return 20 * math.log10(outgoing_amplitude), 20 * math.log10(incoming_amplitude)


def progressive_spectrum_blend(
    outgoing: AudioSegment, incoming: AudioSegment, overlap_ms: int, fade_shape: str,
) -> tuple[AudioSegment, AudioSegment]:
    """Trade spectrum with a musical curve while avoiding a double-kick clash."""
    chunks = min(48, max(12, overlap_ms // 80))
    out_parts: list[AudioSegment] = []
    in_parts: list[AudioSegment] = []
    for index in range(chunks):
        start = index * overlap_ms // chunks
        end = (index + 1) * overlap_ms // chunks
        progress = (index + 1) / chunks
        low_pass_hz = int(18_000 - 13_500 * progress)
        high_pass_hz = int(300 - 250 * progress)
        outgoing_gain, incoming_gain = fade_gains(progress, fade_shape)
        out_parts.append(low_pass_filter(outgoing[start:end], max(3_000, low_pass_hz)).apply_gain(outgoing_gain))
        in_parts.append(high_pass_filter(incoming[start:end], max(45, high_pass_hz)).apply_gain(incoming_gain))
    filtered_outgoing = sum(out_parts, AudioSegment.empty())
    filtered_incoming = sum(in_parts, AudioSegment.empty())
    return filtered_outgoing, filtered_incoming


def loop_phrase(outgoing: AudioSegment, loop_ms: int, overlap_ms: int) -> AudioSegment:
    """Repeat an exact beat-aligned bar, with tiny joins to avoid clicks."""
    if loop_ms < 180 or loop_ms > overlap_ms:
        return outgoing
    phrase = outgoing[-loop_ms:]
    rendered = phrase
    while len(rendered) < overlap_ms:
        rendered = rendered.append(phrase, crossfade=min(18, len(phrase) // 8))
    return rendered[-overlap_ms:]


def blend(master: AudioSegment, incoming: AudioSegment, transition: dict[str, Any]) -> tuple[AudioSegment, float]:
    overlap_ms = min(len(master) // 2, len(incoming) // 2, int(float(transition["overlapSeconds"]) * 1000))
    if overlap_ms < 400:
        raise ValueError("planned transition is too short to render")
    outgoing = master[-overlap_ms:]
    loop_ms = min(overlap_ms, int(float(transition.get("loopSeconds", 0)) * 1000))
    if transition.get("beatMatched", False):
        outgoing = loop_phrase(outgoing, loop_ms, overlap_ms)
    filtered_outgoing, filtered_incoming = progressive_spectrum_blend(
        outgoing, incoming[:overlap_ms], overlap_ms, str(transition.get("fadeShape", "equal-power")),
    )
    bridge = filtered_outgoing.overlay(filtered_incoming)
    transition["renderQualityScore"] = monitor_transition(
        filtered_outgoing, filtered_incoming, bridge, float(transition["qualityScore"])
    )
    return master[:-overlap_ms] + bridge + incoming[overlap_ms:], overlap_ms / 1000


def render_mix(mix: dict[str, Any], playlist: list[dict[str, Any]]) -> tuple[str, float]:
    if not playlist:
        raise ValueError("mix has no playable tracks")
    mix_id = str(mix["id"])
    master: AudioSegment | None = None
    timeline_end = 0.0
    for item in playlist:
        ensure_job_active("mix", mix_id)
        transition = item.get("transitionIn")
        clip = clip_for(item["track"], item["sourceStartSeconds"], item["sourceEndSeconds"])
        trim_db = float(item.get("gainDb", gain_trim_db(item["track"])))
        clip = clip.apply_gain(trim_db)
        if transition:
            clip = pitch_preserving_time_stretch(clip, float(transition["tempoFactor"]))
        if master is None:
            master = clip
            item["startSeconds"], item["endSeconds"] = 0.0, round(len(clip) / 1000, 2)
            timeline_end = item["endSeconds"]
            continue
        master, overlap = blend(master, clip, transition)
        start = timeline_end - overlap
        item["startSeconds"], item["endSeconds"] = round(start, 2), round(start + len(clip) / 1000, 2)
        timeline_end = item["endSeconds"]
    if master is None:
        raise ValueError("mix renderer produced no audio")
    master = master_bus(master)
    ensure_job_active("mix", mix_id)
    mixes_dir = DATA_ROOT / "mixes"
    mixes_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{mix['id']}.mp3"
    master.export(mixes_dir / filename, format="mp3", bitrate="192k", parameters=["-ar", "44100"])
    return f"/media/mixes/{filename}", round(len(master) / 1000, 2)


@app.post("/prepare-mix")
def prepare_mix(request: dict[str, Any]) -> dict[str, Any]:
    mix, tracks = request["mix"], request["tracks"]
    mix_id = str(mix["id"])
    ensure_job_active("mix", mix_id)
    options = mix["options"]
    candidates = ordered_tracks(tracks, options["genreOrder"])
    if not candidates:
        raise HTTPException(422, "The preparation does not contain any tracks")
    for candidate in candidates:
        ensure_job_active("mix", mix_id)
        ensure_track_loudness(candidate)
    max_rejected = math.floor(len(candidates) * (100 - options["acceptancePercentage"]) / 100)
    kept, rejected = [], []
    for candidate in candidates:
        ensure_job_active("mix", mix_id)
        if candidate["durationSeconds"] < options["minTrackSeconds"] + 0.25:
            if len(rejected) >= max_rejected:
                raise HTTPException(422, "The selected minimum duration leaves too few playable tracks for this acceptance percentage")
            rejected.append(candidate["id"])
            continue
        if not kept:
            kept.append(candidate)
            continue
        screening = transition_between(kept[-1], candidate, 0.0, options["minTrackSeconds"], options["maxTrackSeconds"])
        if screening is None:
            if len(rejected) < max_rejected:
                rejected.append(candidate["id"])
                continue
            raise HTTPException(422, "A track cannot meet the requested minimum play time after its phrase-safe entry. Lower the minimum or allow more tracks to be left out.")
        score = screening["qualityScore"]
        if score < 54 and len(rejected) < max_rejected:
            rejected.append(candidate["id"])
        else:
            kept.append(candidate)
    if not kept:
        raise HTTPException(422, "No tracks meet the requested minimum duration")
    playlist = [
        {"track": track, "startSeconds": 0.0, "endSeconds": 0.0, "sourceStartSeconds": 0.0,
         "sourceEndSeconds": 0.0, "deckBpm": float(track["bpm"]), "gainDb": gain_trim_db(track), "transitionIn": None}
        for track in kept
    ]
    for index in range(len(playlist) - 1):
        ensure_job_active("mix", mix_id)
        current, upcoming = playlist[index], playlist[index + 1]
        transition = transition_between(
            current["track"], upcoming["track"], current["sourceStartSeconds"],
            options["minTrackSeconds"], options["maxTrackSeconds"], current["deckBpm"],
        )
        if transition is None:
            raise HTTPException(422, "A selected track has no phrase-safe exit within the requested duration range")
        current["sourceEndSeconds"] = transition["exitAtSeconds"]
        upcoming["sourceStartSeconds"] = transition["enterAtSeconds"]
        upcoming["deckBpm"] = round(float(upcoming["track"]["bpm"]) * float(transition["tempoFactor"]), 3)
        upcoming["transitionIn"] = transition
    final_track = playlist[-1]
    final_end = min(
        final_track["track"]["durationSeconds"] - 0.25,
        final_track["sourceStartSeconds"] + options["maxTrackSeconds"],
    )
    if final_end - final_track["sourceStartSeconds"] < options["minTrackSeconds"]:
        raise HTTPException(422, "The final track cannot meet the selected minimum duration")
    final_track["sourceEndSeconds"] = round(final_end, 2)
    audio_url, duration = render_mix(mix, playlist)
    return {
        "playlist": playlist, "rejectedTrackIds": rejected, "durationSeconds": duration, "audioUrl": audio_url,
    }


@app.post("/next-transition")
def next_transition(request: dict[str, Any]) -> dict[str, Any]:
    playlist = request["mix"].get("playlist", [])
    index = int(request["currentTrackIndex"])
    if index < 0 or index >= len(playlist) - 1:
        raise HTTPException(422, "There is no next track available")
    current, upcoming = playlist[index], playlist[index + 1]
    elapsed = max(0.0, float(request.get("playbackSeconds", 0)) - current["startSeconds"])
    incoming_transition = current.get("transitionIn") or {}
    source_position = current.get("sourceStartSeconds", 0.0) + elapsed * float(incoming_transition.get("tempoFactor", 1.0))
    transition = transition_between(
        current["track"], upcoming["track"], source_position, 4.0, 14.0,
        float(current.get("deckBpm") or current["track"]["bpm"]),
    )
    if transition is None:
        raise HTTPException(422, "There is no natural skip point left in this track")
    outgoing = clip_for(current["track"], source_position, transition["exitAtSeconds"]).apply_gain(
        float(current.get("gainDb", gain_trim_db(current["track"])))
    )
    incoming_end = min(upcoming["track"]["durationSeconds"] - 0.25, transition["enterAtSeconds"] + 12)
    incoming = pitch_preserving_time_stretch(
        clip_for(upcoming["track"], transition["enterAtSeconds"], incoming_end).apply_gain(
            float(upcoming.get("gainDb", gain_trim_db(upcoming["track"])))
        ), transition["tempoFactor"],
    )
    bridge, _ = blend(outgoing, incoming, transition)
    skips_dir = DATA_ROOT / "skips"
    skips_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{request['mix']['id']}-{uuid.uuid4()}.mp3"
    bridge.export(skips_dir / filename, format="mp3", bitrate="192k", parameters=["-ar", "44100"])
    resume_after = max(0.0, (incoming_end - transition["enterAtSeconds"]) / transition["tempoFactor"])
    return {
        "transition": transition,
        "audioUrl": f"/media/skips/{filename}",
        "resumeAtSeconds": round(upcoming["startSeconds"] + resume_after, 2),
    }
