from __future__ import annotations

import base64
import hashlib
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
PHRASE_BARS = 8
BEAT_ANCHOR_TOLERANCE_SECONDS = 0.035
# A double kick becomes noticeable well before the previous 45 ms tolerance.
# This is deliberately a hard gate: when a reliable grid cannot maintain this
# residual, a short phrase-boundary hand-off is more musical than a drifting
# overlay.
MAX_BEAT_OVERLAY_RESIDUAL_MS = 22.0
WAVEFORM_SAMPLES_PER_BEAT = 64
WAVEFORM_DETAIL_VERSION = 2
VECTOR_COLLECTION = "track_embeddings"
VECTOR_SIZE = 512
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus"}
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
DJ_TARGET_LUFS = float(os.environ.get("DJ_TARGET_LUFS", "-14"))
DJ_MAX_TRIM_DB = float(os.environ.get("DJ_MAX_TRIM_DB", "8"))
MASTER_TARGET_DBFS = float(os.environ.get("MASTER_TARGET_DBFS", "-14"))
MASTER_CEILING_DBFS = float(os.environ.get("MASTER_CEILING_DBFS", "-1"))
# Keep one extra dB before the final master limiter. Unlike a limiter applied
# after AudioSegment.overlay, this guard runs on the two floating-point deck
# signals before they can saturate their integer PCM container.
TRANSITION_OVERLAY_CEILING_DBFS = min(-1.0, MASTER_CEILING_DBFS - 1.0)
TRANSITION_POLICY_PATH = Path(
    os.environ.get("TRANSITION_POLICY_PATH", str(DATA_ROOT / "models" / "transition-policy-v1.json"))
).resolve()
TRANSITION_FEATURE_VERSION = "dj-attatouille-transition-v1"
TRANSITION_POLICY_FEATURES = [
    "tempo_ratio", "harmonic_compatibility", "outgoing_energy", "incoming_energy", "energy_delta",
    "outgoing_energy_slope", "incoming_energy_slope", "slope_delta", "outgoing_bass", "incoming_bass",
    "bass_collision", "outgoing_drums", "incoming_drums", "outgoing_vocals", "incoming_vocals",
    "vocal_collision", "outgoing_spectral_density", "incoming_spectral_density",
    "outgoing_harmonic_density", "incoming_harmonic_density", "outgoing_novelty", "incoming_novelty",
    "outgoing_loopability", "incoming_cue_confidence",
    "outgoing_elapsed_fraction", "outgoing_remaining_fraction", "incoming_entry_fraction",
    "genre_compatibility", "outgoing_cue_confidence",
]
CANCELLED_JOBS: set[tuple[str, str]] = set()
CANCELLED_JOBS_LOCK = threading.Lock()
# all-in-one-infer normally reconstructs its checkpoint on every API call.
# The worker is deliberately long lived, so retain one model per device/profile
# and avoid repeatedly moving the same Harmonix weights into unified memory.
HARMONIX_MODELS: dict[tuple[str, str], Any] = {}
HARMONIX_MODELS_LOCK = threading.Lock()
HARMONIX_INFERENCE_LOCK = threading.Lock()
TRANSITION_POLICY_CACHE: tuple[int, dict[str, Any] | None] = (-1, None)
TRANSITION_POLICY_LOCK = threading.Lock()
DEFAULT_COMPATIBILITY_PROFILE = {
    "energy": {"weight": 0.19, "targetDelta": 0.10, "scale": 0.24},
    "trajectory": {"weight": 0.14, "targetDelta": 0.12, "scale": 0.34},
    "bass": {"weight": 0.13, "targetDelta": 0.10, "scale": 0.25},
    "drums": {"weight": 0.10, "targetDelta": 0.10, "scale": 0.27},
    "vocals": {"weight": 0.13, "targetDelta": 0.20, "scale": 0.30},
    "spectral": {"weight": 0.11, "targetDelta": 0.12, "scale": 0.28},
    "harmonic": {"weight": 0.07, "targetDelta": 0.10, "scale": 0.30},
    "novelty": {"weight": 0.13, "targetDelta": 0.16, "scale": 0.34},
}
DURATION_JOURNEY = (0.58, 0.32, 0.74, 0.46, 0.84, 0.27, 0.66, 0.40)
# v5 fixes a timeline bug in the prior high-resolution waveform encoder: it
# asked an ~43 fps RMS feature stream for 64 samples per beat, producing more
# bins than frames and leaving a stretched tail of zero-valued samples.  The
# new envelope is binned directly from the decoded audio timeline.
ANALYSIS_CACHE_VERSION = "track-analysis-v5-source-timed-waveform"


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
    cached_track_count: int = 0,
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
        "cachedTrackCount": max(0, int(cached_track_count)),
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
    point = get_indexed_track(track_id)
    return point[0] if point else None


def get_indexed_track(track_id: str) -> tuple[np.ndarray, dict[str, Any]] | None:
    result = qdrant_request("POST", f"/collections/{VECTOR_COLLECTION}/points", {
        "ids": [track_id], "with_vector": True, "with_payload": True,
    })
    points = (result or {}).get("result", [])
    if not points:
        return None
    vector = points[0].get("vector")
    if not vector:
        return None
    return np.asarray(vector, dtype=np.float32), dict(points[0].get("payload") or {})


def analysis_cache_key(path: Path) -> str:
    """Fingerprint source bytes plus every setting that changes analysis."""
    stat = path.stat()
    try:
        relative = path.resolve().relative_to(MUSIC_ROOT).as_posix()
    except ValueError:
        relative = path.resolve().as_posix()
    descriptor = {
        "version": ANALYSIS_CACHE_VERSION,
        "relativePath": relative,
        "size": stat.st_size,
        "mtimeNs": stat.st_mtime_ns,
        "analysisModel": ANALYSIS_MODEL,
        "harmonixModel": HARMONIX_MODEL or "device-default",
        "harmonixProfile": HARMONIX_PROFILE,
        "analysisSampleRate": ANALYSIS_SAMPLE_RATE,
        "proxySampleRate": HARMONIX_PROXY_SAMPLE_RATE,
    }
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()


def analysis_cache_path(path: Path) -> Path:
    return DATA_ROOT / "analysis-cache" / f"{analysis_cache_key(path)}.json"


def read_analysis_cache(path: Path) -> tuple[dict[str, Any], np.ndarray] | None:
    try:
        payload = json.loads(analysis_cache_path(path).read_text(encoding="utf-8"))
        if payload.get("version") != ANALYSIS_CACHE_VERSION:
            return None
        track = payload["track"]
        vector = np.asarray(payload["embedding"], dtype=np.float32)
        if not isinstance(track, dict) or vector.shape != (VECTOR_SIZE,) or not np.isfinite(vector).all():
            return None
        return track, vector
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def write_analysis_cache(path: Path, track: dict[str, Any], vector: np.ndarray) -> None:
    """Atomically persist one completed track before advancing to the next."""
    destination = analysis_cache_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    template = dict(track)
    for key in ("id", "artworkUrl", "embeddingIndexed"):
        template.pop(key, None)
    payload = {
        "version": ANALYSIS_CACHE_VERSION,
        "track": template,
        "embedding": np.asarray(vector, dtype=np.float32).tolist(),
    }
    temporary = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(jsonable(payload), allow_nan=False), encoding="utf-8")
    temporary.replace(destination)


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


def waveform_points(signal: np.ndarray, count: int = 512) -> list[float]:
    """Create an amplitude envelope whose bins span the exact source timeline.

    ``np.array_split`` is only safe while the source contains at least as many
    frames as requested display points.  A detailed 64-samples-per-beat view
    exceeds the low-rate RMS frame count, which used to compress the real
    signal into the beginning of the display and append zero bins.  Binning
    decoded PCM directly preserves the source-time mapping used by beatGrid.
    """
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    count = max(1, int(count))
    if not values.size:
        return [0.0] * count
    magnitude = np.abs(values)
    if values.size < count:
        # This is mainly useful for tiny unit-test signals and protects callers
        # that provide a feature stream instead of PCM. Interpolation retains
        # the full timeline rather than appending empty chunks.
        positions = np.linspace(0, values.size - 1, count, dtype=np.float32)
        points = np.interp(positions, np.arange(values.size), magnitude)
    else:
        boundaries = np.linspace(0, values.size, count + 1, dtype=np.int64)
        squared = np.square(values)
        sums = np.add.reduceat(squared, boundaries[:-1])
        points = np.sqrt(sums / np.maximum(1, np.diff(boundaries)))
    peak = float(points.max())
    return [round(float(point / peak), 3) if peak else 0.0 for point in points]


def detailed_waveform(signal: np.ndarray, duration: float, bpm: float, detected_beats: int) -> str:
    """Encode a full-resolution signal with 64 samples per beat.

    A byte per sample keeps the preparation document small enough for large
    crates, unlike storing thousands of BSON doubles for every track.
    """
    estimated_beats = duration * max(bpm, 60.0) / 60.0
    point_count = max(1_024, math.ceil(max(float(detected_beats), estimated_beats) * WAVEFORM_SAMPLES_PER_BEAT))
    values = np.asarray(waveform_points(signal, point_count), dtype=np.float32)
    encoded = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8).tobytes()
    return base64.b64encode(encoded).decode("ascii")


def feature_window(values: np.ndarray, start: float, end: float, hop_length: int, sr: int) -> np.ndarray:
    """Return the analysis frames belonging to an audible time window."""
    frame_start = max(0, int(math.floor(start * sr / hop_length)))
    frame_end = min(values.shape[-1], max(frame_start + 1, int(math.ceil(end * sr / hop_length))))
    return values[..., frame_start:frame_end]


def normalise_phrase_feature(values: list[float]) -> np.ndarray:
    """Robustly map a per-phrase feature into [0, 1] within one track."""
    array = np.asarray(values, dtype=np.float32)
    if not array.size:
        return array
    floor, ceiling = np.percentile(array, [10, 92])
    if ceiling - floor < 1e-8:
        return np.full(array.shape, 0.5, dtype=np.float32)
    return np.clip((array - floor) / (ceiling - floor), 0, 1)


def repaired_downbeat_grid(downbeats: np.ndarray, beats: np.ndarray | None = None) -> np.ndarray:
    """Fill downbeats omitted during quiet breakdowns without moving anchors."""
    grid = np.asarray(sorted({round(float(point), 4) for point in downbeats if point >= 0}), dtype=float)
    if grid.size < 2:
        return grid
    beat_grid = np.asarray(beats if beats is not None else [], dtype=float)
    beat_diffs = np.diff(beat_grid)
    beat_diffs = beat_diffs[(beat_diffs > 0.18) & (beat_diffs < 1.5)]
    if beat_diffs.size:
        expected_bar = float(np.median(beat_diffs) * 4)
    else:
        downbeat_diffs = np.diff(grid)
        expected_bar = float(np.percentile(downbeat_diffs[downbeat_diffs > 0.5], 30))
    if expected_bar <= 0:
        return grid
    repaired = [float(grid[0])]
    for first, second in zip(grid, grid[1:]):
        gap = float(second - first)
        bars = max(1, min(32, int(round(gap / expected_bar))))
        local_bar = gap / bars
        if bars > 1 and abs(local_bar - expected_bar) / expected_bar <= 0.12:
            repaired.extend(float(first + local_bar * step) for step in range(1, bars))
        repaired.append(float(second))
    return np.asarray(repaired, dtype=float)


def phrase_state_boundaries(
    downbeats: np.ndarray, duration: float, segments: list[dict[str, Any]], beats: np.ndarray | None = None,
) -> list[float]:
    """Find eight-bar ``big 1`` candidates from a downbeat grid.

    A downbeat detector already gives a bar clock.  We test every possible
    eight-bar phase and anchor the phrase clock to the phase most supported by
    Harmonix section changes.  This makes the controller reason in DJ phrases
    (the big 1) instead of treating semantic section names as cue points.
    """
    grid = repaired_downbeat_grid(downbeats, beats)
    grid = grid[(grid >= 0) & (grid <= duration)]
    if grid.size < PHRASE_BARS + 1:
        return []
    structure_points = np.asarray(
        [float(segment[edge]) for segment in segments for edge in ("start", "end") if 0 < float(segment[edge]) < duration],
        dtype=float,
    )
    best_offset, best_score = 0, -1.0
    for offset in range(PHRASE_BARS):
        candidates = grid[offset::PHRASE_BARS]
        if not candidates.size:
            continue
        if structure_points.size:
            distance = np.min(np.abs(candidates[:, None] - structure_points[None, :]), axis=1)
            score = float(np.sum(np.exp(-distance / 0.38)))
        else:
            score = 0.0
        # In a tie, prefer the earliest tracked downbeat. It is normally the
        # track's first bar and makes preparations deterministic.
        score -= offset * 1e-4
        if score > best_score:
            best_offset, best_score = offset, score
    # Do not prepend a partial phrase when the best phase starts later than the
    # first detected bar; every adjacent pair below must represent eight bars.
    return [round(point, 3) for point in grid[best_offset::PHRASE_BARS].tolist()]


def build_phrase_states(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    rms: np.ndarray,
    spectral: np.ndarray,
    downbeats: np.ndarray,
    duration: float,
    segments: list[dict[str, Any]],
    beats: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Persist a compact local DJ state for every eight-bar phrase.

    These are intentionally actionable measurements, not another loose
    ``verse/chorus`` classifier: energy trajectory, bass/drum/vocal-band
    activity, spectral density, novelty and loopability tell the transition
    policy whether two phrase starts can share a mixer.
    """
    boundaries = phrase_state_boundaries(downbeats, duration, segments, beats)
    if len(boundaries) < 2:
        return []
    # A small mel representation is sufficient for phrase-scale activity and
    # costs far less than the source separation we deliberately removed from
    # the default analysis profile.
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=32, fmax=min(8_000, sr // 2 - 1), hop_length=hop_length, power=2,
    )
    mel_frequencies = librosa.mel_frequencies(n_mels=mel.shape[0], fmin=0, fmax=min(8_000, sr // 2 - 1))
    total = np.maximum(mel.sum(axis=0), 1e-9)
    bass_frames = mel[mel_frequencies <= 240].sum(axis=0) / total
    mid_frames = mel[(mel_frequencies >= 220) & (mel_frequencies <= 4_500)].sum(axis=0) / total
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    onset_scale = float(np.percentile(onset, 90)) if onset.size else 0.0
    onset_normalised = np.clip(onset / max(onset_scale, 1e-6), 0, 1)
    shared_frames = min(mid_frames.size, onset_normalised.size, flatness.size)
    # This remains a lightweight proxy rather than a separated vocal stem, but
    # unlike the old mean-onset multiplier it retains phrase-level evidence.
    # Sustained, harmonic mid-band content scores higher than percussive frames.
    vocal_frames = (
        mid_frames[:shared_frames]
        * (0.35 + 0.65 * (1 - onset_normalised[:shared_frames]))
        * (0.40 + 0.60 * (1 - np.clip(flatness[:shared_frames], 0, 1)))
    )
    vocal_threshold = max(0.08, float(np.percentile(vocal_frames, 62))) if vocal_frames.size else 1.0

    raw: list[dict[str, float]] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        energy_frames = feature_window(rms, start, end, hop_length, sr)
        onset_frames = feature_window(onset, start, end, hop_length, sr)
        bass_window = feature_window(bass_frames, start, end, hop_length, sr)
        vocal_window = feature_window(vocal_frames, start, end, hop_length, sr)
        flat_window = feature_window(flatness, start, end, hop_length, sr)
        centroid_window = feature_window(spectral, start, end, hop_length, sr)
        split = max(1, energy_frames.size // 3)
        vocal_edge = max(1, vocal_window.size // 4)
        energy_slope = float(np.mean(energy_frames[-split:]) - np.mean(energy_frames[:split])) if energy_frames.size else 0.0
        raw.append({
            "start": start,
            "end": end,
            "phraseIndex": float(index),
            "energy": float(np.mean(energy_frames)) if energy_frames.size else 0.0,
            "energySlope": energy_slope,
            "bassActivity": float(np.mean(bass_window)) if bass_window.size else 0.0,
            "drumActivity": float(np.mean(onset_frames)) if onset_frames.size else 0.0,
            # These are deliberately named activity rather than vocal stems.
            # Head/tail/continuity let the planner avoid cutting a sustained
            # lyric even when the phrase-average score alone looks harmless.
            "vocalActivity": float(np.mean(vocal_window)) if vocal_window.size else 0.0,
            "vocalHeadActivity": float(np.mean(vocal_window[:vocal_edge])) if vocal_window.size else 0.0,
            "vocalTailActivity": float(np.mean(vocal_window[-vocal_edge:])) if vocal_window.size else 0.0,
            "vocalContinuity": float(np.mean(vocal_window >= vocal_threshold)) if vocal_window.size else 0.0,
            "spectralDensity": float(np.mean(centroid_window)) if centroid_window.size else 0.0,
            "harmonicDensity": float(1 - np.mean(flat_window)) if flat_window.size else 0.0,
            "steady": float(np.std(energy_frames) / max(np.mean(energy_frames), 1e-6)) if energy_frames.size else 1.0,
        })
    for key in (
        "energy", "bassActivity", "drumActivity", "vocalActivity", "vocalHeadActivity",
        "vocalTailActivity", "spectralDensity", "harmonicDensity",
    ):
        values = normalise_phrase_feature([state[key] for state in raw])
        for state, value in zip(raw, values):
            state[key] = float(value)
    slope_scale = float(np.percentile(np.abs([state["energySlope"] for state in raw]), 90)) if raw else 0.0
    phrase_states: list[dict[str, Any]] = []
    for index, state in enumerate(raw):
        previous = raw[index - 1] if index else None
        following = raw[index + 1] if index + 1 < len(raw) else None
        novelty_in = 0.0 if previous is None else abs(state["energy"] - previous["energy"]) * 0.65 + abs(state["spectralDensity"] - previous["spectralDensity"]) * 0.35
        novelty_out = 0.0 if following is None else abs(following["energy"] - state["energy"]) * 0.65 + abs(following["spectralDensity"] - state["spectralDensity"]) * 0.35
        nearby_structure = any(abs(state["start"] - float(segment[edge])) < 0.45 for segment in segments for edge in ("start", "end"))
        loopability = np.clip(
            0.50 * state["drumActivity"] + 0.30 * (1 - min(1.0, state["steady"] * 3)) + 0.20 * (1 - state["vocalActivity"]), 0, 1,
        )
        cue_confidence = np.clip(0.28 + 0.28 * novelty_out + 0.22 * loopability + (0.22 if nearby_structure else 0), 0, 1)
        phrase_states.append({
            "start": round(state["start"], 3), "end": round(state["end"], 3), "phraseIndex": int(state["phraseIndex"]),
            "energy": round(state["energy"], 3), "energySlope": round(float(np.clip(state["energySlope"] / max(slope_scale, 1e-6), -1, 1)), 3),
            "bassActivity": round(state["bassActivity"], 3), "drumActivity": round(state["drumActivity"], 3),
            "vocalActivity": round(state["vocalActivity"], 3), "spectralDensity": round(state["spectralDensity"], 3),
            "vocalHeadActivity": round(state["vocalHeadActivity"], 3),
            "vocalTailActivity": round(state["vocalTailActivity"], 3),
            "vocalContinuity": round(float(np.clip(state["vocalContinuity"], 0, 1)), 3),
            "harmonicDensity": round(state["harmonicDensity"], 3), "noveltyIn": round(float(np.clip(novelty_in, 0, 1)), 3),
            "noveltyOut": round(float(np.clip(novelty_out, 0, 1)), 3), "loopability": round(float(loopability), 3),
            "cueConfidence": round(float(cue_confidence), 3),
        })
    return phrase_states


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


def analyse_track(path: Path, preparation_id: str) -> tuple[dict[str, Any], str]:
    track_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{preparation_id}:{path.relative_to(MUSIC_ROOT)}"))
    metadata, tag_genres, artwork_url = tags_for(path, track_id)
    cached = read_analysis_cache(path)
    if cached is not None:
        cached_track, vector = cached
        track = dict(cached_track)
        track.update({"id": track_id, "artworkUrl": artwork_url})
        track["embeddingIndexed"] = index_vector(track_id, vector, {
            "preparationId": preparation_id, "title": track["title"], "artist": track["artist"],
            "genres": track["genres"], "bpm": track["bpm"], "key": track["key"],
            "energy": track["energy"], "loudnessLufs": track.get("loudnessLufs"),
        })
        return track, "cache"

    # A timed-out legacy preparation may already have completed its expensive
    # Harmonix pass and indexed the embedding before per-track cache existed.
    # Retain those stored model outputs and rebuild only the cheap DSP details.
    indexed_track = get_indexed_track(track_id)
    indexed_vector, indexed_payload = indexed_track if indexed_track is not None else (None, {})
    y, sr = librosa.load(path, sr=ANALYSIS_SAMPLE_RATE, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    hop = 512
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, trim=False)
    bpm = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 0.0
    if bpm < 55 and bpm > 0:
        bpm *= 2
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    # We only need stable pitch-class energy for key compatibility. Explicitly
    # pinning concert tuning avoids librosa trying to infer it from silent
    # frames, which otherwise emits one warning per empty analysis window.
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, tuning=0.0)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24, hop_length=hop)
    spectral = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    deep = None if indexed_vector is not None else harmonix(path, y, sr)
    if deep and deep["bpm"] > 0:
        bpm = deep["bpm"]
    elif indexed_payload.get("bpm"):
        bpm = float(indexed_payload["bpm"])
    if deep and deep.get("beats"):
        beat_times = np.asarray(deep["beats"], dtype=float)
    downbeats = np.asarray(deep.get("downbeats", []) if deep else [], dtype=float)
    if not downbeats.size:
        downbeats = approximate_downbeats(beat_times)
    raw_segments = deep["segments"] if deep and deep["segments"] else fallback_segments(duration, rms, hop, sr, beat_times)
    segments = with_segment_energy(raw_segments, rms, hop, sr)
    phrase_states = build_phrase_states(y, sr, hop, rms, spectral, downbeats, duration, segments, beat_times)
    phrase_starts = [float(state["start"]) for state in phrase_states]
    phrase_ends = [float(state["end"]) for state in phrase_states]
    entries, exits = beat_safe_points(beat_times, downbeats, duration, segments)
    # Explicit phrase boundaries are stronger DJ cues than an arbitrary
    # semantic-section edge. Keep both so old/short/non-4/4 material still has
    # a safe fallback, while the transition controller can prefer the big 1.
    entries = sorted({*entries, *(round(point, 2) for point in phrase_starts if 3 < point < duration - 15)})
    exits = sorted({*exits, *(round(point, 2) for point in phrase_ends if 15 < point < duration - 3)})
    measured_energy = float(np.clip(np.percentile(rms, 75) * 4.0, 0, 1)) if rms.size else 0.0
    energy = float(indexed_payload.get("energy", measured_energy))
    loudness_lufs = float(indexed_payload["loudnessLufs"]) if indexed_payload.get("loudnessLufs") is not None else integrated_loudness(path, y)
    stored_genres = indexed_payload.get("genres")
    genres = list(stored_genres) if isinstance(stored_genres, list) and stored_genres else tag_genres or heuristic_genres(
        bpm, float(spectral.mean()) if spectral.size else 0, energy,
    )
    key = str(indexed_payload.get("key") or estimate_key(chroma))
    vector = indexed_vector if indexed_vector is not None else project_embedding(
        deep.get("embedding") if deep else None, mfcc, chroma, spectral,
    )
    indexed = index_vector(track_id, vector, {
        "preparationId": preparation_id, "title": metadata["title"], "artist": metadata["artist"],
        "genres": genres, "bpm": bpm, "key": key, "energy": energy, "loudnessLufs": loudness_lufs,
    })
    first_drop = next((segment["start"] for segment in segments if segment["label"] in {"chorus", "drop"}), None)
    track = {
        "id": track_id,
        "relativePath": path.relative_to(MUSIC_ROOT).as_posix(),
        "title": metadata["title"], "artist": metadata["artist"], "album": metadata["album"],
        "artworkUrl": artwork_url, "durationSeconds": round(duration, 2), "bpm": round(bpm, 2),
        "key": key, "energy": round(energy, 3), "loudnessLufs": loudness_lufs,
        # Render both waveform resolutions from decoded PCM, not analysis-rate
        # RMS frames. This makes every waveform x-coordinate share the same
        # zero and duration as the stored beat and downbeat grids.
        "waveform": waveform_points(y), "waveformDetail": detailed_waveform(y, duration, bpm, len(beat_times)),
        "waveformDetailVersion": WAVEFORM_DETAIL_VERSION,
        "beatGrid": [round(float(point), 3) for point in beat_times], "downbeats": [round(float(point), 3) for point in downbeats],
        "genres": genres, "segments": segments, "phraseStates": phrase_states,
        "cues": {
            "introEnd": entries[0], "firstDrop": first_drop, "safeEntries": entries, "safeExits": exits,
            "phraseBoundaries": sorted({round(point, 3) for point in phrase_starts + phrase_ends}),
        },
        "embeddingIndexed": indexed,
    }
    write_analysis_cache(path, track, vector)
    return track, "indexed" if indexed_vector is not None else "analysed"


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        device = harmonix_device() if ANALYSIS_MODEL != "librosa" else "cpu"
    except RuntimeError:
        device = "unavailable"
    return {
        "status": "ok",
        "engine": "local",
        "analysisDevice": device,
        "analysisProfile": HARMONIX_PROFILE if ANALYSIS_MODEL != "librosa" else "librosa",
        "transitionPolicy": transition_policy_status(),
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
    reused_count = 0
    total = len(paths)
    report_preparation_progress(
        preparation_id, progress=8, discovered_track_count=total,
        analysed_track_count=0, failed_track_count=0, current_track=None,
        message=f"Found {total} track{'s' if total != 1 else ''}. Starting local analysis.",
        cached_track_count=0,
    )
    for path in paths:
        processed = len(tracks) + len(failures)
        report_preparation_progress(
            preparation_id, progress=8 + round(processed / total * 87),
            discovered_track_count=total, analysed_track_count=len(tracks),
            failed_track_count=len(failures), current_track=path.name,
            message=f"Preparing track {processed + 1} of {total}; {reused_count} reused from cache",
            cached_track_count=reused_count,
        )
        try:
            ensure_job_active("preparation", preparation_id)
            track, source_kind = analyse_track(path, preparation_id)
            tracks.append(track)
            if source_kind != "analysed":
                reused_count += 1
        except HTTPException:
            raise
        except Exception as problem:
            failures.append(f"{path.name}: {problem}")
        processed = len(tracks) + len(failures)
        report_preparation_progress(
            preparation_id, progress=8 + round(processed / total * 87),
            discovered_track_count=total, analysed_track_count=len(tracks),
            failed_track_count=len(failures), current_track=path.name,
            message=(
                f"Prepared {processed} of {total}; {reused_count} reused"
                if len(failures) == 0
                else f"Prepared {len(tracks)} of {total}; {reused_count} reused; {len(failures)} skipped"
            ),
            cached_track_count=reused_count,
        )
    ensure_job_active("preparation", preparation_id)
    if not tracks:
        raise HTTPException(422, "No audio files could be decoded. Ensure FFmpeg supports the library format.")
    genres = sorted({genre for track in tracks for genre in track.get("genres", [])})
    report_preparation_progress(
        preparation_id, progress=97, discovered_track_count=total,
        analysed_track_count=len(tracks), failed_track_count=len(failures), current_track=None,
        message="Finalising music features and mix data",
        cached_track_count=reused_count,
    )
    return {
        "tracks": tracks, "genres": genres, "cachedTrackCount": reused_count,
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


def phrase_states_for(track: dict[str, Any]) -> list[dict[str, Any]]:
    return [state for state in track.get("phraseStates", []) if float(state.get("end", 0)) > float(state.get("start", 0))]


def derived_phrase_boundaries(track: dict[str, Any]) -> list[float]:
    """Return persisted phrase boundaries, or derive an eight-bar legacy grid."""
    states = phrase_states_for(track)
    if states:
        return sorted({float(state[edge]) for state in states for edge in ("start", "end")})
    stored = [float(point) for point in track.get("cues", {}).get("phraseBoundaries", [])]
    if stored:
        return sorted(set(stored))
    downbeats = [float(point) for point in track.get("downbeats", [])]
    return downbeats[::PHRASE_BARS] if downbeats else []


def phrase_state_at_boundary(track: dict[str, Any], point: float, role: str) -> dict[str, Any] | None:
    states = phrase_states_for(track)
    edge = "end" if role == "exit" else "start"
    if not states:
        return None
    tolerance = max(0.08, beat_duration(track) * 0.3)
    matching = [state for state in states if abs(float(state[edge]) - point) <= tolerance]
    return min(matching, key=lambda state: abs(float(state[edge]) - point)) if matching else None


def _gelu_tanh(values: np.ndarray) -> np.ndarray:
    return 0.5 * values * (
        1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (values + 0.044715 * np.power(values, 3)))
    )


def load_transition_policy() -> dict[str, Any] | None:
    """Hot-load the compact exported policy; no trainer/critic is imported."""
    global TRANSITION_POLICY_CACHE
    try:
        modified = TRANSITION_POLICY_PATH.stat().st_mtime_ns
    except OSError:
        modified = -1
    with TRANSITION_POLICY_LOCK:
        if TRANSITION_POLICY_CACHE[0] == modified:
            return TRANSITION_POLICY_CACHE[1]
        policy: dict[str, Any] | None = None
        if modified >= 0:
            try:
                candidate = json.loads(TRANSITION_POLICY_PATH.read_text(encoding="utf-8"))
                if candidate.get("featureVersion") != TRANSITION_FEATURE_VERSION:
                    raise ValueError("transition policy feature version does not match the worker")
                if candidate.get("featureNames") != TRANSITION_POLICY_FEATURES:
                    raise ValueError("transition policy feature order does not match the worker")
                if len(candidate.get("layers", [])) != 3:
                    raise ValueError("transition policy must contain three dense layers")
                policy = candidate
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                policy = None
        TRANSITION_POLICY_CACHE = (modified, policy)
        return policy


def transition_policy_status() -> str:
    policy = load_transition_policy()
    return str(policy.get("policyVersion", "loaded")) if policy else "deterministic-fallback"


def policy_state_vector(
    first: dict[str, Any],
    second: dict[str, Any],
    outgoing_state: dict[str, Any] | None,
    incoming_state: dict[str, Any] | None,
    source_bpm: float,
    source_start: float = 0.0,
    exit_at: float | None = None,
    entry_at: float | None = None,
) -> np.ndarray:
    outgoing = outgoing_state or {}
    incoming = incoming_state or {}
    outgoing_energy = float(outgoing.get("energy", first.get("energy", 0.5)))
    incoming_energy = float(incoming.get("energy", second.get("energy", 0.5)))
    outgoing_slope = float(outgoing.get("energySlope", 0.0))
    incoming_slope = float(incoming.get("energySlope", 0.0))
    outgoing_bass = float(outgoing.get("bassActivity", 0.5))
    incoming_bass = float(incoming.get("bassActivity", 0.5))
    outgoing_vocals = float(outgoing.get("vocalActivity", 0.25))
    incoming_vocals = float(incoming.get("vocalActivity", 0.25))
    exit_point = float(exit_at if exit_at is not None else outgoing.get("end", first.get("durationSeconds", 1.0)))
    entry_point = float(entry_at if entry_at is not None else incoming.get("start", 0.0))
    outgoing_duration = max(float(first.get("durationSeconds", 1.0)), 1.0)
    incoming_duration = max(float(second.get("durationSeconds", 1.0)), 1.0)
    elapsed_fraction = float(np.clip((exit_point - source_start) / max(outgoing_duration - source_start, 1.0), 0, 1))
    remaining_fraction = float(np.clip((outgoing_duration - exit_point) / outgoing_duration, 0, 1))
    genre_compatibility = 1.0 if set(first.get("genres", [])) & set(second.get("genres", [])) else 0.0
    return np.asarray([
        float(second.get("bpm", source_bpm)) / max(source_bpm, 1.0),
        key_score(str(first.get("key", "")), str(second.get("key", ""))),
        outgoing_energy, incoming_energy, abs(outgoing_energy - incoming_energy),
        outgoing_slope, incoming_slope, abs(outgoing_slope - incoming_slope),
        outgoing_bass, incoming_bass, outgoing_bass * incoming_bass,
        float(outgoing.get("drumActivity", 0.5)), float(incoming.get("drumActivity", 0.5)),
        outgoing_vocals, incoming_vocals, outgoing_vocals * incoming_vocals,
        float(outgoing.get("spectralDensity", 0.5)), float(incoming.get("spectralDensity", 0.5)),
        float(outgoing.get("harmonicDensity", 0.5)), float(incoming.get("harmonicDensity", 0.5)),
        float(outgoing.get("noveltyOut", 0.0)), float(incoming.get("noveltyIn", 0.0)),
        float(outgoing.get("loopability", 0.5)), float(incoming.get("cueConfidence", 0.5)),
        elapsed_fraction, remaining_fraction, float(np.clip(entry_point / incoming_duration, 0, 1)),
        genre_compatibility, float(outgoing.get("cueConfidence", 0.5)),
    ], dtype=np.float32)


def transition_policy_controls(features: np.ndarray) -> dict[str, float] | None:
    policy = load_transition_policy()
    if policy is None:
        return None
    try:
        mean = np.asarray(policy["featureMean"], dtype=np.float32)
        scale = np.asarray(policy["featureScale"], dtype=np.float32)
        hidden = np.clip((features - mean) / np.maximum(scale, 1e-4), -5.0, 5.0)
        for index, layer in enumerate(policy["layers"]):
            hidden = np.asarray(layer["weight"], dtype=np.float32) @ hidden + np.asarray(layer["bias"], dtype=np.float32)
            if index < 2:
                hidden = _gelu_tanh(hidden)
        unit = 1.0 / (1.0 + np.exp(-np.clip(hidden, -30, 30)))
        lower = np.asarray(policy["controlLower"], dtype=np.float32)
        upper = np.asarray(policy["controlUpper"], dtype=np.float32)
        values = lower + unit * (upper - lower)
        return {str(name): float(value) for name, value in zip(policy["controlNames"], values)}
    except (KeyError, TypeError, ValueError):
        return None


def vocal_boundary_risk(
    outgoing_state: dict[str, Any] | None,
    incoming_state: dict[str, Any] | None,
) -> tuple[float, float, float]:
    """Return outgoing lyric-cut, incoming vocal-head and overlap risks."""
    outgoing = outgoing_state or {}
    incoming = incoming_state or {}
    outgoing_activity = float(outgoing.get("vocalActivity", 0.25))
    incoming_activity = float(incoming.get("vocalActivity", 0.25))
    outgoing_tail = float(outgoing.get("vocalTailActivity", outgoing_activity))
    incoming_head = float(incoming.get("vocalHeadActivity", incoming_activity))
    continuity = float(outgoing.get("vocalContinuity", outgoing_activity))
    lyric_cut = float(np.clip(outgoing_tail * (0.45 + 0.55 * continuity), 0, 1))
    overlap = float(np.clip(max(outgoing_activity * incoming_activity, outgoing_tail * incoming_head), 0, 1))
    return lyric_cut, float(np.clip(incoming_head, 0, 1)), overlap


def transition_compatibility_score(
    first: dict[str, Any],
    second: dict[str, Any],
    outgoing_state: dict[str, Any] | None,
    incoming_state: dict[str, Any] | None,
    harmonic_compatibility: float | None = None,
) -> tuple[float, dict[str, float]]:
    """Score a cue pair using feature weights learned from professional sets.

    The target for a feature is not assumed to be zero. If professionals tend
    to introduce a controlled energy or spectrum contrast, the exported
    profile can reward that instead of blindly maximizing vector similarity.
    Beat, bar and phrase constraints remain hard gates outside this score.
    """
    outgoing = outgoing_state or {}
    incoming = incoming_state or {}
    harmonic = harmonic_compatibility
    if harmonic is None:
        harmonic = key_score(str(first.get("key", "")), str(second.get("key", "")))
    observed = {
        "energy": abs(float(outgoing.get("energy", first.get("energy", 0.5))) - float(incoming.get("energy", second.get("energy", 0.5)))),
        "trajectory": abs(float(outgoing.get("energySlope", 0.0)) - float(incoming.get("energySlope", 0.0))),
        "bass": abs(float(outgoing.get("bassActivity", 0.5)) - float(incoming.get("bassActivity", 0.5))),
        "drums": abs(float(outgoing.get("drumActivity", 0.5)) - float(incoming.get("drumActivity", 0.5))),
        "vocals": abs(float(outgoing.get("vocalActivity", 0.25)) - float(incoming.get("vocalActivity", 0.25))),
        "spectral": abs(float(outgoing.get("spectralDensity", 0.5)) - float(incoming.get("spectralDensity", 0.5))),
        "harmonic": 1.0 - float(np.clip(harmonic, 0, 1)),
        "novelty": abs(float(outgoing.get("noveltyOut", 0.0)) - float(incoming.get("noveltyIn", 0.0))),
    }
    policy = load_transition_policy() or {}
    exported = policy.get("compatibilityProfile", {}).get("features", {})
    profile = exported if isinstance(exported, dict) and exported else DEFAULT_COMPATIBILITY_PROFILE
    weighted = 0.0
    total_weight = 0.0
    component_scores: dict[str, float] = {}
    for name, value in observed.items():
        settings = profile.get(name, DEFAULT_COMPATIBILITY_PROFILE[name])
        try:
            weight = max(0.0, float(settings.get("weight", 0.0)))
            target = float(settings.get("targetDelta", 0.0))
            scale = max(0.035, float(settings.get("scale", 0.25)))
        except (AttributeError, TypeError, ValueError):
            settings = DEFAULT_COMPATIBILITY_PROFILE[name]
            weight = float(settings["weight"])
            target = float(settings["targetDelta"])
            scale = float(settings["scale"])
        fit = math.exp(-0.5 * ((value - target) / scale) ** 2)
        component_scores[name] = round(float(np.clip(fit, 0, 1)), 4)
        weighted += weight * fit
        total_weight += weight
    return float(np.clip(weighted / max(total_weight, 1e-8), 0, 1)), component_scores


def near_grid(point: float, grid: list[float] | np.ndarray, tolerance: float) -> bool:
    values = np.asarray(grid, dtype=float)
    return bool(values.size and float(np.min(np.abs(values - point))) <= tolerance)


def _anchored_beat_span(track: dict[str, Any], start: float, end: float) -> np.ndarray | None:
    """Return a beat sequence whose first and last beats lock to a cue span.

    Phrase labels are deliberately allowed a little semantic flexibility, but
    a two-deck overlay is not: its musical endpoints must actually land on
    detected beats.  Returning ``None`` here turns an uncertain candidate into
    a filtered hand-off instead of an optimistic, drifting blend.
    """
    grid = np.asarray(sorted({float(point) for point in track.get("beatGrid", [])}), dtype=float)
    if grid.size < 5:
        return None
    first_index = int(np.argmin(np.abs(grid - start)))
    last_index = int(np.argmin(np.abs(grid - end)))
    if (
        first_index >= last_index
        or abs(float(grid[first_index]) - start) > BEAT_ANCHOR_TOLERANCE_SECONDS
        or abs(float(grid[last_index]) - end) > BEAT_ANCHOR_TOLERANCE_SECONDS
    ):
        return None
    beats = grid[first_index:last_index + 1]
    return beats if beats.size >= 5 else None


def beat_overlay_fit(
    first: dict[str, Any], second: dict[str, Any], outgoing_start: float, outgoing_end: float,
    incoming_start: float, incoming_end: float, outgoing_factor: float,
) -> dict[str, float] | None:
    """Fit the incoming atempo factor against every beat in an overlay.

    The former check compared two sequences after subtracting their respective
    first beat.  That measured drift but accidentally discarded a phase error
    at the real crossfade boundary.  This fit keeps that phase term, derives
    the factor from all paired beats, and rejects a missing/unequal grid rather
    than silently pairing arbitrary beat indices.
    """
    outgoing_beats = _anchored_beat_span(first, outgoing_start, outgoing_end)
    incoming_beats = _anchored_beat_span(second, incoming_start, incoming_end)
    if outgoing_beats is None or incoming_beats is None or outgoing_beats.size != incoming_beats.size:
        return None
    outgoing_rendered = (outgoing_beats - outgoing_start) / max(outgoing_factor, 1e-6)
    incoming_source = incoming_beats - incoming_start
    # Least-squares fit through the actual overlap origin.  The first beat is
    # intentionally retained so a cue that begins between beats is penalised.
    denominator = float(np.dot(incoming_source, outgoing_rendered))
    if denominator <= 1e-8:
        return None
    incoming_factor = float(np.dot(incoming_source, incoming_source) / denominator)
    if not math.isfinite(incoming_factor) or incoming_factor <= 0:
        return None
    incoming_rendered = incoming_source / incoming_factor
    residual = np.abs(outgoing_rendered - incoming_rendered)
    return {
        "tempoFactor": incoming_factor,
        "residualMs": float(np.percentile(residual, 95) * 1000),
        "phaseErrorMs": float(residual[0] * 1000),
        "beatCount": float(outgoing_beats.size),
    }


def transition_grid_error_ms(
    first: dict[str, Any], second: dict[str, Any], outgoing_start: float, outgoing_end: float,
    incoming_start: float, incoming_end: float, outgoing_factor: float, incoming_factor: float,
) -> float:
    """Measure boundary-inclusive beat residual using a supplied tempo factor."""
    outgoing_beats = _anchored_beat_span(first, outgoing_start, outgoing_end)
    incoming_beats = _anchored_beat_span(second, incoming_start, incoming_end)
    if outgoing_beats is None or incoming_beats is None or outgoing_beats.size != incoming_beats.size:
        return 999.0
    outgoing = (outgoing_beats - outgoing_start) / max(outgoing_factor, 1e-6)
    incoming = (incoming_beats - incoming_start) / max(incoming_factor, 1e-6)
    return round(float(np.percentile(np.abs(outgoing - incoming), 95) * 1000), 2)


def safe_exit_for(
    track: dict[str, Any],
    source_start: float,
    minimum: float,
    maximum: float,
    duration_preference: float | None = None,
) -> float | None:
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
    phrase_exits = [
        float(state["end"]) for state in phrase_states_for(track)
        if lower <= float(state["end"]) <= upper and float(state.get("cueConfidence", 1.0)) >= 0.34
    ]
    if not phrase_exits:
        phrase_exits = [point for point in derived_phrase_boundaries(track) if lower <= point <= upper]
    downbeats = [float(point) for point in track.get("downbeats", []) if lower <= float(point) <= upper]
    # Once an eight-bar clock is available, arbitrary downbeats are not mix
    # points. They remain a fallback only for short or irregular material.
    candidates = sorted({round(point, 3) for point in (phrase_exits or safe_exits or downbeats)})
    has_explicit_phrase_clock = bool(phrase_states_for(track) or track.get("cues", {}).get("phraseBoundaries"))
    preference = float(np.clip(duration_preference if duration_preference is not None else 0.5, 0.10, 0.90))
    target = lower + (upper - lower) * preference
    beat = beat_duration(track)

    def penalty(point: float) -> float:
        before = segment_at(track, max(source_start, point - beat / 2))
        after = segment_at(track, min(track["durationSeconds"] - 0.01, point + beat / 2))
        state = phrase_state_at_boundary(track, point, "exit")
        value = abs(point - target)
        # Never abandon a chorus/drop halfway through. Its final phrase boundary
        # is allowed. Phrase-scale evidence can also release a long semantic
        # section when its current eight bars are losing energy or lead into a
        # major change; a sustained peak remains protected.
        if is_protected_section(before):
            sustained_peak = bool(
                state
                and float(state.get("energy", 1.0)) >= 0.80
                and float(state.get("noveltyOut", 0.0)) < 0.30
                and float(state.get("energySlope", 0.0)) > -0.22
            )
            if point < float(before["end"]) - beat * 1.1 and (state is None or sustained_peak):
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
        if state:
            value -= float(state.get("cueConfidence", 0.5)) * 6
            value -= float(state.get("loopability", 0.5)) * 2
            lyric_cut, _, _ = vocal_boundary_risk(state, None)
            # A phrase boundary is not automatically the end of a sentence.
            # Strong sustained vocal evidence makes another phrase exit much
            # preferable, while still allowing the natural end of a file.
            if lyric_cut >= 0.34 and point < natural_end - beat * 1.1:
                value += 34.0 * lyric_cut
        return value

    viable = [point for point in candidates if penalty(point) < 10_000]
    if viable:
        return min(viable, key=penalty)
    if has_explicit_phrase_clock:
        # Once the richer clock exists, a semantic edge between phrases is not
        # an acceptable escape hatch. Reordering/acceptance should choose a
        # different track instead of manufacturing an off-phrase transition.
        if upper >= natural_end - 0.5:
            return round(natural_end, 3)
        return None
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
    entries = [float(point) for point in second["cues"].get("safeEntries", []) if 0 <= float(point) <= latest_entry]
    phrase_entries = [
        float(state["start"]) for state in phrase_states_for(second)
        if 0 <= float(state["start"]) <= latest_entry and float(state.get("cueConfidence", 1.0)) >= 0.30
    ]
    if not phrase_entries:
        phrase_entries = [point for point in derived_phrase_boundaries(second) if 0 <= point <= latest_entry]
    downbeats = [float(point) for point in second.get("downbeats", []) if 0 <= float(point) <= latest_entry]
    candidates = sorted({round(point, 3) for point in (phrase_entries or entries or downbeats)})
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
        state = phrase_state_at_boundary(second, point, "entry")
        if state:
            # A clean intro has space for the outgoing record. Vocal-band and
            # bass-heavy starts remain valid, but are reserved for guarded EQ.
            value += float(state.get("bassActivity", 0.5)) * 5
            value += float(state.get("vocalActivity", 0.5)) * 8
            value += float(state.get("vocalHeadActivity", state.get("vocalActivity", 0.5))) * 10
            value -= float(state.get("cueConfidence", 0.5)) * 5
        return value + point / 3200  # prefer an earlier equivalent phase only

    return min(candidates, key=penalty)


def learned_cue_pair(
    first: dict[str, Any],
    second: dict[str, Any],
    source_start: float,
    minimum: float,
    maximum: float,
    source_bpm: float,
    fallback_exit: float,
    fallback_entry: float,
    duration_preference: float = 0.5,
) -> tuple[float, float, dict[str, float] | None]:
    """Let the compact policy rank safe phrase candidates, never raw samples."""
    if load_transition_policy() is None:
        return fallback_exit, fallback_entry, None
    natural_end = float(first["durationSeconds"]) - 0.25
    lower = source_start + min(minimum, max(0.5, natural_end - source_start))
    upper = min(natural_end, source_start + maximum)
    target = lower + (upper - lower) * float(np.clip(duration_preference, 0.10, 0.90))
    beat = beat_duration(first, source_bpm)
    exit_points = {
        fallback_exit,
        *(float(state["end"]) for state in phrase_states_for(first)),
        *(float(point) for point in first.get("cues", {}).get("safeExits", [])),
    }
    safe_exits: list[float] = []
    for point in exit_points:
        if not lower <= point <= upper:
            continue
        before = segment_at(first, max(source_start, point - beat / 2))
        if (is_build_section(before) or is_protected_section(before)) and point < float(before["end"]) - beat * 1.1:
            continue
        safe_exits.append(point)
    safe_exits = sorted(safe_exits, key=lambda point: abs(point - target))[:10] or [fallback_exit]

    best: tuple[float, float, dict[str, float], float] | None = None
    latest_entry = float(second["durationSeconds"]) - min(minimum, float(second["durationSeconds"]) - 0.5) - 0.25
    for exit_point in safe_exits:
        default_entry = safe_entry_for(first, second, exit_point, minimum)
        entry_points = {
            *(float(state["start"]) for state in phrase_states_for(second)),
            *(float(point) for point in second.get("cues", {}).get("safeEntries", [])),
        }
        if default_entry is not None:
            entry_points.add(default_entry)
        outgoing_state = phrase_state_at_boundary(first, exit_point, "exit")
        outgoing_energy = float((outgoing_state or {}).get("energy", section_energy(first, exit_point - beat / 2)))
        candidates: list[tuple[float, float, float]] = []
        for entry_point in entry_points:
            if not 0 <= entry_point <= latest_entry:
                continue
            incoming_section = segment_at(second, entry_point + beat_duration(second) / 2)
            incoming_energy = section_energy(second, entry_point + beat_duration(second) / 2)
            if is_build_section(incoming_section):
                continue
            if is_protected_section(incoming_section) and outgoing_energy < 0.70:
                continue
            incoming_state = phrase_state_at_boundary(second, entry_point, "entry")
            compatibility, _ = transition_compatibility_score(first, second, outgoing_state, incoming_state)
            _, incoming_head, vocal_overlap = vocal_boundary_risk(outgoing_state, incoming_state)
            cue = float((incoming_state or {}).get("cueConfidence", 0.5))
            pre_score = 0.65 * compatibility + 0.20 * cue + 0.15 * (1 - max(incoming_head, vocal_overlap))
            candidates.append((entry_point, incoming_energy, pre_score))
        candidates.sort(key=lambda item: item[2], reverse=True)
        for entry_point, incoming_energy, _ in candidates[:12]:
            incoming_state = phrase_state_at_boundary(second, entry_point, "entry")
            controls = transition_policy_controls(policy_state_vector(
                first, second, outgoing_state, incoming_state, source_bpm,
                source_start=source_start, exit_at=exit_point, entry_at=entry_point,
            ))
            if controls is None:
                continue
            phrases = int(np.clip(round(controls.get("overlap_phrases", 1.0)), 1, 4))
            first_boundaries = [point for point in derived_phrase_boundaries(first) if point < exit_point - beat * 0.3]
            second_boundaries = [point for point in derived_phrase_boundaries(second) if point > entry_point + beat * 0.3]
            if len(first_boundaries) < phrases or len(second_boundaries) < phrases:
                continue
            outgoing_start = first_boundaries[-phrases]
            incoming_end = second_boundaries[phrases - 1]
            outgoing_factor = source_bpm / max(float(first.get("bpm") or source_bpm), 1.0)
            outgoing_duration = (exit_point - outgoing_start) / max(outgoing_factor, 1e-6)
            incoming_duration = incoming_end - entry_point
            nominal_factor = incoming_duration / max(outgoing_duration, 1e-6)
            beat_fit = beat_overlay_fit(
                first, second, outgoing_start, exit_point, entry_point, incoming_end, outgoing_factor,
            )
            requested_factor = float(beat_fit["tempoFactor"]) if beat_fit else nominal_factor
            global_factor = source_bpm / max(float(second.get("bpm") or source_bpm), 1.0)
            tolerance = max(0.10, beat * 0.32)
            if not (
                beat_fit is not None
                and
                0.92 <= requested_factor <= 1.08
                and abs(requested_factor / max(global_factor, 1e-6) - 1) <= 0.035
                and near_grid(exit_point, derived_phrase_boundaries(first), tolerance)
                and near_grid(entry_point, derived_phrase_boundaries(second), tolerance)
                and near_grid(outgoing_start, first.get("downbeats", []), tolerance)
                and near_grid(entry_point, second.get("downbeats", []), tolerance)
                and float(beat_fit["residualMs"]) <= MAX_BEAT_OVERLAY_RESIDUAL_MS
            ):
                continue
            compatibility, _ = transition_compatibility_score(first, second, outgoing_state, incoming_state)
            bass_collision = float((outgoing_state or {}).get("bassActivity", 0.5)) * float((incoming_state or {}).get("bassActivity", 0.5))
            lyric_cut, incoming_head, vocal_collision = vocal_boundary_risk(outgoing_state, incoming_state)
            cue_confidence = 0.5 * (
                float((outgoing_state or {}).get("cueConfidence", 0.5))
                + float((incoming_state or {}).get("cueConfidence", 0.5))
            )
            trajectory_penalty = 0.0
            if (
                outgoing_energy >= 0.76 and incoming_energy >= 0.70
                and float((outgoing_state or {}).get("energySlope", 0.0)) < -0.12
            ):
                # Do not bury the audible build-down/tail of a peak behind a
                # new high-energy phrase; select another entry or exit.
                trajectory_penalty = 0.35
            duration_fit = 1.0 - min(1.0, abs(exit_point - target) / max(upper - lower, 1.0))
            vocal_boundary_penalty = 0.52 * lyric_cut + 0.16 * max(0.0, incoming_head - 0.55)
            score = (
                0.42 * float(controls.get("timing_score", 0.5))
                + 0.26 * compatibility + 0.12 * cue_confidence + 0.08 * duration_fit
                + 0.06 * (1 - bass_collision) + 0.06 * (1 - vocal_collision)
                - trajectory_penalty - vocal_boundary_penalty
            )
            if best is None or score > best[3]:
                best = (exit_point, entry_point, controls, score)
    if best is None:
        fallback_out = phrase_state_at_boundary(first, fallback_exit, "exit")
        fallback_in = phrase_state_at_boundary(second, fallback_entry, "entry")
        controls = transition_policy_controls(policy_state_vector(
            first, second, fallback_out, fallback_in, source_bpm,
            source_start=source_start, exit_at=fallback_exit, entry_at=fallback_entry,
        ))
        return fallback_exit, fallback_entry, controls
    return best[0], best[1], best[2]


def transition_between(
    first: dict[str, Any],
    second: dict[str, Any],
    source_start: float = 0.0,
    minimum: float = 2.0,
    maximum: float = 14.0,
    running_bpm: float | None = None,
    duration_preference: float = 0.5,
) -> dict[str, Any] | None:
    exit_at = safe_exit_for(first, source_start, minimum, maximum, duration_preference)
    if exit_at is None:
        return None
    entry_at = safe_entry_for(first, second, exit_at, minimum)
    if entry_at is None:
        return None
    source_bpm = float(running_bpm or first["bpm"] or 120.0)
    exit_at, entry_at, learned_controls = learned_cue_pair(
        first, second, source_start, minimum, maximum, source_bpm, exit_at, entry_at,
        duration_preference,
    )
    outgoing_factor = source_bpm / max(float(first.get("bpm") or source_bpm), 1.0)
    tempo_ratio = second["bpm"] / source_bpm if source_bpm else 1.0
    global_factor = source_bpm / max(float(second["bpm"]), 1.0)

    outgoing_state = phrase_state_at_boundary(first, exit_at, "exit")
    incoming_state = phrase_state_at_boundary(second, entry_at, "entry")
    requested_phrases = int(round((learned_controls or {}).get("overlap_phrases", 1.0)))
    requested_phrases = int(np.clip(requested_phrases, 1, 4))

    first_boundaries = derived_phrase_boundaries(first)
    second_boundaries = derived_phrase_boundaries(second)
    tolerance = max(0.10, beat_duration(first, source_bpm) * 0.32)
    outgoing_candidates = [point for point in first_boundaries if point < exit_at - tolerance]
    incoming_candidates = [point for point in second_boundaries if point > entry_at + tolerance]
    selected_phrases = min(requested_phrases, len(outgoing_candidates), len(incoming_candidates))
    outgoing_phrase_start = outgoing_candidates[-selected_phrases] if selected_phrases else None
    incoming_phrase_end = incoming_candidates[selected_phrases - 1] if selected_phrases else None
    phrase_matched = bool(
        outgoing_phrase_start is not None
        and incoming_phrase_end is not None
        and near_grid(exit_at, first_boundaries, tolerance)
        and near_grid(entry_at, second_boundaries, tolerance)
    )
    # Match the actual detected 8-bar spans. This removes small BPM-estimator
    # rounding errors: both phrase endpoints land together after atempo, not
    # just the first kick of the overlap.
    requested_factor = global_factor
    beat_fit: dict[str, float] | None = None
    overlap = min(1.25, max(0.55, 60.0 / source_bpm * 2))
    if phrase_matched and outgoing_phrase_start is not None and incoming_phrase_end is not None:
        outgoing_phrase_duration = (exit_at - outgoing_phrase_start) / max(outgoing_factor, 1e-6)
        incoming_phrase_duration = incoming_phrase_end - entry_at
        if outgoing_phrase_duration > 2 and incoming_phrase_duration > 2:
            nominal_factor = incoming_phrase_duration / outgoing_phrase_duration
            beat_fit = beat_overlay_fit(
                first, second, outgoing_phrase_start, exit_at, entry_at, incoming_phrase_end, outgoing_factor,
            )
            requested_factor = float(beat_fit["tempoFactor"]) if beat_fit else nominal_factor
            overlap = outgoing_phrase_duration

    # Exact beat overlays are only enabled when tempo, bar phase, phrase phase,
    # and measured beat residual all pass. Otherwise the records meet at their
    # phrase boundaries without laying two drifting kick grids together.
    factor_is_safe = 0.92 <= requested_factor <= 1.08 and abs(requested_factor / max(global_factor, 1e-6) - 1) <= 0.035
    tempo_factor = requested_factor if factor_is_safe else 1.0
    bar_matched = bool(
        phrase_matched
        and outgoing_phrase_start is not None
        and (
            near_grid(outgoing_phrase_start, first.get("downbeats", []), tolerance)
            or near_grid(outgoing_phrase_start, first.get("beatGrid", []), tolerance)
        )
        and (
            near_grid(entry_at, second.get("downbeats", []), tolerance)
            or near_grid(entry_at, second.get("beatGrid", []), tolerance)
        )
    )
    beat_alignment_error_ms = float(beat_fit["residualMs"]) if beat_fit else 999.0
    beat_phase_error_ms = float(beat_fit["phaseErrorMs"]) if beat_fit else 999.0
    beat_matched = (
        factor_is_safe and bar_matched and beat_fit is not None
        and beat_alignment_error_ms <= MAX_BEAT_OVERLAY_RESIDUAL_MS
    )
    if not beat_matched:
        tempo_factor = 1.0
        overlap = min(1.25, max(0.55, 60.0 / source_bpm * 2))

    tempo_score = max(0.0, 1 - min(abs(tempo_ratio - 1), 0.25) / 0.25)
    harmonic = key_score(first["key"], second["key"])
    outgoing_energy = float(outgoing_state.get("energy")) if outgoing_state else section_energy(first, max(source_start, exit_at - beat_duration(first, source_bpm)))
    incoming_energy = float(incoming_state.get("energy")) if incoming_state else section_energy(second, entry_at + beat_duration(second) / 2)
    outgoing_section = segment_at(first, max(source_start, exit_at - beat_duration(first, source_bpm)))
    incoming_section = segment_at(second, entry_at + beat_duration(second) / 2)
    outgoing_bass = float((outgoing_state or {}).get("bassActivity", 0.5))
    incoming_bass = float((incoming_state or {}).get("bassActivity", 0.5))
    outgoing_vocals = float((outgoing_state or {}).get("vocalActivity", 0.25))
    bass_clash_risk = outgoing_bass * incoming_bass
    lyric_cut_risk, incoming_vocal_head, vocal_clash_risk = vocal_boundary_risk(outgoing_state, incoming_state)
    learned_compatibility, compatibility_components = transition_compatibility_score(
        first, second, outgoing_state, incoming_state, harmonic,
    )
    phrase_compatibility = float(np.clip(
        0.72 * learned_compatibility + 0.14 * (1 - bass_clash_risk) + 0.14 * (1 - vocal_clash_risk),
        0, 1,
    ))
    shared_genre = set(first["genres"]) & set(second["genres"])
    genre_score = 1.0 if shared_genre else 0.55
    similarity = vector_score(first["id"], second["id"])
    phase_score = 1.0 if beat_matched and phrase_matched else 0.35 if phrase_matched else 0.0
    cue_score = 0.5 * (
        float((outgoing_state or {}).get("cueConfidence", 0.5))
        + float((incoming_state or {}).get("cueConfidence", 0.5))
    )
    score = 100 * (
        0.18 * tempo_score + 0.26 * phase_score + 0.28 * phrase_compatibility
        + 0.10 * harmonic + 0.07 * genre_score + 0.05 * similarity + 0.06 * cue_score
    )
    guard_outgoing = section_label(outgoing_section) in SENSITIVE_HANDOFF_SECTIONS and incoming_energy >= 0.70
    if not beat_matched:
        technique = "phrase-boundary-cut"
        fade_shape = "fast-outgoing"
    elif lyric_cut_risk >= 0.34:
        technique = "vocal-carry-bed"
        fade_shape = "vocal-carry"
    elif vocal_clash_risk >= 0.30:
        technique = "vocal-guarded-eq"
        fade_shape = "long-release"
    elif guard_outgoing or incoming_energy > outgoing_energy + 0.18:
        technique = "build-and-bass-swap"
        fade_shape = "incoming-lift"
    elif bass_clash_risk >= 0.24:
        technique = "bass-swap"
        fade_shape = "equal-power"
    else:
        technique = "long-eq-blend"
        fade_shape = "equal-power"
    if guard_outgoing and lyric_cut_risk < 0.34:
        fade_shape = "fast-outgoing"
    elif lyric_cut_risk >= 0.34 and beat_matched:
        fade_shape = "vocal-carry"
    elif outgoing_energy > incoming_energy + 0.22 and beat_matched:
        fade_shape = "long-release"
    overlap_bars = PHRASE_BARS * selected_phrases if beat_matched else 0
    loop_seconds = 0.0
    bass_swap_progress = float((learned_controls or {}).get(
        "bass_swap_progress",
        0.62 if technique == "build-and-bass-swap" else 0.50 if technique == "bass-swap" else 0.68,
    ))
    fade_out_curve = float((learned_controls or {}).get("fade_out_curve", 1.0))
    fade_in_curve = float((learned_controls or {}).get("fade_in_curve", 1.0))
    outgoing_low_db = float((learned_controls or {}).get("outgoing_low_db", -18.0))
    incoming_low_db = float((learned_controls or {}).get("incoming_low_db", -18.0))
    outgoing_mid_db = float((learned_controls or {}).get("outgoing_mid_db", -3.0))
    incoming_mid_db = float((learned_controls or {}).get("incoming_mid_db", -3.0))
    outgoing_high_db = float((learned_controls or {}).get("outgoing_high_db", -5.0))
    incoming_high_db = float((learned_controls or {}).get("incoming_high_db", -5.0))
    if (
        beat_matched and learned_controls
        and learned_controls.get("loop_probability", 0.0) >= 0.65
        and float((outgoing_state or {}).get("loopability", 0.0)) >= 0.65
        and outgoing_vocals < 0.36
        and float((outgoing_state or {}).get("vocalTailActivity", outgoing_vocals)) < 0.32
    ):
        loop_seconds = round(4 * 60.0 / source_bpm, 3)
    if beat_matched:
        style = f"{overlap_bars}-bar learned beat/bar/phrase locked blend" if learned_controls else "8-bar beat/bar/phrase locked blend"
    elif phrase_matched:
        style = "phrase-boundary filtered hand-off"
    else:
        style = "protected emergency hand-off"
    notes = [
        f"harmonic {harmonic:.0%}", f"learned phrase compatibility {phrase_compatibility:.0%}",
        f"embedding affinity {similarity:.0%} (low-weight tie-breaker)",
        "high-energy phrases protected", f"{technique} controller",
    ]
    if beat_matched:
        notes.extend([
            f"pitch-preserving tempo {(tempo_factor - 1):+.1%}",
            f"{overlap_bars}-bar phrase overlay",
            f"95th percentile beat residual {beat_alignment_error_ms:.1f} ms; boundary phase {beat_phase_error_ms:.1f} ms",
        ])
    else:
        notes.append("overlap disabled because beat/bar/phrase lock did not pass")
    if guard_outgoing:
        notes.append("outgoing build/breakdown is removed early before the incoming high-energy phrase")
    if lyric_cut_risk >= 0.34:
        notes.append("outgoing vocal is carried to the phrase tail while the incoming bed stays ducked")
    if learned_controls:
        notes.append(f"runtime policy {transition_policy_status()} (critic and trainer offline)")
    return {
        "fromTrackId": first["id"], "toTrackId": second["id"], "exitAtSeconds": round(exit_at, 2),
        "enterAtSeconds": round(entry_at, 2), "overlapSeconds": round(overlap, 3), "bpmRatio": round(tempo_ratio, 4),
        "tempoFactor": round(tempo_factor, 4), "loopSeconds": round(loop_seconds, 3), "beatMatched": beat_matched,
        "barMatched": bar_matched, "phraseMatched": phrase_matched,
        "beatAlignmentErrorMs": round(beat_alignment_error_ms, 2),
        "beatPhaseErrorMs": round(beat_phase_error_ms, 2), "overlapBars": overlap_bars,
        "fadeShape": fade_shape, "technique": technique, "bassSwapProgress": bass_swap_progress,
        "vocalClashRisk": round(vocal_clash_risk, 3),
        "vocalBoundaryRisk": round(lyric_cut_risk, 3), "incomingVocalHead": round(incoming_vocal_head, 3),
        "learnedCompatibility": round(learned_compatibility, 4),
        "compatibilityComponents": compatibility_components,
        "fadeOutCurve": round(fade_out_curve, 4), "fadeInCurve": round(fade_in_curve, 4),
        "outgoingLowDb": round(outgoing_low_db, 3), "incomingLowDb": round(incoming_low_db, 3),
        "outgoingMidDb": round(outgoing_mid_db, 3), "incomingMidDb": round(incoming_mid_db, 3),
        "outgoingHighDb": round(outgoing_high_db, 3), "incomingHighDb": round(incoming_high_db, 3),
        "timingScore": round(float((learned_controls or {}).get("timing_score", 0.0)), 4),
        "policyVersion": transition_policy_status(),
        "spectrumPlan": "phrase-locked three-band hand-off with a single bass owner", "style": style,
        "qualityScore": round(score, 1), "renderQualityScore": None, "notes": notes,
    }


def ordered_tracks(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one continuous, low-to-peak energy progression.

    Genre describes a track but is not a sequencing constraint: a genre bucket
    can contain either warm-up material or a peak-time record. Ordering the
    whole crate by measured energy keeps adjacent records close in intensity;
    BPM and title only make equal-energy ties reproducible.
    """
    def energy(track: dict[str, Any]) -> float:
        try:
            value = float(track.get("energy", 0.5))
        except (TypeError, ValueError):
            value = 0.5
        return float(np.clip(value, 0.0, 1.0)) if math.isfinite(value) else 0.5

    def bpm(track: dict[str, Any]) -> float:
        try:
            value = float(track.get("bpm", 120.0))
        except (TypeError, ValueError):
            value = 120.0
        return value if math.isfinite(value) else 120.0

    return sorted(
        tracks,
        key=lambda track: (energy(track), bpm(track), str(track.get("title", "")).lower(), str(track.get("id", ""))),
    )


def duration_preference_for(index: int, current: dict[str, Any], upcoming: dict[str, Any]) -> float:
    """Shape deck time into a deterministic rise/dip journey, not one midpoint."""
    preference = DURATION_JOURNEY[index % len(DURATION_JOURNEY)]
    current_energy = float(current.get("energy", 0.5))
    incoming_energy = float(upcoming.get("energy", 0.5))
    if current_energy >= 0.78:
        preference -= 0.10
    if incoming_energy >= current_energy + 0.16:
        preference -= 0.07
    elif current_energy <= 0.42 and incoming_energy <= current_energy:
        preference += 0.08
    return float(np.clip(preference, 0.18, 0.88))


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


def overlay_with_headroom(
    outgoing: AudioSegment, incoming: AudioSegment, ceiling_dbfs: float = TRANSITION_OVERLAY_CEILING_DBFS,
) -> tuple[AudioSegment, dict[str, float]]:
    """Sum two decks in float and apply gain before integer PCM can clip.

    ``AudioSegment.overlay`` mixes in the segment's integer sample format. If
    two aligned kicks momentarily sum above full scale, clipping occurs there
    and a later master limiter can only limit the already-distorted result.
    This function observes the actual sample sum first, applies only the gain
    necessary to keep a conservative transition ceiling, then converts once.
    """
    if not outgoing or not incoming:
        bridge = outgoing.overlay(incoming)
        return bridge, {"preOverlayPeakDbfs": float(bridge.max_dBFS), "overlayGainDb": 0.0}
    incoming = incoming.set_frame_rate(outgoing.frame_rate).set_channels(outgoing.channels).set_sample_width(outgoing.sample_width)
    channels = max(1, outgoing.channels)
    raw_outgoing = np.asarray(outgoing.get_array_of_samples())
    raw_incoming = np.asarray(incoming.get_array_of_samples())
    frames = min(raw_outgoing.size // channels, raw_incoming.size // channels)
    if frames <= 0:
        bridge = outgoing.overlay(incoming)
        return bridge, {"preOverlayPeakDbfs": float(bridge.max_dBFS), "overlayGainDb": 0.0}
    sample_count = frames * channels
    # max_possible_amplitude matches pydub's actual integer container width,
    # including its 24-bit-to-32-bit conversion behaviour.
    scale = max(float(outgoing.max_possible_amplitude), 1.0)
    mixed = raw_outgoing[:sample_count].astype(np.float64) / scale
    mixed += raw_incoming[:sample_count].astype(np.float64) / scale
    pre_overlay_peak = float(np.max(np.abs(mixed)))
    ceiling = 10 ** (min(-0.05, float(ceiling_dbfs)) / 20)
    linear_gain = min(1.0, ceiling / max(pre_overlay_peak, 1e-12))
    guarded = mixed * linear_gain
    dtype = raw_outgoing.dtype
    limits = np.iinfo(dtype)
    container = np.clip(np.rint(guarded * scale), limits.min, limits.max).astype(dtype)
    bridge = outgoing._spawn(container.tobytes())
    return bridge, {
        "preOverlayPeakDbfs": round(20 * math.log10(max(pre_overlay_peak, 1e-12)), 3),
        "overlayGainDb": round(20 * math.log10(max(linear_gain, 1e-12)), 3),
    }


def low_frequency_phase_offset_ms(
    outgoing: AudioSegment, incoming: AudioSegment, maximum_offset_ms: int = 24,
) -> int:
    """Find a small, confidence-gated kick-phase correction on rendered audio.

    Beat grids constrain the planned timing, while this final pass observes the
    actual low-frequency transients after FFmpeg's pitch-preserving stretch.
    It searches only a tiny neighbourhood and retains zero offset unless the
    low-band onset correlation has a clear improvement, so unrelated basslines
    cannot pull an otherwise valid transition off-grid.
    """
    if not outgoing or not incoming or outgoing.frame_rate <= 0:
        return 0

    def onset_envelope(segment: AudioSegment) -> np.ndarray:
        filtered = low_pass_filter(segment, 240)
        raw = np.asarray(filtered.get_array_of_samples(), dtype=np.float32)
        if filtered.channels > 1:
            raw = raw.reshape((-1, filtered.channels)).mean(axis=1)
        if raw.size < filtered.frame_rate // 2:
            return np.empty(0, dtype=np.float32)
        frame = max(1, round(filtered.frame_rate * 0.005))
        usable = raw.size // frame * frame
        rms = np.sqrt(np.mean(np.square(raw[:usable].reshape((-1, frame))), axis=1) + 1e-9)
        # Kicks have a short positive envelope edge.  Retain a small RMS term
        # for rounder four-on-the-floor kicks whose attack was compressed.
        changes = np.maximum(0.0, np.diff(np.log(rms + 1e-6), prepend=np.log(rms[0] + 1e-6)))
        envelope = changes + 0.12 * rms / max(float(np.percentile(rms, 90)), 1e-6)
        return (envelope - np.mean(envelope)) / max(float(np.std(envelope)), 1e-6)

    out = onset_envelope(outgoing)
    inc = onset_envelope(incoming)
    size = min(out.size, inc.size)
    if size < 48:
        return 0
    out, inc = out[:size], inc[:size]
    frames = max(1, round(maximum_offset_ms / 5))

    def score(offset: int) -> float:
        if offset > 0:
            left, right = out[offset:], inc[:-offset]
        elif offset < 0:
            left, right = out[:offset], inc[-offset:]
        else:
            left, right = out, inc
        if left.size < 32:
            return -1.0
        return float(np.dot(left, right) / max(left.size, 1))

    baseline = score(0)
    best_offset = max(range(-frames, frames + 1), key=score)
    best_score = score(best_offset)
    # A phase correction must be both meaningful and supported by an audible
    # rhythmic correlation. Otherwise keeping the grid-derived zero offset is
    # safer for breaks, vocal intros, and syncopated basslines.
    if best_offset == 0 or best_score < 0.18 or best_score - baseline < 0.045:
        return 0
    return int(best_offset * 5)


def apply_phase_offset(segment: AudioSegment, offset_ms: int) -> AudioSegment:
    """Delay (positive) or advance (negative) a deck without changing length."""
    if offset_ms == 0:
        return segment
    amount = min(abs(int(offset_ms)), max(0, len(segment) // 8))
    if amount == 0:
        return segment
    silence = AudioSegment.silent(duration=amount, frame_rate=segment.frame_rate)
    silence = silence.set_channels(segment.channels).set_sample_width(segment.sample_width)
    if offset_ms > 0:
        return (silence + segment)[:len(segment)]
    return (segment[amount:] + silence)[:len(segment)]


def fade_gains(
    progress: float,
    fade_shape: str,
    fade_out_curve: float | None = None,
    fade_in_curve: float | None = None,
) -> tuple[float, float]:
    """Return non-linear equal-power gains for a phrase-aware crossfade."""
    if fade_shape == "vocal-carry":
        # Keep the outgoing sentence intelligible through most of the phrase,
        # then release it quickly at the musical boundary. The incoming deck
        # can establish rhythm underneath without masking the words.
        outgoing_progress = float(np.clip((progress - 0.62) / 0.38, 0, 1)) ** 0.78
        incoming_progress = progress ** 1.28
    elif fade_out_curve is not None and fade_in_curve is not None:
        outgoing_progress = progress ** float(np.clip(fade_out_curve, 0.35, 2.5))
        incoming_progress = progress ** float(np.clip(fade_in_curve, 0.35, 2.5))
    elif fade_shape == "fast-outgoing":
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
    outgoing: AudioSegment, incoming: AudioSegment, overlap_ms: int, transition: dict[str, Any],
) -> tuple[AudioSegment, AudioSegment]:
    """Trade spectrum on a phrase clock while keeping one bass owner.

    The old blend gradually released incoming bass while outgoing bass stayed
    present, creating a long double-kick region.  This envelope makes a smooth
    but decisive bass swap at a configured bar position and applies all other
    gain/filter changes continuously around it.
    """
    fade_shape = str(transition.get("fadeShape", "equal-power"))
    technique = str(transition.get("technique", "long-eq-blend"))
    swap_at = float(np.clip(transition.get("bassSwapProgress", 0.6), 0.35, 0.8))
    fade_out_curve = transition.get("fadeOutCurve")
    fade_in_curve = transition.get("fadeInCurve")
    outgoing_low_db = float(transition.get("outgoingLowDb", -18.0))
    incoming_low_db = float(transition.get("incomingLowDb", -18.0))
    outgoing_mid_db = float(transition.get("outgoingMidDb", -3.0))
    incoming_mid_db = float(transition.get("incomingMidDb", -3.0))
    outgoing_high_db = float(transition.get("outgoingHighDb", -5.0))
    incoming_high_db = float(transition.get("incomingHighDb", -5.0))
    chunks = min(128, max(32, overlap_ms // 55))
    out_parts: list[AudioSegment] = []
    in_parts: list[AudioSegment] = []
    for index in range(chunks):
        start = index * overlap_ms // chunks
        end = (index + 1) * overlap_ms // chunks
        progress = (index + 1) / chunks
        swap_phase = float(np.clip((progress - (swap_at - 0.10)) / 0.20, 0, 1))
        swap_curve = swap_phase * swap_phase * (3 - 2 * swap_phase)
        if technique == "vocal-carry-bed":
            outgoing_low_pass_hz = int(18_500 - 5_000 * progress)
        else:
            outgoing_low_pass_hz = int(18_500 - 13_500 * progress)
        outgoing_high_pass_hz = int(45 + max(120, abs(outgoing_low_db) * 16) * swap_curve)
        incoming_high_pass_hz = int(45 + max(120, abs(incoming_low_db) * 16) * (1 - swap_curve))
        outgoing_gain, incoming_gain = fade_gains(
            progress, fade_shape,
            float(fade_out_curve) if fade_out_curve is not None else None,
            float(fade_in_curve) if fade_in_curve is not None else None,
        )
        # The learned broad-band targets complement the low-frequency handoff.
        # Weight them conservatively to avoid zipper noise in short chunks.
        outgoing_gain += progress * (0.18 * outgoing_mid_db + 0.08 * outgoing_high_db)
        incoming_gain += (1 - progress) * (0.18 * incoming_mid_db + 0.08 * incoming_high_db)
        if technique == "vocal-guarded-eq":
            # Hold the incoming record deeper until the outgoing vocal has
            # cleared, then complete the same phrase-boundary bass exchange.
            incoming_gain -= (1 - swap_curve) * 4.5
        elif technique == "vocal-carry-bed":
            carry_release = float(np.clip((progress - 0.58) / 0.30, 0, 1))
            carry_release = carry_release * carry_release * (3 - 2 * carry_release)
            incoming_gain -= (1 - carry_release) * 6.0
        outgoing_chunk = low_pass_filter(outgoing[start:end], max(4_000, outgoing_low_pass_hz))
        outgoing_chunk = high_pass_filter(outgoing_chunk, max(45, outgoing_high_pass_hz)).apply_gain(outgoing_gain)
        incoming_chunk = high_pass_filter(incoming[start:end], max(45, incoming_high_pass_hz)).apply_gain(incoming_gain)
        out_parts.append(outgoing_chunk)
        in_parts.append(incoming_chunk)
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
        phase_offset_ms = low_frequency_phase_offset_ms(outgoing, incoming[:overlap_ms])
        if phase_offset_ms:
            incoming = apply_phase_offset(incoming, phase_offset_ms)
        transition["lowFrequencyPhaseOffsetMs"] = phase_offset_ms
    filtered_outgoing, filtered_incoming = progressive_spectrum_blend(
        outgoing, incoming[:overlap_ms], overlap_ms, transition,
    )
    bridge, overlay_metrics = overlay_with_headroom(filtered_outgoing, filtered_incoming)
    transition.update(overlay_metrics)
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
    candidates = ordered_tracks(tracks)
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
        screening = transition_between(
            kept[-1], candidate, 0.0, options["minTrackSeconds"], options["maxTrackSeconds"],
            duration_preference=duration_preference_for(len(kept) - 1, kept[-1], candidate),
        )
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
            duration_preference_for(index, current["track"], upcoming["track"]),
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
        0.22,
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
