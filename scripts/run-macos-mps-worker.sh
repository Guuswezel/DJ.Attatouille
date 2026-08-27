#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
venv_path="${MPS_VENV_PATH:-$project_root/.venv-mps}"

if [ -f "$project_root/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$project_root/.env"
  set +a
fi
if [ ! -x "$venv_path/bin/python" ]; then
  echo "Run ./scripts/setup-macos-mps-worker.sh first." >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "FFmpeg is required. Install it with: brew install ffmpeg libsndfile" >&2
  exit 1
fi

music_path="${MUSIC_LIBRARY_PATH:-$project_root/music}"
data_path="${DJ_DATA_PATH:-$project_root/.dj-attatouille-data}"
case "$music_path" in /*) ;; *) music_path="$project_root/$music_path" ;; esac
case "$data_path" in /*) ;; *) data_path="$project_root/$data_path" ;; esac
mkdir -p "$data_path"

export MUSIC_ROOT="$music_path"
export REQUEST_MUSIC_ROOT=/music
export DATA_ROOT="$data_path"
# Do not inherit the Docker-only qdrant hostname from .env. Native macOS code
# reaches the loopback ports published by docker-compose.macos.yml instead.
export QDRANT_URL="${MPS_QDRANT_URL:-http://127.0.0.1:${QDRANT_HOST_PORT:-6333}}"
export BACKEND_URL="${MPS_BACKEND_URL:-http://127.0.0.1:${BACKEND_HOST_PORT:-8080}}"
export ANALYSIS_MODEL="${ANALYSIS_MODEL:-harmonix}"
export ANALYSIS_DEVICE=mps
# Fast uses lightweight harmonic/percussive proxy stems for preparation; full
# remains available for a deliberately slow HTDemucs source-separation pass.
export HARMONIX_PROFILE="${HARMONIX_PROFILE:-fast}"
export ANALYSIS_SAMPLE_RATE="${ANALYSIS_SAMPLE_RATE:-22050}"
export HARMONIX_PROXY_SAMPLE_RATE="${HARMONIX_PROXY_SAMPLE_RATE:-22050}"
# all-in-one-infer imports matplotlib; keep its cache in the shared writable
# app-data folder rather than macOS's protected home-directory location.
export MPLCONFIGDIR="$data_path/.matplotlib"
mkdir -p "$MPLCONFIGDIR"
# A few auxiliary operators still lack an MPS kernel in PyTorch. This keeps
# model inference on Metal while safely using CPU only for those operations.
export PYTORCH_ENABLE_MPS_FALLBACK=1

cd "$project_root/worker"
exec "$venv_path/bin/uvicorn" app.main:app --host 127.0.0.1 --port 8090
