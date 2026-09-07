#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
venv_path="${MPS_VENV_PATH:-$project_root/.venv-mps}"
export DJ_YT_DLP_BIN="${DJ_YT_DLP_BIN:-$project_root/.tools/yt-dlp_macos}"

if [ -f "$project_root/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$project_root/.env"
  set +a
fi
if [ ! -x "$venv_path/bin/python" ]; then
  echo "Run ./scripts/setup-macos-transition-trainer.sh first." >&2
  exit 1
fi
export PYTHONPATH="$project_root/trainer"
if ! "$venv_path/bin/python" -c "from dj_train.dataset import javascript_runtime_args, yt_dlp_command; yt_dlp_command(); javascript_runtime_args()" >/dev/null 2>&1; then
  echo "The YouTube downloader or its JavaScript runtime is unavailable. Run ./scripts/setup-macos-transition-trainer.sh once." >&2
  exit 1
fi

training_path="${TRANSITION_TRAINING_PATH:-$project_root/.dj-attatouille-training}"
data_path="${DJ_DATA_PATH:-$project_root/.dj-attatouille-data}"
case "$training_path" in /*) ;; *) training_path="$project_root/$training_path" ;; esac
case "$data_path" in /*) ;; *) data_path="$project_root/$data_path" ;; esac
mkdir -p "$training_path" "$data_path/models"

export TRANSITION_DATASET_ROOT="$training_path"
export TRANSITION_POLICY_OUTPUT="$data_path/models/transition-policy-v1.json"
export TRANSITION_TRAIN_DEVICE="${TRANSITION_TRAIN_DEVICE:-mps}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$project_root"
exec "$venv_path/bin/python" -m dj_train.cli "$@"
