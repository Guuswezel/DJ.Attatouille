#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
venv_path="${MPS_VENV_PATH:-$project_root/.venv-mps}"
yt_dlp_binary="${DJ_YT_DLP_BIN:-$project_root/.tools/yt-dlp_macos}"

if [ ! -x "$venv_path/bin/python" ]; then
  "$project_root/scripts/setup-macos-mps-worker.sh"
fi
"$venv_path/bin/pip" install --upgrade -r "$project_root/trainer/requirements.txt"
mkdir -p "$(dirname "$yt_dlp_binary")"
curl --fail --location --retry 3 \
  https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos \
  --output "$yt_dlp_binary.download"
chmod 755 "$yt_dlp_binary.download"
mv "$yt_dlp_binary.download" "$yt_dlp_binary"
DJ_YT_DLP_BIN="$yt_dlp_binary" PYTHONPATH="$project_root/trainer" "$venv_path/bin/python" -c "import torch; from dj_train.dataset import javascript_runtime_args, yt_dlp_command; assert torch.backends.mps.is_available(), 'MPS is unavailable'; print('Transition trainer ready:', yt_dlp_command(), javascript_runtime_args())"
"$yt_dlp_binary" --version
