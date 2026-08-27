#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
# The Command Line Tools interpreter is arm64 on Apple Silicon and PyTorch
# 2.5 supports its Python 3.9 runtime. A native Homebrew Python can be passed
# explicitly with PYTHON_BIN when desired.
python_bin="${PYTHON_BIN:-/usr/bin/python3}"
venv_path="${MPS_VENV_PATH:-$project_root/.venv-mps}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "A native arm64 Python 3.9+ interpreter is required. Set PYTHON_BIN to one." >&2
  exit 1
fi
python_architecture="$($python_bin -c 'import platform; print(platform.machine())')"
if [ "$python_architecture" != "arm64" ]; then
  echo "$python_bin is $python_architecture. MPS requires an arm64 Python, not an Intel/Rosetta interpreter." >&2
  exit 1
fi
if ! "$python_bin" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)'; then
  echo "Python 3.9 or newer is required for the MPS worker." >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "FFmpeg is required. Install it with: brew install ffmpeg libsndfile" >&2
  exit 1
fi

"$python_bin" -m venv "$venv_path"
"$venv_path/bin/python" -m pip install --upgrade pip
"$venv_path/bin/pip" install -r "$project_root/worker/requirements.txt"
# PyPI publishes the Apple Silicon build with Metal/MPS support; unlike the
# Docker image it must not use the Linux CPU or CUDA wheel indexes.
"$venv_path/bin/pip" install "torch==2.5.1" "torchaudio==2.5.1" all-in-one-infer
"$venv_path/bin/python" -c "import torch; assert torch.backends.mps.is_available(), 'MPS is not available to this Python installation'; print('MPS ready:', torch.backends.mps.is_available())"

echo "MPS worker environment ready at $venv_path"
