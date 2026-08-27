# DJ.Attatouille

A local-first, standalone music player for small parties. It prepares a mounted music library, plans a genre-led mix, renders a continuous MP3 with monitored transitions, and serves a responsive two-deck player.

No API key is required and music is never uploaded. MongoDB stores preparation and mix records; Qdrant stores local music embeddings for transition similarity search.

## Start it

1. Copy `.env.example` to `.env`.
2. Set `MUSIC_LIBRARY_PATH` to the absolute path of the folder that contains the music you want to use. Docker mounts it read-only as `/music`.
3. Run `docker compose up --build`.
4. Open `http://localhost:4173` (or the `DJ_PORT` set in `.env`).

The app can prepare `/music` or any nested folder such as `/music/Friday`. Multiple preparations and mixes stay available in the sidebar while work continues in the background.

## What preparation extracts

- Metadata and embedded cover art
- BPM, musical key, EBU R128 integrated loudness, energy, beat/downbeat/bar grids, explicit eight-bar phrase states, a compact waveform envelope, and a high-resolution 8-bit signal with at least 16 samples per beat
- Genres from tags, with an offline acoustic fallback
- A 512D track vector in Qdrant, derived from Harmonix/all-in-one embeddings where available and MFCC/chroma/spectral features otherwise

Harmonix is the default analysis model, via the maintained `all-in-one-infer` implementation. It runs locally, retains its downloaded weights in the local cache, and needs no API key. Its pure-PyTorch attention path avoids the legacy NATTEN and native `madmom` build chain. `ANALYSIS_DEVICE=auto` uses CUDA automatically when the container has an NVIDIA GPU; set `ANALYSIS_MODEL=librosa` only when deliberately opting into the lightweight local fallback.

Preparation uses `HARMONIX_PROFILE=fast` by default. It decodes a 22.05 kHz mono proxy, separates only harmonic/percussive and frequency bands, and feeds those four compact spectral trends into the learned Harmonix heads. This preserves beat/downbeat grids, phrase/section energy, genre features, and embeddings while skipping full-track HTDemucs source separation—the step responsible for multi-minute preparation times. The proxy is re-clocked to Harmonix's 44.1 kHz feature timing, so cue timestamps remain exact. Set `HARMONIX_PROFILE=full` only when you want the slower full HTDemucs stem pass; it is unnecessary for normal mix preparation.

### NVIDIA GPU acceleration

On a Linux NVIDIA machine with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed, build and run with the GPU override:

> **macOS Docker Desktop:** do not use this override. It requires an NVIDIA GPU driver and will fail with `could not select device driver "nvidia"`. Use the normal `docker compose up -d --build` command instead; Docker's Linux VM cannot access the Apple GPU.

```sh
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

That override installs the CUDA build of PyTorch, exposes all NVIDIA GPUs to the worker, and requires `ANALYSIS_DEVICE=cuda`, so the Harmonix model uses the GPU rather than silently falling back. Core decoding, metadata reads, loudness measurement, and some librosa feature extraction remain CPU/IO work; the substantially heavier Harmonix structure and embedding inference runs on CUDA. Docker Desktop for macOS cannot pass through the Apple GPU to Linux containers, so the standard Compose stack intentionally uses CPU there.

### Apple Silicon GPU acceleration (M1/M2/M3/M4)

On an Apple Silicon Mac, the worker must run natively so PyTorch can use Metal/MPS; Docker Desktop cannot forward that GPU into its Linux VM. The rest of the application remains in Docker. This mode uses `./.dj-attatouille-data` as a shared generated-media folder and leaves the standard Docker volume untouched.

First install native prerequisites and create the MPS environment:

```sh
brew install ffmpeg libsndfile
./scripts/setup-macos-mps-worker.sh
```

The setup uses macOS's native arm64 Python 3.9+ by default. Do not point `PYTHON_BIN` at an Intel/Rosetta Python under `/usr/local`; it cannot access MPS.

Then use two terminals. Start the native MPS worker in the first (keep it open):

```sh
./scripts/run-macos-mps-worker.sh
```

Start the Docker services with the macOS override in the second:

```sh
docker compose -f docker-compose.yml -f docker-compose.macos.yml up -d --build
```

Verify `http://localhost:8090/health` returns `"analysisDevice":"mps"`. The override deliberately does not start the Linux worker, exposes backend/Qdrant only on `127.0.0.1` for the native worker, and maps the API to the native worker through `host.docker.internal`.

Current PyTorch MPS cannot execute one large HTDemucs convolution. The default fast profile avoids that stage entirely: it derives compact proxy stems on CPU, then runs Harmonix’s structure and embedding model on Metal. If you explicitly select `HARMONIX_PROFILE=full`, its four-stem HTDemucs pass stays on CPU before the Harmonix pass. The MPS default is `harmonix-fold0`, one learned Harmonix fold that fits M2 unified memory reliably. Set `HARMONIX_MODEL=harmonix-all` only on an Apple Silicon machine with sufficient free unified memory for all eight folds.

## Mix and playback behavior

The mix form lets a user drag genres into the desired journey, select the minimum/maximum time on deck, and choose the lowest acceptable percentage of tracks to keep. Preparation establishes an explicit eight-bar phrase clock from the downbeat grid, anchored by learned structure changes. Each phrase persists its energy/slope, bass, drums, vocal-band activity, spectral/harmonic density, novelty, loopability, and cue confidence. The controller pairs outgoing and incoming phrase states while protecting the middle of drops, peaks, and builds.

Tempo corrections use FFmpeg's pitch-preserving `atempo` filter—never a sample-rate change—so BPM changes do not shift voices, keys, or other frequencies. An overlapping transition is allowed only when beat phase, bar phase, and both eight-bar phrase endpoints lock. Tempo is derived from the actual detected phrase spans; the 95th-percentile rendered beat-grid residual must stay below 45 ms. Valid transitions use a full eight-bar deterministic EQ blend with one bass owner and a smooth bass swap. If that timing gate fails, the records meet at a phrase boundary without overlapping drifting kick grids. Every rendered overlap is level/clipping monitored before the mix is marked ready.

Preparation also measures whole-track EBU R128 loudness locally. The renderer uses it as a virtual DJ gain knob: it applies a bounded per-track trim toward `DJ_TARGET_LUFS` before tempo processing and transitions, including on a live Next bridge. A conservative final master gain plus a −1 dBFS look-ahead limiter protects summed transition peaks without flattening the per-track level balance. The deck display shows the applied auto-gain; the target and bounds can be adjusted in `.env`.

Desktop shows the current and next decks. Mobile intentionally reduces this to the current rotating artwork plus volume, play/pause, and Next. Play and pause fade over two seconds. Next asks the backend to render a fresh short bridge from a safe current exit to the next track; it plays that bridge before resuming the prepared mix rather than jumping directly.

## Learning transitions from professional mixes

Transition learning is an explicit offline workflow. It is not started by normal Compose playback and it never runs in the live worker. A professional DJ mix and its timed tracklist are treated as weak supervision: each timestamp opens a search region, and a per-bar localizer refines the likely transition area using energy, bass movement, spectral flux, timbre change, beats, bars, and phrase context.

Every professional and artificial example passes through the same canonical contract before the critic sees it:

- 44.1 kHz stereo PCM, the same high/low bandwidth limits, −14 LUFS target, and −1 dB true-peak ceiling
- pitch-preserving beat-warped 16 kHz model input with 64 bars: 16 bars before, 32 transition bars, and 16 bars after
- a common 120-BPM model clock with exactly 8,000 samples per beat, plus 64-bin mel and 24-value MIR features per bar
- identical random gain, broad EQ, bandwidth, quantization/codec proxy, noise, and mastering augmentation in both domains

The offline stages are deliberately separate:

1. A multiple-instance localizer learns exact transition regions from weak ±45-second timestamp bags and negatives far away from a tracklist boundary.
2. A context critic learns professional versus artificial transitions and diagnostic heads for phrase alignment, energy smoothness, bass separation, spectral smoothness, duration, strength, and location.
3. A differentiable PyTorch mixer lets a compact policy learn phrase-candidate timing, overlap length, non-linear fade curves, three-band handoff, bass-swap location, and a relaxed loop decision against the frozen critic. At runtime it ranks only already-safe phrase candidates; tempo itself remains a pitch-preserving hard constraint.
4. Optional 1–5 human ratings calibrate the critic before another policy optimization pass.

The exported `transition-policy-v1.json` contains only a small three-layer policy. The live worker hot-loads that JSON with NumPy; PyTorch training models, the professional audio, localizer, and critic remain offline. Beat residual, downbeat, bar, phrase, tempo-range, high-energy protection, and peak gates still have final authority over every learned proposal.

### Apple Silicon/M2 training

Use the native Metal environment because Docker Desktop cannot expose the M2 GPU:

```sh
./scripts/setup-macos-transition-trainer.sh

# Put a timed tracklist at training-input/my-set.txt, then ingest a public mix.
./scripts/transition-trainer-macos.sh ingest-professional \
  --name "Reference set 01" \
  --url "https://www.youtube.com/watch?v=..." \
  --tracklist training-input/my-set.txt

# Generate artificial examples from the same local tracks used by the mixer.
./scripts/transition-trainer-macos.sh synthesize --folder music/Party1

# Check that both domains have exactly the same representation, then train.
./scripts/transition-trainer-macos.sh verify
./scripts/transition-trainer-macos.sh train --stage all --device mps
```

For human tuning, `export-previews` writes the same pitch-preserving canonical WAV representation for both domains. Ratings use JSONL rows such as `{"sampleId":"…","rating":4}` and are applied with `human-feedback --feedback ratings.jsonl`; run the policy stage again afterward.

The worker notices the exported policy in `.dj-attatouille-data/models/transition-policy-v1.json` without a restart. Run `curl http://127.0.0.1:8090/health`; `transitionPolicy` changes from `deterministic-fallback` to the exported policy version.

The downloader only handles publicly accessible, non-DRM media. A local audio file can be used instead with `--audio /path/to/mix.mp3`. Make sure you have the right to use reference recordings for training.

### Compose training profile

The same commands are available in a CPU-compatible, opt-in Compose profile. Put tracklists in `training-input/` so they appear under `/input`:

```sh
docker compose --profile training run --rm trainer ingest-professional \
  --name "Reference set 01" --url "https://www.youtube.com/watch?v=..." \
  --tracklist /input/my-set.txt
docker compose --profile training run --rm trainer synthesize --folder /music/Party1
docker compose --profile training run --rm trainer verify
docker compose --profile training run --rm trainer train --stage all
```

On Apple Silicon this Compose profile is useful for reproducibility but is CPU-only; use the native commands above for MPS. Training data persists in `.dj-attatouille-training`, while only the compact policy is exported into the runtime data folder.

## Architecture

| Component | Responsibility |
| --- | --- |
| `frontend/` | React, Vite, Tailwind player and preparation UI |
| `backend/src/routes.rs` | Rust HTTP handlers and background orchestration |
| `backend/src/models.rs` | API, MongoDB, and worker contract types |
| `backend/src/repository.rs` | MongoDB queries and updates |
| `backend/src/worker_client.rs` | Typed calls to the local analysis/render worker |
| `worker/` | Python audio analysis, Qdrant indexing, mix rendering, and skip bridges |
| `trainer/` | Offline weak localization, canonical datasets, critic, differentiable mixer, policy training/export |
| `docker-compose.yml` | Frontend, Rust API, worker, MongoDB, and Qdrant |

## Validation

```sh
cd backend && cargo check
cd ../frontend && npm install && npm run build
cd ../worker && python -m unittest discover -s tests
cd ../trainer && PYTHONPATH=. python -m unittest discover -s tests
docker compose config --quiet
```

The real-audio Party1 regression renders three fixed local tracks and fails if either transition loses its beat/bar/phrase lock or its acoustic quality threshold:

```sh
MUSIC_ROOT="$PWD/music" DATA_ROOT="$PWD/.dj-attatouille-data" \
QDRANT_URL=http://127.0.0.1:6333 \
./.venv-mps/bin/python scripts/validate-party1-transitions.py
```

The final worker test command is intended to run inside the Compose worker image (Python 3.11) when the host does not provide its audio dependencies:

```sh
docker compose run --rm worker python -m unittest discover -s tests
```
