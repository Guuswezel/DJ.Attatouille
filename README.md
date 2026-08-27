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
- BPM, musical key, EBU R128 integrated loudness, energy, beat-aligned phrase boundaries, segment energy, a compact waveform envelope, and a high-resolution 8-bit signal with at least 16 samples per beat
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

The mix form lets a user drag genres into the desired journey, select the minimum/maximum time on deck, and choose the lowest acceptable percentage of tracks to keep. The worker records beat grids and downbeats, then selects phrase-ending downbeats while protecting the middle of chorus/drop/peak sections. It pairs the outgoing and incoming section-energy profiles, scores key, tempo, genre, and embedding affinity, and only overlays beat grids when a bounded tempo correction can lock them together.

Tempo corrections use FFmpeg's pitch-preserving `atempo` filter—never a sample-rate change—so BPM changes do not shift voices, keys, or other frequencies. During a beat-aligned blend the renderer uses a 2/4/8-bar downbeat overlap, progressive outgoing low-pass/incoming bass-release filtering, and a beat-aligned one-bar loop only where the outgoing phrase is safe to repeat. Larger BPM gaps use a short filtered phrase hand-off instead of two drifting rhythms. Every rendered overlap is level/clipping monitored before the mix is marked ready.

Preparation also measures whole-track EBU R128 loudness locally. The renderer uses it as a virtual DJ gain knob: it applies a bounded per-track trim toward `DJ_TARGET_LUFS` before tempo processing and transitions, including on a live Next bridge. A conservative final master gain plus a −1 dBFS look-ahead limiter protects summed transition peaks without flattening the per-track level balance. The deck display shows the applied auto-gain; the target and bounds can be adjusted in `.env`.

Desktop shows the current and next decks. Mobile intentionally reduces this to the current rotating artwork plus volume, play/pause, and Next. Play and pause fade over two seconds. Next asks the backend to render a fresh short bridge from a safe current exit to the next track; it plays that bridge before resuming the prepared mix rather than jumping directly.

## Architecture

| Component | Responsibility |
| --- | --- |
| `frontend/` | React, Vite, Tailwind player and preparation UI |
| `backend/src/routes.rs` | Rust HTTP handlers and background orchestration |
| `backend/src/models.rs` | API, MongoDB, and worker contract types |
| `backend/src/repository.rs` | MongoDB queries and updates |
| `backend/src/worker_client.rs` | Typed calls to the local analysis/render worker |
| `worker/` | Python audio analysis, Qdrant indexing, mix rendering, and skip bridges |
| `docker-compose.yml` | Frontend, Rust API, worker, MongoDB, and Qdrant |

## Validation

```sh
cd backend && cargo check
cd ../frontend && npm install && npm run build
cd ../worker && python -m unittest discover -s tests
docker compose config --quiet
```

The final worker test command is intended to run inside the Compose worker image (Python 3.11) when the host does not provide its audio dependencies:

```sh
docker compose run --rm worker python -m unittest discover -s tests
```
