from __future__ import annotations

import os
from pathlib import Path

FEATURE_VERSION = "dj-attatouille-transition-v1"
SAMPLE_RATE = 44_100
FEATURE_SAMPLE_RATE = 16_000
CHANNELS = 2
MEL_BINS = 64
MIR_DIM = 24
CONTEXT_BEFORE_BARS = 16
TRANSITION_BARS = 32
CONTEXT_AFTER_BARS = 16
TOTAL_BARS = CONTEXT_BEFORE_BARS + TRANSITION_BARS + CONTEXT_AFTER_BARS
# Training audio is moved to a common 120-BPM clock with a phase vocoder, not
# sample-rate conversion. Every beat therefore has 8,000 samples while pitch
# and meaningful low/mid/high frequency bands remain unchanged.
MODEL_BPM = 120
SAMPLES_PER_BEAT = int(FEATURE_SAMPLE_RATE * 60 / MODEL_BPM)
SAMPLES_PER_BAR = SAMPLES_PER_BEAT * 4
TARGET_LUFS = -14.0
TRUE_PEAK_DB = -1.0

DATASET_ROOT = Path(os.environ.get("TRANSITION_DATASET_ROOT", "/training")).resolve()
MUSIC_ROOT = Path(os.environ.get("MUSIC_ROOT", "/music")).resolve()
POLICY_OUTPUT = Path(
    os.environ.get("TRANSITION_POLICY_OUTPUT", str(DATASET_ROOT / "exports" / "transition-policy-v1.json"))
).resolve()

POLICY_FEATURE_NAMES = [
    "tempo_ratio",
    "harmonic_compatibility",
    "outgoing_energy",
    "incoming_energy",
    "energy_delta",
    "outgoing_energy_slope",
    "incoming_energy_slope",
    "slope_delta",
    "outgoing_bass",
    "incoming_bass",
    "bass_collision",
    "outgoing_drums",
    "incoming_drums",
    "outgoing_vocals",
    "incoming_vocals",
    "vocal_collision",
    "outgoing_spectral_density",
    "incoming_spectral_density",
    "outgoing_harmonic_density",
    "incoming_harmonic_density",
    "outgoing_novelty",
    "incoming_novelty",
    "outgoing_loopability",
    "incoming_cue_confidence",
    "outgoing_elapsed_fraction",
    "outgoing_remaining_fraction",
    "incoming_entry_fraction",
    "genre_compatibility",
    "outgoing_cue_confidence",
]

POLICY_CONTROL_NAMES = [
    "overlap_phrases",
    "bass_swap_progress",
    "fade_out_curve",
    "fade_in_curve",
    "outgoing_low_db",
    "incoming_low_db",
    "outgoing_mid_db",
    "incoming_mid_db",
    "outgoing_high_db",
    "incoming_high_db",
    "loop_probability",
    "timing_score",
]
