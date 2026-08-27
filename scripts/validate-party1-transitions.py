#!/usr/bin/env python3
"""Render and verify a deterministic three-track Party1 transition chain."""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import librosa
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "worker"))

from pydub import AudioSegment  # noqa: E402

from app.main import (  # noqa: E402
    ANALYSIS_SAMPLE_RATE,
    MUSIC_ROOT,
    blend,
    build_phrase_states,
    clip_for,
    gain_trim_db,
    master_bus,
    pitch_preserving_time_stretch,
    transition_between,
)

TRACK_TITLES = [
    "Strobe (Original Mix)",
    "Lucky Day (Remix) (Ian Pooley Remix)",
    "Channeling (Original Mix)",
]


def main() -> None:
    api_url = os.environ.get("DJ_API_URL", "http://127.0.0.1:8080").rstrip("/")
    with urllib.request.urlopen(f"{api_url}/api/overview", timeout=15) as response:
        overview = json.load(response)
    preparation = next(
        item for item in overview["preparations"]
        if item["sourcePath"] == "/music/Party1" and item["status"] == "ready"
    )
    by_title = {track["title"]: track for track in preparation["tracks"]}
    tracks = [by_title[title] for title in TRACK_TITLES]
    # Existing preparations can predate phrase-state persistence. Recompute
    # only the cheap DSP state for these three real tracks while reusing their
    # already measured Harmonix beat/downbeat grids and structural segments.
    for track in tracks:
        source = MUSIC_ROOT / track["relativePath"]
        y, sr = librosa.load(source, sr=ANALYSIS_SAMPLE_RATE, mono=True)
        hop = 512
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        spectral = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        phrase_states = build_phrase_states(
            y, sr, hop, rms, spectral, np.asarray(track["downbeats"], dtype=float),
            float(track["durationSeconds"]), track["segments"], np.asarray(track["beatGrid"], dtype=float),
        )
        if not phrase_states:
            raise RuntimeError(f"No eight-bar phrase states were extracted for {track['title']}")
        track["phraseStates"] = phrase_states
        track["cues"]["phraseBoundaries"] = sorted({
            float(state[edge]) for state in phrase_states for edge in ("start", "end")
        })

    previews: list[AudioSegment] = []
    measurements: list[dict[str, object]] = []
    source_start = 0.0
    running_bpm = float(tracks[0]["bpm"])
    for first, second in zip(tracks, tracks[1:]):
        plan = transition_between(first, second, source_start, 90.0, 180.0, running_bpm)
        if plan is None:
            raise RuntimeError(f"No transition found for {first['title']} -> {second['title']}")
        if not (plan["beatMatched"] and plan["barMatched"] and plan["phraseMatched"]):
            raise RuntimeError(f"Timing lock failed: {json.dumps(plan, indent=2)}")
        if plan["overlapBars"] not in {8, 16, 24, 32} or float(plan["beatAlignmentErrorMs"]) > 45:
            raise RuntimeError(f"Phrase overlap is outside the DJ timing gate: {json.dumps(plan, indent=2)}")

        outgoing_factor = running_bpm / max(float(first["bpm"]), 1.0)
        overlap = float(plan["overlapSeconds"])
        outgoing_source_span = overlap * outgoing_factor
        outgoing_start = max(source_start, float(plan["exitAtSeconds"]) - outgoing_source_span - 8 * outgoing_factor)
        outgoing = clip_for(first, outgoing_start, float(plan["exitAtSeconds"])).apply_gain(gain_trim_db(first))

        incoming_factor = float(plan["tempoFactor"])
        incoming_end = min(
            float(second["durationSeconds"]) - 0.25,
            float(plan["enterAtSeconds"]) + (overlap + 8) * incoming_factor,
        )
        incoming = clip_for(second, float(plan["enterAtSeconds"]), incoming_end).apply_gain(gain_trim_db(second))
        incoming = pitch_preserving_time_stretch(incoming, incoming_factor)
        phrase_duration_error_ms = abs(outgoing_source_span / outgoing_factor * 1000 - overlap * 1000)
        if phrase_duration_error_ms > 2:
            raise RuntimeError(f"Rendered phrase duration error is {phrase_duration_error_ms:.2f} ms")

        preview, _ = blend(outgoing, incoming, plan)
        if float(plan["renderQualityScore"]) < 70:
            raise RuntimeError(f"Rendered transition quality failed: {json.dumps(plan, indent=2)}")
        previews.append(preview)
        measurements.append({
            "from": first["title"], "to": second["title"],
            "exitAtSeconds": plan["exitAtSeconds"], "enterAtSeconds": plan["enterAtSeconds"],
            "tempoFactor": plan["tempoFactor"], "overlapBars": plan["overlapBars"],
            "beatAlignmentErrorMs": plan["beatAlignmentErrorMs"],
            "renderQualityScore": plan["renderQualityScore"], "technique": plan["technique"],
            "outgoingPhraseStates": len(first["phraseStates"]),
            "incomingPhraseStates": len(second["phraseStates"]),
        })
        source_start = float(plan["enterAtSeconds"])
        running_bpm = float(second["bpm"]) * incoming_factor

    combined = previews[0] + AudioSegment.silent(duration=1_500) + previews[1]
    combined = master_bus(combined)
    output_dir = Path(os.environ.get("DJ_VALIDATION_DIR", PROJECT_ROOT / ".dj-attatouille-data" / "validation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "party1-three-track-transitions.mp3"
    combined.export(output_path, format="mp3", bitrate="192k", parameters=["-ar", "44100"])
    print(json.dumps({"output": str(output_path), "transitions": measurements}, indent=2))


if __name__ == "__main__":
    main()
