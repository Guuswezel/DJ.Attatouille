#!/usr/bin/env python3
"""Generate the figures embedded in PROJECT_PAPER.md from local artifacts.

The script intentionally uses only derived, low-resolution visualisations of
the local canonical training tensors.  It does not export or reproduce audio.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dj-attatouille-paper-matplotlib")

import matplotlib

# This report is rendered in headless CI/terminal environments as well as on a
# developer workstation.  Selecting Agg before pyplot prevents macOS GUI
# backend initialisation during figure generation.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
IMAGE_DIR = PAPER_DIR / "images"
TRAINING_ROOT = ROOT / ".dj-attatouille-training"
POLICY_PATH = ROOT / ".dj-attatouille-data" / "models" / "transition-policy-v1.json"
SAMPLE_RATE = 16_000
SAMPLES_PER_BEAT = 8_000
SAMPLES_PER_BAR = 32_000

COLORS = {
    "ink": "#162033",
    "navy": "#204A87",
    "blue": "#3B82F6",
    "teal": "#0F9D8A",
    "orange": "#EF8354",
    "purple": "#805AD5",
    "red": "#C2414B",
    "muted": "#64748B",
    "grid": "#D9E2EC",
    "paper": "#FBFCFE",
}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": COLORS["grid"],
            "axes.linewidth": 0.8,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["ink"],
            "figure.facecolor": COLORS["paper"],
            "axes.facecolor": COLORS["paper"],
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGE_DIR / name, dpi=180, bbox_inches="tight", facecolor=COLORS["paper"])
    plt.close(fig)


def manifest_rows() -> list[dict]:
    path = TRAINING_ROOT / "manifest.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def policy() -> dict:
    return json.loads(POLICY_PATH.read_text())


def low_passed(values: np.ndarray, cutoff_hz: float = 180.0) -> np.ndarray:
    """Retain the kick/bass band for a readable beat-alignment plot."""
    signal = values.astype(np.float32)
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / SAMPLE_RATE)
    # A 60 Hz cosine roll-off avoids an artificial hard-filter edge while
    # removing higher-frequency waveform detail that obscures kick placement.
    mask = np.clip((cutoff_hz + 60.0 - frequencies) / 60.0, 0.0, 1.0)
    return np.fft.irfft(np.fft.rfft(signal) * mask, n=signal.size).astype(np.float32)


def waveform_fit(rows: list[dict]) -> None:
    sample = next(row for row in rows if row["domain"] == "synthetic")
    archive = np.load(TRAINING_ROOT / sample["sample"])
    # Bars 16--20 are the start of the 32-bar synthetic crossfade.  The rows
    # have already been beat-warped onto the common 120-BPM canonical clock.
    bars = slice(16, 20)
    outgoing = archive["outgoing_audio"][bars].astype(np.float32).reshape(-1)
    incoming = archive["incoming_audio"][bars].astype(np.float32).reshape(-1)
    outgoing = low_passed(outgoing)
    incoming = low_passed(incoming)
    time = np.arange(outgoing.size, dtype=np.float32) / SAMPLE_RATE
    step = 32  # 500 Hz display resolution; it deliberately avoids audio export.
    offset_seconds = 0.080
    shifted = np.interp(time, time + offset_seconds, incoming, left=0.0, right=0.0)

    def envelope(signal: np.ndarray) -> np.ndarray:
        window = 640  # 40 ms RMS: preserves kick placement while removing cycles.
        return np.sqrt(np.convolve(np.square(signal), np.ones(window) / window, mode="same") + 1e-9)

    out_env, in_env, shifted_env = map(envelope, (outgoing, incoming, shifted))
    fig, (ax_wave, ax_env) = plt.subplots(
        2, 1, figsize=(12, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.05, 0.72], "hspace": 0.10}
    )
    for ax in (ax_wave, ax_env):
        for beat in np.arange(0, time[-1] + 0.001, 0.5):
            ax.axvline(beat, color=COLORS["grid"], lw=0.65, zorder=0)
        for bar in np.arange(0, time[-1] + 0.001, 2.0):
            ax.axvline(bar, color="#A9BCD0", lw=1.1, zorder=0)
    ax_wave.plot(time[::step], outgoing[::step], color=COLORS["navy"], lw=0.9, label="Outgoing deck — canonical grid")
    ax_wave.plot(time[::step], incoming[::step], color=COLORS["teal"], lw=0.9, label="Incoming deck — fitted grid")
    ax_wave.plot(time[::step], shifted[::step], color=COLORS["orange"], lw=0.85, ls="--", label="Incoming — illustrative 80 ms phase error")
    ax_wave.set_ylabel("Low-frequency\namplitude")
    ax_wave.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=8.7)
    ax_wave.set_title("Beat-fitted canonical waveforms at the start of a transition")
    ax_env.plot(time[::step], out_env[::step], color=COLORS["navy"], lw=1.35, label="Outgoing kick envelope")
    ax_env.plot(time[::step], in_env[::step], color=COLORS["teal"], lw=1.35, label="Incoming fitted envelope")
    ax_env.plot(time[::step], shifted_env[::step], color=COLORS["orange"], lw=1.0, ls="--", label="Illustrative shifted envelope")
    ax_env.set_ylabel("40 ms RMS\nenvelope")
    ax_env.set_xlabel("Canonical time (seconds; 120 BPM, vertical thin lines = beats)")
    ax_env.set_xlim(0, 8)
    ax_env.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=8.4)
    ax_env.text(
        0.01, -0.43,
        "Source: synthetic Party1 training example; 16 kHz canonical tensors. The dashed trace is a visual counterexample, not an additional training sample.",
        transform=ax_env.transAxes, color=COLORS["muted"], fontsize=8.3,
    )
    save(fig, "waveform-fit.png")


def training_shape(rows: list[dict]) -> None:
    # The trainer reports mean epoch loss to stdout. These are the recorded
    # checkpoints from the completed 36/48/72-epoch training run, retained
    # here rather than interpolated into an artificial full history.
    curves = {
        "Localizer": ([1, 12, 24, 36], [0.731740, 0.691487, 0.630270, 0.521045], COLORS["navy"]),
        "Critic": ([1, 12, 24, 36, 48], [1.276337, 1.084038, 0.791204, 0.404377, 0.363582], COLORS["teal"]),
        "Policy": ([1, 12, 24, 36, 48, 60, 72], [7.353894, 7.310984, 7.277055, 7.278036, 7.273751, 7.257222, 7.262072], COLORS["purple"]),
    }
    counts = {domain: sum(row["domain"] == domain for row in rows) for domain in ["professional", "negative", "synthetic"]}
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.8), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    for ax, (name, (epochs, losses, color)) in zip(axes[:3], curves.items()):
        ax.plot(epochs, losses, marker="o", color=color, lw=2.1, markersize=4.8)
        ax.fill_between(epochs, losses, max(losses) * 1.03, color=color, alpha=0.08)
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean training loss")
        ax.grid(axis="y", color=COLORS["grid"], lw=0.8)
        ax.set_xlim(0, max(epochs) + 2)
        ax.annotate(f"{losses[-1]:.3f}", (epochs[-1], losses[-1]), xytext=(-8, -17), textcoords="offset points", color=color, fontweight="bold")
    labels = ["Professional\nweak positives", "Far-away\nnegatives", "Synthetic\ndeck pairs"]
    values = [counts["professional"], counts["negative"], counts["synthetic"]]
    bars = axes[3].bar(labels, values, color=[COLORS["navy"], COLORS["orange"], COLORS["teal"]], width=0.68)
    axes[3].set_title("Training corpus")
    axes[3].set_ylabel("Examples")
    axes[3].set_ylim(0, max(values) * 1.22)
    axes[3].grid(axis="y", color=COLORS["grid"], lw=0.8)
    for bar, value in zip(bars, values):
        axes[3].text(bar.get_x() + bar.get_width() / 2, value + 4, str(value), ha="center", color=COLORS["ink"], fontweight="bold")
    fig.suptitle("Recorded training shape and local dataset composition", x=0.5, y=1.03, fontsize=13, fontweight="bold")
    fig.text(
        0.5, -0.06,
        "Loss scales differ by task and are not directly comparable. Points are recorded console checkpoints from the completed run, not interpolated epochs.",
        ha="center", color=COLORS["muted"], fontsize=8.6,
    )
    save(fig, "training-shape.png")


def compatibility_profile(export: dict) -> None:
    features = export["compatibilityProfile"]["features"]
    names = list(features)
    weights = [features[name]["weight"] for name in names]
    targets = [features[name]["targetDelta"] for name in names]
    display_names = [name.replace("trajectory", "energy\ntrajectory").replace("spectral", "spectral\ndensity") for name in names]
    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(12, 4.8))
    bars = ax1.bar(x, weights, color=COLORS["navy"], alpha=0.86, label="Learned relative weight")
    ax1.set_ylabel("Relative compatibility weight", color=COLORS["navy"])
    ax1.tick_params(axis="y", labelcolor=COLORS["navy"])
    ax1.set_xticks(x, display_names)
    ax1.set_ylim(0, max(weights) * 1.28)
    ax1.grid(axis="y", color=COLORS["grid"], lw=0.8)
    ax2 = ax1.twinx()
    ax2.plot(x, targets, color=COLORS["orange"], marker="o", lw=2.0, label="Professional target delta")
    ax2.set_ylabel("Target feature difference", color=COLORS["orange"])
    ax2.tick_params(axis="y", labelcolor=COLORS["orange"])
    ax2.set_ylim(0, max(targets) * 1.35)
    for bar, weight in zip(bars, weights):
        ax1.text(bar.get_x() + bar.get_width() / 2, weight + 0.008, f"{weight:.3f}", ha="center", va="bottom", fontsize=8, color=COLORS["navy"])
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=True)
    ax1.set_title("Exported compatibility profile learned from 138 professional contexts")
    fig.text(
        0.5, -0.05,
        "The runtime score rewards differences close to each learned target; hard beat, bar, phrase, tempo and peak gates are separate and cannot be overridden by these weights.",
        ha="center", color=COLORS["muted"], fontsize=8.5,
    )
    save(fig, "compatibility-profile.png")


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(13, 7.1))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        patch = plt.Rectangle((x, y), w, h, facecolor="white", edgecolor=color, lw=1.8, joinstyle="round")
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", weight="bold", color=color, fontsize=10)
        ax.text(x + w / 2, y + h / 2 - 0.18, body, ha="center", va="center", color=COLORS["ink"], fontsize=8.2, linespacing=1.32)

    def arrow(start: tuple[float, float], end: tuple[float, float], text: str = "", y_offset: float = 0.0) -> None:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": COLORS["muted"], "lw": 1.5})
        if text:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + y_offset, text, ha="center", va="center", fontsize=7.8, color=COLORS["muted"], bbox={"facecolor": COLORS["paper"], "edgecolor": "none", "pad": 1.5})

    ax.text(0.35, 7.65, "DJ.Attatouille: local learning and non-destructive playback", fontsize=14, weight="bold", color=COLORS["ink"])
    ax.text(0.35, 7.24, "The saved mix is a versioned recipe; original audio stays in the mounted library and no full rendered master is required.", fontsize=9.3, color=COLORS["muted"])

    box(0.35, 4.7, 2.15, 1.45, "Original music library", "read-only local tracks\nstreamed at play time\nno copied master", COLORS["navy"])
    box(3.05, 4.7, 2.35, 1.45, "Analysis + planner", "energy-first ordering\nbeat / bar / phrase gates\noverlap validation", COLORS["teal"])
    box(6.0, 4.7, 2.15, 1.45, "Playback recipe", "source cues + timeline\ngain, tempo and loops\nfade + EQ curves", COLORS["purple"])
    box(8.85, 4.7, 2.75, 1.45, "Two-deck Web Audio", "stream originals + short fallback\nsample-clock automation\nlimiter + virtual seek", COLORS["orange"])
    arrow((2.5, 5.42), (3.05, 5.42), "analyse")
    arrow((5.4, 5.42), (6.0, 5.42), "store plan")
    arrow((8.15, 5.42), (8.85, 5.42), "execute")
    box(0.45, 1.2, 2.15, 1.55, "Professional mixes", "timed tracklists\nweak transition labels", COLORS["navy"])
    box(3.35, 1.2, 2.45, 1.55, "Canonical dataset", "64 bars at 120 BPM\n16 kHz, mel + MIR\nshared augmentation", COLORS["teal"])
    box(6.55, 1.2, 2.0, 1.55, "PyTorch trainer", "localizer → critic\n→ differentiable policy", COLORS["purple"])
    box(9.3, 1.2, 2.55, 1.55, "Policy export", "transition-policy-v2\nNumPy hot-load\nhard safety gates", COLORS["orange"])
    arrow((2.6, 1.98), (3.35, 1.98), "ingest")
    arrow((5.8, 1.98), (6.55, 1.98), "train")
    arrow((8.55, 1.98), (9.3, 1.98), "JSON export")
    arrow((10.58, 2.75), (4.22, 4.7), "candidate ranking only", -0.18)
    arrow((1.52, 4.7), (4.58, 2.75), "bounded synthetic examples", 0.15)
    ax.text(6.5, 0.35, "Runtime guardrail: energy-first sequence → fit every boundary beat → residual ≤ 22 ms → optional ±24 ms phase correction → schedule recipe.", ha="center", fontsize=8.7, color=COLORS["muted"])
    save(fig, "architecture.png")


def main() -> None:
    configure()
    rows = manifest_rows()
    export = policy()
    waveform_fit(rows)
    training_shape(rows)
    compatibility_profile(export)
    architecture()
    print(f"Generated 4 figures in {IMAGE_DIR}")


if __name__ == "__main__":
    main()
