from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .canonical import inspect_sample
from .config import DATASET_ROOT, FEATURE_SAMPLE_RATE, POLICY_OUTPUT
from .dataset import (
    dataset_summary,
    ingest_professional_mix,
    read_manifest,
    synthesize_transitions,
)
from .train import train_critic, train_localizer, train_policy, tune_from_human_feedback


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus"}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="DJ.Attatouille offline transition trainer")
    result.add_argument("--root", type=Path, default=DATASET_ROOT, help="dataset/output root")
    commands = result.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest-professional", help="download/canonicalize a DJ mix and weak timed tracklist")
    ingest.add_argument("--name", required=True)
    source = ingest.add_mutually_exclusive_group(required=True)
    source.add_argument("--url")
    source.add_argument("--audio", type=Path)
    ingest.add_argument("--tracklist", required=True, type=Path)

    synthetic = commands.add_parser("synthesize", help="make baseline artificial transitions from clean local tracks")
    synthetic_source = synthetic.add_mutually_exclusive_group(required=True)
    synthetic_source.add_argument("--folder", type=Path)
    synthetic_source.add_argument("--tracks", type=Path, nargs="+")
    synthetic.add_argument("--limit", type=int, default=0)

    train = commands.add_parser("train", help="run one stage or the complete offline pipeline")
    train.add_argument("--stage", choices=["localizer", "critic", "policy", "all"], default="all")
    train.add_argument("--epochs", type=int, default=0, help="override the stage default")
    train.add_argument("--batch-size", type=int, default=0)
    train.add_argument("--device", default=os.environ.get("TRANSITION_TRAIN_DEVICE", "auto"))
    train.add_argument("--policy-output", type=Path, default=POLICY_OUTPUT)

    feedback = commands.add_parser("human-feedback", help="calibrate the critic from JSONL sample ratings")
    feedback.add_argument("--feedback", type=Path, required=True)
    feedback.add_argument("--epochs", type=int, default=8)
    feedback.add_argument("--device", default=os.environ.get("TRANSITION_TRAIN_DEVICE", "auto"))

    commands.add_parser("status", help="summarize the canonical dataset")
    previews = commands.add_parser("export-previews", help="write listenable canonical WAVs for rating")
    previews.add_argument("--destination", type=Path)
    previews.add_argument("--domain", choices=["professional", "synthetic"])
    verify = commands.add_parser("verify", help="validate that every domain uses the same feature/audio shapes")
    verify.add_argument("--show", type=int, default=5)
    return result


def _tracks(args: argparse.Namespace) -> list[Path]:
    if args.tracks:
        paths = list(args.tracks)
    else:
        paths = sorted(path for path in args.folder.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS)
    if args.limit:
        paths = paths[:args.limit]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths


def run(args: argparse.Namespace) -> Any:
    root = args.root.resolve()
    if args.command == "ingest-professional":
        return ingest_professional_mix(
            root=root, name=args.name, tracklist_path=args.tracklist,
            url=args.url, audio_path=args.audio,
        )
    if args.command == "synthesize":
        return synthesize_transitions(root, _tracks(args))
    if args.command == "status":
        return dataset_summary(root)
    if args.command == "export-previews":
        destination = (args.destination or (root / "previews")).resolve()
        exported: list[str] = []
        for row in read_manifest(root):
            if row["domain"] == "negative" or (args.domain and row["domain"] != args.domain):
                continue
            with np.load(root / row["sample"], allow_pickle=False) as sample:
                audio = sample["audio"].astype(np.float32).reshape(-1)
            output = destination / row["domain"] / f"{row['id']}.wav"
            output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output, audio, FEATURE_SAMPLE_RATE, subtype="PCM_16")
            exported.append(str(output))
        return {"exported": len(exported), "destination": str(destination), "files": exported}
    if args.command == "verify":
        rows = read_manifest(root)
        inspections = [inspect_sample(root / row["sample"]) for row in rows]
        shapes = {
            (tuple(item["audioShape"]), tuple(item["melShape"]), tuple(item["mirShape"]))
            for item in inspections
        }
        versions = {item["featureVersion"] for item in inspections}
        if len(shapes) > 1 or len(versions) > 1:
            raise RuntimeError(f"Canonical representation mismatch: shapes={shapes}, versions={versions}")
        return {**dataset_summary(root), "representations": len(shapes), "examples": inspections[:args.show]}
    if args.command == "human-feedback":
        return tune_from_human_feedback(root, args.feedback, epochs=args.epochs, device_name=args.device)
    if args.command == "train":
        reports: list[dict[str, Any]] = []
        stages = [args.stage] if args.stage != "all" else ["localizer", "critic", "policy"]
        for stage in stages:
            if stage == "localizer":
                reports.append(train_localizer(
                    root, epochs=args.epochs or 20, batch_size=args.batch_size or 8, device_name=args.device,
                ))
            elif stage == "critic":
                reports.append(train_critic(
                    root, epochs=args.epochs or 24, batch_size=args.batch_size or 8, device_name=args.device,
                ))
            else:
                reports.append(train_policy(
                    root, epochs=args.epochs or 30, batch_size=args.batch_size or 2,
                    device_name=args.device, policy_output=args.policy_output,
                ))
        return reports
    raise RuntimeError(f"Unknown command: {args.command}")


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(run(args), indent=2, default=str))


if __name__ == "__main__":
    main()
