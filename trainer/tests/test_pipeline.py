from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from dj_train.canonical import features_from_bars, inspect_sample, save_sample
from dj_train.config import (
    FEATURE_VERSION,
    FEATURE_SAMPLE_RATE,
    MEL_BINS,
    MIR_DIM,
    POLICY_CONTROL_NAMES,
    POLICY_FEATURE_NAMES,
    SAMPLES_PER_BAR,
    TOTAL_BARS,
)
from dj_train.dataset import (
    baseline_mix,
    domain_augmentation,
    feature_augmentation,
    ingest_professional_mix,
    javascript_runtime_args,
    normalize_public_url,
)
from dj_train.models import (
    DifferentiableMixer,
    TransitionCritic,
    TransitionLocalizer,
    TransitionPolicy,
    export_policy,
)
from dj_train.tracklist import parse_tracklist
from dj_train.train import collate_common_fields


def signal(frequency: float = 110.0) -> np.ndarray:
    points = np.arange(SAMPLES_PER_BAR, dtype=np.float32) / FEATURE_SAMPLE_RATE
    bar = (0.35 * np.sin(2 * np.pi * frequency * points)).astype(np.float32)
    return np.tile(bar, (TOTAL_BARS, 1))


class TracklistTests(unittest.TestCase):
    def test_parses_weak_timestamps_in_common_formats(self) -> None:
        entries = parse_tracklist("00:00 First\n04:21 - Second\n1:02:03 | Third")
        self.assertEqual([entry.seconds for entry in entries], [0.0, 261.0, 3723.0])
        self.assertEqual(entries[1].title, "Second")

    def test_cleans_markdown_and_shell_escaped_youtube_urls(self) -> None:
        copied = r"[https://www.youtube.com/watch\?v\=01qjuSpFUY0](https://www.youtube.com/watch\?v\=01qjuSpFUY0)"
        self.assertEqual(
            normalize_public_url(copied),
            "https://www.youtube.com/watch?v=01qjuSpFUY0",
        )

    @patch("dj_train.dataset.subprocess.run")
    @patch("dj_train.dataset.shutil.which")
    def test_selects_supported_node_for_youtube_challenges(self, which, run) -> None:
        which.side_effect = lambda runtime: "/usr/local/bin/node" if runtime == "node" else None
        run.return_value.returncode = 0
        run.return_value.stdout = "v26.6.0\n"
        run.return_value.stderr = ""
        self.assertEqual(
            javascript_runtime_args(),
            ["--js-runtimes", "node:/usr/local/bin/node"],
        )

    def test_mixed_domains_collate_only_shared_training_fields(self) -> None:
        batch = collate_common_fields([
            {"domain": "professional", "mel": np.zeros((2, 3), dtype=np.float32)},
            {
                "domain": "synthetic",
                "mel": np.ones((2, 3), dtype=np.float32),
                "outgoing_audio": np.zeros((2, 4), dtype=np.float32),
            },
        ])
        self.assertEqual(set(batch), {"domain", "mel"})
        self.assertEqual(tuple(batch["mel"].shape), (2, 2, 3))


class CanonicalRepresentationTests(unittest.TestCase):
    def test_feature_contract_is_fixed_for_every_domain(self) -> None:
        bars = signal()
        mel, mir = features_from_bars(bars)
        self.assertEqual(mel.shape, (TOTAL_BARS, MEL_BINS))
        self.assertEqual(mir.shape, (TOTAL_BARS, MIR_DIM))
        with tempfile.TemporaryDirectory() as directory:
            professional = Path(directory) / "professional.npz"
            synthetic = Path(directory) / "synthetic.npz"
            for path, domain in ((professional, "professional"), (synthetic, "synthetic")):
                save_sample(
                    path, bars=bars, mask=np.ones(TOTAL_BARS),
                    metadata={"domain": domain},
                    arrays={"weak_positive_mask": np.ones(TOTAL_BARS)},
                )
            first, second = inspect_sample(professional), inspect_sample(synthetic)
            self.assertEqual(first["featureVersion"], FEATURE_VERSION)
            self.assertEqual(first["audioShape"], second["audioShape"])
            self.assertEqual(first["melShape"], second["melShape"])
            self.assertEqual(first["mirShape"], second["mirShape"])

    def test_domain_augmentation_is_shape_safe_and_peak_bounded(self) -> None:
        augmented = domain_augmentation(signal(), np.random.default_rng(4))
        self.assertEqual(augmented.shape, (TOTAL_BARS, SAMPLES_PER_BAR))
        self.assertLessEqual(float(np.max(np.abs(augmented))), 0.991)

    def test_feature_augmentation_preserves_canonical_shapes_and_bounds(self) -> None:
        mel, mir = features_from_bars(signal())
        augmented_mel, augmented_mir = feature_augmentation(
            mel, mir, np.random.default_rng(4),
        )
        self.assertEqual(augmented_mel.shape, mel.shape)
        self.assertEqual(augmented_mir.shape, mir.shape)
        self.assertTrue(np.isfinite(augmented_mel).all())
        self.assertTrue(np.isfinite(augmented_mir).all())
        self.assertGreaterEqual(float(augmented_mel.min()), 0.0)
        self.assertLessEqual(float(augmented_mel.max()), 1.0)

    def test_local_mix_and_weak_tracklist_produce_a_professional_sample(self) -> None:
        sample_rate = 22_050
        duration = 20
        points = np.arange(sample_rate * duration, dtype=np.float32) / sample_rate
        audio = 0.2 * np.sin(2 * np.pi * 110 * points)
        for second in np.arange(0, duration, 0.5):
            start = int(second * sample_rate)
            audio[start:start + 180] += np.hanning(180).astype(np.float32) * 0.7
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reference.wav"
            tracklist = root / "tracklist.txt"
            sf.write(source, audio, sample_rate)
            tracklist.write_text("00:00 First\n00:10 Second\n", encoding="utf-8")
            rows = ingest_professional_mix(
                root=root / "dataset", name="Reference", tracklist_path=tracklist,
                audio_path=source,
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["domain"], "professional")
            inspection = inspect_sample(root / "dataset" / rows[0]["sample"])
            self.assertEqual(inspection["audioShape"], [TOTAL_BARS, SAMPLES_PER_BAR])
            self.assertEqual(inspection["melShape"], [TOTAL_BARS, MEL_BINS])


class ModelTests(unittest.TestCase):
    def test_localizer_backpropagates_through_temporal_context_features(self) -> None:
        model = TransitionLocalizer(width=32)
        mel = torch.zeros(2, TOTAL_BARS, MEL_BINS)
        mir = torch.zeros(2, TOTAL_BARS, MIR_DIM)
        loss = model(mel, mir).sigmoid().mean()
        loss.backward()
        self.assertIsNotNone(model.encoder.temporal[0].projection.weight.grad)

    def test_differentiable_mixer_backpropagates_to_policy_controls(self) -> None:
        outgoing = torch.from_numpy(signal()).unsqueeze(0)
        incoming = torch.from_numpy(signal(220)).unsqueeze(0)
        policy = TransitionPolicy()
        state = torch.zeros(1, len(POLICY_FEATURE_NAMES))
        controls = policy(state)
        rendered = DifferentiableMixer()(outgoing, incoming, controls)
        loss = rendered.square().mean()
        loss.backward()
        self.assertEqual(rendered.shape, outgoing.shape)
        self.assertIsNotNone(policy.output.weight.grad)
        self.assertGreater(float(policy.output.weight.grad.abs().sum()), 0.0)

    def test_critic_exposes_realism_location_and_diagnostics(self) -> None:
        critic = TransitionCritic(width=32)
        output = critic(
            torch.zeros(2, TOTAL_BARS, MEL_BINS),
            torch.zeros(2, TOTAL_BARS, MIR_DIM),
            torch.ones(2, TOTAL_BARS),
        )
        self.assertEqual(output["realism"].shape, (2,))
        self.assertEqual(output["location"].shape, (2, TOTAL_BARS))
        self.assertEqual(output["auxiliary"].shape, (2, 6))

    def test_export_contains_only_the_compact_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = export_policy(TransitionPolicy(), Path(directory) / "policy.json")
            payload = json.loads(destination.read_text())
            self.assertEqual(payload["featureNames"], POLICY_FEATURE_NAMES)
            self.assertEqual(payload["controlNames"], POLICY_CONTROL_NAMES)
            self.assertEqual(len(payload["layers"]), 3)
            self.assertNotIn("critic", payload)
            self.assertNotIn("localizer", payload)

    def test_baseline_artificial_mix_uses_the_same_bar_shape(self) -> None:
        rendered, controls = baseline_mix(signal(), signal(220))
        self.assertEqual(rendered.shape, (TOTAL_BARS, SAMPLES_PER_BAR))
        self.assertEqual(controls["overlapPhrases"], 4.0)


if __name__ == "__main__":
    unittest.main()
