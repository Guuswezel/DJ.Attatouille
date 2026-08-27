import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from app.main import (
    detailed_waveform,
    gain_trim_db,
    harmonix_device,
    harmonix_model_name,
    harmonix_proxy_stems,
    project_embedding,
    report_preparation_progress,
    resolve_music_source,
    safe_exit_for,
    transition_between,
    waveform_points,
)


def track(track_id: str, bpm: float, key: str, energy: float) -> dict:
    return {
        "id": track_id,
        "title": track_id,
        "durationSeconds": 240.0,
        "bpm": bpm,
        "key": key,
        "energy": energy,
        "genres": ["house"],
        "beatGrid": [float(point) for point in range(0, 241)],
        "downbeats": [float(point) for point in range(0, 241, 2)],
        "segments": [
            {"start": 0.0, "end": 64.0, "label": "intro", "energy": 0.25},
            {"start": 64.0, "end": 128.0, "label": "verse", "energy": energy},
            {"start": 128.0, "end": 192.0, "label": "chorus", "energy": 0.92},
            {"start": 192.0, "end": 240.0, "label": "outro", "energy": 0.28},
        ],
        "cues": {"safeEntries": [8.0, 16.0], "safeExits": [60.0, 96.0, 120.0, 150.0]},
    }


class TransitionPlanningTests(unittest.TestCase):
    def test_exit_is_within_requested_deck_time(self) -> None:
        selected = safe_exit_for(track("one", 120, "C", 0.5), source_start=12, minimum=90, maximum=120)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertGreaterEqual(selected, 102)
        self.assertLessEqual(selected, 132)

    @patch("app.main.vector_score", return_value=0.8)
    def test_transition_uses_pitch_preserving_beat_overlay_when_bpm_is_close(self, _: object) -> None:
        plan = transition_between(track("one", 120, "C", 0.5), track("two", 124, "G", 0.6), 0, 90, 150)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertGreaterEqual(plan["tempoFactor"], 0.92)
        self.assertLessEqual(plan["tempoFactor"], 1.08)
        self.assertTrue(plan["beatMatched"])
        self.assertIn(plan["overlapBars"], {2, 4, 8})
        self.assertGreaterEqual(plan["exitAtSeconds"], 90)
        self.assertLessEqual(plan["exitAtSeconds"], 150)
        self.assertIn("renderQualityScore", plan)

    @patch("app.main.vector_score", return_value=0.8)
    def test_large_tempo_gap_never_overlays_drifting_beats(self, _: object) -> None:
        plan = transition_between(track("one", 120, "C", 0.5), track("two", 150, "G", 0.6), 0, 90, 150)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertFalse(plan["beatMatched"])
        self.assertEqual(plan["tempoFactor"], 1.0)
        self.assertLess(plan["overlapSeconds"], 2.0)

    def test_protected_chorus_is_not_used_as_a_mid_section_exit(self) -> None:
        protected = track("one", 120, "C", 0.5)
        protected["cues"]["safeExits"] = [150.0, 190.0]
        selected = safe_exit_for(protected, source_start=160.0, minimum=4.0, maximum=35.0)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertGreaterEqual(selected, 192.0)

    @patch("app.main.vector_score", return_value=0.8)
    def test_maximum_longer_than_track_uses_its_natural_end_when_grid_is_missing(self, _: object) -> None:
        short = track("short", 120, "C", 0.5)
        short["durationSeconds"] = 105.0
        short["beatGrid"] = []
        short["downbeats"] = []
        short["cues"]["safeExits"] = [48.0]
        plan = transition_between(short, track("next", 122, "G", 0.55), 0, 90, 180)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan["exitAtSeconds"], 104.75)

    @patch("app.main.vector_score", return_value=0.8)
    def test_high_energy_phrase_can_enter_later_and_cuts_an_outgoing_build_early(self, _: object) -> None:
        outgoing = track("build", 120, "C", 0.9)
        outgoing["segments"][2] = {"start": 128.0, "end": 192.0, "label": "build", "energy": 0.9}
        outgoing["cues"]["safeExits"] = [192.0]
        outgoing["downbeats"] = [192.0]
        incoming = track("peak", 122, "G", 0.5)
        plan = transition_between(outgoing, incoming, 0, 90, 200)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertGreaterEqual(plan["enterAtSeconds"], 128.0)
        self.assertEqual(plan["fadeShape"], "fast-outgoing")
        self.assertEqual(plan["overlapBars"], 2)

    def test_waveform_is_normalised_for_the_ui(self) -> None:
        points = waveform_points(np.array([0.0, 0.5, 1.0, 0.25]), count=4)
        self.assertEqual(len(points), 4)
        self.assertEqual(max(points), 1.0)
        self.assertEqual(points[0], 0.0)

    def test_detailed_waveform_keeps_sixteen_samples_per_beat(self) -> None:
        encoded = detailed_waveform(np.array([0.0, 0.5, 1.0, 0.25]), duration=120.0, bpm=150.0, detected_beats=300)
        self.assertEqual(len(base64.b64decode(encoded)), 4_800)

    def test_fast_harmonix_proxy_preserves_four_timed_spectral_lanes(self) -> None:
        source_rate = 22_050
        points = np.arange(source_rate * 2, dtype=np.float32) / source_rate
        source = (
            0.35 * np.sin(2 * np.pi * 110 * points)
            + 0.20 * np.sin(2 * np.pi * 2_200 * points)
        ).astype(np.float32)
        stems = harmonix_proxy_stems(source, source_rate)
        self.assertEqual(set(stems), {"bass", "drums", "other", "vocals"})
        self.assertTrue(all(stem.dtype == np.int16 for stem in stems.values()))
        # Proxy stems are re-clocked to Harmonix's 44.1 kHz / 100 fps model
        # input while maintaining the exact two-second music duration.
        self.assertTrue(all(len(stem) == 44_100 * 2 for stem in stems.values()))

    def test_harmonix_ensemble_embeddings_are_pooled_into_the_qdrant_vector(self) -> None:
        raw = np.ones((2, 6, 24), dtype=np.float32)
        vector = project_embedding(raw, np.zeros((20, 1)), np.zeros((12, 1)), np.zeros(1))
        self.assertEqual(vector.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)

    def test_dj_gain_trim_matches_loudness_with_a_safe_limit(self) -> None:
        self.assertEqual(gain_trim_db({"loudnessLufs": -20.0}), 6.0)
        self.assertEqual(gain_trim_db({"loudnessLufs": -40.0}), 8.0)
        self.assertEqual(gain_trim_db({}), 0.0)

    @patch("app.main.urllib.request.urlopen")
    @patch("app.main.BACKEND_URL", "http://api")
    def test_analysis_progress_callback_includes_track_counts_and_current_file(self, mocked_open: object) -> None:
        mocked_open.return_value.__enter__.return_value = object()  # type: ignore[attr-defined]
        report_preparation_progress(
            "prep id", progress=105, discovered_track_count=7,
            analysed_track_count=3, failed_track_count=1, current_track="four.mp3",
            message="Analysing track 5 of 7",
        )
        request = mocked_open.call_args.args[0]  # type: ignore[attr-defined]
        self.assertEqual(request.full_url, "http://api/api/preparations/prep%20id/progress")
        payload = json.loads(request.data.decode())
        self.assertEqual(payload["progress"], 99)
        self.assertEqual(payload["discoveredTrackCount"], 7)
        self.assertEqual(payload["analysedTrackCount"], 3)
        self.assertEqual(payload["failedTrackCount"], 1)
        self.assertEqual(payload["currentTrack"], "four.mp3")

    @patch("app.main.ANALYSIS_DEVICE", "mps")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_harmonix_selects_mps_when_the_native_metal_backend_is_available(self, _: object, __: object) -> None:
        self.assertEqual(harmonix_device(), "mps")

    @patch("app.main.HARMONIX_MODEL", "")
    def test_mps_uses_a_memory_safe_harmonix_fold_by_default(self) -> None:
        self.assertEqual(harmonix_model_name("mps"), "harmonix-fold0")
        self.assertEqual(harmonix_model_name("cpu"), "harmonix-all")

    def test_native_worker_maps_the_api_music_root_into_its_local_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            local_library = Path(directory).resolve()
            song = local_library / "Party" / "song.mp3"
            song.parent.mkdir()
            song.touch()
            with patch("app.main.REQUEST_MUSIC_ROOT", Path("/music")), patch("app.main.MUSIC_ROOT", local_library):
                self.assertEqual(resolve_music_source("/music/Party/song.mp3"), song)


if __name__ == "__main__":
    unittest.main()
