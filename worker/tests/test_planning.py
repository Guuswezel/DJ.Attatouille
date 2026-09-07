import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pydub import AudioSegment

from app.main import (
    TRANSITION_POLICY_FEATURES,
    analyse_track,
    detailed_waveform,
    fade_gains,
    gain_trim_db,
    harmonix_device,
    harmonix_model_name,
    harmonix_proxy_stems,
    ordered_tracks,
    overlay_with_headroom,
    phrase_state_boundaries,
    project_embedding,
    read_analysis_cache,
    repaired_downbeat_grid,
    report_preparation_progress,
    resolve_music_source,
    safe_exit_for,
    transition_compatibility_score,
    transition_between,
    transition_policy_controls,
    waveform_points,
    write_analysis_cache,
)


def track(track_id: str, bpm: float, key: str, energy: float) -> dict:
    phrase_states = [
        {
            "start": float(start), "end": float(start + 16), "phraseIndex": index,
            "energy": 0.25 if start < 64 or start >= 192 else 0.92 if start >= 128 else energy,
            "energySlope": 0.0, "bassActivity": 0.45, "drumActivity": 0.75,
            "vocalActivity": 0.15, "spectralDensity": 0.55, "harmonicDensity": 0.6,
            "noveltyIn": 0.2, "noveltyOut": 0.2, "loopability": 0.8, "cueConfidence": 0.8,
        }
        for index, start in enumerate(range(0, 240, 16))
    ]
    return {
        "id": track_id,
        "title": track_id,
        "durationSeconds": 240.0,
        "bpm": bpm,
        "key": key,
        "energy": energy,
        "genres": ["house"],
        "beatGrid": [point / 2 for point in range(0, 481)],
        "downbeats": [float(point) for point in range(0, 241, 2)],
        "segments": [
            {"start": 0.0, "end": 64.0, "label": "intro", "energy": 0.25},
            {"start": 64.0, "end": 128.0, "label": "verse", "energy": energy},
            {"start": 128.0, "end": 192.0, "label": "chorus", "energy": 0.92},
            {"start": 192.0, "end": 240.0, "label": "outro", "energy": 0.28},
        ],
        "phraseStates": phrase_states,
        "cues": {
            "safeEntries": [8.0, 16.0], "safeExits": [60.0, 96.0, 120.0, 150.0],
            "phraseBoundaries": [float(point) for point in range(0, 241, 16)],
        },
    }


class TransitionPlanningTests(unittest.TestCase):
    def test_exit_is_within_requested_deck_time(self) -> None:
        selected = safe_exit_for(track("one", 120, "C", 0.5), source_start=12, minimum=90, maximum=120)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertGreaterEqual(selected, 102)
        self.assertLessEqual(selected, 132)

    def test_duration_preference_selects_different_safe_phrases(self) -> None:
        source = track("journey", 120, "C", 0.5)
        source["segments"] = [{"start": 0.0, "end": 240.0, "label": "verse", "energy": 0.5}]
        short = safe_exit_for(source, 0, 32, 112, duration_preference=0.2)
        long = safe_exit_for(source, 0, 32, 112, duration_preference=0.8)
        self.assertIsNotNone(short)
        self.assertIsNotNone(long)
        assert short is not None and long is not None
        self.assertLess(short, long)

    def test_exit_avoids_a_sustained_vocal_tail(self) -> None:
        source = track("vocal", 120, "C", 0.5)
        source["segments"] = [{"start": 0.0, "end": 240.0, "label": "verse", "energy": 0.5}]
        risky = next(state for state in source["phraseStates"] if state["end"] == 64.0)
        risky.update({"vocalActivity": 1.0, "vocalTailActivity": 1.0, "vocalContinuity": 1.0})
        selected = safe_exit_for(source, 0, 50, 90, duration_preference=0.2)
        self.assertEqual(selected, 80.0)

    def test_energy_order_is_monotonic_across_genres(self) -> None:
        candidates = [track(f"track-{index}", 120, "C", energy) for index, energy in enumerate([0.8, 0.2, 0.6, 0.4])]
        for candidate, genre in zip(candidates, ["techno", "ambient", "house", "disco"]):
            candidate["genres"] = [genre]
        energies = [item["energy"] for item in ordered_tracks(candidates)]
        self.assertEqual(energies, [0.2, 0.4, 0.6, 0.8])

    def test_energy_order_ignores_genre_even_when_the_old_journey_conflicts(self) -> None:
        warmup = track("warmup", 110, "C", 0.20)
        peak = track("peak", 132, "C", 0.90)
        warmup["genres"], peak["genres"] = ["techno"], ["ambient"]
        self.assertEqual([item["id"] for item in ordered_tracks([peak, warmup])], ["warmup", "peak"])

    def test_float_overlay_applies_headroom_before_pcm_saturation(self) -> None:
        samples = np.full(2_048, 29_000, dtype=np.int16)
        outgoing = AudioSegment(samples.tobytes(), frame_rate=44_100, sample_width=2, channels=1)
        bridge, metrics = overlay_with_headroom(outgoing, outgoing, ceiling_dbfs=-2.0)
        self.assertLess(metrics["preOverlayPeakDbfs"], 6.0)
        self.assertLess(metrics["overlayGainDb"], -3.0)
        self.assertLessEqual(bridge.max_dBFS, -1.95)

    def test_professional_compatibility_target_can_reward_controlled_contrast(self) -> None:
        first = track("one", 120, "C", 0.4)
        second = track("two", 120, "C", 0.7)
        outgoing = first["phraseStates"][4]
        incoming = second["phraseStates"][4].copy()
        outgoing["energy"] = 0.4
        incoming["energy"] = 0.7
        features = {
            name: {"weight": 1.0 if name == "energy" else 0.0, "targetDelta": 0.3, "scale": 0.05}
            for name in ("energy", "trajectory", "bass", "drums", "vocals", "spectral", "harmonic", "novelty")
        }
        with patch("app.main.load_transition_policy", return_value={"compatibilityProfile": {"features": features}}):
            score, components = transition_compatibility_score(first, second, outgoing, incoming)
        self.assertGreater(score, 0.99)
        self.assertGreater(components["energy"], 0.99)

    def test_vocal_carry_does_not_fade_the_sentence_at_mid_phrase(self) -> None:
        vocal_out, _ = fade_gains(0.5, "vocal-carry", 0.7, 0.7)
        regular_out, _ = fade_gains(0.5, "equal-power", 1.0, 1.0)
        self.assertGreater(vocal_out, -0.1)
        self.assertLess(regular_out, -2.5)

    @patch("app.main.vector_score", return_value=0.8)
    def test_transition_uses_pitch_preserving_beat_overlay_when_bpm_is_close(self, _: object) -> None:
        plan = transition_between(track("one", 120, "C", 0.5), track("two", 124, "G", 0.6), 0, 90, 150)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertGreaterEqual(plan["tempoFactor"], 0.92)
        self.assertLessEqual(plan["tempoFactor"], 1.08)
        self.assertTrue(plan["beatMatched"])
        self.assertTrue(plan["barMatched"])
        self.assertTrue(plan["phraseMatched"])
        self.assertEqual(plan["overlapBars"], 8)
        self.assertLessEqual(plan["beatAlignmentErrorMs"], 22)
        self.assertLessEqual(plan["beatPhaseErrorMs"], 22)
        self.assertGreaterEqual(plan["exitAtSeconds"], 90)
        self.assertLessEqual(plan["exitAtSeconds"], 150)
        self.assertIn("renderQualityScore", plan)

    def test_runtime_policy_loader_evaluates_only_the_exported_dense_policy(self) -> None:
        controls = [
            "overlap_phrases", "bass_swap_progress", "fade_out_curve", "fade_in_curve",
            "outgoing_low_db", "incoming_low_db", "outgoing_mid_db", "incoming_mid_db",
            "outgoing_high_db", "incoming_high_db", "loop_probability",
            "timing_score",
        ]
        widths = [len(TRANSITION_POLICY_FEATURES), 3, 3, len(controls)]
        layers = [
            {"weight": np.zeros((widths[index + 1], widths[index])).tolist(), "bias": np.zeros(widths[index + 1]).tolist()}
            for index in range(3)
        ]
        payload = {
            "featureVersion": "dj-attatouille-transition-v1",
            "policyVersion": "test-policy",
            "featureNames": TRANSITION_POLICY_FEATURES,
            "controlNames": controls,
            "featureMean": np.zeros(widths[0]).tolist(),
            "featureScale": np.ones(widths[0]).tolist(),
            "controlLower": [1, .35, .5, .5, -24, -24, -9, -9, -12, -12, 0, 0],
            "controlUpper": [4, .8, 1.8, 1.8, -6, -6, 0, 0, 0, 0, 1, 1],
            "layers": layers,
        }
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "policy.json"
            policy_path.write_text(json.dumps(payload))
            with patch("app.main.TRANSITION_POLICY_PATH", policy_path), patch("app.main.TRANSITION_POLICY_CACHE", (-2, None)):
                predicted = transition_policy_controls(np.zeros(widths[0], dtype=np.float32))
        self.assertIsNotNone(predicted)
        assert predicted is not None
        self.assertAlmostEqual(predicted["overlap_phrases"], 2.5)
        self.assertAlmostEqual(predicted["bass_swap_progress"], 0.575)
        self.assertEqual(set(predicted), set(controls))

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
        short["phraseStates"] = []
        short["cues"]["phraseBoundaries"] = []
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
        # Finish the preceding phrase exactly where the protected build starts;
        # none of that build can sit behind the incoming high-energy phrase.
        self.assertEqual(plan["exitAtSeconds"], 128.0)
        self.assertGreaterEqual(plan["enterAtSeconds"], 128.0)
        self.assertTrue(plan["beatMatched"])
        self.assertTrue(plan["barMatched"])
        self.assertTrue(plan["phraseMatched"])
        self.assertEqual(plan["overlapBars"], 8)

    def test_waveform_is_normalised_for_the_ui(self) -> None:
        points = waveform_points(np.array([0.0, 0.5, 1.0, 0.25]), count=4)
        self.assertEqual(len(points), 4)
        self.assertEqual(max(points), 1.0)
        self.assertEqual(points[0], 0.0)

    def test_waveform_detail_never_appends_empty_bins_when_more_points_are_requested(self) -> None:
        points = waveform_points(np.array([0.0, 0.5, 1.0, 0.25]), count=8)
        self.assertEqual(len(points), 8)
        self.assertEqual(max(points), 1.0)
        # The old array_split implementation put four samples at the beginning
        # and four zero bins at the end, which stretched the visual timeline.
        self.assertGreater(points[-1], 0.0)
        self.assertGreater(points[-2], 0.0)

    def test_detailed_waveform_keeps_sixty_four_samples_per_beat(self) -> None:
        encoded = detailed_waveform(np.array([0.0, 0.5, 1.0, 0.25]), duration=120.0, bpm=150.0, detected_beats=300)
        values = base64.b64decode(encoded)
        self.assertEqual(len(values), 19_200)
        self.assertGreater(values[-1], 0)

    def test_phrase_clock_uses_eight_bar_big_one_boundaries(self) -> None:
        downbeats = np.arange(0.0, 130.0, 2.0)
        segments = [
            {"start": 0.0, "end": 64.0, "label": "intro"},
            {"start": 64.0, "end": 128.0, "label": "chorus"},
        ]
        boundaries = phrase_state_boundaries(downbeats, 128.0, segments)
        self.assertEqual(boundaries[:5], [0.0, 16.0, 32.0, 48.0, 64.0])

    def test_phrase_clock_repairs_downbeats_missing_during_a_breakdown(self) -> None:
        beats = np.arange(0.0, 66.0, 0.5)
        downbeats = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 18.0, 20.0])
        repaired = repaired_downbeat_grid(downbeats, beats)
        self.assertTrue(np.allclose(repaired[:11], np.arange(0.0, 22.0, 2.0)))

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
            message="Analysing track 5 of 7", cached_track_count=2,
        )
        request = mocked_open.call_args.args[0]  # type: ignore[attr-defined]
        self.assertEqual(request.full_url, "http://api/api/preparations/prep%20id/progress")
        payload = json.loads(request.data.decode())
        self.assertEqual(payload["progress"], 99)
        self.assertEqual(payload["discoveredTrackCount"], 7)
        self.assertEqual(payload["analysedTrackCount"], 3)
        self.assertEqual(payload["failedTrackCount"], 1)
        self.assertEqual(payload["cachedTrackCount"], 2)
        self.assertEqual(payload["currentTrack"], "four.mp3")

    def test_completed_track_cache_is_reused_without_decoding_or_harmonix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            music_root = root / "music"
            data_root = root / "data"
            song = music_root / "Party" / "song.mp3"
            song.parent.mkdir(parents=True)
            song.write_bytes(b"unchanged source audio")
            cached_track = {
                "relativePath": "Party/song.mp3", "title": "Cached song", "artist": "Artist",
                "album": "Album", "durationSeconds": 180.0, "bpm": 124.0, "key": "Am",
                "energy": 0.7, "loudnessLufs": -14.0, "waveform": [0.2, 0.8],
                "waveformDetail": None, "beatGrid": [0.0, 0.484], "downbeats": [0.0],
                "genres": ["house"], "segments": [], "phraseStates": [],
                "cues": {"introEnd": 0.0, "firstDrop": None, "safeEntries": [0.0],
                         "safeExits": [176.0], "phraseBoundaries": [0.0, 176.0]},
            }
            vector = np.ones(512, dtype=np.float32) / np.sqrt(512)
            with patch("app.main.MUSIC_ROOT", music_root), patch("app.main.DATA_ROOT", data_root):
                write_analysis_cache(song, cached_track, vector)
                self.assertIsNotNone(read_analysis_cache(song))
                tags = ({"title": "Tagged", "artist": "Artist", "album": "Album"}, ["house"], "/artwork/new")
                with patch("app.main.tags_for", return_value=tags), \
                     patch("app.main.index_vector", return_value=True), \
                     patch("app.main.librosa.load") as load_audio, \
                     patch("app.main.harmonix") as run_harmonix:
                    restored, source = analyse_track(song, "new-preparation")
                load_audio.assert_not_called()
                run_harmonix.assert_not_called()
                self.assertEqual(source, "cache")
                self.assertEqual(restored["title"], "Cached song")
                self.assertEqual(restored["artworkUrl"], "/artwork/new")
                self.assertTrue(restored["embeddingIndexed"])

                # A changed source must not accidentally inherit stale features.
                song.write_bytes(b"changed source audio with a different size")
                self.assertIsNone(read_analysis_cache(song))

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
