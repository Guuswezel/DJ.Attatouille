use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Track {
    pub id: String,
    pub relative_path: String,
    pub title: String,
    pub artist: String,
    pub album: String,
    pub artwork_url: Option<String>,
    pub duration_seconds: f64,
    pub bpm: f64,
    pub key: String,
    pub energy: f64,
    #[serde(default)]
    pub loudness_lufs: Option<f64>,
    #[serde(default)]
    pub waveform: Vec<f64>,
    #[serde(default)]
    pub waveform_detail: Option<String>,
    /// Waveform v2 is binned directly from decoded audio, sharing the exact
    /// source timeline with the beat grid. Older preparations fall back to a
    /// client-side decode while their stale cached waveform is refreshed.
    #[serde(default)]
    pub waveform_detail_version: Option<u8>,
    #[serde(default)]
    pub beat_grid: Vec<f64>,
    #[serde(default)]
    pub downbeats: Vec<f64>,
    pub genres: Vec<String>,
    pub segments: Vec<Segment>,
    #[serde(default)]
    pub phrase_states: Vec<PhraseState>,
    pub cues: TrackCues,
    pub embedding_indexed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub label: String,
    pub energy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhraseState {
    pub start: f64,
    pub end: f64,
    pub phrase_index: usize,
    pub energy: f64,
    pub energy_slope: f64,
    pub bass_activity: f64,
    pub drum_activity: f64,
    pub vocal_activity: f64,
    #[serde(default)]
    pub vocal_head_activity: f64,
    #[serde(default)]
    pub vocal_tail_activity: f64,
    #[serde(default)]
    pub vocal_continuity: f64,
    pub spectral_density: f64,
    pub harmonic_density: f64,
    pub novelty_in: f64,
    pub novelty_out: f64,
    pub loopability: f64,
    pub cue_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct TrackCues {
    pub intro_end: f64,
    pub first_drop: Option<f64>,
    pub safe_entries: Vec<f64>,
    pub safe_exits: Vec<f64>,
    #[serde(default)]
    pub phrase_boundaries: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Preparation {
    pub id: String,
    pub source_path: String,
    pub label: String,
    pub status: String,
    pub progress: u8,
    pub message: String,
    pub track_count: usize,
    /// Progress details are written by the analysis worker as it scans and
    /// processes the folder. Defaults retain compatibility with preparations
    /// created before live progress reporting existed.
    #[serde(default)]
    pub discovered_track_count: usize,
    #[serde(default)]
    pub analysed_track_count: usize,
    #[serde(default)]
    pub failed_track_count: usize,
    #[serde(default)]
    pub cached_track_count: usize,
    #[serde(default)]
    pub current_track: Option<String>,
    pub genres: Vec<String>,
    pub tracks: Vec<Track>,
    pub model_report: Option<ModelReport>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelReport {
    pub structure_model: String,
    pub embedding_model: String,
    pub genre_source: String,
    pub qdrant_collection: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreatePreparation {
    pub source_path: Option<String>,
    pub label: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreparationProgressUpdate {
    pub progress: u8,
    pub discovered_track_count: usize,
    pub analysed_track_count: usize,
    #[serde(default)]
    pub failed_track_count: usize,
    #[serde(default)]
    pub cached_track_count: usize,
    #[serde(default)]
    pub current_track: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MixOptions {
    pub preparation_id: String,
    pub min_track_seconds: u32,
    pub max_track_seconds: u32,
    pub acceptance_percentage: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlaylistItem {
    pub track: Track,
    pub start_seconds: f64,
    pub end_seconds: f64,
    #[serde(default)]
    pub source_start_seconds: f64,
    #[serde(default)]
    pub source_end_seconds: f64,
    #[serde(default)]
    pub deck_bpm: f64,
    #[serde(default)]
    pub gain_db: f64,
    pub transition_in: Option<Transition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Transition {
    pub from_track_id: String,
    pub to_track_id: String,
    pub exit_at_seconds: f64,
    pub enter_at_seconds: f64,
    pub overlap_seconds: f64,
    pub bpm_ratio: f64,
    #[serde(default = "one")]
    pub tempo_factor: f64,
    #[serde(default)]
    pub loop_seconds: f64,
    #[serde(default)]
    pub beat_matched: bool,
    #[serde(default)]
    pub bar_matched: bool,
    #[serde(default)]
    pub phrase_matched: bool,
    #[serde(default)]
    pub beat_alignment_error_ms: f64,
    #[serde(default)]
    pub overlap_bars: u8,
    #[serde(default)]
    pub fade_shape: String,
    #[serde(default)]
    pub spectrum_plan: String,
    #[serde(default)]
    pub technique: String,
    #[serde(default)]
    pub bass_swap_progress: f64,
    #[serde(default)]
    pub vocal_clash_risk: f64,
    #[serde(default)]
    pub vocal_boundary_risk: f64,
    #[serde(default)]
    pub incoming_vocal_head: f64,
    #[serde(default)]
    pub learned_compatibility: f64,
    #[serde(default = "one")]
    pub fade_out_curve: f64,
    #[serde(default = "one")]
    pub fade_in_curve: f64,
    #[serde(default)]
    pub outgoing_low_db: f64,
    #[serde(default)]
    pub incoming_low_db: f64,
    #[serde(default)]
    pub outgoing_mid_db: f64,
    #[serde(default)]
    pub incoming_mid_db: f64,
    #[serde(default)]
    pub outgoing_high_db: f64,
    #[serde(default)]
    pub incoming_high_db: f64,
    #[serde(default)]
    pub timing_score: f64,
    #[serde(default)]
    pub policy_version: String,
    pub style: String,
    pub quality_score: f64,
    #[serde(default)]
    pub render_quality_score: Option<f64>,
    #[serde(default)]
    pub pre_overlay_peak_dbfs: Option<f64>,
    #[serde(default)]
    pub overlay_gain_db: Option<f64>,
    pub notes: Vec<String>,
}

fn one() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Mix {
    pub id: String,
    pub name: String,
    pub status: String,
    pub progress: u8,
    pub message: String,
    pub options: MixOptions,
    pub playlist: Vec<PlaylistItem>,
    pub rejected_track_ids: Vec<String>,
    pub duration_seconds: f64,
    pub audio_url: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkipRequest {
    pub current_track_index: usize,
    pub playback_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkipPlan {
    pub transition: Transition,
    pub audio_url: String,
    pub resume_at_seconds: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisResult {
    pub tracks: Vec<Track>,
    pub genres: Vec<String>,
    pub model_report: ModelReport,
    #[serde(default)]
    pub failures: Vec<String>,
    #[serde(default)]
    pub cached_track_count: usize,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MixRender {
    pub playlist: Vec<PlaylistItem>,
    pub rejected_track_ids: Vec<String>,
    pub duration_seconds: f64,
    pub audio_url: String,
}
