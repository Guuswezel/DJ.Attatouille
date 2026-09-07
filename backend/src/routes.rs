use std::{io::ErrorKind, path::Path as FilePath, sync::Arc};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use chrono::Utc;
use mongodb::bson::{doc, Bson};
use serde_json::json;
use tracing::error;
use uuid::Uuid;

use crate::{
    error::{ApiError, ApiResult},
    models::{
        CreatePreparation, Mix, MixOptions, Preparation, PreparationProgressUpdate, SkipPlan,
        SkipRequest,
    },
    repository,
    state::AppState,
    worker_client,
};

pub async fn health() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(json!({ "status": "ok", "service": "dj-attatouille-api" })),
    )
}

pub async fn overview(State(state): State<Arc<AppState>>) -> ApiResult<serde_json::Value> {
    let preparations = repository::preparations(&state).await?;
    let mixes = repository::mixes(&state).await?;
    let active_jobs = preparations
        .iter()
        .filter(|item| is_active(&item.status))
        .count()
        + mixes.iter().filter(|item| is_active(&item.status)).count();
    Ok(Json(
        json!({ "preparations": preparations, "mixes": mixes, "activeJobs": active_jobs }),
    ))
}

pub async fn list_preparations(State(state): State<Arc<AppState>>) -> ApiResult<Vec<Preparation>> {
    Ok(Json(repository::preparations(&state).await?))
}

pub async fn get_preparation(
    Path(id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> ApiResult<Preparation> {
    repository::find_preparation(&state, &id).await.map(Json)
}

pub async fn delete_preparation(
    Path(id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> ApiResult<serde_json::Value> {
    let preparation = repository::find_preparation(&state, &id).await?;
    if is_active(&preparation.status) {
        request_worker_cancellation(&state, "preparation", &id).await;
    }

    let related_mixes = repository::mixes_for_preparation(&state, &id).await?;

    let deleted_mix_ids = related_mixes
        .iter()
        .map(|mix| mix.id.clone())
        .collect::<Vec<_>>();
    for mix in &related_mixes {
        if is_active(&mix.status) {
            request_worker_cancellation(&state, "mix", &mix.id).await;
        }
        repository::delete_mix(&state, &mix.id).await?;
        cleanup_mix_assets(&state, &mix.id).await;
    }
    repository::delete_preparation(&state, &id).await?;

    if let Err(problem) = worker_client::cleanup_preparation(&state, &preparation).await {
        // The preparation has already been removed from the user-visible database.
        // Asset cleanup is deliberately best effort so an unavailable worker cannot
        // make deletion appear to fail.
        error!(%id, error = ?problem, "preparation asset cleanup failed");
    }

    Ok(Json(
        json!({ "deletedPreparationId": id, "deletedMixIds": deleted_mix_ids }),
    ))
}

pub async fn create_preparation(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreatePreparation>,
) -> ApiResult<Preparation> {
    let source_path = request.source_path.unwrap_or_else(|| "/music".into());
    if source_path != "/music" && !source_path.starts_with("/music/") {
        return Err(ApiError::bad_request(
            "The library folder must be inside the mounted /music path.",
        ));
    }
    let label = request.label.unwrap_or_else(|| "Music library".into());
    if label.trim().is_empty() {
        return Err(ApiError::bad_request("Give this music folder a name."));
    }

    let now = Utc::now().to_rfc3339();
    let preparation = Preparation {
        id: Uuid::new_v4().to_string(),
        label: label.trim().to_owned(),
        source_path,
        status: "queued".into(),
        progress: 0,
        message: "Waiting for the local analyser".into(),
        track_count: 0,
        discovered_track_count: 0,
        analysed_track_count: 0,
        failed_track_count: 0,
        cached_track_count: 0,
        current_track: None,
        genres: vec![],
        tracks: vec![],
        model_report: None,
        created_at: now.clone(),
        updated_at: now,
    };
    state
        .preparations
        .insert_one(&preparation)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let job_state = state.clone();
    let job_id = preparation.id.clone();
    tokio::spawn(async move { run_analysis(job_state, job_id).await });
    Ok(Json(preparation))
}

/// Receives small, local-only status updates from the Python analysis worker.
/// The browser observes these fields through its existing overview polling.
pub async fn update_preparation_progress(
    Path(id): Path<String>,
    State(state): State<Arc<AppState>>,
    Json(update): Json<PreparationProgressUpdate>,
) -> ApiResult<serde_json::Value> {
    let preparation = repository::find_preparation(&state, &id).await?;
    if !is_active(&preparation.status) {
        // A cancelled/deleted worker can race one final callback. Do not let it
        // overwrite a completed job, but acknowledge the harmless request.
        return Ok(Json(json!({ "status": preparation.status })));
    }

    let mut changes = doc! {
        "progress": i32::from(update.progress.min(99)),
        "discoveredTrackCount": update.discovered_track_count as i64,
        "analysedTrackCount": update.analysed_track_count as i64,
        "failedTrackCount": update.failed_track_count as i64,
        "cachedTrackCount": update.cached_track_count as i64,
        "message": update.message,
        "updatedAt": Utc::now().to_rfc3339(),
    };
    changes.insert(
        "currentTrack",
        update.current_track.map(Bson::String).unwrap_or(Bson::Null),
    );
    repository::update_preparation(&state, &id, doc! { "$set": changes }).await?;
    Ok(Json(json!({ "status": "updated" })))
}

pub async fn retry_preparation(
    Path(id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> ApiResult<Preparation> {
    let preparation = repository::find_preparation(&state, &id).await?;
    if preparation.status != "failed" {
        return Err(ApiError::bad_request(
            "Only a failed preparation can be resumed.",
        ));
    }
    repository::update_preparation(
        &state,
        &id,
        doc! { "$set": doc! {
            "status": "queued", "progress": 0i32,
            "message": "Waiting to resume from cached track analysis",
            "currentTrack": Bson::Null,
            "updatedAt": Utc::now().to_rfc3339(),
        } },
    )
    .await?;
    let job_state = state.clone();
    let job_id = id.clone();
    tokio::spawn(async move { run_analysis(job_state, job_id).await });
    repository::find_preparation(&state, &id).await.map(Json)
}

pub async fn list_mixes(State(state): State<Arc<AppState>>) -> ApiResult<Vec<Mix>> {
    Ok(Json(repository::mixes(&state).await?))
}

pub async fn get_mix(Path(id): Path<String>, State(state): State<Arc<AppState>>) -> ApiResult<Mix> {
    repository::find_mix(&state, &id).await.map(Json)
}

pub async fn delete_mix(
    Path(id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> ApiResult<serde_json::Value> {
    let mix = repository::find_mix(&state, &id).await?;
    if is_active(&mix.status) {
        request_worker_cancellation(&state, "mix", &id).await;
    }
    repository::delete_mix(&state, &id).await?;
    cleanup_mix_assets(&state, &id).await;
    Ok(Json(json!({ "deletedMixId": id })))
}

pub async fn create_mix(
    State(state): State<Arc<AppState>>,
    Json(options): Json<MixOptions>,
) -> ApiResult<Mix> {
    if options.min_track_seconds < 20
        || options.max_track_seconds < options.min_track_seconds
        || options.acceptance_percentage < 50
        || options.acceptance_percentage > 100
    {
        return Err(ApiError::bad_request(
            "Use 20+ seconds, a valid duration range, and an acceptance percentage between 50 and 100.",
        ));
    }
    let preparation = repository::find_preparation(&state, &options.preparation_id).await?;
    if preparation.status != "ready" {
        return Err(ApiError::bad_request(
            "This library is still being prepared.",
        ));
    }
    let now = Utc::now().to_rfc3339();
    let mix = Mix {
        id: Uuid::new_v4().to_string(),
        name: format!("{} mix", preparation.label),
        status: "queued".into(),
        progress: 0,
        message: "Waiting for the mix engine".into(),
        options,
        playlist: vec![],
        rejected_track_ids: vec![],
        duration_seconds: 0.0,
        audio_url: None,
        created_at: now.clone(),
        updated_at: now,
    };
    state
        .mixes
        .insert_one(&mix)
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let job_state = state.clone();
    let job_id = mix.id.clone();
    tokio::spawn(async move { run_mix(job_state, job_id).await });
    Ok(Json(mix))
}

pub async fn next_transition(
    Path(id): Path<String>,
    State(state): State<Arc<AppState>>,
    Json(request): Json<SkipRequest>,
) -> ApiResult<SkipPlan> {
    let mix = repository::find_mix(&state, &id).await?;
    if mix.status != "ready" {
        return Err(ApiError::bad_request("The mix is not ready yet."));
    }
    worker_client::plan_skip(&state, &mix, &request)
        .await
        .map(Json)
        .map_err(|error| {
            error!(error = ?error, "safe skip planning failed");
            ApiError::internal("The transition engine could not plan a safe skip.")
        })
}

async fn run_analysis(state: Arc<AppState>, preparation_id: String) {
    let result: anyhow::Result<()> = async {
        let preparation = repository::find_preparation(&state, &preparation_id).await?;
        repository::update_preparation(&state, &preparation_id, doc! { "$set": doc! {
            "status": "running", "progress": 5i32, "message": "Scanning the music folder", "updatedAt": Utc::now().to_rfc3339()
        } }).await?;
        let result = worker_client::analyse(&state, &preparation).await?;
        let warning = if result.failures.is_empty() {
            "Data preparation done".to_owned()
        } else {
            format!("Data preparation done; {} unreadable file(s) skipped", result.failures.len())
        };
        repository::update_preparation(&state, &preparation_id, doc! { "$set": doc! {
            "status": "ready", "progress": 100i32, "message": warning,
            "tracks": mongodb::bson::to_bson(&result.tracks)?, "trackCount": result.tracks.len() as i64,
            "discoveredTrackCount": (result.tracks.len() + result.failures.len()) as i64,
            "analysedTrackCount": result.tracks.len() as i64,
            "failedTrackCount": result.failures.len() as i64,
            "cachedTrackCount": result.cached_track_count as i64,
            "currentTrack": Bson::Null,
            "genres": mongodb::bson::to_bson(&result.genres)?, "modelReport": mongodb::bson::to_bson(&result.model_report)?,
            "updatedAt": Utc::now().to_rfc3339()
        } }).await?;
        Ok(())
    }
    .await;
    if let Err(problem) = result {
        error!(%preparation_id, error = ?problem, "analysis job failed");
        let _ = repository::update_preparation(&state, &preparation_id, doc! { "$set": doc! {
            "status": "failed", "currentTrack": Bson::Null, "message": problem.to_string(), "updatedAt": Utc::now().to_rfc3339()
        } }).await;
    }
}

async fn run_mix(state: Arc<AppState>, mix_id: String) {
    let result: anyhow::Result<()> = async {
        let mix = repository::find_mix(&state, &mix_id).await?;
        let preparation = repository::find_preparation(&state, &mix.options.preparation_id).await?;
        repository::update_mix(&state, &mix_id, doc! { "$set": doc! {
            "status": "running", "progress": 5i32, "message": "Scoring and rendering monitored transitions", "updatedAt": Utc::now().to_rfc3339()
        } }).await?;
        let render = worker_client::render_mix(&state, &mix, &preparation.tracks).await?;
        repository::update_mix(&state, &mix_id, doc! { "$set": doc! {
            "status": "ready", "progress": 100i32, "message": "Mix ready",
            "playlist": mongodb::bson::to_bson(&render.playlist)?, "rejectedTrackIds": mongodb::bson::to_bson(&render.rejected_track_ids)?,
            "durationSeconds": render.duration_seconds, "audioUrl": render.audio_url, "updatedAt": Utc::now().to_rfc3339()
        } }).await?;
        Ok(())
    }
    .await;
    if let Err(problem) = result {
        error!(%mix_id, error = ?problem, "mix job failed");
        let _ = repository::update_mix(&state, &mix_id, doc! { "$set": doc! {
            "status": "failed", "message": problem.to_string(), "updatedAt": Utc::now().to_rfc3339()
        } }).await;
    }
}

fn is_active(status: &str) -> bool {
    matches!(status, "queued" | "running")
}

async fn request_worker_cancellation(state: &AppState, job_type: &str, job_id: &str) {
    if let Err(problem) = worker_client::cancel_job(state, job_type, job_id).await {
        // Deletion is still allowed even if the worker has already stopped or
        // is unavailable. Removing the database job prevents it reappearing.
        error!(%job_type, %job_id, error = ?problem, "worker cancellation request failed");
    }
}

async fn cleanup_mix_assets(state: &AppState, mix_id: &str) {
    // IDs originate from UUIDs generated by this service. Validate before forming
    // filenames so cleanup can never escape the dedicated generated-audio folders.
    if Uuid::parse_str(mix_id).is_err() {
        error!(%mix_id, "refusing to clean generated audio for a non-UUID mix id");
        return;
    }

    remove_if_present(&state.data_root.join("mixes").join(format!("{mix_id}.mp3"))).await;
    let skips_dir = state.data_root.join("skips");
    let prefix = format!("{mix_id}-");
    let Ok(mut entries) = tokio::fs::read_dir(&skips_dir).await else {
        return;
    };
    while let Ok(Some(entry)) = entries.next_entry().await {
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            continue;
        };
        if name.starts_with(&prefix) && name.ends_with(".mp3") {
            remove_if_present(&entry.path()).await;
        }
    }
}

async fn remove_if_present(path: &FilePath) {
    if let Err(problem) = tokio::fs::remove_file(path).await {
        if problem.kind() != ErrorKind::NotFound {
            error!(path = %path.display(), error = ?problem, "could not remove generated asset");
        }
    }
}
