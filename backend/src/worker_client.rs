use anyhow::Context;
use serde_json::json;
use tokio::time::{timeout, Duration};

use crate::{
    models::{AnalysisResult, Mix, MixRender, Preparation, SkipPlan, SkipRequest},
    state::AppState,
};

pub async fn analyse(
    state: &AppState,
    preparation: &Preparation,
) -> anyhow::Result<AnalysisResult> {
    request(
        state,
        "/analyze",
        json!({ "preparationId": preparation.id, "sourcePath": preparation.source_path }),
        "starting analysis worker",
    )
    .await
}

pub async fn cleanup_preparation(
    state: &AppState,
    preparation: &Preparation,
) -> anyhow::Result<()> {
    let response = state
        .http
        .post(format!("{}/cleanup-preparation", state.worker_url))
        .json(&json!({
            "preparationId": preparation.id,
            "trackIds": preparation.tracks.iter().map(|track| &track.id).collect::<Vec<_>>(),
        }))
        .send()
        .await
        .context("cleaning preparation assets")?;
    if !response.status().is_success() {
        anyhow::bail!("cleanup worker returned {}", response.status());
    }
    Ok(())
}

pub async fn cancel_job(state: &AppState, job_type: &str, job_id: &str) -> anyhow::Result<()> {
    let response = timeout(
        Duration::from_secs(3),
        state
            .http
            .post(format!("{}/cancel-job", state.worker_url))
            .json(&json!({ "jobType": job_type, "jobId": job_id }))
            .send(),
    )
    .await
    .context("timed out while requesting worker cancellation")?
    .context("requesting worker cancellation")?;
    if !response.status().is_success() {
        anyhow::bail!("worker cancellation returned {}", response.status());
    }
    Ok(())
}

pub async fn render_mix(
    state: &AppState,
    mix: &Mix,
    tracks: &[crate::models::Track],
) -> anyhow::Result<MixRender> {
    request(
        state,
        "/prepare-mix",
        json!({ "mix": mix, "tracks": tracks }),
        "starting mix worker",
    )
    .await
}

pub async fn plan_skip(
    state: &AppState,
    mix: &Mix,
    request_body: &SkipRequest,
) -> anyhow::Result<SkipPlan> {
    request(
        state,
        "/next-transition",
        json!({
            "mix": mix,
            "currentTrackIndex": request_body.current_track_index,
            "playbackSeconds": request_body.playback_seconds,
        }),
        "asking transition engine",
    )
    .await
}

async fn request<T>(
    state: &AppState,
    path: &str,
    body: serde_json::Value,
    context: &str,
) -> anyhow::Result<T>
where
    T: serde::de::DeserializeOwned,
{
    let response = state
        .http
        .post(format!("{}{}", state.worker_url, path))
        .json(&body)
        .send()
        .await
        .with_context(|| context.to_owned())?;
    let status = response.status();
    if !status.is_success() {
        let detail = response.text().await.unwrap_or_default();
        anyhow::bail!("worker returned {status}: {detail}");
    }
    response.json().await.context("reading worker response")
}
