use futures::TryStreamExt;
use mongodb::{
    bson::{doc, Document},
    Collection,
};
use serde::{Deserialize, Serialize};

use crate::{
    error::ApiError,
    models::{Mix, Preparation},
    state::AppState,
};

pub async fn preparations(state: &AppState) -> anyhow::Result<Vec<Preparation>> {
    collect(&state.preparations).await
}

pub async fn mixes(state: &AppState) -> anyhow::Result<Vec<Mix>> {
    collect(&state.mixes).await
}

pub async fn find_preparation(state: &AppState, id: &str) -> Result<Preparation, ApiError> {
    state
        .preparations
        .find_one(doc! { "id": id })
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .ok_or_else(|| ApiError::not_found("Preparation not found"))
}

pub async fn find_mix(state: &AppState, id: &str) -> Result<Mix, ApiError> {
    state
        .mixes
        .find_one(doc! { "id": id })
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
        .ok_or_else(|| ApiError::not_found("Mix not found"))
}

pub async fn update_preparation(
    state: &AppState,
    id: &str,
    update: Document,
) -> anyhow::Result<()> {
    state
        .preparations
        .update_one(doc! { "id": id }, update)
        .await?;
    Ok(())
}

pub async fn update_mix(state: &AppState, id: &str, update: Document) -> anyhow::Result<()> {
    state.mixes.update_one(doc! { "id": id }, update).await?;
    Ok(())
}

pub async fn mixes_for_preparation(
    state: &AppState,
    preparation_id: &str,
) -> Result<Vec<Mix>, ApiError> {
    let mut cursor = state
        .mixes
        .find(doc! { "options.preparationId": preparation_id })
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let mut rows = vec![];
    while let Some(row) = cursor
        .try_next()
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?
    {
        rows.push(row);
    }
    Ok(rows)
}

pub async fn delete_preparation(state: &AppState, id: &str) -> Result<Preparation, ApiError> {
    let preparation = find_preparation(state, id).await?;
    state
        .preparations
        .delete_one(doc! { "id": id })
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    Ok(preparation)
}

pub async fn delete_mix(state: &AppState, id: &str) -> Result<Mix, ApiError> {
    let mix = find_mix(state, id).await?;
    state
        .mixes
        .delete_one(doc! { "id": id })
        .await
        .map_err(|error| ApiError::internal(error.to_string()))?;
    Ok(mix)
}

async fn collect<T>(collection: &Collection<T>) -> anyhow::Result<Vec<T>>
where
    T: Serialize + for<'de> Deserialize<'de> + Unpin + Send + Sync,
{
    let mut cursor = collection
        .find(doc! {})
        .sort(doc! { "createdAt": -1 })
        .await?;
    let mut rows = vec![];
    while let Some(row) = cursor.try_next().await? {
        rows.push(row);
    }
    Ok(rows)
}
