mod error;
mod models;
mod repository;
mod routes;
mod state;
mod worker_client;

use std::{env, net::SocketAddr};

use axum::{
    routing::{get, post},
    Router,
};
use tower_http::{cors::CorsLayer, services::ServeDir, trace::TraceLayer};
use tracing::info;

use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let state = AppState::connect().await?;
    let data_root = env::var("DATA_ROOT").unwrap_or_else(|_| "/data".into());
    let music_root = env::var("MUSIC_ROOT").unwrap_or_else(|_| "/music".into());

    let app = Router::new()
        .route("/api/health", get(routes::health))
        .route("/api/overview", get(routes::overview))
        .route(
            "/api/preparations",
            get(routes::list_preparations).post(routes::create_preparation),
        )
        .route(
            "/api/preparations/:id",
            get(routes::get_preparation).delete(routes::delete_preparation),
        )
        .route(
            "/api/preparations/:id/progress",
            post(routes::update_preparation_progress),
        )
        .route(
            "/api/mixes",
            get(routes::list_mixes).post(routes::create_mix),
        )
        .route(
            "/api/mixes/:id",
            get(routes::get_mix).delete(routes::delete_mix),
        )
        .route(
            "/api/mixes/:id/next-transition",
            post(routes::next_transition),
        )
        .nest_service("/media", ServeDir::new(data_root))
        .nest_service("/library", ServeDir::new(music_root))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let address = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!(%address, "DJ.Attatouille API is ready");
    let listener = tokio::net::TcpListener::bind(address).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
