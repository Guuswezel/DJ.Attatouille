use std::{env, path::PathBuf, sync::Arc};

use anyhow::Context;
use mongodb::{bson::doc, options::ClientOptions, Client, Collection};
use reqwest::Client as HttpClient;
use tokio::time::{sleep, Duration};

use crate::models::{Mix, Preparation};

#[derive(Clone)]
pub struct AppState {
    pub preparations: Collection<Preparation>,
    pub mixes: Collection<Mix>,
    pub http: HttpClient,
    pub worker_url: String,
    pub data_root: PathBuf,
}

impl AppState {
    pub async fn connect() -> anyhow::Result<Arc<Self>> {
        let mongo_url =
            env::var("MONGO_URL").unwrap_or_else(|_| "mongodb://localhost:27017".into());
        let mut options = ClientOptions::parse(&mongo_url).await?;
        options.app_name = Some("dj-attatouille".into());
        let client = Client::with_options(options)?;

        for attempt in 1..=20 {
            if client
                .database("admin")
                .run_command(doc! { "ping": 1 })
                .await
                .is_ok()
            {
                break;
            }
            if attempt == 20 {
                anyhow::bail!("MongoDB did not become ready");
            }
            sleep(Duration::from_millis(500)).await;
        }

        let database = client.database("dj_attatouille");
        Ok(Arc::new(Self {
            preparations: database.collection("preparations"),
            mixes: database.collection("mixes"),
            http: HttpClient::builder()
                .timeout(Duration::from_secs(60 * 60))
                .build()
                .context("building worker client")?,
            worker_url: env::var("WORKER_URL").unwrap_or_else(|_| "http://localhost:8090".into()),
            data_root: PathBuf::from(env::var("DATA_ROOT").unwrap_or_else(|_| "/data".into())),
        }))
    }
}
