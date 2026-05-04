use std::{env, net::SocketAddr, path::PathBuf, sync::Arc, time::Duration};

use axum::{
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use fraud::{index::Index, payload::FraudRequest, vector::vectorize};
use serde::Serialize;
use tracing::{error, info, warn};

#[derive(Clone)]
struct AppState {
    index: Option<Arc<Index>>,
    knn_timeout: Option<Duration>,
}

#[derive(Serialize)]
struct FraudResponse {
    approved: bool,
    fraud_score: f32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let index_path = env::var_os("INDEX_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/index/data.bin"));
    let index = match Index::open(&index_path) {
        Ok(index) => {
            info!(path = %index_path.display(), records = index.len(), "index loaded");
            Some(Arc::new(index))
        }
        Err(err) => {
            warn!(path = %index_path.display(), error = %err, "index unavailable; using approve fallback");
            None
        }
    };

    let knn_timeout = env::var("KNN_TIMEOUT_US")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|micros| *micros > 0)
        .map(Duration::from_micros);

    let state = AppState { index, knn_timeout };
    let app = Router::new()
        .route("/ready", get(ready))
        .route("/fraud-score", post(fraud_score))
        .with_state(state);

    let addr: SocketAddr = env::var("API_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8080".to_string())
        .parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!(%addr, "api listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn ready() -> impl IntoResponse {
    StatusCode::NO_CONTENT
}

async fn fraud_score(State(state): State<AppState>, body: Bytes) -> impl IntoResponse {
    let request = match serde_json::from_slice::<FraudRequest>(&body) {
        Ok(request) => request,
        Err(err) => {
            error!(error = %err, "invalid payload; using approve fallback");
            return Json(fallback_response());
        }
    };

    let response = match std::panic::catch_unwind(|| score_request(&state, &request)) {
        Ok(response) => response,
        Err(_) => {
            error!(id = %request.id, "panic while scoring request; using approve fallback");
            fallback_response()
        }
    };
    Json(response)
}

fn score_request(state: &AppState, request: &FraudRequest) -> FraudResponse {
    let vector = vectorize(request);
    let fraud_score = state
        .index
        .as_ref()
        .and_then(|index| index.fraud_score(&vector, state.knn_timeout))
        .unwrap_or(0.0);

    FraudResponse {
        approved: fraud_score < 0.6,
        fraud_score,
    }
}

fn fallback_response() -> FraudResponse {
    FraudResponse {
        approved: true,
        fraud_score: 0.0,
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
