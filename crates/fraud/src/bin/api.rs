use std::{
    env, net::SocketAddr, os::unix::fs::PermissionsExt, path::PathBuf, sync::Arc, time::Duration,
};

use axum::{
    body::Bytes,
    extract::State,
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use fraud::{
    fast_parse::parse_and_vectorize,
    index::{Index, SearchResult},
};
use mimalloc::MiMalloc;
use tracing::{error, info, warn};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Clone)]
struct AppState {
    index: Arc<Index>,
    knn_timeout: Duration,
}

const DEFAULT_KNN_TIMEOUT_US: u64 = 1_000;

const FRAUD_RESPONSES: [&[u8]; 6] = [
    b"{\"approved\":true,\"fraud_score\":0.0}",
    b"{\"approved\":true,\"fraud_score\":0.2}",
    b"{\"approved\":true,\"fraud_score\":0.4}",
    b"{\"approved\":false,\"fraud_score\":0.6}",
    b"{\"approved\":false,\"fraud_score\":0.8}",
    b"{\"approved\":false,\"fraud_score\":1.0}",
];
const FRAUD_FALLBACK: &[u8] = FRAUD_RESPONSES[0];

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let index_path = env::var_os("INDEX_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/index/data.bin"));
    let index = Arc::new(Index::open(&index_path)?);
    info!(path = %index_path.display(), records = index.len(), "index loaded");

    let state = AppState {
        index,
        knn_timeout: configured_timeout(),
    };
    let app = Router::new()
        .route("/ready", get(ready))
        .route("/fraud-score", post(fraud_score))
        .with_state(state);

    serve(app).await
}

async fn serve(app: Router) -> anyhow::Result<()> {
    if let Some(socket_path) = env::var_os("SOCKET_PATH").map(PathBuf::from) {
        if let Some(parent) = socket_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let _ = std::fs::remove_file(&socket_path);
        let listener = tokio::net::UnixListener::bind(&socket_path)?;
        std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o666))?;
        info!(path = %socket_path.display(), "api listening on unix socket");
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await?;
        return Ok(());
    }

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

fn configured_timeout() -> Duration {
    Duration::from_micros(
        env::var("KNN_TIMEOUT_US")
            .ok()
            .and_then(|raw| raw.parse::<u64>().ok())
            .filter(|micros| *micros > 0)
            .unwrap_or(DEFAULT_KNN_TIMEOUT_US),
    )
}

async fn ready() -> impl IntoResponse {
    StatusCode::NO_CONTENT
}

async fn fraud_score(State(state): State<AppState>, body: Bytes) -> Response {
    let vector = match parse_and_vectorize(&body) {
        Some(v) => v,
        None => {
            error!("invalid payload; using approve fallback");
            return json_response(FRAUD_FALLBACK);
        }
    };
    let count = match state.index.fraud_score(&vector, Some(state.knn_timeout)) {
        SearchResult::Score(score) => ((score * 5.0).round() as usize).min(5),
        SearchResult::TimedOut => {
            warn!("knn timed out; using approve fallback");
            return json_response(FRAUD_FALLBACK);
        }
    };
    json_response(FRAUD_RESPONSES[count])
}

fn json_response(body: &'static [u8]) -> Response {
    let mut response = Response::new(axum::body::Body::from(Bytes::from_static(body)));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
