//! `nvsim-server` — Axum host fronting the deterministic nvsim pipeline.
//!
//! ADR-092 §6.2 — REST control plane + binary WebSocket data plane.
//! Same `(scene, config, seed)` produces byte-identical witnesses across
//! the WASM transport (in-browser worker) and this WS transport — the
//! determinism contract the dashboard's Verify panel asserts.
//!
//! ## Routes
//!
//! | Method | Path                    | Purpose                          |
//! |--------|-------------------------|----------------------------------|
//! | GET    | /api/health             | liveness + nvsim version + magic |
//! | GET    | /api/scene              | current scene (JSON)             |
//! | PUT    | /api/scene              | replace scene                    |
//! | GET    | /api/config             | current `PipelineConfig`         |
//! | PUT    | /api/config             | replace config                   |
//! | GET    | /api/seed               | current seed (hex)               |
//! | PUT    | /api/seed               | set seed                         |
//! | POST   | /api/run                | start a run                      |
//! | POST   | /api/pause              | pause                            |
//! | POST   | /api/reset              | reset to t=0                     |
//! | POST   | /api/step               | single step                      |
//! | POST   | /api/witness/generate   | run N frames + return SHA-256    |
//! | POST   | /api/witness/verify     | re-derive + compare expected     |
//! | POST   | /api/witness/reference  | run canonical Proof::generate    |
//! | POST   | /api/export-proof       | proof bundle as JSON             |
//! | GET    | /ws/stream              | binary MagFrame batch stream     |

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::{header, HeaderValue, StatusCode},
    response::IntoResponse,
    routing::{get, post, put},
    Json, Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tower_http::{
    cors::{AllowOrigin, Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::{info, warn};

use nvsim::{
    pipeline::{Pipeline, PipelineConfig},
    proof::Proof,
    scene::Scene,
};

#[derive(Parser, Debug)]
#[command(name = "nvsim-server", version)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:7878")]
    listen: SocketAddr,

    /// Browser origins allowed by CORS; defaults to the local Vite dashboard and RuView Pages.
    #[arg(
        long,
        default_value = "http://localhost:5173,http://127.0.0.1:5173,https://ruvnet.github.io"
    )]
    allowed_origin: String,

    /// Require `Authorization: Bearer <token>` on state-mutating routes
    /// (PUT/POST). Strongly recommended before binding to anything other
    /// than 127.0.0.1. Can also be set via the NVSIM_TOKEN env var.
    #[arg(long, env = "NVSIM_TOKEN", hide_env_values = true)]
    token: Option<String>,
}

#[derive(Debug, Clone)]
struct AppState {
    inner: Arc<Mutex<RunState>>,
}

#[derive(Debug, Clone)]
struct RunState {
    scene: Scene,
    config: PipelineConfig,
    seed: u64,
    running: bool,
    frames_emitted: u64,
}

impl AppState {
    fn new() -> Self {
        let scene = Proof::reference_scene().expect("reference scene parses");
        Self {
            inner: Arc::new(Mutex::new(RunState {
                scene,
                config: PipelineConfig::default(),
                seed: Proof::SEED,
                running: false,
                frames_emitted: 0,
            })),
        }
    }
}

#[derive(Serialize)]
struct HealthBody {
    nvsim_version: &'static str,
    magic: u32,
    frame_bytes: usize,
    expected_witness_hex: &'static str,
}

#[derive(Serialize)]
struct SeedBody {
    seed_hex: String,
}

#[derive(Deserialize)]
struct SeedReq {
    seed_hex: String,
}

#[derive(Deserialize, Default)]
struct WitnessReq {
    samples: Option<usize>,
}

#[derive(Serialize)]
struct WitnessBody {
    witness_hex: String,
    samples: usize,
    seed_hex: String,
}

#[derive(Deserialize)]
struct VerifyReq {
    expected_hex: String,
    samples: Option<usize>,
}

#[derive(Serialize)]
struct VerifyBody {
    ok: bool,
    actual_hex: String,
    expected_hex: String,
}

/// Incoming request body for the `/step` endpoint.
/// Fields are optional; unused ones are reserved for future extensions.
#[derive(Deserialize)]
#[allow(dead_code)]
struct StepReq {
    direction: Option<String>,
    dt_ms: Option<f64>,
}

#[derive(Serialize)]
struct ProofBundle {
    kind: &'static str,
    nvsim_version: &'static str,
    seed_hex: String,
    n_samples: usize,
    witness_hex: String,
    expected_hex: &'static str,
    ok: bool,
    ts: String,
}

const EXPECTED_WITNESS_HEX: &str =
    "cc8de9b01b0ff5bd97a6c17848a3f156c174ea7589d0888164a441584ec593b4";

/// Bearer-token gate for state-mutating routes when `--token` is configured.
async fn require_bearer(
    State(token): State<String>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let authorized = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .map(|candidate| candidate == token)
        .unwrap_or(false);

    if authorized {
        next.run(req).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": "missing or invalid bearer token"})),
        )
            .into_response()
    }
}

fn build_cors_layer(allowed_origin: &str) -> CorsLayer {
    let allowed_origin = allowed_origin.trim();
    let allow_origin = if allowed_origin == "*" {
        AllowOrigin::any()
    } else {
        let origins = allowed_origin
            .split(',')
            .filter_map(|origin| origin.trim().parse::<HeaderValue>().ok());
        AllowOrigin::list(origins)
    };

    CorsLayer::new()
        .allow_origin(allow_origin)
        .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE])
        .allow_methods(Any)
}

fn build_router(state: AppState, token: Option<String>) -> Router {
    let read_only: Router<AppState> = Router::new()
        .route("/api/health", get(health))
        .route("/api/scene", get(get_scene))
        .route("/api/config", get(get_config))
        .route("/api/seed", get(get_seed))
        .route("/ws/stream", get(ws_handler));

    let mut mutating: Router<AppState> = Router::new()
        .route("/api/scene", put(put_scene))
        .route("/api/config", put(put_config))
        .route("/api/seed", put(put_seed))
        .route("/api/run", post(run_pipe))
        .route("/api/pause", post(pause_pipe))
        .route("/api/reset", post(reset_pipe))
        .route("/api/step", post(step_pipe))
        .route("/api/witness/generate", post(witness_generate))
        .route("/api/witness/verify", post(witness_verify))
        .route("/api/witness/reference", post(witness_reference))
        .route("/api/export-proof", post(export_proof));

    if let Some(token) = token {
        mutating = mutating.layer(axum::middleware::from_fn_with_state(token, require_bearer));
    }

    read_only.merge(mutating).with_state(state)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nvsim_server=info,tower_http=info".into()),
        )
        .init();

    let args = Args::parse();
    let state = AppState::new();
    let cors = build_cors_layer(&args.allowed_origin);

    if args.token.is_some() {
        info!("nvsim-server: bearer auth ENABLED on state-mutating routes");
    } else if !args.listen.ip().is_loopback() {
        warn!(
            "nvsim-server bound to {} with NO --token/NVSIM_TOKEN set — anyone who can reach this address can mutate simulator state (PUT scene/config/seed, POST run/pause/reset/step/witness/export-proof). Set NVSIM_TOKEN before exposing this outside localhost.",
            args.listen
        );
    }

    let app = build_router(state, args.token)
        .layer(cors)
        .layer(TraceLayer::new_for_http());

    info!("nvsim-server listening on http://{}", args.listen);
    let listener = tokio::net::TcpListener::bind(args.listen)
        .await
        .expect("bind listener");
    axum::serve(listener, app).await.expect("axum serve");
}

async fn health() -> Json<HealthBody> {
    Json(HealthBody {
        nvsim_version: env!("CARGO_PKG_VERSION"),
        magic: nvsim::MAG_FRAME_MAGIC,
        frame_bytes: nvsim::frame::MAG_FRAME_BYTES,
        expected_witness_hex: EXPECTED_WITNESS_HEX,
    })
}

async fn get_scene(State(s): State<AppState>) -> Json<Scene> {
    Json(s.inner.lock().await.scene.clone())
}

async fn put_scene(
    State(s): State<AppState>,
    Json(scene): Json<Scene>,
) -> Result<&'static str, AppError> {
    s.inner.lock().await.scene = scene;
    Ok("ok")
}

async fn get_config(State(s): State<AppState>) -> Json<PipelineConfig> {
    Json(s.inner.lock().await.config)
}

async fn put_config(
    State(s): State<AppState>,
    Json(cfg): Json<PipelineConfig>,
) -> Result<&'static str, AppError> {
    s.inner.lock().await.config = cfg;
    Ok("ok")
}

async fn get_seed(State(s): State<AppState>) -> Json<SeedBody> {
    let seed = s.inner.lock().await.seed;
    Json(SeedBody {
        seed_hex: format!("0x{:016X}", seed),
    })
}

async fn put_seed(
    State(s): State<AppState>,
    Json(req): Json<SeedReq>,
) -> Result<&'static str, AppError> {
    let raw = req.seed_hex.trim().trim_start_matches("0x");
    let seed = u64::from_str_radix(raw, 16).map_err(|e| AppError::BadInput(e.to_string()))?;
    s.inner.lock().await.seed = seed;
    Ok("ok")
}

async fn run_pipe(State(s): State<AppState>) -> &'static str {
    s.inner.lock().await.running = true;
    "running"
}

async fn pause_pipe(State(s): State<AppState>) -> &'static str {
    s.inner.lock().await.running = false;
    "paused"
}

async fn reset_pipe(State(s): State<AppState>) -> &'static str {
    let mut g = s.inner.lock().await;
    g.frames_emitted = 0;
    g.running = false;
    "reset"
}

async fn step_pipe(
    State(s): State<AppState>,
    Json(_req): Json<StepReq>,
) -> Result<&'static str, AppError> {
    s.inner.lock().await.frames_emitted += 1;
    Ok("ok")
}

async fn witness_generate(
    State(s): State<AppState>,
    Json(req): Json<WitnessReq>,
) -> Json<WitnessBody> {
    let n = req.samples.unwrap_or(256);
    let g = s.inner.lock().await;
    let pipeline = Pipeline::new(g.scene.clone(), g.config, g.seed);
    let (_, witness) = pipeline.run_with_witness(n);
    Json(WitnessBody {
        witness_hex: Proof::hex(&witness),
        samples: n,
        seed_hex: format!("0x{:016X}", g.seed),
    })
}

async fn witness_verify(
    State(_s): State<AppState>,
    Json(req): Json<VerifyReq>,
) -> Result<Json<VerifyBody>, AppError> {
    // ADR-092 §6.3 — verify always runs the *canonical* reference scene
    // (Proof::generate) so it matches Proof::EXPECTED_WITNESS_HEX. The
    // user's working scene/config/seed don't enter this check.
    let _samples = req.samples.unwrap_or(Proof::N_SAMPLES);
    let actual = Proof::generate().map_err(|e| AppError::Internal(e.to_string()))?;
    let actual_hex = Proof::hex(&actual);
    let expected_hex = req.expected_hex.trim().to_lowercase();
    let ok = actual_hex == expected_hex;
    Ok(Json(VerifyBody {
        ok,
        actual_hex,
        expected_hex,
    }))
}

async fn witness_reference() -> Result<Json<WitnessBody>, AppError> {
    let actual = Proof::generate().map_err(|e| AppError::Internal(e.to_string()))?;
    Ok(Json(WitnessBody {
        witness_hex: Proof::hex(&actual),
        samples: Proof::N_SAMPLES,
        seed_hex: format!("0x{:016X}", Proof::SEED),
    }))
}

async fn export_proof(State(_s): State<AppState>) -> Result<Json<ProofBundle>, AppError> {
    let actual = Proof::generate().map_err(|e| AppError::Internal(e.to_string()))?;
    let actual_hex = Proof::hex(&actual);
    let ok = actual_hex == EXPECTED_WITNESS_HEX;
    Ok(Json(ProofBundle {
        kind: "nvsim-proof-bundle",
        nvsim_version: env!("CARGO_PKG_VERSION"),
        seed_hex: format!("0x{:016X}", Proof::SEED),
        n_samples: Proof::N_SAMPLES,
        witness_hex: actual_hex,
        expected_hex: EXPECTED_WITNESS_HEX,
        ok,
        ts: chrono_like_now(),
    }))
}

fn chrono_like_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}-unix")
}

async fn ws_handler(ws: WebSocketUpgrade, State(s): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, s))
}

async fn handle_ws(mut socket: WebSocket, state: AppState) {
    info!("ws/stream client connected");
    // Build the pipeline on connect — single instance per client; the
    // server doesn't multiplex pipelines because the sim is fast enough
    // to spin one up per client without measurable latency.
    let (scene, config, seed) = {
        let g = state.inner.lock().await;
        (g.scene.clone(), g.config, g.seed)
    };
    let pipeline = Pipeline::new(scene, config, seed);
    let mut tick = tokio::time::interval(std::time::Duration::from_millis(16));
    let batch_size = 32usize;

    loop {
        tokio::select! {
            _ = tick.tick() => {
                let running = { state.inner.lock().await.running };
                if !running { continue; }

                let frames = pipeline.run(batch_size);
                let mut bytes = Vec::with_capacity(frames.len() * nvsim::frame::MAG_FRAME_BYTES);
                for f in &frames { bytes.extend_from_slice(&f.to_bytes()); }
                if socket.send(Message::Binary(bytes)).await.is_err() {
                    warn!("ws/stream client closed");
                    return;
                }

                let mut g = state.inner.lock().await;
                g.frames_emitted = g.frames_emitted.saturating_add(frames.len() as u64);
            }
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => {
                        info!("ws/stream client disconnected");
                        return;
                    }
                    Some(Ok(_)) => { /* ignore inbound messages in V1 */ }
                    Some(Err(e)) => {
                        warn!(?e, "ws/stream socket error");
                        return;
                    }
                }
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum AppError {
    #[error("bad input: {0}")]
    BadInput(String),
    #[error("internal: {0}")]
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (code, msg) = match &self {
            AppError::BadInput(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            AppError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };
        (code, msg).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{Method, Request};
    use clap::CommandFactory;
    use tower::ServiceExt;

    #[test]
    fn allowed_origin_defaults_to_dashboard_allowlist() {
        let args = Args::try_parse_from(["nvsim-server"]).expect("default arguments parse");
        assert_eq!(
            args.allowed_origin,
            "http://localhost:5173,http://127.0.0.1:5173,https://ruvnet.github.io"
        );
    }

    #[test]
    fn token_can_be_supplied_on_the_command_line() {
        let args = Args::try_parse_from(["nvsim-server", "--token", "test-secret"])
            .expect("token argument parses");
        assert_eq!(args.token.as_deref(), Some("test-secret"));
    }

    #[test]
    fn token_arg_hides_env_value_in_help() {
        let command = Args::command();
        let token_arg = command
            .get_arguments()
            .find(|arg| arg.get_id() == "token")
            .expect("token argument exists");
        assert!(token_arg.is_hide_env_values_set());
    }

    #[tokio::test]
    async fn configured_token_rejects_unauthenticated_mutation_with_json_error() {
        let response = build_router(AppState::new(), Some("test-secret".into()))
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/run")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).unwrap(),
            serde_json::json!({"error": "missing or invalid bearer token"})
        );
    }

    #[tokio::test]
    async fn configured_token_allows_authenticated_mutation() {
        let response = build_router(AppState::new(), Some("test-secret".into()))
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/run")
                    .header(header::AUTHORIZATION, "Bearer test-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn configured_token_leaves_read_only_routes_open() {
        let response = build_router(AppState::new(), Some("test-secret".into()))
            .oneshot(
                Request::builder()
                    .uri("/api/scene")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn tokenless_dev_mode_leaves_mutating_routes_open() {
        let response = build_router(AppState::new(), None)
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/run")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn cors_list_allows_only_listed_origins() {
        let app = build_router(AppState::new(), None).layer(build_cors_layer(
            "https://dashboard.example, invalid origin",
        ));
        let preflight = |origin: &'static str| {
            Request::builder()
                .method(Method::OPTIONS)
                .uri("/api/run")
                .header(header::ORIGIN, origin)
                .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
                .body(Body::empty())
                .unwrap()
        };

        let allowed = app.clone().oneshot(preflight("https://dashboard.example"));
        let denied = app.oneshot(preflight("https://other.example"));
        let (allowed, denied) = tokio::join!(allowed, denied);

        assert_eq!(
            allowed.unwrap().headers().get(header::ACCESS_CONTROL_ALLOW_ORIGIN),
            Some(&HeaderValue::from_static("https://dashboard.example"))
        );
        assert!(denied
            .unwrap()
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .is_none());
    }

    #[tokio::test]
    async fn cors_wildcard_remains_an_explicit_opt_in() {
        let response = build_router(AppState::new(), None)
            .layer(build_cors_layer(" * "))
            .oneshot(
                Request::builder()
                    .method(Method::OPTIONS)
                    .uri("/api/run")
                    .header(header::ORIGIN, "https://any.example")
                    .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            response.headers().get(header::ACCESS_CONTROL_ALLOW_ORIGIN),
            Some(&HeaderValue::from_static("*"))
        );
    }

    #[tokio::test]
    async fn cors_explicitly_allows_bearer_request_headers() {
        let response = build_router(AppState::new(), None)
            .layer(build_cors_layer("https://dashboard.example"))
            .oneshot(
                Request::builder()
                    .method(Method::OPTIONS)
                    .uri("/api/run")
                    .header(header::ORIGIN, "https://dashboard.example")
                    .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
                    .header(
                        header::ACCESS_CONTROL_REQUEST_HEADERS,
                        "authorization,content-type",
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            response.headers().get(header::ACCESS_CONTROL_ALLOW_HEADERS),
            Some(&HeaderValue::from_static("authorization,content-type"))
        );
    }
}
