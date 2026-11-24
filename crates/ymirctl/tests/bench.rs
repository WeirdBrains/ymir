//! `ymirctl bench` against a mock bridge: no model, no real network.

use axum::Router;
use axum::extract::Json;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use serde_json::{Value, json};
use ymir_core::client::BridgeClient;
use ymirctl::bench::{BenchArgs, BridgeBackend, Suite, Via, run};

fn answer_for(prompt: &str) -> &'static str {
    if prompt.contains("17 * 23") {
        "396"
    } else if prompt.contains("capital of Australia") {
        "Canberra"
    } else if prompt.contains("JSON") {
        "{\"name\":\"Ada\",\"age\":36,\"tags\":[\"x\",\"y\"]}"
    } else if prompt.contains("fib(n)") {
        "```python\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```"
    } else {
        "I am a mock answer with a few words in it."
    }
}

async fn chat(Json(body): Json<Value>) -> impl IntoResponse {
    let prompt = body["messages"][0]["content"].as_str().unwrap_or_default().to_string();
    let text = answer_for(&prompt);
    let first = json!({"choices":[{"delta":{"content":text},"finish_reason":null,"index":0}]});
    let last = json!({"choices":[{"delta":{},"finish_reason":"stop","index":0}],
        "usage":{"prompt_tokens": 50, "completion_tokens": 12, "prompt_tokens_details": {"cached_tokens": 0}},
        "weirdnet":{"served_by":"real-host-1","standin":null,"cache_hit":false}});
    let sse = format!("data: {first}\n\ndata: {last}\n\ndata: [DONE]\n\n");
    ([("content-type", "text/event-stream")], sse)
}

async fn models() -> Json<Value> {
    Json(json!({"data":[
        {"id":"chatty","weirdnet":{"purpose":"chat","state":"warm","replicas":1}},
        {"id":"cold","weirdnet":{"purpose":"chat","state":"loadable","replicas":0}},
        {"id":"vec","weirdnet":{"purpose":"embedding","state":"warm","replicas":1}},
    ]}))
}

async fn embeddings() -> Json<Value> {
    Json(json!({"data":[{"embedding":[0.1,0.2,0.3,0.4]}]}))
}

#[tokio::test]
async fn benchmarks_each_model_and_hides_node_names() {
    let app = Router::new().route("/v1/models", get(models)).route("/v1/chat/completions", post(chat)).route("/v1/embeddings", post(embeddings));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let dir = tempfile::tempdir().unwrap();
    let args = BenchArgs {
        via: Via::Bridge,
        suite: Suite::Standard,
        repeats: 2,
        timeout: 20,
        busy_retries: 0,
        busy_wait: 0,
        out_dir: dir.path().to_path_buf(),
        ..Default::default()
    };
    let backend = BridgeBackend(BridgeClient::new(port, "0123456789abcdef0123"));
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let (json_path, md_path, csv_path) = run(&backend, &args, &mut out, &mut err).await.unwrap();

    let report: Value = serde_json::from_str(&std::fs::read_to_string(json_path).unwrap()).unwrap();
    let md = std::fs::read_to_string(md_path).unwrap();
    let csv = std::fs::read_to_string(csv_path).unwrap();
    assert!(
        !md.contains("real-host-1") && !report.to_string().contains("real-host-1") && !csv.contains("real-host-1"),
        "node names are anonymised by default"
    );
    assert!(md.contains("node-a"));
    let rows = report["rows"].as_array().unwrap();
    let chatty = rows.iter().find(|m| m["model"] == "chatty").unwrap();
    assert_eq!(chatty["samples"], 6);
    assert!(chatty["decode_tps"]["median"].as_f64().unwrap() > 0.0);
    let q = &chatty["quality"];
    // Two samples of each question; the code one is scored only where python3 and the sandbox are.
    assert_eq!(q["ran"], if ymirctl::bench::python_available() { 8 } else { 6 });
    let passed = q["passed"].as_u64().unwrap();
    // arithmetic_1, json, fib (if python3 exists) and the capital are answered right; the rest are not.
    assert!((3..=4).contains(&passed), "passed {passed}");
    let cold = rows.iter().find(|m| m["model"] == "cold").unwrap();
    assert!(cold["skipped"].as_str().unwrap().contains("not loaded"));
    let vec = rows.iter().find(|m| m["model"] == "vec").unwrap();
    assert_eq!(vec["embedding"]["dims"], 4);

    // Over the bridge the pool picks the node, so asking for one is refused.
    let mut pinned = args.clone();
    pinned.node = vec!["m1".into()];
    assert!(run(&backend, &pinned, &mut Vec::new(), &mut Vec::new()).await.is_err());
}
