//! `ymirctl mcp`: this node as a Model Context Protocol server, so tools such as Cursor, Claude
//! Desktop and VS Code can see the pool and use its models.
//!
//! - Transports: stdio (the default; the client starts `ymirctl mcp`), or streamable HTTP on a
//!   loopback address with `--http`, where clients send the bridge token as a bearer token.
//! - Read tools: status, model nodes, linked devices, the pool's models, chat (text, images and
//!   audio, through the local bridge or straight over Weirdnet), completions, embeddings, service
//!   status, Hugging Face model search and the model files on this node.
//! - Mutating tools (load, unload, move, pull) are marked "MUTATES THE POOL" in their description, carry
//!   `readOnlyHint: false` and do nothing without `confirm: true`. Deleting model files
//!   (`ymir_remove_model`) is not offered unless the server starts with `--allow-remove`.
//!   `--read-only` hides every mutating tool.
//!
//! Everything goes through the same control socket, bridge and `models` code `ymirctl` uses;
//! nothing here reimplements the pool. The protocol layer (JSON-RPC, stdio, HTTP guard, resources)
//! is `weirdnet-mcp` 0.6.
//!
//! Resources (read-only): `ymir://status`, `ymir://nodes`, `ymir://models` and `ymir://models/local`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use base64::Engine as _;
use serde_json::{Value, json};
use std::net::SocketAddr;
use weirdnet_mcp::http::{Auth, Guard};
use weirdnet_mcp::{
    CallToolResult, Param, ParamType as T, Resource, ResourceContents, ResourceError, Server, ServerInfo, ToolError, ToolSpec, Tools,
};
use ymir_core::Paths;
use ymir_core::control::{self, ControlRequest, ModelState, StatusReport};

use crate::models::{DirArgs, ListArgs, ModelsCmd, NodeArgs, PullArgs, RemoveArgs, SearchArgs};
use crate::{ServiceArgs, ServiceCmd};

/// Where the streamable HTTP endpoint lives.
pub const HTTP_PATH: &str = "/mcp";

/// The default `--http` address.
pub const DEFAULT_HTTP: &str = "127.0.0.1:8485";

/// An image attached to a chat message may be this large (as in Bragi).
pub const MAX_IMAGE_BYTES: u64 = 5 << 20;
/// An audio file attached to a chat message may be this large (as in Bragi).
pub const MAX_AUDIO_BYTES: u64 = 10 << 20;

#[derive(clap::Args, Debug, Clone, Default)]
pub struct McpArgs {
    /// Serve streamable HTTP at http://ADDR/mcp instead of stdio. ADDR must be a loopback address
    /// (default 127.0.0.1:8485). Clients send the bridge token (`ymirctl bridge token`) as a bearer
    /// token.
    #[arg(long, value_name = "ADDR", num_args = 0..=1, default_missing_value = DEFAULT_HTTP)]
    pub http: Option<String>,
    /// Offer no tool that changes anything: no load, unload, pull, or `ymir_speak` (it writes
    /// audio files).
    #[arg(long)]
    pub read_only: bool,
    /// Also offer `ymir_remove_model`, which deletes downloaded model files (it still needs `confirm`).
    #[arg(long)]
    pub allow_remove: bool,
    /// A directory `ymir_transcribe` and `ymir_speak` may read files from and `ymir_speak` may
    /// write audio to (repeatable). Without any, anywhere this user can; with some, only under
    /// them.
    #[arg(long = "root", value_name = "DIR")]
    pub roots: Vec<PathBuf>,
}

/// Runs `ymirctl mcp`.
pub async fn run(explicit_home: Option<PathBuf>, paths: Paths, args: McpArgs) -> Result<()> {
    if args.read_only && args.allow_remove {
        bail!("--read-only and --allow-remove contradict each other");
    }
    let tools = YmirTools::new(paths.clone(), explicit_home, Policy { read_only: args.read_only, allow_remove: args.allow_remove })
        .with_roots(&args.roots)?;
    let server = Arc::new(server(tools));
    let mode = if args.read_only {
        ", read-only"
    } else if args.allow_remove {
        ", model removal on"
    } else {
        ""
    };
    match args.http {
        None => {
            eprintln!("ymirctl {}: MCP server on stdio{mode}", env!("CARGO_PKG_VERSION"));
            serve_stdio(server, tokio::io::stdin(), tokio::io::stdout()).await
        }
        Some(addr) => {
            let addr = loopback_addr(&addr)?;
            let token = ymir_core::token::load(&paths)?.context("no bridge token yet; run `ymirctl setup` or start ymird once")?;
            let listener = tokio::net::TcpListener::bind(addr).await.with_context(|| format!("binding {addr}"))?;
            eprintln!(
                "ymirctl {}: MCP server on http://{}{}{mode} (bearer token: `ymirctl bridge token`)",
                env!("CARGO_PKG_VERSION"),
                listener.local_addr()?,
                HTTP_PATH
            );
            serve_http(listener, server, token, async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await
        }
    }
}

/// The protocol server for `tools`.
pub fn server(tools: YmirTools) -> Server<YmirTools> {
    let info = ServerInfo {
        name: "ymir".into(),
        title: "Ymir".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        instructions: "Ymir is a pool of local models. Start with ymir_status or ymir_models, then use ymir_chat, ymir_complete or \
            ymir_embed. Tools marked MUTATES THE POOL change the pool and do nothing unless confirm=true: only pass it after the user \
            has agreed to that exact change."
            .into(),
    };
    Server::new(info, tools)
}

/// Serves newline-delimited JSON-RPC on `input` and `output` until `input` ends and every request
/// has been answered. Requests run concurrently, so a slow chat does not block a ping.
pub async fn serve_stdio<R, W>(server: Arc<Server<YmirTools>>, input: R, output: W) -> Result<()>
where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    weirdnet_mcp::stdio::serve(server, input, output).await.context("serving the MCP client")
}

/// Parses `--http ADDR` and refuses anything but a loopback address.
pub fn loopback_addr(s: &str) -> Result<SocketAddr> {
    let s = s.trim();
    let addr: SocketAddr = match s.strip_prefix("localhost:") {
        Some(port) => SocketAddr::from(([127, 0, 0, 1], port.parse().with_context(|| format!("bad port in {s}"))?)),
        None => s.parse().with_context(|| format!("`{s}` is not an address like 127.0.0.1:8485"))?,
    };
    if !addr.ip().is_loopback() {
        bail!("the MCP server only listens on loopback addresses (127.0.0.1 or ::1), not {}", addr.ip());
    }
    Ok(addr)
}

/// Serves stateless streamable HTTP at `/mcp` on `listener`, which must be bound to a loopback
/// address, until `shutdown`. `weirdnet-mcp`'s guard checks `Host`, `Origin` and the bearer token;
/// the loopback bind check is ours (the guard does not look at the bind address).
pub async fn serve_http(
    listener: tokio::net::TcpListener,
    server: Arc<Server<YmirTools>>,
    token: String,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<()> {
    if !listener.local_addr()?.ip().is_loopback() {
        bail!("refusing to serve MCP on {}: loopback only", listener.local_addr()?);
    }
    let app = weirdnet_mcp::http::router(HTTP_PATH, server, Guard::loopback(Auth::token(token)));
    axum::serve(listener, app).with_graceful_shutdown(shutdown).await?;
    Ok(())
}

/// What the server offers and allows.
#[derive(Clone, Copy, Debug, Default)]
pub struct Policy {
    pub read_only: bool,
    pub allow_remove: bool,
}

const fn p_model(desc: &'static str) -> Param {
    Param::required("model", T::String, desc)
}

const CONFIRM: Param =
    Param::optional("confirm", T::Boolean, "Must be true or nothing happens. Pass it only after the user agreed to this exact change.");

const fn spec(
    name: &'static str,
    description: &'static str,
    params: &'static [Param],
    read_only: bool,
    destructive: bool,
    open_world: bool,
) -> ToolSpec {
    ToolSpec { name, description, params, auth: false, read_only, destructive, open_world }
}

const NO_PARAMS: &[Param] = &[];

/// An integer of at least 1.
const POSITIVE_INT: T = T::Json(r#"{"type":"integer","minimum":1}"#);

const MESSAGE: &[Param] = &[Param::required("role", T::String, ""), Param::required("content", T::Any, "")];

const SAMPLING: [Param; 2] = [
    Param::optional("temperature", T::Number, "Sampling temperature."),
    Param::optional("max_tokens", POSITIVE_INT, "Longest answer, in tokens (at least 1)."),
];

const CHAT_PARAMS: &[Param] = &[
    Param::optional("prompt", T::String, "The user message. Use this or `messages`."),
    Param::optional("system", T::String, "A system message to put first."),
    Param::optional("model", T::String, "Model id (see ymir_models). Default: the pool's main model."),
    SAMPLING[0],
    SAMPLING[1],
    Param::optional(
        "messages",
        T::Array(&T::Object(MESSAGE)),
        "OpenAI-style messages ({role, content}); the last one is the user's turn. content is a string, or a list of parts: {type:text,text}, {type:image_url,image_url:{url:<data: URL>}}, {type:input_audio,input_audio:{data:<base64>,format:wav|mp3}}. Use this or `prompt`.",
    ),
    Param::optional(
        "images",
        T::StringArray,
        "With `prompt`: image files on this machine (PNG, JPEG, GIF or WebP, up to 5 MB each) or data: URLs to send with it. Needs a model that takes images.",
    ),
    Param::optional(
        "audio",
        T::StringArray,
        "With `prompt`: audio files on this machine (WAV or MP3, up to 10 MB each) to send with it. Needs a model that takes audio.",
    ),
    Param::optional("wait", T::Boolean, "Wait for the model to load instead of taking a stand-in's answer (bridge route)."),
    Param::optional(
        "route",
        T::Enum(&["bridge", "weirdnet"]),
        "bridge (default): through this node's loopback OpenAI bridge. weirdnet: straight to the pool's inference nodes over Weirdnet as a synced conversation (text only; needs a linked device identity).",
    ),
    Param::optional("conversation", T::String, "weirdnet route: continue this conversation (send only the new messages). Default: a new one."),
];

const COMPLETE_PARAMS: &[Param] =
    &[Param::required("prompt", T::String, "Text to continue."), Param::optional("model", T::String, "Model id."), SAMPLING[0], SAMPLING[1]];

const EMBED_PARAMS: &[Param] = &[
    Param::required("input", T::AnyOf(&[T::String, T::StringArray]), "A string or a list of strings to embed."),
    Param::optional("model", T::String, "An embedding model. Default: the pool's embedding model."),
];

const SERVICE_PARAMS: &[Param] = &[Param::optional("system", T::Boolean, "The machine's service instead of this user's.")];

const SEARCH_PARAMS: &[Param] = &[
    Param::required("query", T::String, "Words to search the Hugging Face Hub for, such as `qwen3 8b`."),
    Param::optional("format", T::Enum(&["gguf", "mlx", "all"]), "gguf, mlx or all (default: GGUF, plus MLX on Apple Silicon)."),
    Param::optional("limit", T::Integer, "Repos to show per format (default 5)."),
    Param::optional(
        "context",
        T::Integer,
        "Context length to size the memory estimate for (default 32768, the length a quant is picked at; ymird then runs the model with as much as fits).",
    ),
];

const LOAD_PARAMS: &[Param] = &[
    p_model("The model id (see ymir_models)."),
    Param::optional("node", T::String, "The model node to load it on (see ymir_nodes). Default: the pool picks."),
    CONFIRM,
];

const UNLOAD_PARAMS: &[Param] = &[p_model("The model id."), CONFIRM];

const MOVE_PARAMS: &[Param] = &[
    p_model("The model id (see ymir_models)."),
    Param::required("to", T::String, "The model node to move it to (see ymir_nodes); it needs its own copy of the model."),
    Param::optional("from", T::String, "The node to move it from. Default: where it is loaded."),
    Param::optional("grace", T::Integer, "Seconds to let the old node finish what it has before it unloads (default 60, at most 600)."),
    CONFIRM,
];

const PULL_PARAMS: &[Param] = &[
    Param::required("repo", T::String, "The Hugging Face repo, `owner/name` (see ymir_models_search)."),
    Param::optional("quant", T::String, "The quant to take, such as Q4_K_M (default: the best that fits this node)."),
    Param::optional("file", T::String, "One exact file in the repo."),
    Param::optional(
        "context",
        T::Integer,
        "A fixed context for the entry (default: none, so ymird runs the model at its full trained length, cut only where the KV cache would not fit).",
    ),
    Param::optional("force", T::Boolean, "Download the smallest quant even though none fits this node."),
    CONFIRM,
];

const TRANSCRIBE_PARAMS: &[Param] = &[
    Param::optional(
        "path",
        T::String,
        "An audio file (wav, mp3, flac, ogg, m4a) or a video file (mp4, mov, mkv, webm; its first audio stream) on this machine, under the allowed roots, up to 512 MB. Use this or `audio`. URLs are never fetched.",
    ),
    Param::optional("audio", T::String, "The audio itself, base64 (wav, mp3, flac or ogg), up to 31 MB. Use this or `path`."),
    Param::optional("model", T::String, "A speech-to-text model (catalog purpose transcribe). Default: the pool's pick."),
    Param::optional("language", T::String, "The spoken language (ISO 639-1, such as en). Default: the model detects it."),
    Param::optional("prompt", T::String, "Names and terms to guide the transcript."),
    Param::optional(
        "format",
        T::Enum(&["text", "json", "srt", "vtt"]),
        "text (default), json (with timed segments), srt or vtt (subtitles in `subtitles`).",
    ),
    Param::optional(
        "chunk",
        T::Enum(&["off", "auto"]),
        "off (default): one request. auto: cut long audio at silences to the transcriber's reported audio_max_seconds (refused when it reports none).",
    ),
];

const SPEAK_TURN: &[Param] =
    &[Param::required("speaker", T::Integer, "The speaker number, from 0."), Param::required("text", T::String, "What this speaker says.")];

const SPEAK_PARAMS: &[Param] = &[
    Param::optional("text", T::String, "The text to speak (paragraphs separated by blank lines). Use this or `script`."),
    Param::optional("script", T::Array(&T::Object(SPEAK_TURN)), "A multi-speaker script, turn by turn. Needs a model that speaks scripts."),
    Param::optional("voice", T::String, "A preset voice of the model (ymir_models lists them). Default: the model's own."),
    Param::optional("voices", T::Any, "For a script: preset voices per speaker, as {\"0\": \"vivian\", \"1\": \"ryan\"}."),
    Param::optional("design", T::String, "Design a voice from this description instead, when the model designs voices."),
    Param::optional(
        "clone_path",
        T::String,
        "Clone the voice in this reference clip (wav, mp3, flac, ogg; at most 30 MB) on this machine, under the allowed roots. URLs are never fetched. Needs consent=true.",
    ),
    Param::optional("clone_audio", T::String, "The reference clip itself, base64, instead of clone_path. Needs consent=true."),
    Param::optional("clone_text", T::String, "What the reference clip says (some models need it)."),
    Param::optional("consent", T::Boolean, "For a cloned voice: true only when the user confirmed the voice is theirs or they may use it."),
    Param::optional("instructions", T::String, "Style or emotion, when the model takes instructions."),
    Param::optional("language", T::String, "The language (ISO 639-1, such as en)."),
    Param::optional("speed", T::Number, "Speaking speed, 0.25 to 4 (1: the model's pace), when the model takes one."),
    Param::optional("seed", T::Integer, "A seed, for a repeatable take."),
    Param::optional("model", T::String, "A speaking model (catalog purpose speak). Default: the pool's pick."),
    Param::optional("format", T::Enum(&["wav", "mp3", "opus", "flac", "pcm16"]), "The audio format (default wav)."),
    Param::optional(
        "chunk",
        T::Enum(&["auto", "off"]),
        "auto (default): split long text to the model's max_input_chars and stitch. off: one request.",
    ),
    Param::optional(
        "out",
        T::String,
        "Write the audio to this file (under the allowed roots; its folder must exist; needs confirm=true). Without it the audio comes back as base64.",
    ),
    CONFIRM,
];

const REMOVE_PARAMS: &[Param] = &[Param::required("id", T::String, "A model id or file name from ymir_models_local."), CONFIRM];

const READ_TOOLS: &[ToolSpec] = &[
    spec(
        "ymir_status",
        "This node: name, version, roles, bridge port, main model, the models it knows with their state, the model nodes, alerts and its Weirdnet identity.",
        NO_PARAMS,
        true,
        false,
        false,
    ),
    spec("ymir_nodes", "The model nodes the pool routes to: serving or draining, models ready, requests in flight.", NO_PARAMS, true, false, false),
    spec(
        "ymir_devices",
        "The devices linked to this Weirdnet identity, which are connected now, and the addresses this node listens on.",
        NO_PARAMS,
        true,
        false,
        false,
    ),
    spec(
        "ymir_models",
        "The pool's models: purpose (chat, embedding), state (warm, busy, loading, loadable), how many replicas serve it, where it came from (hub, upload, train) and its license when known, what it takes (images, audio) and where this node sees it placed.",
        NO_PARAMS,
        true,
        false,
        false,
    ),
    spec(
        "ymir_chat",
        "Ask a pool model and get its answer, with optional images and audio. Goes through the local bridge by default, or straight over Weirdnet with route=weirdnet. Uses the pool's compute; the weirdnet route stores the conversation.",
        CHAT_PARAMS,
        false,
        false,
        false,
    ),
    spec(
        "ymir_complete",
        "Plain text completion: continue a prompt with a pool model (through the local bridge).",
        COMPLETE_PARAMS,
        true,
        false,
        false,
    ),
    spec(
        "ymir_embed",
        "Embedding vectors for one or more strings from the pool's embedding model (through the local bridge).",
        EMBED_PARAMS,
        true,
        false,
        false,
    ),
    spec(
        "ymir_service_status",
        "Whether ymird is installed as a service here, whether it is running, and whether it answers.",
        SERVICE_PARAMS,
        true,
        false,
        false,
    ),
    spec(
        "ymir_models_search",
        "Search the Hugging Face Hub for GGUF and MLX models, with a verdict on whether each fits this node. Reaches the internet; downloads nothing.",
        SEARCH_PARAMS,
        true,
        false,
        true,
    ),
    spec("ymir_models_local", "The model files downloaded on this node and whether node.toml uses each.", NO_PARAMS, true, false, false),
    spec(
        "ymir_transcribe",
        "Speech to text: the transcript of an audio or video file (or base64 audio) from the pool's speech-to-text model, as text, JSON with timed segments, SRT or VTT. Reads the file here; writes nothing.",
        TRANSCRIBE_PARAMS,
        true,
        false,
        false,
    ),
];

const MUTATING_TOOLS: &[ToolSpec] = &[
    spec(
        "ymir_load_model",
        "MUTATES THE POOL (needs confirm=true): load a model on a model node. Also clears a crash loop. Uses node memory.",
        LOAD_PARAMS,
        false,
        false,
        false,
    ),
    spec(
        "ymir_unload_model",
        "MUTATES THE POOL (needs confirm=true): unload a model everywhere it is loaded. Requests for it wait for the next load.",
        UNLOAD_PARAMS,
        false,
        true,
        false,
    ),
    spec(
        "ymir_move_model",
        "MUTATES THE POOL (needs confirm=true): move a loaded model to another model node. It loads there, new requests go there, and the old node unloads it once what it has finishes (at most `grace` seconds). Uses the new node's memory.",
        MOVE_PARAMS,
        false,
        false,
        false,
    ),
    spec(
        "ymir_pull_model",
        "MUTATES THIS NODE (needs confirm=true): download a model from Hugging Face into this node's models folder (many GB). Without confirm it only shows what would be downloaded. node.toml is never edited; the answer includes the entry to add.",
        PULL_PARAMS,
        false,
        false,
        true,
    ),
    spec(
        "ymir_speak",
        "WRITES A FILE (needs confirm=true with `out`): text to speech from the pool's speaking model, with a preset, designed or cloned voice, or a multi-speaker script. Long text is split and stitched. Returns the file's path (or base64 audio without `out`), its length and the usage.",
        SPEAK_PARAMS,
        false,
        false,
        false,
    ),
];

const REMOVE_TOOL: ToolSpec = spec(
    "ymir_remove_model",
    "DELETES FILES (needs confirm=true): delete a downloaded model's files from this node. Without confirm it only shows what would be deleted. Never edits node.toml.",
    REMOVE_PARAMS,
    false,
    true,
    false,
);

/// The names of the tools that change anything.
pub fn mutating_tools() -> Vec<&'static str> {
    MUTATING_TOOLS.iter().chain([&REMOVE_TOOL]).map(|t| t.name).collect()
}

struct Ctx {
    paths: Paths,
    explicit_home: Option<PathBuf>,
    /// Where `ymir_transcribe` and `ymir_speak` may read and `ymir_speak` may write (canonical);
    /// empty: anywhere.
    roots: Vec<PathBuf>,
}

/// File types `ymir_transcribe` reads, checked before anything is opened.
const TRANSCRIBE_EXTS: &[&str] = &["wav", "mp3", "flac", "ogg", "oga", "opus", "m4a", "mp4", "mov", "mkv", "webm"];
/// What `ymir_speak` reads as a reference clip.
const CLIP_EXTS: &[&str] = &["wav", "mp3", "flac", "ogg", "oga", "opus"];
/// What `ymir_speak` writes, by format.
fn out_exts(format: crate::speak::OutFormat) -> &'static [&'static str] {
    use crate::speak::OutFormat::*;
    match format {
        Wav => &["wav"],
        Mp3 => &["mp3"],
        Opus => &["opus", "ogg", "oga"],
        Flac => &["flac"],
        Pcm16 => &["pcm", "raw"],
    }
}

/// Read-only resources: URI, name, description and the read tool that produces each.
const RESOURCES: &[(&str, &str, &str, &str)] = &[
    (
        "ymir://status",
        "Pool status",
        "This node: roles, main model, the models it knows with their state, model nodes, alerts and identity.",
        "ymir_status",
    ),
    ("ymir://nodes", "Model nodes", "The model nodes the pool routes to: serving or draining, models ready, requests in flight.", "ymir_nodes"),
    ("ymir://models", "Pool models", "The pool's models: purpose, state, replicas, what each takes and where it is placed.", "ymir_models"),
    ("ymir://models/local", "Local model files", "The model files downloaded on this node and whether node.toml uses each.", "ymir_models_local"),
];

/// The tool table for one node.
pub struct YmirTools {
    ctx: Arc<Ctx>,
    specs: Vec<ToolSpec>,
}

fn str_arg<'a>(a: &'a Value, k: &str) -> Option<&'a str> {
    a[k].as_str().filter(|s| !s.is_empty())
}

fn req_str<'a>(a: &'a Value, k: &str) -> Result<&'a str> {
    str_arg(a, k).with_context(|| format!("`{k}` is required"))
}

fn state_name(s: &ModelState) -> String {
    match s {
        ModelState::Failed(why) => format!("failed: {why}"),
        other => format!("{other:?}").to_lowercase(),
    }
}

/// Copies `temperature` and `max_tokens` into an OpenAI body.
fn sampling(args: &Value, body: &mut Value) {
    for k in ["temperature", "max_tokens"] {
        if args[k].is_number() {
            body[k] = args[k].clone();
        }
    }
}

/// Refuses a mutating call that did not say `confirm: true`.
fn confirmed(args: &Value, what: &str) -> Result<()> {
    if args["confirm"].as_bool() == Some(true) {
        return Ok(());
    }
    bail!("not done: {what} changes the pool. Ask the user, then call again with confirm=true")
}

/// Text a `models` subcommand printed, without progress redraws (`\r`) and blank edges.
fn tidy(out: Vec<u8>) -> String {
    let s = String::from_utf8_lossy(&out);
    let lines: Vec<&str> = s.lines().map(|l| l.rsplit('\r').next().unwrap_or(l)).collect();
    // A long download prints a progress line often; keep the head and the last lines.
    if lines.len() > 80 {
        let mut kept: Vec<&str> = lines[..20].to_vec();
        kept.push("[...]");
        kept.extend(&lines[lines.len() - 50..]);
        return kept.join("\n");
    }
    lines.join("\n")
}

/// The content part for an image: a `data:` URL as given, or a file read and encoded.
fn image_part(src: &str) -> Result<Value> {
    if let Some(rest) = src.strip_prefix("data:") {
        if !rest.starts_with("image/") || !rest.contains(";base64,") {
            bail!("an image data URL looks like data:image/png;base64,...");
        }
        if rest.len() as u64 > MAX_IMAGE_BYTES * 4 / 3 + 64 {
            bail!("the image is over 5 MB");
        }
        return Ok(json!({"type": "image_url", "image_url": {"url": src}}));
    }
    let (bytes, ext) = read_media(src, MAX_IMAGE_BYTES, "image")?;
    let mime = match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        other => bail!("{src}: .{other} is not an image type the pool takes (PNG, JPEG, GIF, WebP)"),
    };
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    Ok(json!({"type": "image_url", "image_url": {"url": format!("data:{mime};base64,{b64}")}}))
}

/// The content part for an audio file.
fn audio_part(src: &str) -> Result<Value> {
    let (bytes, ext) = read_media(src, MAX_AUDIO_BYTES, "audio file")?;
    let format = match ext.as_str() {
        "wav" => "wav",
        "mp3" => "mp3",
        other => bail!("{src}: .{other} is not an audio type the pool takes (WAV, MP3)"),
    };
    Ok(json!({"type": "input_audio", "input_audio": {"data": base64::engine::general_purpose::STANDARD.encode(bytes), "format": format}}))
}

fn read_media(src: &str, max: u64, what: &str) -> Result<(Vec<u8>, String)> {
    let path = Path::new(src);
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
    // Check the type before reading anything, so the tool cannot be pointed at arbitrary files.
    if !matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "gif" | "webp" | "wav" | "mp3") {
        bail!("{src}: not a {what} type the pool takes");
    }
    let meta = std::fs::metadata(path).with_context(|| format!("reading {src}"))?;
    if !meta.is_file() {
        bail!("{src} is not a file");
    }
    if meta.len() > max {
        bail!("{src} is {} MB; the limit for an {what} is {} MB", meta.len() >> 20, max >> 20);
    }
    Ok((std::fs::read(path).with_context(|| format!("reading {src}"))?, ext))
}

fn new_conversation_id() -> String {
    let mut b = [0u8; 8];
    // A clock-based id is fine if the OS has no randomness to give.
    if getrandom::fill(&mut b).is_err() {
        b = (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or_default() as u64).to_le_bytes();
    }
    format!("mcp-{}", b.iter().map(|x| format!("{x:02x}")).collect::<String>())
}

impl Tools for YmirTools {
    fn specs(&self) -> &[ToolSpec] {
        &self.specs
    }

    async fn call_async(&self, tool: &ToolSpec, args: &Value, _token: Option<&str>) -> Result<CallToolResult, ToolError> {
        self.dispatch(tool.name, args).await.map(|v| CallToolResult::json(&v)).map_err(|e| ToolError(format!("{e:#}")))
    }

    fn has_resources(&self) -> bool {
        true
    }

    async fn list_resources(&self, _token: Option<&str>) -> Result<Vec<Resource>, ResourceError> {
        Ok(RESOURCES
            .iter()
            .map(|(uri, name, description, _)| Resource::new(*uri, *name).with_description(*description).with_mime_type("application/json"))
            .collect())
    }

    async fn read_resource(&self, uri: &str, _token: Option<&str>) -> Result<Vec<ResourceContents>, ResourceError> {
        let (_, _, _, tool) = RESOURCES.iter().find(|r| r.0 == uri).ok_or(ResourceError::NotFound)?;
        let v = self.dispatch(tool, &json!({})).await.map_err(|e| ResourceError::Failed(format!("{e:#}")))?;
        Ok(vec![ResourceContents::json(uri, &v)])
    }
}

impl YmirTools {
    /// The tools `policy` allows, for the node at `paths`.
    pub fn new(paths: Paths, explicit_home: Option<PathBuf>, policy: Policy) -> Self {
        let mut specs = READ_TOOLS.to_vec();
        if !policy.read_only {
            specs.extend_from_slice(MUTATING_TOOLS);
            if policy.allow_remove {
                specs.push(REMOVE_TOOL);
            }
        }
        Self { ctx: Arc::new(Ctx { paths, explicit_home, roots: Vec::new() }), specs }
    }

    /// Restricts the files `ymir_transcribe` reads to these directories.
    pub fn with_roots(mut self, roots: &[PathBuf]) -> Result<Self> {
        let roots = roots.iter().map(|r| r.canonicalize().with_context(|| format!("--root {}", r.display()))).collect::<Result<Vec<_>>>()?;
        if let Some(ctx) = Arc::get_mut(&mut self.ctx) {
            ctx.roots = roots;
        }
        Ok(self)
    }

    /// The bytes of a file `ymir_transcribe` may read: no URL, an audio or video type (checked
    /// before opening), under a root when roots are set (after resolving links), a regular file.
    fn transcribe_file(&self, src: &str) -> Result<Vec<u8>> {
        if src.contains("://") || src.starts_with("data:") {
            bail!("{src}: URLs are never fetched; pass a file on this machine, or the audio as base64 in `audio`");
        }
        let path = Path::new(src);
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
        if !TRANSCRIBE_EXTS.contains(&ext.as_str()) {
            bail!("{src}: not an audio or video type ymir_transcribe reads ({})", TRANSCRIBE_EXTS.join(", "));
        }
        let real = path.canonicalize().with_context(|| format!("reading {src}"))?;
        self.under_roots(&real, src)?;
        crate::transcribe::read_capped(&real)
    }

    fn under_roots(&self, real: &Path, src: &str) -> Result<()> {
        if !self.ctx.roots.is_empty() && !self.ctx.roots.iter().any(|r| real.starts_with(r)) {
            bail!("{src} is outside the allowed roots");
        }
        Ok(())
    }

    /// A reference clip `ymir_speak` may read: no URL, an audio type (checked before opening),
    /// under a root (links resolved), at most 30 MB.
    fn clip_file(&self, src: &str) -> Result<Vec<u8>> {
        if src.contains("://") || src.starts_with("data:") {
            bail!("{src}: URLs are never fetched; pass a file on this machine, or the clip as base64 in `clone_audio`");
        }
        let path = Path::new(src);
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
        if !CLIP_EXTS.contains(&ext.as_str()) {
            bail!("{src}: not an audio type ymir_speak clones from ({})", CLIP_EXTS.join(", "));
        }
        let real = path.canonicalize().with_context(|| format!("reading {src}"))?;
        self.under_roots(&real, src)?;
        let f = std::fs::File::open(&real).with_context(|| format!("reading {src}"))?;
        if !f.metadata()?.is_file() {
            bail!("{src} is not a regular file");
        }
        let mut b = Vec::new();
        std::io::Read::read_to_end(&mut std::io::Read::take(f, weirdnet_agent::remote::infer::v2::MAX_REFERENCE_BYTES as u64 + 1), &mut b)?;
        Ok(b)
    }

    /// Where `ymir_speak` may write: no URL, the format's extension, a folder that exists under a
    /// root (links resolved), and not a link or anything but a regular file if it is there.
    fn out_file(&self, dst: &str, format: crate::speak::OutFormat) -> Result<PathBuf> {
        if dst.contains("://") || dst.starts_with("data:") {
            bail!("{dst}: write to a file on this machine");
        }
        let path = Path::new(dst);
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
        let exts = out_exts(format);
        if !exts.contains(&ext.as_str()) {
            bail!("{dst}: a {} file ends in .{}", format.wire(), exts.join(" or ."));
        }
        let name = path.file_name().with_context(|| format!("{dst} has no file name"))?;
        let parent = match path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        let dir = parent.canonicalize().with_context(|| format!("{}: the folder must exist", parent.display()))?;
        let real = dir.join(name);
        self.under_roots(&real, dst)?;
        if let Ok(m) = std::fs::symlink_metadata(&real)
            && !m.file_type().is_file()
        {
            bail!("{dst} is there and is not a regular file; not writing over it");
        }
        Ok(real)
    }

    async fn speak(&self, args: &Value) -> Result<Value> {
        use crate::speak::{Chunking, Options, OutFormat, Text};
        use weirdnet_agent::remote::infer::v2::Voice;
        let format = match str_arg(args, "format") {
            None => OutFormat::Wav,
            Some(f) => OutFormat::parse(f).with_context(|| format!("unknown format `{f}` (wav, mp3, opus, flac, pcm16)"))?,
        };
        // Checked before anything is spoken.
        let out = match str_arg(args, "out") {
            Some(o) => {
                confirmed(args, &format!("writing {o}"))?;
                Some(self.out_file(o, format)?)
            }
            None => None,
        };
        let text = match (str_arg(args, "text"), args.get("script").filter(|v| !v.is_null())) {
            (Some(t), None) => Text::Plain(t.to_string()),
            (None, Some(s)) => {
                let turns = s.as_array().context("`script` is a list of {speaker, text}")?;
                let mut out = Vec::with_capacity(turns.len());
                for t in turns {
                    let sp = t["speaker"].as_u64().filter(|n| *n <= u64::from(u8::MAX)).context("every turn needs a `speaker` number (0 to 255)")?;
                    let tx = t["text"].as_str().context("every turn needs `text`")?;
                    out.push((sp as u8, tx.to_string()));
                }
                Text::Script(out)
            }
            (Some(_), Some(_)) => bail!("give `text` or `script`, not both"),
            (None, None) => bail!("`text` or `script` is required"),
        };
        let mut voice = str_arg(args, "voice").map(Voice::preset);
        let mut voices = Vec::new();
        if let Some(map) = args.get("voices").and_then(Value::as_object) {
            for (k, v) in map {
                let n: u8 = k.trim_start_matches(['S', 's']).parse().with_context(|| format!("`voices` key `{k}` is not a speaker number"))?;
                voices.push((n, Voice::preset(v.as_str().with_context(|| format!("`voices.{k}` is a preset name"))?)));
            }
        }
        let clip = match (str_arg(args, "clone_path"), str_arg(args, "clone_audio")) {
            (Some(p), None) => Some(self.clip_file(p)?),
            (None, Some(b)) => {
                if b.starts_with("data:") || b.contains("://") {
                    bail!("`clone_audio` is the clip as base64, not a URL");
                }
                Some(base64::engine::general_purpose::STANDARD.decode(b.trim()).context("`clone_audio` is not base64")?)
            }
            (Some(_), Some(_)) => bail!("give `clone_path` or `clone_audio`, not both"),
            (None, None) => None,
        };
        let design = str_arg(args, "design");
        if [voice.is_some(), design.is_some(), clip.is_some()].iter().filter(|b| **b).count() > 1 {
            bail!("give one of `voice`, `design` and a clone");
        }
        if let Some(d) = design {
            voice = Some(Voice::design(d));
        }
        if let Some(c) = clip {
            voice = Some(crate::speak::clone_voice(c, str_arg(args, "clone_text").map(str::to_string), args["consent"].as_bool() == Some(true))?);
        }
        let speed = match args["speed"].as_f64() {
            Some(s) if !(0.25..=4.0).contains(&s) => bail!("`speed` is 0.25 to 4"),
            s => s.map(|s| (s * 1000.0).round() as u32),
        };
        let chunk = match str_arg(args, "chunk") {
            None | Some("auto") => Chunking::Auto,
            Some("off") => Chunking::Off,
            Some(c) => bail!("unknown chunk mode `{c}` (auto, off)"),
        };
        let opts = Options {
            text: Some(text),
            model: str_arg(args, "model").map(str::to_string),
            voice,
            voices,
            instructions: str_arg(args, "instructions").map(str::to_string),
            language: str_arg(args, "language").map(str::to_string),
            speed,
            seed: args["seed"].as_u64(),
            format,
            chunk,
            pause_ms: weirdnet_media::speak::DEFAULT_PAUSE_MS,
            crossfade_ms: weirdnet_media::speak::DEFAULT_CROSSFADE_MS,
        };
        let o = crate::speak::speak(&self.ctx.paths.socket(), &opts, None, &mut |_| {}).await?;
        let mut v = crate::speak::summary(&o);
        match out {
            Some(path) => {
                std::fs::write(&path, &o.audio).with_context(|| format!("writing {}", path.display()))?;
                v["path"] = path.display().to_string().into();
            }
            None => v["audio_base64"] = base64::engine::general_purpose::STANDARD.encode(&o.audio).into(),
        }
        Ok(v)
    }

    async fn transcribe(&self, args: &Value) -> Result<Value> {
        use crate::transcribe::{Chunking, Options, OutFormat};
        let bytes = match (str_arg(args, "path"), str_arg(args, "audio")) {
            (Some(p), None) => self.transcribe_file(p)?,
            (None, Some(b)) => {
                if b.len() as u64 > (control::MAX_TRANSCRIBE_BYTES / 3 + 1) * 4 {
                    bail!("`audio` is over the {} byte limit of one request", control::MAX_TRANSCRIBE_BYTES);
                }
                base64::engine::general_purpose::STANDARD.decode(b.trim()).context("`audio` is not base64")?
            }
            (Some(_), Some(_)) => bail!("give `path` or `audio`, not both"),
            (None, None) => bail!("`path` or `audio` is required"),
        };
        let format = match str_arg(args, "format") {
            None => OutFormat::Text,
            Some(f) => OutFormat::parse(f).with_context(|| format!("unknown format `{f}` (text, json, srt, vtt)"))?,
        };
        let chunk = match str_arg(args, "chunk") {
            None | Some("off") => Chunking::Off,
            Some("auto") => Chunking::Auto,
            Some(c) => bail!("unknown chunk mode `{c}` (off, auto)"),
        };
        let opts = Options {
            model: str_arg(args, "model").map(str::to_string),
            language: str_arg(args, "language").map(str::to_string),
            prompt: str_arg(args, "prompt").map(str::to_string),
            timestamps: format != OutFormat::Text,
            chunk,
            normalize: true,
        };
        let o = crate::transcribe::transcribe(&self.ctx.paths.socket(), &opts, bytes, None, &mut |_| {}).await?;
        Ok(crate::transcribe::to_json(&o, format))
    }

    async fn status_report(&self) -> Result<StatusReport> {
        let r = control::request(&self.ctx.paths.socket(), &ControlRequest::Status).await.context("ymird is not answering; is it running?")?;
        r.status.context(r.message.unwrap_or_else(|| "no status".into()))
    }

    async fn control(&self, req: ControlRequest) -> Result<control::ControlResponse> {
        let r = control::request(&self.ctx.paths.socket(), &req).await.context("ymird is not answering; is it running?")?;
        if !r.ok {
            bail!("{}", r.message.unwrap_or_default());
        }
        Ok(r)
    }

    async fn dispatch(&self, name: &str, args: &Value) -> Result<Value> {
        match name {
            "ymir_status" => Ok(serde_json::to_value(self.status_report().await?)?),
            "ymir_nodes" => Ok(json!({"nodes": self.status_report().await?.nodes})),
            "ymir_devices" => self.devices().await,
            "ymir_models" => self.models().await,
            "ymir_chat" => self.chat(args).await,
            "ymir_complete" => self.complete(args).await,
            "ymir_embed" => self.embed(args).await,
            "ymir_transcribe" => self.transcribe(args).await,
            "ymir_speak" => self.speak(args).await,
            "ymir_service_status" => {
                let a = ServiceArgs { system: args["system"].as_bool() == Some(true), ..Default::default() };
                let lines = crate::service_lines(self.ctx.explicit_home.as_deref(), self.ctx.paths.clone(), ServiceCmd::Status { args: a }).await?;
                Ok(json!({"lines": lines}))
            }
            "ymir_models_search" => self.search(args).await,
            "ymir_models_local" => {
                self.models_cmd(ModelsCmd::List(ListArgs { dir: DirArgs { dir: None }, node: NodeArgs::default() })).await.map(|t| json!({"text": t}))
            }
            "ymir_load_model" => {
                confirmed(args, &format!("loading {} uses node memory and", str_arg(args, "model").unwrap_or("a model")))?;
                let req = ControlRequest::Load { model: req_str(args, "model")?.to_string(), node: str_arg(args, "node").map(str::to_string) };
                Ok(json!({"message": self.control(req).await?.message.unwrap_or_default()}))
            }
            "ymir_unload_model" => {
                confirmed(args, &format!("unloading {}", str_arg(args, "model").unwrap_or("a model")))?;
                let req = ControlRequest::Unload { model: req_str(args, "model")?.to_string() };
                Ok(json!({"message": self.control(req).await?.message.unwrap_or_default()}))
            }
            "ymir_move_model" => {
                confirmed(args, &format!("moving {} loads it on another node and", str_arg(args, "model").unwrap_or("a model")))?;
                let req = ControlRequest::Move {
                    model: req_str(args, "model")?.to_string(),
                    to: req_str(args, "to")?.to_string(),
                    from: str_arg(args, "from").map(str::to_string),
                    grace_secs: args["grace"].as_u64().map_or(60, |g| g.min(600)),
                };
                Ok(json!({"message": self.control(req).await?.message.unwrap_or_default()}))
            }
            "ymir_pull_model" => self.pull(args).await,
            "ymir_remove_model" => self.remove(args).await,
            other => bail!("unknown tool `{other}`"),
        }
    }

    async fn devices(&self) -> Result<Value> {
        let s = self.status_report().await?;
        let Some(n) = s.network else {
            return Ok(json!({"network": null, "message": "Weirdnet is off on this node (link it with `ymirctl pool link`)"}));
        };
        let devices: Vec<Value> = n
            .devices
            .iter()
            .map(|d| json!({"peer": d.peer, "label": d.label, "active": d.active, "this_device": d.me, "connected": n.connected.contains(&d.peer)}))
            .collect();
        Ok(json!({
            "user": n.user,
            "peer": n.peer,
            "linked": n.devices.iter().filter(|d| d.active && !d.me).count(),
            "connected": n.connected.len(),
            "devices": devices,
            "listening": n.listening,
        }))
    }

    /// The pool catalog from the bridge, with this node's view of where each model is placed.
    async fn models(&self) -> Result<Value> {
        let status = self.status_report().await.ok();
        let placements = |id: &str| -> Vec<Value> {
            status
                .iter()
                .flat_map(|s| s.models.iter())
                .filter(|m| m.id == id)
                .map(|m| json!({"node": m.node, "state": state_name(&m.state), "engine": m.engine.name(), "in_flight": m.in_flight, "purpose": m.purpose}))
                .collect()
        };
        // What a model takes, as this node's supervisors report it (null: not known here).
        let takes = |id: &str| -> Value {
            let caps = status.iter().flat_map(|s| s.models.iter()).filter(|m| m.id == id).find_map(|m| m.capabilities);
            json!({"images": caps.map(|c| c.vision), "audio": caps.map(|c| c.audio)})
        };
        let catalog = match crate::bridge(&self.ctx.paths).await {
            Ok(b) => b.models().await,
            Err(e) => Err(e),
        };
        match catalog {
            Ok(v) => {
                let models: Vec<Value> = v["data"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(|m| {
                        let id = m["id"].as_str().unwrap_or_default();
                        let w = &m["weirdnet"];
                        json!({
                            "id": id,
                            "purpose": w["purpose"],
                            "state": w["state"].as_str().unwrap_or("on disk"),
                            "replicas": w["replicas"],
                            "context": w["context"],
                            "origin": w["origin"],
                            "license": w["license"],
                            "takes": takes(id),
                            "placements": placements(id),
                        })
                    })
                    .collect();
                Ok(json!({"source": "bridge", "models": models}))
            }
            // No app role here (or the bridge is down): what the control socket knows.
            Err(bridge_err) => {
                let Some(s) = &status else { return Err(bridge_err) };
                let mut ids: Vec<&str> = s.models.iter().map(|m| m.id.as_str()).collect();
                ids.sort();
                ids.dedup();
                let models: Vec<Value> = ids
                    .into_iter()
                    .map(|id| {
                        let here: Vec<_> = s.models.iter().filter(|m| m.id == id).collect();
                        let replicas = here.iter().filter(|m| m.state == ModelState::Ready).count();
                        json!({
                            "id": id,
                            "purpose": here.first().map(|m| m.purpose.clone()),
                            "state": if replicas > 0 { "warm" } else { "on disk" },
                            "replicas": replicas,
                            "takes": takes(id),
                            "placements": placements(id),
                        })
                    })
                    .collect();
                Ok(json!({"source": "control", "note": format!("the bridge is not available: {bridge_err:#}"), "models": models}))
            }
        }
    }

    fn messages(args: &Value) -> Result<Vec<Value>> {
        let mut messages = Vec::new();
        if let Some(s) = str_arg(args, "system") {
            messages.push(json!({"role": "system", "content": s}));
        }
        let strings = |k: &str| -> Result<Vec<String>> {
            match &args[k] {
                Value::Null => Ok(vec![]),
                Value::Array(a) => a.iter().map(|v| v.as_str().map(str::to_string).with_context(|| format!("`{k}` holds strings"))).collect(),
                _ => bail!("`{k}` is a list of strings"),
            }
        };
        let (images, audio) = (strings("images")?, strings("audio")?);
        match (args["messages"].as_array(), str_arg(args, "prompt")) {
            (Some(m), None) if !m.is_empty() => {
                if !images.is_empty() || !audio.is_empty() {
                    bail!("`images` and `audio` go with `prompt`; with `messages`, put the parts in the message");
                }
                messages.extend(m.iter().cloned());
            }
            (None | Some(_), Some(p)) if args["messages"].as_array().is_none_or(Vec::is_empty) => {
                if images.is_empty() && audio.is_empty() {
                    messages.push(json!({"role": "user", "content": p}));
                } else {
                    let mut parts = vec![json!({"type": "text", "text": p})];
                    for i in &images {
                        parts.push(image_part(i)?);
                    }
                    for a in &audio {
                        parts.push(audio_part(a)?);
                    }
                    messages.push(json!({"role": "user", "content": parts}));
                }
            }
            (Some(_), Some(_)) => bail!("give `prompt` or `messages`, not both"),
            _ => bail!("`prompt` or `messages` is required"),
        }
        Ok(messages)
    }

    async fn chat(&self, args: &Value) -> Result<Value> {
        let messages = Self::messages(args)?;
        match str_arg(args, "route").unwrap_or("bridge") {
            "bridge" => {
                let mut body = json!({"messages": messages, "stream": false});
                if let Some(m) = str_arg(args, "model") {
                    body["model"] = m.into();
                }
                sampling(args, &mut body);
                if args["wait"].as_bool() == Some(true) {
                    body["wait_for_model"] = true.into();
                }
                let v = crate::bridge(&self.ctx.paths).await?.post("chat/completions", &body).await?;
                let choice = &v["choices"][0];
                let mut out = json!({
                    "route": "bridge",
                    "text": choice["message"]["content"].as_str().unwrap_or_default(),
                    "model": v["model"],
                    "finish_reason": choice["finish_reason"],
                    "usage": v["usage"],
                });
                for k in ["cache_hit", "standin"] {
                    if !v["weirdnet"][k].is_null() {
                        out[k] = v["weirdnet"][k].clone();
                    }
                }
                Ok(out)
            }
            "weirdnet" => self.chat_weirdnet(args, messages).await,
            other => bail!("unknown route `{other}` (bridge or weirdnet)"),
        }
    }

    /// Over Weirdnet: earlier messages go into a conversation, then the pool answers its last user
    /// message from that conversation (the answer syncs back like `ymirctl context ask`).
    async fn chat_weirdnet(&self, args: &Value, mut messages: Vec<Value>) -> Result<Value> {
        let text_of = |m: &Value| -> Result<String> {
            m["content"].as_str().map(str::to_string).context("the weirdnet route carries text only: each message's content must be a string")
        };
        let last = messages.pop().context("no message")?;
        if last["role"].as_str() != Some("user") {
            bail!("the last message must be the user's");
        }
        let text = text_of(&last)?;
        let conversation = str_arg(args, "conversation").map_or_else(new_conversation_id, str::to_string);
        for m in &messages {
            let role = m["role"].as_str().unwrap_or("user").to_string();
            self.control(ControlRequest::ContextAdd { conversation: conversation.clone(), role, text: text_of(m)? }).await?;
        }
        let r = self
            .control(ControlRequest::Chat { conversation: conversation.clone(), text, model: str_arg(args, "model").map(str::to_string) })
            .await?;
        let c = r.chat.context("ymird sent no answer")?;
        Ok(json!({
            "route": "weirdnet",
            "conversation": conversation,
            "text": c.text,
            "model": c.model,
            "node": c.node,
            "resumes": c.resumes,
            "path": c.path,
        }))
    }

    async fn complete(&self, args: &Value) -> Result<Value> {
        let mut body = json!({"prompt": req_str(args, "prompt")?, "stream": false});
        if let Some(m) = str_arg(args, "model") {
            body["model"] = m.into();
        }
        sampling(args, &mut body);
        let v = crate::bridge(&self.ctx.paths).await?.post("completions", &body).await?;
        let choice = &v["choices"][0];
        Ok(
            json!({"text": choice["text"].as_str().unwrap_or_default(), "model": v["model"], "finish_reason": choice["finish_reason"], "usage": v["usage"]}),
        )
    }

    async fn embed(&self, args: &Value) -> Result<Value> {
        let input = Some(&args["input"]).filter(|v| v.is_string() || v.is_array()).context("`input` (a string or a list of strings) is required")?;
        let mut body = json!({"input": input});
        if let Some(m) = str_arg(args, "model") {
            body["model"] = m.into();
        }
        let v = crate::bridge(&self.ctx.paths).await?.post("embeddings", &body).await?;
        let embeddings: Vec<Value> = v["data"].as_array().into_iter().flatten().map(|d| d["embedding"].clone()).collect();
        Ok(json!({"model": v["model"], "embeddings": embeddings, "usage": v["usage"]}))
    }

    /// Runs a `models` subcommand on a blocking thread and returns what it printed. Nothing is
    /// read from the terminal: a prompt gets an empty answer, which every command treats as "no".
    async fn models_cmd(&self, cmd: ModelsCmd) -> Result<String> {
        let paths = self.ctx.paths.clone();
        let handle = tokio::runtime::Handle::current();
        tokio::task::spawn_blocking(move || {
            let mut out = Vec::new();
            let r = handle.block_on(crate::models::run(&paths, &mut out, &mut std::io::empty(), cmd));
            let text = tidy(out);
            match r {
                Ok(()) => Ok(text),
                Err(e) if text.is_empty() => Err(e),
                Err(e) => Err(e.context(text)),
            }
        })
        .await?
    }

    async fn search(&self, args: &Value) -> Result<Value> {
        let format = match str_arg(args, "format") {
            None => None,
            Some("gguf") => Some(crate::models::Format::Gguf),
            Some("mlx") => Some(crate::models::Format::Mlx),
            Some("all") => Some(crate::models::Format::All),
            Some(o) => bail!("unknown format `{o}` (gguf, mlx or all)"),
        };
        let a = SearchArgs {
            query: req_str(args, "query")?.split_whitespace().map(str::to_string).collect(),
            format,
            limit: args["limit"].as_u64().map_or(5, |n| n.clamp(1, 25) as usize),
            context: args["context"].as_u64(),
            node: NodeArgs::default(),
        };
        Ok(json!({"text": self.models_cmd(ModelsCmd::Search(a)).await?}))
    }

    fn pull_args(args: &Value, dry_run: bool) -> Result<PullArgs> {
        Ok(PullArgs {
            repo: req_str(args, "repo")?.to_string(),
            // A restricted license is accepted by the operator at the CLI, never by a tool call.
            accept_license: None,
            quant: str_arg(args, "quant").map(str::to_string),
            file: str_arg(args, "file").map(str::to_string),
            context: args["context"].as_u64(),
            id: None,
            role: None,
            // node.toml is never edited from here.
            register: false,
            dry_run,
            force: args["force"].as_bool() == Some(true),
            serve: false,
            load: false,
            adapter_for: None,
            dir: DirArgs { dir: None },
            node: NodeArgs::default(),
        })
    }

    async fn pull(&self, args: &Value) -> Result<Value> {
        if args["confirm"].as_bool() != Some(true) {
            let text = self.models_cmd(ModelsCmd::Pull(Self::pull_args(args, true)?)).await?;
            return Ok(json!({
                "downloaded": false,
                "text": text,
                "message": "Nothing was downloaded. Show this to the user; if they agree, call ymir_pull_model again with confirm=true.",
            }));
        }
        let text = self.models_cmd(ModelsCmd::Pull(Self::pull_args(args, false)?)).await?;
        Ok(json!({"downloaded": true, "text": text}))
    }

    async fn remove(&self, args: &Value) -> Result<Value> {
        let a = RemoveArgs {
            id: req_str(args, "id")?.to_string(),
            yes: args["confirm"].as_bool() == Some(true),
            dir: DirArgs { dir: None },
            node: NodeArgs::default(),
        };
        let deleting = a.yes;
        let text = self.models_cmd(ModelsCmd::Remove(a)).await?;
        if deleting {
            Ok(json!({"deleted": true, "text": text}))
        } else {
            Ok(
                json!({"deleted": false, "text": text, "message": "Nothing was deleted. If the user agrees, call ymir_remove_model again with confirm=true."}),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_specs() -> impl Iterator<Item = &'static ToolSpec> {
        READ_TOOLS.iter().chain(MUTATING_TOOLS).chain([&REMOVE_TOOL])
    }

    #[test]
    fn mutating_tools_are_marked_gated_and_never_read_only() {
        for t in MUTATING_TOOLS.iter().chain([&REMOVE_TOOL]) {
            assert!(!t.read_only, "{}", t.name);
            assert!(t.description.contains("needs confirm=true"), "{}", t.name);
            assert!(t.params.iter().any(|p| p.name == "confirm" && !p.required), "{}", t.name);
        }
        for t in READ_TOOLS {
            assert!(t.params.iter().all(|p| p.name != "confirm"), "{}", t.name);
            assert!(!t.description.contains("MUTATES"), "{}", t.name);
        }
        assert_eq!(
            mutating_tools(),
            ["ymir_load_model", "ymir_unload_model", "ymir_move_model", "ymir_pull_model", "ymir_speak", "ymir_remove_model"]
        );
    }

    #[test]
    fn names_are_unique_and_prefixed() {
        let mut names: Vec<_> = all_specs().map(|t| t.name).collect();
        assert!(names.iter().all(|n| n.starts_with("ymir_")));
        names.sort();
        let n = names.len();
        names.dedup();
        assert_eq!(n, names.len());
    }

    #[test]
    fn schema_check_and_shapes() {
        let checked = ToolSpec::check(&all_specs().copied().collect::<Vec<_>>());
        assert!(checked.is_ok(), "{checked:?}");
        let s = ToolSpec::to_json(&READ_TOOLS[4])["inputSchema"].clone();
        assert_eq!(s["properties"]["messages"]["type"], "array");
        assert_eq!(s["properties"]["messages"]["items"]["required"], json!(["role", "content"]));
        assert_eq!(s["properties"]["route"]["enum"], json!(["bridge", "weirdnet"]));
        assert_eq!(s["properties"]["images"]["items"]["type"], "string");
        assert_eq!(s["properties"]["max_tokens"]["minimum"], 1);
        let e = ToolSpec::to_json(&READ_TOOLS[6])["inputSchema"].clone();
        assert!(e["properties"]["input"]["anyOf"].is_array());
        assert_eq!(e["required"], json!(["input"]));
        let f = ToolSpec::to_json(&READ_TOOLS[8])["inputSchema"].clone();
        assert_eq!(f["properties"]["format"]["enum"], json!(["gguf", "mlx", "all"]));
    }

    #[test]
    fn only_loopback_addresses() {
        assert_eq!(loopback_addr("127.0.0.1:9").unwrap().port(), 9);
        assert!(loopback_addr("[::1]:9").is_ok());
        assert!(loopback_addr("localhost:9").is_ok());
        assert!(loopback_addr("0.0.0.0:9").is_err());
        assert!(loopback_addr("192.0.2.10:9").is_err());
        assert!(loopback_addr("nonsense").is_err());
    }

    #[test]
    fn confirm_must_be_literally_true() {
        assert!(confirmed(&json!({}), "x").is_err());
        assert!(confirmed(&json!({"confirm": false}), "x").is_err());
        assert!(confirmed(&json!({"confirm": "true"}), "x").is_err());
        assert!(confirmed(&json!({"confirm": true}), "x").is_ok());
    }

    #[test]
    fn chat_messages_from_prompt_images_and_audio() {
        let dir = tempfile::tempdir().unwrap();
        let png = dir.path().join("a.png");
        std::fs::write(&png, b"\x89PNG fake").unwrap();
        let wav = dir.path().join("a.wav");
        std::fs::write(&wav, b"RIFF fake").unwrap();
        let m = YmirTools::messages(&json!({"system": "s", "prompt": "look", "images": [png.to_str().unwrap()], "audio": [wav.to_str().unwrap()]}))
            .unwrap();
        assert_eq!(m[0], json!({"role": "system", "content": "s"}));
        let parts = m[1]["content"].as_array().unwrap();
        assert_eq!(parts[0], json!({"type": "text", "text": "look"}));
        assert!(parts[1]["image_url"]["url"].as_str().unwrap().starts_with("data:image/png;base64,"));
        assert_eq!(parts[2]["type"], "input_audio");
        assert_eq!(parts[2]["input_audio"]["format"], "wav");
        // A plain prompt stays a string.
        assert_eq!(YmirTools::messages(&json!({"prompt": "hi"})).unwrap()[0]["content"], "hi");
        // Mixed or missing input is refused.
        assert!(YmirTools::messages(&json!({})).is_err());
        assert!(YmirTools::messages(&json!({"prompt": "a", "messages": [{"role": "user", "content": "b"}]})).is_err());
        assert!(YmirTools::messages(&json!({"messages": [{"role": "user", "content": "b"}], "images": ["x.png"]})).is_err());
    }

    #[test]
    fn transcribe_files_are_checked_before_they_are_read() {
        let dir = tempfile::tempdir().unwrap();
        let inside = dir.path().join("in");
        std::fs::create_dir(&inside).unwrap();
        let ok = inside.join("talk.wav");
        std::fs::write(&ok, b"RIFF").unwrap();
        let outside = dir.path().join("other.wav");
        std::fs::write(&outside, b"RIFF").unwrap();
        let secret = inside.join("notes.txt");
        std::fs::write(&secret, b"private").unwrap();
        let paths = Paths::new(dir.path().join("home"));
        let open = YmirTools::new(paths.clone(), None, Policy::default());
        assert_eq!(open.transcribe_file(outside.to_str().unwrap()).unwrap(), b"RIFF");
        let t = YmirTools::new(paths, None, Policy { read_only: true, allow_remove: false }).with_roots(std::slice::from_ref(&inside)).unwrap();
        assert!(t.specs.iter().any(|s| s.name == "ymir_transcribe"), "offered under --read-only");
        assert_eq!(t.transcribe_file(ok.to_str().unwrap()).unwrap(), b"RIFF");
        let e = |p: &str| t.transcribe_file(p).unwrap_err().to_string();
        assert!(e(outside.to_str().unwrap()).contains("outside the allowed roots"));
        let sneaky = format!("{}/../other.wav", inside.display());
        assert!(e(&sneaky).contains("outside the allowed roots"));
        assert!(e(secret.to_str().unwrap()).contains("not an audio or video type"));
        assert!(e("https://example.com/a.wav").contains("never fetched"));
        assert!(e("file:///etc/passwd.wav").contains("never fetched"));
        assert!(e(&format!("{}/none.wav", inside.display())).contains("reading"));
        #[cfg(unix)]
        {
            let link = inside.join("link.wav");
            std::os::unix::fs::symlink(&outside, &link).unwrap();
            assert!(e(link.to_str().unwrap()).contains("outside the allowed roots"), "links are resolved first");
        }
    }

    #[test]
    fn media_files_are_checked_before_they_are_read() {
        let dir = tempfile::tempdir().unwrap();
        let secret = dir.path().join("notes.txt");
        std::fs::write(&secret, b"private").unwrap();
        assert!(image_part(secret.to_str().unwrap()).unwrap_err().to_string().contains("not a image type"));
        let big = dir.path().join("big.png");
        std::fs::write(&big, vec![0u8; (MAX_IMAGE_BYTES + 1) as usize]).unwrap();
        assert!(image_part(big.to_str().unwrap()).unwrap_err().to_string().contains("limit"));
        let mp4 = dir.path().join("a.png");
        std::fs::write(&mp4, b"x").unwrap();
        assert!(audio_part(mp4.to_str().unwrap()).unwrap_err().to_string().contains("not an audio type"));
        assert!(image_part("data:text/plain;base64,AAAA").is_err());
        assert!(image_part("data:image/png;base64,AAAA").is_ok());
        assert!(image_part("/no/such/file.png").is_err());
    }

    #[test]
    fn progress_redraws_are_collapsed() {
        assert_eq!(tidy(b"a\nx 10%\rx 50%\rx 100%\nb\n".to_vec()), "a\nx 100%\nb");
    }

    #[test]
    fn cli_flags() {
        use clap::Parser;
        let p = |a: &[&str]| crate::Cli::try_parse_from(["ymirctl", "mcp"].iter().chain(a)).map(|c| c.cmd);
        let crate::Cmd::Mcp(m) = p(&[]).unwrap() else { panic!() };
        assert!(!m.read_only && !m.allow_remove && m.http.is_none());
        let crate::Cmd::Mcp(m) = p(&["--http", "--allow-remove"]).unwrap() else { panic!() };
        assert!(m.allow_remove);
        assert_eq!(m.http.as_deref(), Some(DEFAULT_HTTP));
        let crate::Cmd::Mcp(m) = p(&["--http", "127.0.0.1:9000", "--read-only"]).unwrap() else { panic!() };
        assert_eq!(m.http.as_deref(), Some("127.0.0.1:9000"));
        assert!(m.read_only);
    }
}
