//! `/ymir/control/1`, local transport: JSON lines over a Unix socket in the node's private state
//! directory (mode 0600), or on Windows a local-only named pipe derived from that path (see
//! [`pipe_name`]; the pipe's default DACL lets only its owner, SYSTEM and Administrators write).
//! Over Weirdnet (M3) the same messages go to admin devices on the roster.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use weirdnet_backends::EngineKind;
pub use weirdnet_backends::ModelState;

use crate::{Coordinator, LocalModelNode, ModelNode};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ControlRequest {
    Status,
    /// An operator's load: also clears a crash loop.
    Load {
        model: String,
        #[serde(default)]
        node: Option<String>,
    },
    Unload {
        model: String,
    },
    /// Move a model to another model node (M6): load it on `to`, then unload it from `from` (or
    /// where it is loaded) once what it has there finishes, at most `grace_secs`.
    Move {
        model: String,
        to: String,
        #[serde(default)]
        from: Option<String>,
        #[serde(default = "move_grace")]
        grace_secs: u64,
    },
    /// Stop sending a model node new requests (it finishes what it has), or resume it.
    Drain {
        node: String,
        #[serde(default = "yes")]
        on: bool,
    },
    /// Hold a model node's GPU for `holder` (training plan P3a): its models unload and nothing is
    /// placed there until `Release`. `node` defaults to this box's model role. Refused when the
    /// node is the only one serving a loaded model, unless `force`.
    Reserve {
        #[serde(default)]
        node: Option<String>,
        holder: String,
        #[serde(default)]
        force: bool,
    },
    /// End `holder`'s reservation of `node` (default this box's model role).
    Release {
        #[serde(default)]
        node: Option<String>,
        holder: String,
    },
    /// A device this user just linked from here (`ymirctl pool add`) joins the roster.
    PoolApprove {
        cert: weirdnet_net::user::NodeCert,
    },
    /// The sync bundle for a cert-only link to `peer` as `role` (`inference` or `model`), made
    /// by the running node so its grant and any first rotation land in the live roster
    /// (`ymirctl pool add --service`). The answer's message is the bundle, hex; it holds epoch
    /// keys for an inference node. Local socket only.
    PoolServiceBundle {
        peer: String,
        role: String,
    },
    /// Revoke a device (`ymirctl pool revoke`).
    PoolRevoke {
        peer: String,
    },
    /// Share the pool with another Weirdnet identity (`ymirctl pool share`): record the guest
    /// grant in the live, synced pool records.
    PoolShare {
        grant: crate::guests::GuestGrant,
    },
    /// Revoke a guest identity's grant (`ymirctl pool unshare`).
    PoolUnshare {
        user_id: String,
    },
    /// Wake a Bragi device (peer id or label) through its relay so it syncs (`ymirctl pool
    /// wake`); `reason` is `message` or `connection`.
    PoolWake {
        device: String,
        #[serde(default = "message")]
        reason: String,
    },
    /// Add a message to a conversation in this node's context store (it syncs to the user's
    /// other inference and app nodes).
    ContextAdd {
        conversation: String,
        #[serde(default = "user")]
        role: String,
        text: String,
    },
    /// The conversations this node holds, or one of them in full.
    ContextShow {
        #[serde(default)]
        conversation: Option<String>,
    },
    /// Tombstone a conversation everywhere.
    ContextDelete {
        conversation: String,
    },
    /// Give or withdraw a conversation's training consent (training plan P2). It syncs.
    ContextTrain {
        conversation: String,
        on: bool,
    },
    /// The conversations whose training consent is on, with their raw messages, for
    /// `ymirctl context export --training`, which strips and writes them. Local transport only.
    ContextForTraining,
    /// Add `text` to `conversation` as the user and ask the pool to answer it from the
    /// conversation (a `context_ref`): the inference node that answers writes the answer back.
    Chat {
        conversation: String,
        text: String,
        #[serde(default)]
        model: Option<String>,
    },
    /// Split models (M7): bring the group for `model` up or down, or list the groups.
    Group {
        /// `up`, `down` or `status`.
        action: String,
        #[serde(default)]
        model: Option<String>,
    },
    /// Fan `prompts` out over the pool at batch priority and wait for all the answers.
    Batch {
        prompts: Vec<String>,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        concurrency: Option<usize>,
        #[serde(default)]
        retries: Option<u32>,
    },
    /// Fan `tasks` (each with its own messages, sampling and seed) out over the pool at batch
    /// priority and wait for all the answers (`ymirctl batch --jsonl`, in chunks).
    BatchTasks {
        tasks: Vec<crate::batch::BatchTask>,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        concurrency: Option<usize>,
        #[serde(default)]
        retries: Option<u32>,
        /// Job id prefix, stable across chunks and resumed runs (default: a fresh one).
        #[serde(default)]
        job: Option<String>,
        /// Wait for the model to load rather than take a stand-in.
        #[serde(default)]
        wait_for_model: bool,
        #[serde(default)]
        cache: crate::batch::CacheMode,
    },
    /// Manage model files on the user's model nodes over Weirdnet (`ymirctl models --node`). The
    /// answer is a stream of [`crate::models::ModelsLine`]s (a pull's progress, then the result), not
    /// one [`ControlResponse`].
    Models {
        /// A roster label, a peer id, `auto` (best fit, pulls only) or `all`.
        node: String,
        request: crate::models::ModelsOp,
    },
    /// A training run on another of the user's model nodes over Weirdnet (`ymirctl train run
    /// --node`). The answer is a stream of [`crate::train_net::TrainLine`]s ending with `done` or
    /// `failed`; closing the connection cancels the run.
    Train {
        /// A roster label or a peer id.
        node: String,
        request: crate::train_net::TrainOp,
    },
    /// A call on the synced training job board this daemon holds (`ymirctl train run --job`
    /// on Weirdnet; see [`crate::train_jobs_ctl`]).
    Jobs {
        call: crate::train_jobs_ctl::JobCall,
    },
    /// Another model node's training status over Weirdnet (`ymirctl train status --node`,
    /// `/ymir/train-status/1`).
    TrainStatus {
        /// A roster label or a peer id.
        node: String,
    },
    /// The versioned rollouts in force (P4b), or, with `set`, replace them live (new
    /// conversations follow at once). `ymirctl models rollout` also writes them to node.toml.
    Rollouts {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        set: Option<Vec<crate::rollout::Rollout>>,
    },
    /// Another inference node's rollouts over Weirdnet (`ymirctl models rollout --inference NODE`,
    /// `/ymir/rollouts/1`): read them, or change them by that node's rules.
    RemoteRollouts {
        /// A roster label or a peer id.
        node: String,
        request: crate::rollouts_net::Request,
    },
    /// Speech to text through the pool (audio design milestone 1): `bytes` of audio follow this
    /// line on the connection, raw (local transport only; see [`transcribe`]). Goes through the
    /// app role's pool, so the inference role routes it as it routes infer/2 `Transcribe`.
    Transcribe(TranscribeJob),
    /// The transcriber a request would use and its audio limits from catalog/1, so a client can
    /// plan chunks (audio design milestone 3). Nothing is sent to a model.
    TranscribeTarget {
        #[serde(default)]
        model: Option<String>,
    },
    /// Text to speech through the pool (`weirdnet-agent` 0.5.11 infer/2 `Speak`): the reference
    /// clips of cloned voices follow this line raw, `clips[i]` bytes for the i-th cloned voice
    /// (local transport only; see [`speak`]). The answer's audio follows the answer line, raw
    /// ([`SpeechReply::bytes`]). Neither text, clip nor audio is logged.
    Speak(SpeakJob),
    /// The speaker a speech request would use and its catalog/1 profile, so a client can plan
    /// long text (`max_input_chars`) and pick formats. Nothing is sent to a model.
    SpeakTarget(SpeakJob),
    /// Stop `ymird` cleanly (engines too), as SIGTERM does. Local transport only: this is how
    /// `ymirctl service stop` stops a Windows scheduled task, which has no signal to send.
    Shutdown,
}

fn user() -> String {
    "user".into()
}

fn message() -> String {
    "message".into()
}

fn move_grace() -> u64 {
    60
}

fn yes() -> bool {
    true
}

/// A model node, for `ymirctl nodes`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeSummary {
    pub name: String,
    /// This box's own model role.
    #[serde(default)]
    pub local: bool,
    pub draining: bool,
    /// Who holds the node's GPU, when it is reserved (a training run).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reserved: Option<String>,
    pub models: usize,
    pub ready: usize,
    pub in_flight: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeModel {
    pub node: String,
    pub id: String,
    pub engine: EngineKind,
    pub state: ModelState,
    pub in_flight: usize,
    pub swap_group: Option<String>,
    pub purpose: String,
    /// The engine process, while one runs.
    pub pid: Option<u32>,
    #[serde(default)]
    pub last_load_secs: Option<f32>,
    /// A stock server Ymir forwards to but doesn't own.
    #[serde(default)]
    pub wrapped: bool,
    /// Engine crashes in the current window.
    #[serde(default)]
    pub crashes: usize,
    /// What the model takes (chat, embeddings, images, audio), when the node reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<weirdnet_backends::Capabilities>,
    /// For a LoRA adapter: the base model it runs on (it shares the base's engine and state).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<String>,
    /// The context it is set to launch with (or runs with, once loaded), when one is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<u32>,
    /// The concurrent sequences it is set to run, when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slots: Option<u32>,
    /// The context per request the engine reported when the model last loaded (llama-server
    /// `/props`, vLLM `max_model_len`; `weirdnet-backends` 0.4.9), when measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_tokens: Option<u32>,
    /// The weights' revision the node reports (node.toml's `revision`, or an upload's), which
    /// keys the answer cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatusReport {
    pub node: String,
    pub version: String,
    pub pid: u32,
    pub roles: Vec<String>,
    /// The loopback bridge port, when the app role is on.
    pub bridge_port: Option<u16>,
    pub main_model: Option<String>,
    pub models: Vec<NodeModel>,
    /// The model nodes the inference role routes to (this box's own and the user's other devices').
    #[serde(default)]
    pub nodes: Vec<NodeSummary>,
    /// Alert keys currently failing.
    #[serde(default)]
    pub alerts: Vec<String>,
    /// This device on Weirdnet, once it has an identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network: Option<NetworkStatus>,
    /// The semantic answer cache, when it is on (M6; pool mode shows its members and counters).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<crate::semantic::CacheStatus>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetworkStatus {
    pub peer: String,
    pub user: String,
    pub listening: Vec<String>,
    pub connected: Vec<String>,
    pub devices: Vec<Device>,
}

/// A device in the user's roster.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Device {
    pub peer: String,
    pub label: String,
    pub active: bool,
    /// This device.
    pub me: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControlResponse {
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<StatusReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat: Option<ChatResult>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conversations: Vec<ConversationView>,
    /// The answer to [`ControlRequest::ContextForTraining`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub training: Vec<crate::context::RawConversation>,
    /// The outcomes of a [`ControlRequest::Batch`], in prompt order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub batch: Vec<crate::batch::BatchItem>,
    /// The split-model groups that are up (`ymirctl group status`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<crate::group::GroupInfo>,
    /// Groups that went down by themselves, newest last.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub group_losses: Vec<crate::group::GroupLoss>,
    /// The rollouts in force, for [`ControlRequest::Rollouts`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollouts: Option<Vec<crate::rollout::Rollout>>,
    /// Another inference node's answer, for [`ControlRequest::RemoteRollouts`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote_rollouts: Option<crate::rollouts_net::Rollouts>,
    /// The answer to [`ControlRequest::TrainStatus`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub train_status: Option<crate::train_net::TrainStatus>,
    /// The answer to [`ControlRequest::Jobs`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job: Option<crate::train_jobs_ctl::JobReply>,
    /// The answer to [`ControlRequest::Transcribe`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcript: Option<TranscribeReply>,
    /// The answer to [`ControlRequest::TranscribeTarget`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcriber: Option<Transcriber>,
    /// The answer to [`ControlRequest::Speak`]; its audio follows the answer line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speech: Option<SpeechReply>,
    /// The answer to [`ControlRequest::SpeakTarget`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<Speaker>,
}

/// The most reference audio one [`ControlRequest::Speak`] carries, all clips together (each clip
/// is also held to infer/2's 30 MiB, `MAX_REFERENCE_BYTES`).
pub const MAX_SPEAK_CLIP_BYTES: u64 = 31 << 20;

/// A speech request for the control socket: the infer/2 request with every cloned voice's audio
/// left empty, and the length of each clip that follows the line, in [`SpeakRequest::all_voices`]
/// order.
///
/// [`SpeakRequest::all_voices`]: weirdnet_agent::remote::infer::v2::SpeakRequest::all_voices
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SpeakJob {
    pub request: weirdnet_agent::remote::infer::v2::SpeakRequest,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub clips: Vec<u64>,
}

impl SpeakJob {
    /// `request` with its clips taken out (to send after the line).
    pub fn split(mut request: weirdnet_agent::remote::infer::v2::SpeakRequest) -> (Self, Vec<Vec<u8>>) {
        use weirdnet_agent::remote::infer::v2::Voice;
        let mut clips = Vec::new();
        let voices = request.voice.iter_mut().chain(request.voices.iter_mut().map(|v| &mut v.voice));
        for v in voices {
            if let Voice::Clone { audio, .. } = v {
                clips.push(std::mem::take(audio));
            }
        }
        request.cert = None;
        (Self { request, clips: clips.iter().map(|c| c.len() as u64).collect() }, clips)
    }

    /// The request with `clips` put back, checked against [`Self::clips`].
    pub fn join(self, clips: Vec<Vec<u8>>) -> Result<weirdnet_agent::remote::infer::v2::SpeakRequest> {
        use weirdnet_agent::remote::infer::v2::Voice;
        let mut request = self.request;
        request.cert = None;
        let mut it = clips.into_iter();
        let voices = request.voice.iter_mut().chain(request.voices.iter_mut().map(|v| &mut v.voice));
        for v in voices {
            if let Voice::Clone { audio, .. } = v {
                *audio = it.next().context("fewer reference clips than cloned voices")?;
            }
        }
        if it.next().is_some() {
            bail!("more reference clips than cloned voices");
        }
        Ok(request)
    }

    fn check_clips(&self) -> Result<u64, String> {
        let cloned = self.request.all_voices().filter(|v| matches!(v, weirdnet_agent::remote::infer::v2::Voice::Clone { .. })).count();
        if cloned != self.clips.len() {
            return Err(format!("{} reference clip lengths for {cloned} cloned voices", self.clips.len()));
        }
        let max = weirdnet_agent::remote::infer::v2::MAX_REFERENCE_BYTES as u64;
        if let Some(c) = self.clips.iter().find(|c| **c == 0 || **c > max) {
            return Err(format!("a reference clip takes 1 to {max} bytes, not {c}"));
        }
        let total: u64 = self.clips.iter().sum();
        if total > MAX_SPEAK_CLIP_BYTES {
            return Err(format!("the reference clips are {total} bytes together; one request carries at most {MAX_SPEAK_CLIP_BYTES}"));
        }
        Ok(total)
    }
}

/// Speech and where it came from; [`Self::bytes`] of audio follow the answer line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechReply {
    pub bytes: u64,
    pub format: weirdnet_agent::speech::AudioFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channels: Option<u16>,
    pub model: String,
    pub node: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<weirdnet_agent::remote::infer::v2::Usage>,
    /// Request fields the model received but does not use (`Meta::ignored`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ignored: Vec<String>,
}

/// The speaker a request would be routed to, and its profile from catalog/1 (absent when no node
/// reports one; never defaulted).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Speaker {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<weirdnet_agent::speech::SpeechProfile>,
}

/// Largest audio clip one [`ControlRequest::Transcribe`] carries: infer/2 frames are at most
/// 32 MiB (`weirdnet-agent` `MAX_FRAME`), less room for the rest of the request. Longer audio is
/// chunked by the client (`ymirctl transcribe --chunk auto`).
pub const MAX_TRANSCRIBE_BYTES: u64 = 31 << 20;

/// A transcription request; the audio follows on the connection.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TranscribeJob {
    /// How many bytes of audio follow the request line.
    pub bytes: u64,
    /// `wav`, `mp3`, `flac`, ...
    pub format: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub timestamps: bool,
}

/// A transcript and where it came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscribeReply {
    pub transcript: weirdnet_agent::remote::infer::v2::Transcript,
    pub model: String,
    pub node: String,
}

/// The transcriber a request would be routed to, and what catalog/1 reports for it (each field
/// absent when no node reports it; never defaulted).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transcriber {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_max_seconds: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_request_bytes: Option<u64>,
}

/// An answer to [`ControlRequest::Chat`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatResult {
    pub text: String,
    /// The inference node that finished it.
    pub node: String,
    pub model: String,
    /// How many times the answer moved to another inference node mid-answer.
    pub resumes: usize,
    /// What happened on the way, in order: `accepted <node>` and `resumed <node> at <offset>`.
    #[serde(default)]
    pub path: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConversationView {
    pub id: String,
    pub title: String,
    pub messages: Vec<(String, String)>,
    /// The user's training consent is on.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub training: bool,
}

impl ControlResponse {
    fn ok(message: impl Into<String>) -> Self {
        Self {
            ok: true,
            message: Some(message.into()),
            status: None,
            chat: None,
            conversations: Vec::new(),
            training: Vec::new(),
            batch: Vec::new(),
            groups: Vec::new(),
            group_losses: Vec::new(),
            rollouts: None,
            remote_rollouts: None,
            train_status: None,
            job: None,
            transcript: None,
            transcriber: None,
            speech: None,
            speaker: None,
        }
    }
    fn err(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            message: Some(message.into()),
            status: None,
            chat: None,
            conversations: Vec::new(),
            training: Vec::new(),
            batch: Vec::new(),
            groups: Vec::new(),
            group_losses: Vec::new(),
            rollouts: None,
            remote_rollouts: None,
            train_status: None,
            job: None,
            transcript: None,
            transcriber: None,
            speech: None,
            speaker: None,
        }
    }
}

impl ControlHandler {
    /// [`ControlRequest::Transcribe`], with its audio read. Neither the audio nor the transcript
    /// is logged.
    pub async fn transcribe(&self, job: TranscribeJob, audio: Vec<u8>) -> ControlResponse {
        use weirdnet_agent::remote::infer::v2::TranscribeRequest;
        let Some(pool) = &self.pool else {
            return ControlResponse::err("transcription needs the app role (it asks the pool like the bridge does)");
        };
        let mut r = TranscribeRequest::new(audio, job.format).with_timestamps(job.timestamps);
        if let Some(m) = job.model {
            r = r.with_model(m);
        }
        if let Some(l) = job.language.filter(|l| !l.trim().is_empty()) {
            r = r.with_language(l);
        }
        if let Some(p) = job.prompt.filter(|p| !p.trim().is_empty()) {
            r = r.with_prompt(p);
        }
        match pool.transcribe(r, &weirdnet_agent::CancelToken::new()).await {
            Ok(t) => {
                let mut resp = ControlResponse::ok("transcribed");
                resp.transcript = Some(TranscribeReply { transcript: t.transcript, model: t.model, node: t.node });
                resp
            }
            Err(e) => ControlResponse::err(e.to_string()),
        }
    }

    /// [`ControlRequest::Speak`], with its clips read: the whole speech, through the pool. The
    /// audio comes back beside the answer (to write after its line).
    pub async fn speak(&self, job: SpeakJob, clips: Vec<Vec<u8>>) -> (ControlResponse, Vec<u8>) {
        let Some(pool) = &self.pool else {
            return (ControlResponse::err("speech needs the app role (it asks the pool like the bridge does)"), Vec::new());
        };
        let request = match job.join(clips) {
            Ok(r) => r,
            Err(e) => return (ControlResponse::err(e.to_string()), Vec::new()),
        };
        match pool.speak(request, &weirdnet_agent::CancelToken::new()).await {
            Ok(s) => {
                let mut resp = ControlResponse::ok("spoken");
                resp.speech = Some(SpeechReply {
                    bytes: s.audio.len() as u64,
                    format: s.format,
                    sample_rate: s.sample_rate,
                    channels: s.channels,
                    model: s.model,
                    node: s.node,
                    usage: s.usage,
                    ignored: s.meta.ignored.clone(),
                });
                (resp, s.audio)
            }
            Err(e) => (ControlResponse::err(e.to_string()), Vec::new()),
        }
    }

    /// [`ControlRequest::SpeakTarget`]: routed as the speech request would be, from catalog/1.
    pub async fn speak_target(&self, job: SpeakJob) -> ControlResponse {
        let Some(pool) = &self.pool else {
            return ControlResponse::err("speech needs the app role (it asks the pool like the bridge does)");
        };
        let catalog = match pool.catalog().await {
            Ok(c) => c,
            Err(e) => return ControlResponse::err(format!("no catalog from the pool: {e}")),
        };
        let mut needs = job.request.needs();
        // The clips are not sent for routing; their lengths are. Length and format are left
        // out: the caller splits long text to the speaker's `max_input_chars` and converts to a
        // format it doesn't make.
        needs.reference_bytes = job.clips.iter().max().copied();
        needs.input_chars = 0;
        needs.format = None;
        match catalog.route_speak(job.request.model.as_deref(), &needs) {
            Ok(id) => {
                let profile = catalog.find(&id).and_then(|m| m.speech.clone());
                let mut resp = ControlResponse::ok("speaker");
                resp.speaker = Some(Speaker { model: id, profile });
                resp
            }
            Err(e) => ControlResponse::err(e.to_string()),
        }
    }

    /// [`ControlRequest::TranscribeTarget`]: routed as a transcription would be, from catalog/1.
    pub async fn transcribe_target(&self, model: Option<&str>) -> ControlResponse {
        let Some(pool) = &self.pool else {
            return ControlResponse::err("transcription needs the app role (it asks the pool like the bridge does)");
        };
        let catalog = match pool.catalog().await {
            Ok(c) => c,
            Err(e) => return ControlResponse::err(format!("no catalog from the pool: {e}")),
        };
        match catalog.route_transcribe(model) {
            Ok(id) => {
                let limits = catalog.find(&id).map(|m| m.limits).unwrap_or_default();
                let mut resp = ControlResponse::ok("transcriber");
                resp.transcriber =
                    Some(Transcriber { model: id, audio_max_seconds: limits.audio_max_seconds, max_request_bytes: limits.max_request_bytes });
                resp
            }
            Err(e) => ControlResponse::err(e.to_string()),
        }
    }
}

/// What the control socket can reach.
pub struct ControlHandler {
    pub node: String,
    pub roles: Vec<String>,
    pub bridge_port: Option<u16>,
    pub main_model: Option<String>,
    pub coordinator: Option<Arc<Coordinator>>,
    pub model_node: Option<Arc<LocalModelNode>>,
    pub alerter: Option<Arc<weirdnet_alerts::Alerter>>,
    pub network: Option<(Arc<crate::membership::Membership>, Arc<crate::membership::SyncNode>)>,
    /// Conversations (inference and app roles on Weirdnet).
    pub context: Option<crate::context::ContextStore>,
    /// The pool the app role asks through.
    pub pool: Option<Arc<weirdnet_agent::pool::Pool>>,
    /// Model management over Weirdnet (needs a device identity).
    pub models: Option<Arc<crate::models::ModelsHub>>,
    /// Training runs on other model nodes over Weirdnet (needs a device identity).
    pub train: Option<Arc<crate::train_net::TrainHub>>,
    /// The synced training job board and this device's id on it (Weirdnet).
    pub jobs: Option<(Arc<crate::train_board::BoardJobStore>, String)>,
    /// Fetches training bundles from the user's other devices (Weirdnet).
    pub blobs: Option<Arc<crate::blobs_net::BlobsClient>>,
    /// Woken by [`ControlRequest::Shutdown`]; `ymird` waits on it next to its signals.
    pub exit: Arc<tokio::sync::Notify>,
}

impl ControlHandler {
    /// The node a `Reserve` or `Release` names, else this box's model role.
    fn reservable(&self, node: Option<&str>) -> Result<Arc<dyn ModelNode>, String> {
        match node {
            Some(name) => self.nodes().into_iter().find(|n| n.name() == name).ok_or_else(|| format!("no model node `{name}` (see `ymirctl nodes`)")),
            None => match &self.model_node {
                Some(n) => Ok(n.clone() as Arc<dyn ModelNode>),
                None => Err("this node runs no model role; name a model node with --node".into()),
            },
        }
    }

    /// Refuses to reserve `n` while it is the only node with a loaded model ready: reserving it
    /// would take that model out of the pool. Without the inference role this box can't see the
    /// other nodes, so it can't tell; the operator decides with `force`.
    fn sole_server_check(&self, n: &dyn ModelNode) -> Result<(), String> {
        let ready: Vec<String> = n.status().into_iter().filter(|s| s.state == ModelState::Ready && s.base.is_none()).map(|s| s.id).collect();
        if ready.is_empty() {
            return Ok(());
        }
        let Some(c) = &self.coordinator else {
            return Err(format!(
                "{} serves {} and this box has no inference role, so it can't see whether another node serves them too; check, then pass --force",
                n.name(),
                ready.join(", ")
            ));
        };
        let alone: Vec<String> = ready
            .into_iter()
            .filter(|id| {
                !c.nodes()
                    .iter()
                    .any(|o| o.name() != n.name() && !o.draining() && o.status().iter().any(|s| &s.id == id && s.state == ModelState::Ready))
            })
            .collect();
        if alone.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "{} is the only node serving {} right now; reserving it takes them out of the pool. Load them elsewhere first, or pass --force",
                n.name(),
                alone.join(", ")
            ))
        }
    }

    fn network_status(&self) -> Option<NetworkStatus> {
        let (m, node) = self.network.as_ref()?;
        let st = node.status();
        let me = m.peer_id().to_string();
        let devices = m
            .roster()
            .certs
            .iter()
            .map(|c| Device { peer: c.peer_id.clone(), label: c.label.clone(), active: c.active(), me: c.peer_id == me })
            .collect();
        Some(NetworkStatus { peer: me, user: m.user_id(), listening: st.listening, connected: st.connected, devices })
    }

    fn nodes(&self) -> Vec<Arc<dyn ModelNode>> {
        match (&self.coordinator, &self.model_node) {
            (Some(c), _) => c.nodes(),
            (None, Some(n)) => vec![n.clone() as Arc<dyn ModelNode>],
            (None, None) => Vec::new(),
        }
    }

    /// Runs a models request and writes its lines to `w` as they happen.
    async fn stream_models<W: AsyncWrite + Unpin>(&self, node: String, op: crate::models::ModelsOp, w: &mut W) -> Result<()> {
        use crate::models::ModelsLine;
        let Some(hub) = self.models.clone() else {
            let line = ModelsLine::Failed {
                message: "model management over Weirdnet needs a device identity on this node (see `ymirctl pool link`)".into(),
            };
            return write_line(w, &line).await;
        };
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            let done = hub.run(&node, op, &tx).await;
            let _ = tx.send(match done {
                Ok(d) => ModelsLine::Done(Box::new(d)),
                Err(message) => ModelsLine::Failed { message },
            });
        });
        while let Some(line) = rx.recv().await {
            write_line(w, &line).await?;
        }
        let _ = task.await;
        Ok(())
    }

    /// Runs a training request on another node and writes its lines to `w`; `closed` resolving
    /// (the asking `ymirctl` went away) cancels it.
    async fn stream_train<W: AsyncWrite + Unpin>(
        &self,
        node: String,
        op: crate::train_net::TrainOp,
        w: &mut W,
        closed: impl std::future::Future<Output = ()>,
    ) -> Result<()> {
        use crate::train_net::TrainLine;
        let Some(hub) = self.train.clone() else {
            let line = TrainLine::Failed { message: "training on another node needs a device identity on this one (see `ymirctl pool link`)".into() };
            return write_line(w, &line).await;
        };
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            let done = hub.run(&node, op, &tx).await;
            let _ = tx.send(match done {
                Ok(record) => TrainLine::Done { record: Box::new(record) },
                Err(message) => TrainLine::Failed { message },
            });
        });
        tokio::pin!(closed);
        loop {
            tokio::select! {
                line = rx.recv() => match line {
                    Some(line) => write_line(w, &line).await?,
                    None => break,
                },
                _ = &mut closed => {
                    // Dropping the stream to the node cancels the run there.
                    task.abort();
                    break;
                }
            }
        }
        let _ = task.await;
        Ok(())
    }

    pub async fn handle(&self, req: ControlRequest) -> ControlResponse {
        match req {
            // Streamed by `stream_models` and `stream_train`, which `connection` calls instead.
            ControlRequest::Models { .. } => ControlResponse::err("models requests are streamed"),
            ControlRequest::Train { .. } => ControlResponse::err("train requests are streamed"),
            ControlRequest::Rollouts { set } => {
                let Some(c) = &self.coordinator else {
                    return ControlResponse::err("rollouts are routed by an inference node; this one has no inference role");
                };
                if let Some(r) = set
                    && let Err(e) = c.set_rollouts(r)
                {
                    return ControlResponse::err(e);
                }
                let mut resp = ControlResponse::ok("rollouts");
                resp.rollouts = Some(c.rollouts());
                resp
            }
            ControlRequest::RemoteRollouts { node, request } => {
                let Some(hub) = &self.models else {
                    return ControlResponse::err("changing rollouts over Weirdnet needs a device identity on this node (see `ymirctl pool link`)");
                };
                match hub.rollouts(&node, request).await {
                    Ok((label, r)) => {
                        let mut resp = ControlResponse::ok(label);
                        resp.remote_rollouts = Some(r);
                        resp
                    }
                    Err(e) => ControlResponse::err(e),
                }
            }
            ControlRequest::Jobs { call } => {
                let Some((store, me)) = &self.jobs else {
                    return ControlResponse::err("this node is not on Weirdnet, so it has no synced training job board");
                };
                let mut resp = ControlResponse::ok("");
                resp.job = Some(
                    crate::train_jobs_ctl::serve(store.as_ref(), self.blobs.as_deref().map(|b| b as &dyn crate::train_job::BlobFetch), me, call)
                        .await,
                );
                resp
            }
            ControlRequest::TrainStatus { node } => {
                let Some(hub) = &self.train else {
                    return ControlResponse::err("training status over Weirdnet needs a device identity on this node (see `ymirctl pool link`)");
                };
                match hub.status(&node).await {
                    Ok((label, s)) => {
                        let mut resp = ControlResponse::ok(label);
                        resp.train_status = Some(s);
                        resp
                    }
                    Err(e) => ControlResponse::err(e),
                }
            }
            ControlRequest::Transcribe(_) => ControlResponse::err("a transcription request carries audio: send it over the control socket"),
            ControlRequest::TranscribeTarget { model } => self.transcribe_target(model.as_deref()).await,
            ControlRequest::Speak(_) => ControlResponse::err("a speech request returns audio: send it over the control socket"),
            ControlRequest::SpeakTarget(job) => self.speak_target(job).await,
            ControlRequest::Shutdown => {
                tracing::info!("shutdown requested over the control socket");
                self.exit.notify_one();
                ControlResponse::ok("ymird is stopping")
            }
            ControlRequest::Status => {
                let models = self
                    .nodes()
                    .iter()
                    .flat_map(|n| {
                        n.status().into_iter().map(|s| NodeModel {
                            revision: n.revision(&s.id),
                            purpose: if crate::model_node::embeds(&s) {
                                "embedding".into()
                            } else if crate::model_node::transcribes(&s) {
                                "transcribe".into()
                            } else {
                                "chat".into()
                            },
                            node: n.name().to_string(),
                            id: s.id,
                            engine: s.engine,
                            state: s.state,
                            in_flight: s.in_flight,
                            swap_group: s.swap_group,
                            pid: s.pid,
                            last_load_secs: s.last_load_secs,
                            wrapped: s.wrapped,
                            crashes: s.crashes,
                            capabilities: s.capabilities,
                            base: s.base,
                            context: s.context,
                            slots: s.slots,
                            context_tokens: s.context_tokens,
                        })
                    })
                    .collect();
                let nodes = self
                    .nodes()
                    .iter()
                    .map(|n| {
                        let st = n.status();
                        NodeSummary {
                            name: n.name().to_string(),
                            local: self.model_node.as_ref().is_some_and(|l| l.name() == n.name()),
                            draining: n.draining(),
                            reserved: n.reserved(),
                            models: st.len(),
                            ready: st.iter().filter(|s| s.state == ModelState::Ready).count(),
                            in_flight: st.iter().map(|s| s.in_flight).sum(),
                        }
                    })
                    .collect();
                ControlResponse {
                    ok: true,
                    message: None,
                    chat: None,
                    conversations: Vec::new(),
                    training: Vec::new(),
                    batch: Vec::new(),
                    groups: Vec::new(),
                    group_losses: Vec::new(),
                    rollouts: None,
                    remote_rollouts: None,
                    train_status: None,
                    job: None,
                    transcript: None,
                    transcriber: None,
                    speech: None,
                    speaker: None,
                    status: Some(StatusReport {
                        node: self.node.clone(),
                        version: env!("CARGO_PKG_VERSION").into(),
                        pid: std::process::id(),
                        roles: self.roles.clone(),
                        bridge_port: self.bridge_port,
                        main_model: self.main_model.clone(),
                        models,
                        nodes,
                        alerts: self.alerter.as_ref().map(|a| a.failing()).unwrap_or_default(),
                        network: self.network_status(),
                        cache: self.coordinator.as_ref().and_then(|c| c.semantic()).map(|s| s.status()),
                    }),
                }
            }
            ControlRequest::Load { model, node } => {
                let result = match (&self.coordinator, &self.model_node) {
                    (Some(c), _) => c.load(&model, node.as_deref(), true).await,
                    (None, Some(n)) => n.load(&model, true).await.map(|_| n.name().to_string()),
                    (None, None) => return ControlResponse::err("this node runs neither the inference nor the model role"),
                };
                match result {
                    Ok(on) => ControlResponse::ok(format!("{model} is ready on {on}")),
                    Err(e) => ControlResponse::err(e.to_string()),
                }
            }
            ControlRequest::Unload { model } => {
                let result = match (&self.coordinator, &self.model_node) {
                    (Some(c), _) => c.unload(&model).await,
                    (None, Some(n)) => n.unload(&model).await.map(|_| vec![n.name().to_string()]),
                    (None, None) => return ControlResponse::err("this node runs neither the inference nor the model role"),
                };
                match result {
                    Ok(on) if on.is_empty() => ControlResponse::ok(format!("{model} was not loaded")),
                    Ok(on) => ControlResponse::ok(format!("{model} unloaded on {}", on.join(", "))),
                    Err(e) => ControlResponse::err(e.to_string()),
                }
            }
            ControlRequest::Move { model, to, from, grace_secs } => {
                let Some(c) = &self.coordinator else { return ControlResponse::err("moving a model needs the inference role") };
                match c.move_model(&model, from.as_deref(), &to, std::time::Duration::from_secs(grace_secs)).await {
                    Ok((from, to)) => ControlResponse::ok(format!("{model} moved from {from} to {to}")),
                    Err(e) => ControlResponse::err(e.to_string()),
                }
            }
            ControlRequest::Drain { node, on } => {
                let n = self.nodes().into_iter().find(|n| n.name() == node);
                let Some(n) = n else { return ControlResponse::err(format!("no model node `{node}` (see `ymirctl nodes`)")) };
                match n.drain(on).await {
                    Ok(()) if on => ControlResponse::ok(format!("{node} is drained: no new requests, nothing in flight")),
                    Ok(()) => ControlResponse::ok(format!("{node} takes requests again")),
                    Err(e) => ControlResponse::err(format!("{node}: {e}")),
                }
            }
            ControlRequest::Reserve { node, holder, force } => {
                let n = match self.reservable(node.as_deref()) {
                    Ok(n) => n,
                    Err(e) => return ControlResponse::err(e),
                };
                if !force && let Err(e) = self.sole_server_check(n.as_ref()) {
                    return ControlResponse::err(e);
                }
                match n.reserve(&holder).await {
                    Ok(()) => ControlResponse::ok(format!(
                        "{} is reserved for {holder}: its models are unloaded and nothing loads there until released",
                        n.name()
                    )),
                    Err(e) => ControlResponse::err(format!("{}: {e}", n.name())),
                }
            }
            ControlRequest::Release { node, holder } => {
                let n = match self.reservable(node.as_deref()) {
                    Ok(n) => n,
                    Err(e) => return ControlResponse::err(e),
                };
                match n.release(&holder).await {
                    Ok(()) => ControlResponse::ok(format!("{} is released: models load there again", n.name())),
                    Err(e) => ControlResponse::err(format!("{}: {e}", n.name())),
                }
            }
            ControlRequest::PoolApprove { cert } => {
                let Some((m, _)) = &self.network else { return ControlResponse::err("this node is not on Weirdnet") };
                let label = cert.label.clone();
                match m.approve(cert) {
                    Ok(()) => ControlResponse::ok(format!("{label} is linked")),
                    Err(e) => ControlResponse::err(format!("{e:#}")),
                }
            }
            ControlRequest::PoolServiceBundle { peer, role } => {
                let Some((m, _)) = &self.network else { return ControlResponse::err("this node is not on Weirdnet") };
                let made = crate::membership::ServiceRole::parse(&role).and_then(|r| m.service_bundle(&peer, r));
                match made {
                    Ok(b) => ControlResponse::ok(hex::encode(b.as_slice())),
                    Err(e) => ControlResponse::err(format!("{e:#}")),
                }
            }
            ControlRequest::ContextAdd { conversation, role, text } => {
                let Some(store) = &self.context else { return no_context() };
                match store.append(&conversation, &role, &text, None, None) {
                    Ok(id) => ControlResponse::ok(id),
                    Err(e) => ControlResponse::err(format!("{e:#}")),
                }
            }
            ControlRequest::ContextShow { conversation } => {
                let Some(store) = &self.context else { return no_context() };
                let all = match conversation {
                    Some(id) => match store.load(&id) {
                        Some(c) => vec![c],
                        None => return ControlResponse::err(format!("no conversation `{id}` here")),
                    },
                    None => store.list(),
                };
                let mut r = ControlResponse::ok(format!("{} conversation(s)", all.len()));
                r.conversations = all
                    .into_iter()
                    .map(|c| ConversationView {
                        title: c.meta.title.clone(),
                        training: store.train(&c.id).on,
                        messages: c.messages.into_iter().map(|m| (m.role, m.text)).collect(),
                        id: c.id,
                    })
                    .collect();
                r
            }
            ControlRequest::ContextDelete { conversation } => {
                let Some(store) = &self.context else { return no_context() };
                match store.delete(&conversation) {
                    Ok(()) => ControlResponse::ok(format!("{conversation} is deleted")),
                    Err(e) => ControlResponse::err(format!("{e:#}")),
                }
            }
            ControlRequest::ContextTrain { conversation, on } => {
                let Some(store) = &self.context else { return no_context() };
                match store.set_train(&conversation, on) {
                    Ok(_) if on => ControlResponse::ok(format!("{conversation} is marked for training")),
                    Ok(_) => ControlResponse::ok(format!("{conversation} is no longer for training")),
                    Err(e) => ControlResponse::err(format!("{e:#}")),
                }
            }
            ControlRequest::ContextForTraining => {
                let Some(store) = &self.context else { return no_context() };
                let all = store.for_training();
                let mut r = ControlResponse::ok(format!("{} conversation(s) marked for training", all.len()));
                r.training = all;
                r
            }
            ControlRequest::Chat { conversation, text, model } => {
                let (Some(store), Some(pool)) = (&self.context, &self.pool) else {
                    return ControlResponse::err("chat needs the app role and a device identity (see `ymirctl pool link`)");
                };
                if let Err(e) = store.append(&conversation, "user", &text, None, None) {
                    return ControlResponse::err(format!("{e:#}"));
                }
                let mut request =
                    weirdnet_agent::remote::infer::v2::Request::context(weirdnet_agent::remote::infer::v2::ContextRef::new(&conversation, None));
                if let Some(m) = model {
                    request = request.with_model(m);
                }
                // Count the times the answer moved to another inference node.
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
                let result = pool.infer(request, &tx, &weirdnet_agent::CancelToken::new()).await;
                drop(tx);
                let (mut resumes, mut path) = (0, Vec::new());
                while let Some(e) = rx.recv().await {
                    match e {
                        weirdnet_agent::pool::PoolEvent::Accepted { node, .. } => path.push(format!("accepted {node}")),
                        weirdnet_agent::pool::PoolEvent::Resumed { node, offset } => {
                            resumes += 1;
                            path.push(format!("resumed {node} at {offset}"));
                        }
                        _ => {}
                    }
                }
                match result {
                    Ok(a) => {
                        let mut r = ControlResponse::ok("answered");
                        r.chat = Some(ChatResult { text: a.text, node: a.node, model: a.model, resumes, path });
                        r
                    }
                    Err(i) => ControlResponse::err(format!("{} (after {} characters)", i.error, i.partial.text.chars().count())),
                }
            }
            ControlRequest::Group { action, model } => {
                let Some(c) = &self.coordinator else {
                    return ControlResponse::err("split models are decided by an inference node; this one has no inference role");
                };
                match (action.as_str(), model) {
                    ("up", Some(m)) => match c.group_up(&m).await {
                        Ok(i) => {
                            let mut r = ControlResponse::ok(format!(
                                "{m} is up: {} leads, with {}",
                                i.leader,
                                i.members.iter().skip(1).cloned().collect::<Vec<_>>().join(", ")
                            ));
                            r.groups = vec![i];
                            r
                        }
                        Err(e) => ControlResponse::err(e.to_string()),
                    },
                    ("down", Some(m)) => match c.group_down(&m).await {
                        Ok(_) => ControlResponse::ok(format!("{m}: group down; normal placement resumes")),
                        Err(e) => ControlResponse::err(e.to_string()),
                    },
                    ("status", _) => {
                        let mut r = ControlResponse::ok("groups");
                        r.groups = c.groups();
                        r.group_losses = c.group_losses();
                        r
                    }
                    _ => ControlResponse::err("usage: group up MODEL | group down MODEL | group status"),
                }
            }
            ControlRequest::Batch { prompts, model, concurrency, retries } => {
                let Some(pool) = &self.pool else { return ControlResponse::err("batch needs the app role") };
                let mut o = crate::batch::BatchOptions::new(format!(
                    "batch-{:x}",
                    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos())
                ));
                o.model = model;
                o.concurrency = concurrency.unwrap_or(o.concurrency);
                o.retries = retries.unwrap_or(o.retries);
                let items = crate::batch::run(pool.clone(), prompts, o, weirdnet_agent::CancelToken::new()).await;
                let failed = items.iter().filter(|i| !i.ok).count();
                let mut r = ControlResponse::ok(format!("{} task(s), {failed} failed", items.len()));
                r.batch = items;
                r
            }
            ControlRequest::BatchTasks { tasks, model, concurrency, retries, job, wait_for_model, cache } => {
                let Some(pool) = &self.pool else { return ControlResponse::err("batch needs the app role") };
                let job = job.unwrap_or_else(|| {
                    format!("batch-{:x}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()))
                });
                let mut o = crate::batch::BatchOptions::new(job);
                o.model = model;
                o.concurrency = concurrency.unwrap_or(o.concurrency);
                o.retries = retries.unwrap_or(o.retries);
                o.wait_for_model = wait_for_model;
                o.cache = cache;
                let items = crate::batch::run_tasks(pool.clone(), tasks, o, weirdnet_agent::CancelToken::new()).await;
                let failed = items.iter().filter(|i| !i.ok).count();
                let mut r = ControlResponse::ok(format!("{} task(s), {failed} failed", items.len()));
                r.batch = items;
                r
            }
            ControlRequest::PoolWake { device, reason } => {
                let Some((m, node)) = &self.network else { return ControlResponse::err("this node is not on Weirdnet") };
                let reason = match reason.as_str() {
                    "message" => weirdnet_push::Reason::MessageWaiting,
                    "connection" => weirdnet_push::Reason::ConnectionRequested,
                    other => return ControlResponse::err(format!("unknown wake reason {other:?} (message or connection)")),
                };
                let devices = crate::wake::wakeable(m);
                let Some(d) = crate::wake::find(&devices, &device) else {
                    return ControlResponse::err(format!("{device} shares no wake card with this node"));
                };
                match crate::wake::wake(node, d, reason).await {
                    Ok(weirdnet_push::relaywake::Delivery::Delivered) => ControlResponse::ok(format!("woke {}", d.label)),
                    Ok(other) => ControlResponse::err(format!("{}: the relay answered {other:?}", d.label)),
                    Err(e) => ControlResponse::err(format!("{e:#}")),
                }
            }
            ControlRequest::PoolRevoke { peer } => {
                let Some((m, _)) = &self.network else { return ControlResponse::err("this node is not on Weirdnet") };
                match m.revoke(&peer) {
                    Ok(c) => ControlResponse::ok(format!("{} ({}) is revoked", c.label, c.peer_id)),
                    Err(e) => ControlResponse::err(format!("{e:#}")),
                }
            }
            ControlRequest::PoolShare { grant } => {
                let Some((m, _)) = &self.network else { return ControlResponse::err("this node is not on Weirdnet") };
                match crate::guests::share(m, &grant) {
                    Ok(()) => ControlResponse::ok(format!("shared with {}", grant.user_id)),
                    Err(e) => ControlResponse::err(format!("{e:#}")),
                }
            }
            ControlRequest::PoolUnshare { user_id } => {
                let Some((m, _)) = &self.network else { return ControlResponse::err("this node is not on Weirdnet") };
                match crate::guests::unshare(m, &user_id, crate::train::now_ms()) {
                    Ok(g) => ControlResponse::ok(format!("{} is no longer shared with", g.user_id)),
                    Err(e) => ControlResponse::err(format!("{e:#}")),
                }
            }
        }
    }
}

/// The control endpoint `ymird` listens on.
#[cfg(unix)]
pub type Listener = tokio::net::UnixListener;

/// The control endpoint `ymird` listens on: the next free instance of its named pipe.
#[cfg(windows)]
pub struct Listener {
    name: String,
    next: tokio::net::windows::named_pipe::NamedPipeServer,
}

/// The local named pipe that stands in for the control socket at `socket` on Windows:
/// `\\.\pipe\ymird-<hash of the socket path>`, so every `YMIR_HOME` gets its own pipe and
/// `ymirctl` finds it from the same path. Defined on every OS so it can be tested anywhere.
pub fn pipe_name(socket: &Path) -> String {
    let abs = std::path::absolute(socket).unwrap_or_else(|_| socket.to_path_buf());
    let key = abs.to_string_lossy().replace('/', "\\").to_lowercase();
    // `canonicalize` on Windows gives `\\?\C:\...`: the same file as `C:\...`.
    let key = key.strip_prefix(r"\\?\").map(str::to_string).unwrap_or(key);
    // FNV-1a, 64 bit: stable across builds and platforms (std's hasher is not).
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in key.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!(r"\\.\pipe\ymird-{h:016x}")
}

/// Binds the socket, replacing a stale one. Fails if another `ymird` answers on it.
#[cfg(unix)]
pub async fn bind(path: &Path) -> Result<Listener> {
    use tokio::net::{UnixListener, UnixStream};
    if path.exists() {
        if UnixStream::connect(path).await.is_ok() {
            bail!("another ymird is already running (control socket {})", path.display());
        }
        std::fs::remove_file(path).with_context(|| format!("removing stale {}", path.display()))?;
    }
    let listener = UnixListener::bind(path).with_context(|| format!("binding {}", path.display()))?;
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(listener)
}

/// Creates the first instance of the control pipe. Fails if another `ymird` holds it.
#[cfg(windows)]
pub async fn bind(path: &Path) -> Result<Listener> {
    use tokio::net::windows::named_pipe::ServerOptions;
    let name = pipe_name(path);
    let next = ServerOptions::new()
        .first_pipe_instance(true)
        .reject_remote_clients(true)
        .create(&name)
        .map_err(|e| anyhow::anyhow!("another ymird is already running (control pipe {name}): {e}"))?;
    Ok(Listener { name, next })
}

#[cfg(unix)]
async fn accept(listener: &mut Listener) -> std::io::Result<tokio::net::UnixStream> {
    listener.accept().await.map(|(s, _)| s)
}

#[cfg(windows)]
async fn accept(listener: &mut Listener) -> std::io::Result<tokio::net::windows::named_pipe::NamedPipeServer> {
    use tokio::net::windows::named_pipe::ServerOptions;
    listener.next.connect().await?;
    let fresh = ServerOptions::new().reject_remote_clients(true).create(&listener.name)?;
    Ok(std::mem::replace(&mut listener.next, fresh))
}

pub async fn serve(mut listener: Listener, handler: Arc<ControlHandler>, mut shutdown: tokio::sync::watch::Receiver<bool>) {
    loop {
        let accepted = tokio::select! {
            accepted = accept(&mut listener) => accepted,
            _ = shutdown.wait_for(|stop| *stop) => return,
        };
        match accepted {
            Ok(stream) => {
                let h = handler.clone();
                tokio::spawn(async move {
                    if let Err(e) = connection(stream, h).await {
                        tracing::debug!("control connection: {e:#}");
                    }
                });
            }
            Err(e) => {
                tracing::debug!("control accept: {e}");
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        }
    }
}

async fn connection<S: AsyncRead + AsyncWrite + Send + 'static>(stream: S, handler: Arc<ControlHandler>) -> Result<()> {
    let (r, mut w) = tokio::io::split(stream);
    let mut lines = BufReader::new(r.take(64 * 1024)).lines();
    while let Some(line) = lines.next_line().await? {
        let resp = match serde_json::from_str::<ControlRequest>(&line) {
            Ok(ControlRequest::Models { node, request }) => {
                handler.stream_models(node, request, &mut w).await?;
                continue;
            }
            Ok(ControlRequest::Transcribe(job)) => {
                if job.bytes == 0 || job.bytes > MAX_TRANSCRIBE_BYTES {
                    // The audio that follows can't be skipped safely: answer and close.
                    let resp = ControlResponse::err(format!(
                        "a transcription request takes 1 to {MAX_TRANSCRIBE_BYTES} bytes of audio, not {} (longer audio: `ymirctl transcribe --chunk auto`)",
                        job.bytes
                    ));
                    let mut out = serde_json::to_vec(&resp)?;
                    out.push(b'\n');
                    w.write_all(&out).await?;
                    break;
                }
                let reader = lines.get_mut();
                let take = reader.get_mut();
                take.set_limit(take.limit().saturating_add(job.bytes));
                let mut audio = vec![0u8; usize::try_from(job.bytes)?];
                reader.read_exact(&mut audio).await?;
                handler.transcribe(job, audio).await
            }
            Ok(ControlRequest::Speak(job)) => {
                let total = match job.check_clips() {
                    Ok(t) => t,
                    Err(e) => {
                        // Clips that follow can't be skipped safely: answer and close.
                        let mut out = serde_json::to_vec(&ControlResponse::err(e))?;
                        out.push(b'\n');
                        w.write_all(&out).await?;
                        break;
                    }
                };
                let reader = lines.get_mut();
                let take = reader.get_mut();
                take.set_limit(take.limit().saturating_add(total));
                let mut clips = Vec::with_capacity(job.clips.len());
                for len in &job.clips {
                    let mut clip = vec![0u8; usize::try_from(*len)?];
                    reader.read_exact(&mut clip).await?;
                    clips.push(clip);
                }
                let (resp, audio) = handler.speak(job, clips).await;
                let mut out = serde_json::to_vec(&resp)?;
                out.push(b'\n');
                w.write_all(&out).await?;
                w.write_all(&audio).await?;
                w.flush().await?;
                continue;
            }
            Ok(ControlRequest::Train { node, request }) => {
                // The caller sends nothing more; its side closing ends the run.
                let closed = async { while let Ok(Some(_)) = lines.next_line().await {} };
                handler.stream_train(node, request, &mut w, closed).await?;
                break;
            }
            Ok(req) => handler.handle(req).await,
            Err(e) => ControlResponse::err(format!("bad request: {e}")),
        };
        let mut out = serde_json::to_vec(&resp)?;
        out.push(b'\n');
        w.write_all(&out).await?;
    }
    Ok(())
}

#[cfg(unix)]
async fn connect(socket: &Path) -> Result<tokio::net::UnixStream> {
    tokio::net::UnixStream::connect(socket).await.with_context(|| format!("ymird is not running (no control socket at {})", socket.display()))
}

#[cfg(windows)]
async fn connect(socket: &Path) -> Result<tokio::net::windows::named_pipe::NamedPipeClient> {
    use tokio::net::windows::named_pipe::ClientOptions;
    const ERROR_PIPE_BUSY: i32 = 231;
    let name = pipe_name(socket);
    let mut tries = 0;
    loop {
        match ClientOptions::new().open(&name) {
            Ok(c) => return Ok(c),
            Err(e) if e.raw_os_error() == Some(ERROR_PIPE_BUSY) && tries < 40 => {
                tries += 1;
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                bail!("ymird is not running (no control pipe {name} for {})", socket.display())
            }
            Err(e) => return Err(e).with_context(|| format!("opening control pipe {name} (an elevated prompt is needed for a system service)")),
        }
    }
}

async fn write_line<W: AsyncWrite + Unpin>(w: &mut W, line: &impl Serialize) -> Result<()> {
    let mut out = serde_json::to_vec(line)?;
    out.push(b'\n');
    w.write_all(&out).await?;
    Ok(())
}

/// Sends a models request to the `ymird` at `socket` and calls `on_line` for each progress line.
/// Returns the final answer.
pub async fn request_models(
    socket: &Path,
    node: &str,
    op: crate::models::ModelsOp,
    mut on_line: impl FnMut(&crate::models::ModelsLine),
) -> Result<crate::models::ModelsDone> {
    use crate::models::ModelsLine;
    let stream = connect(socket).await?;
    let (r, mut w) = tokio::io::split(stream);
    write_line(&mut w, &ControlRequest::Models { node: node.to_string(), request: op }).await?;
    let mut lines = BufReader::new(r).lines();
    while let Some(line) = lines.next_line().await? {
        match serde_json::from_str::<ModelsLine>(&line) {
            Ok(ModelsLine::Done(d)) => return Ok(*d),
            Ok(ModelsLine::Failed { message }) => bail!("{message}"),
            Ok(p) => on_line(&p),
            // An older ymird answers an unknown request with one `ControlResponse` error.
            Err(_) => match serde_json::from_str::<ControlResponse>(&line) {
                Ok(r) if !r.ok => bail!("this ymird does not manage models over Weirdnet (upgrade it): {}", r.message.unwrap_or_default()),
                _ => bail!("ymird sent something this ymirctl does not understand (are they the same version?)"),
            },
        }
    }
    bail!("ymird closed the control connection")
}

/// Sends a training request for another node to the `ymird` at `socket` and calls `on_line` for
/// each line until the run ends. Returns the run's record. Dropping the future cancels the run.
pub async fn request_train(
    socket: &Path,
    node: &str,
    op: crate::train_net::TrainOp,
    mut on_line: impl FnMut(&crate::train_net::TrainLine),
) -> Result<crate::train::RunRecord> {
    use crate::train_net::TrainLine;
    let stream = connect(socket).await?;
    let (r, mut w) = tokio::io::split(stream);
    write_line(&mut w, &ControlRequest::Train { node: node.to_string(), request: op }).await?;
    let mut lines = BufReader::new(r).lines();
    while let Some(line) = lines.next_line().await? {
        match serde_json::from_str::<TrainLine>(&line) {
            Ok(TrainLine::Done { record }) => return Ok(*record),
            Ok(TrainLine::Failed { message }) => bail!("{message}"),
            Ok(l) => on_line(&l),
            Err(_) => match serde_json::from_str::<ControlResponse>(&line) {
                Ok(r) if !r.ok => bail!("this ymird does not start runs on other nodes (upgrade it): {}", r.message.unwrap_or_default()),
                _ => bail!("ymird sent something this ymirctl does not understand (are they the same version?)"),
            },
        }
    }
    bail!("ymird closed the control connection")
}

/// Sends one request to the `ymird` at `socket`.
pub async fn request(socket: &Path, req: &ControlRequest) -> Result<ControlResponse> {
    let stream = connect(socket).await?;
    let (r, mut w) = tokio::io::split(stream);
    let mut line = serde_json::to_vec(req)?;
    line.push(b'\n');
    w.write_all(&line).await?;
    let mut lines = BufReader::new(r).lines();
    let resp = lines.next_line().await?.context("ymird closed the control connection")?;
    Ok(serde_json::from_str(&resp)?)
}

/// Transcribes `audio` through the `ymird` on `socket` ([`ControlRequest::Transcribe`]).
pub async fn transcribe(socket: &Path, mut job: TranscribeJob, audio: &[u8]) -> Result<TranscribeReply> {
    job.bytes = audio.len() as u64;
    if job.bytes > MAX_TRANSCRIBE_BYTES {
        bail!("the audio is {} bytes; one request takes at most {MAX_TRANSCRIBE_BYTES} (use --chunk auto)", job.bytes);
    }
    let stream = connect(socket).await?;
    let (r, mut w) = tokio::io::split(stream);
    let mut line = serde_json::to_vec(&ControlRequest::Transcribe(job))?;
    line.push(b'\n');
    w.write_all(&line).await?;
    w.write_all(audio).await?;
    w.flush().await?;
    let mut lines = BufReader::new(r).lines();
    let resp = lines.next_line().await?.context("ymird closed the control connection")?;
    let r: ControlResponse =
        serde_json::from_str(&resp).context("ymird sent something this ymirctl does not understand (are they the same version?)")?;
    if !r.ok {
        bail!("{}", r.message.unwrap_or_default());
    }
    r.transcript.context("ymird sent no transcript (is it older than this ymirctl?)")
}

/// Speech for `request` through the `ymird` on `socket` ([`ControlRequest::Speak`]): the answer
/// and the audio.
pub async fn speak(socket: &Path, request: weirdnet_agent::remote::infer::v2::SpeakRequest) -> Result<(SpeechReply, Vec<u8>)> {
    let (job, clips) = SpeakJob::split(request);
    job.check_clips().map_err(|e| anyhow::anyhow!(e))?;
    let stream = connect(socket).await?;
    let (r, mut w) = tokio::io::split(stream);
    let mut line = serde_json::to_vec(&ControlRequest::Speak(job))?;
    line.push(b'\n');
    w.write_all(&line).await?;
    for c in &clips {
        w.write_all(c).await?;
    }
    w.flush().await?;
    let mut reader = BufReader::new(r);
    let mut resp = String::new();
    if reader.read_line(&mut resp).await? == 0 {
        bail!("ymird closed the control connection");
    }
    let r: ControlResponse =
        serde_json::from_str(&resp).context("ymird sent something this ymirctl does not understand (are they the same version?)")?;
    if !r.ok {
        bail!("{}", r.message.unwrap_or_default());
    }
    let reply = r.speech.context("ymird sent no speech (is it older than this ymirctl?)")?;
    let mut audio = vec![0u8; usize::try_from(reply.bytes)?];
    reader.read_exact(&mut audio).await.context("ymird closed the connection before all the audio came")?;
    Ok((reply, audio))
}

/// The speaker the pool would use for `request` (its model, or the pool's pick), with its profile.
pub async fn speak_target(socket: &Path, speech: &weirdnet_agent::remote::infer::v2::SpeakRequest) -> Result<Speaker> {
    let (job, _) = SpeakJob::split(speech.clone());
    let r = request(socket, &ControlRequest::SpeakTarget(job)).await?;
    if !r.ok {
        bail!("{}", r.message.unwrap_or_default());
    }
    r.speaker.context("ymird sent no speaker (is it older than this ymirctl?)")
}

/// The transcriber the pool would use for `model` (or its pick), with its audio limits.
pub async fn transcribe_target(socket: &Path, model: Option<&str>) -> Result<Transcriber> {
    let r = request(socket, &ControlRequest::TranscribeTarget { model: model.map(str::to_string) }).await?;
    if !r.ok {
        bail!("{}", r.message.unwrap_or_default());
    }
    r.transcriber.context("ymird sent no transcriber (is it older than this ymirctl?)")
}

/// Whether a `ymird` answers on `socket`.
pub async fn reachable(socket: &Path) -> bool {
    request(socket, &ControlRequest::Status).await.is_ok()
}

fn no_context() -> ControlResponse {
    ControlResponse::err("this node keeps no conversations: it needs the inference or app role and a device identity (see `ymirctl pool link`)")
}

#[cfg(test)]
mod speak_tests {
    use super::*;
    use weirdnet_agent::remote::infer::v2::{ScriptTurn, SpeakRequest, Voice};

    fn clone(n: usize) -> Voice {
        Voice::clone_from(vec![7; n], "wav", None, true)
    }

    #[test]
    fn speak_jobs_carry_clips_beside_the_line() {
        let req = SpeakRequest::script(vec![ScriptTurn::new(0, "a"), ScriptTurn::new(1, "b")])
            .with_voice(clone(3))
            .with_speaker_voice(0, Voice::preset("vivian"))
            .with_speaker_voice(1, clone(5));
        let (job, clips) = SpeakJob::split(req.clone());
        assert_eq!(job.clips, [3, 5]);
        assert_eq!(clips, [vec![7; 3], vec![7; 5]]);
        // The line carries no audio.
        let line = serde_json::to_string(&ControlRequest::Speak(job.clone())).unwrap();
        assert!(line.len() < 1024, "{line}");
        assert_eq!(job.check_clips(), Ok(8));
        assert_eq!(job.clone().join(clips.clone()).unwrap(), req);
        assert!(job.clone().join(clips[..1].to_vec()).is_err());
        assert!(job.clone().join([clips.clone(), vec![vec![1]]].concat()).is_err());

        // Lengths that don't match the voices, or out of range, are refused before reading.
        let mut bad = job.clone();
        bad.clips.pop();
        assert!(bad.check_clips().is_err());
        let mut bad = job.clone();
        bad.clips[0] = 0;
        assert!(bad.check_clips().is_err());
        let mut bad = job;
        bad.clips[0] = weirdnet_agent::remote::infer::v2::MAX_REFERENCE_BYTES as u64 + 1;
        assert!(bad.check_clips().is_err());
    }
}
