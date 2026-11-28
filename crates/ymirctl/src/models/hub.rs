//! What `ymirctl models` still needs from the Hugging Face Hub itself: MLX repos (the
//! `weirdnet-model` Hub reads GGUF only), the token lookup (Ymir's own token file sources, handed to
//! `weirdnet-model`'s `ModelStore` for every GGUF and MLX download) and a free-space check.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde_json::Value;
use weirdnet_model::hf::RepoFile;

/// One search hit.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub id: String,
    pub downloads: u64,
    pub likes: u64,
    /// `false`, or `auto`/`manual`: a gated repo needs a token that has accepted its terms.
    pub gated: bool,
    /// Parameter count when the Hub reports it.
    pub params: Option<u64>,
}

/// A repo's metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct RepoInfo {
    pub id: String,
    pub gated: bool,
    pub files: Vec<RepoFile>,
    pub tags: Vec<String>,
}

pub struct MlxHub {
    base: String,
    /// Only ever sent to the Hub itself, never printed.
    token: Option<String>,
    agent: ureq::Agent,
}

fn encode(s: &str, keep_slash: bool) -> String {
    let mut o = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) || (keep_slash && b == b'/') {
            o.push(b as char);
        } else {
            o.push_str(&format!("%{b:02X}"));
        }
    }
    o
}

pub fn parse_hits(v: &Value) -> Vec<Hit> {
    v.as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| {
            Some(Hit {
                id: m["id"].as_str()?.to_string(),
                downloads: m["downloads"].as_u64().unwrap_or(0),
                likes: m["likes"].as_u64().unwrap_or(0),
                gated: !matches!(m["gated"], Value::Bool(false) | Value::Null),
                params: m["gguf"]["total"].as_u64().or_else(|| m["safetensors"]["total"].as_u64()),
            })
        })
        .collect()
}

pub fn parse_info(id: &str, v: &Value) -> RepoInfo {
    let files = v["siblings"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|s| {
            Some(RepoFile {
                name: s["rfilename"].as_str()?.to_string(),
                size: s["size"].as_u64().unwrap_or(0),
                sha256: s["lfs"]["sha256"].as_str().map(str::to_string),
            })
        })
        .collect();
    RepoInfo {
        id: id.to_string(),
        gated: !matches!(v["gated"], Value::Bool(false) | Value::Null),
        files,
        tags: v["tags"].as_array().into_iter().flatten().filter_map(|t| t.as_str().map(str::to_string)).collect(),
    }
}

impl MlxHub {
    pub fn new(base: &str, token: Option<String>) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder().http_status_as_error(false).build().into();
        Self { base: base.trim_end_matches('/').to_string(), token, agent }
    }

    pub fn has_token(&self) -> bool {
        self.token.is_some()
    }

    fn get(&self, url: &str, range: Option<String>) -> Result<ureq::http::Response<ureq::Body>> {
        let mut req = self.agent.get(url).header("User-Agent", concat!("ymirctl/", env!("CARGO_PKG_VERSION")));
        // The token goes to the Hub only; ureq does not forward it across a redirect to a CDN.
        if let Some(t) = self.token.as_ref().filter(|_| url.starts_with(&self.base)) {
            req = req.header("Authorization", format!("Bearer {t}"));
        }
        if let Some(r) = range {
            req = req.header("Range", r);
        }
        req.call().map_err(|e| anyhow::anyhow!("the Hub could not be reached: {e}"))
    }

    fn ok(&self, resp: &ureq::http::Response<ureq::Body>, what: &str) -> Result<()> {
        match resp.status().as_u16() {
            200 | 206 => Ok(()),
            401 | 403 if self.token.is_none() => bail!(
                "{what}: the Hub refused access (a gated or private repo). Set HF_TOKEN, or put a token in the file named by YMIR_HF_TOKEN_FILE"
            ),
            401 | 403 => {
                bail!("{what}: the Hub refused access with this token (accept the repo's terms on huggingface.co first, or check the token)")
            }
            404 => bail!("{what}: not found on the Hub"),
            429 => bail!("{what}: the Hub is rate limiting this machine; wait a minute and try again (a token raises the limit)"),
            s => bail!("{what}: the Hub answered {s}"),
        }
    }

    /// Repos matching `query`, most downloaded first. `library` is the Hub filter: `gguf` or `mlx`.
    pub fn search(&self, query: &str, library: &str, limit: usize) -> Result<Vec<Hit>> {
        let url = format!(
            "{}/api/models?search={}&filter={library}&sort=downloads&direction=-1&limit={limit}&expand[]=downloads&expand[]=likes&expand[]=gated&expand[]=gguf&expand[]=safetensors",
            self.base,
            encode(query, false)
        );
        let mut resp = self.get(&url, None)?;
        self.ok(&resp, "search")?;
        let text = resp.body_mut().read_to_string().context("reading the search answer")?;
        Ok(parse_hits(&serde_json::from_str(&text).context("the Hub's search answer is not JSON")?))
    }

    /// One small file of a repo (such as `config.json`) at `main`.
    pub fn file(&self, repo: &str, name: &str) -> Result<Vec<u8>> {
        weirdnet_model::hf::check_repo(repo).map_err(|_| anyhow::anyhow!("`{repo}` is not a repo id (use owner/name)"))?;
        let mut resp = self.get(&format!("{}/{repo}/resolve/main/{name}", self.base), None)?;
        self.ok(&resp, name)?;
        resp.body_mut().read_to_vec().with_context(|| format!("reading {name}"))
    }

    /// The license a repo's model card states ([`ymir_core::dataset::hub_license`]).
    pub fn license(&self, repo: &str) -> Result<Option<String>> {
        weirdnet_model::hf::check_repo(repo).map_err(|_| anyhow::anyhow!("`{repo}` is not a repo id (use owner/name)"))?;
        let mut resp = self.get(&format!("{}/api/models/{repo}", self.base), None)?;
        self.ok(&resp, repo)?;
        let text = resp.body_mut().read_to_string().context("reading the repo answer")?;
        Ok(ymir_core::dataset::hub_license(&serde_json::from_str(&text).context("the Hub's repo answer is not JSON")?))
    }

    pub fn repo(&self, repo: &str) -> Result<RepoInfo> {
        weirdnet_model::hf::check_repo(repo).map_err(|_| anyhow::anyhow!("`{repo}` is not a repo id (use owner/name)"))?;
        let mut resp = self.get(&format!("{}/api/models/{repo}?blobs=true", self.base), None)?;
        self.ok(&resp, repo)?;
        let text = resp.body_mut().read_to_string().context("reading the repo answer")?;
        Ok(parse_info(repo, &serde_json::from_str(&text).context("the Hub's repo answer is not JSON")?))
    }
}

/// Free bytes on the disk holding `dir` (which must exist), or `None` where that isn't known.
pub fn free_bytes(dir: &Path) -> Option<u64> {
    disk_space(dir).map(|(free, _)| free)
}

/// Free and total bytes on the disk holding `dir` (which must exist), or `None` where that isn't
/// known. Free is what an unprivileged user may still write.
pub fn disk_space(dir: &Path) -> Option<(u64, u64)> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let c = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
        let mut s = std::mem::MaybeUninit::<libc::statvfs>::uninit();
        // SAFETY: `c` is a valid NUL-terminated path and `s` is a valid out pointer.
        let rc = unsafe { libc::statvfs(c.as_ptr(), s.as_mut_ptr()) };
        if rc != 0 {
            return None;
        }
        // SAFETY: statvfs returned 0, so it filled `s`.
        let s = unsafe { s.assume_init() };
        // The field types differ by platform (u32 on macOS), so the conversion is needed there.
        #[allow(clippy::useless_conversion)]
        let frsize = u64::from(s.f_frsize);
        #[allow(clippy::useless_conversion)]
        Some((u64::from(s.f_bavail).saturating_mul(frsize), u64::from(s.f_blocks).saturating_mul(frsize)))
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        // kernel32 is always linked on Windows, so no extra crate is needed for this one call.
        unsafe extern "system" {
            fn GetDiskFreeSpaceExW(dir: *const u16, available: *mut u64, total: *mut u64, free: *mut u64) -> i32;
        }
        let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        let (mut available, mut total) = (0u64, 0u64);
        // SAFETY: `wide` is a NUL-terminated UTF-16 path; `available` and `total` are valid out pointers and the last one is null.
        let ok = unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &mut available, &mut total, std::ptr::null_mut()) };
        (ok != 0).then_some((available, total))
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = dir;
        None
    }
}

#[cfg(test)]
pub mod mock {
    //! A tiny in-process Hub: JSON routes, files with `Range` support, and a log of the
    //! `Authorization` headers it saw.
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    /// A minimal GGUF header for tests: Llama-like architecture keys, then a tokenizer key.
    pub fn sample_header(layers: u32, kv_heads: u32) -> Vec<u8> {
        fn s(out: &mut Vec<u8>, v: &str) {
            out.extend((v.len() as u64).to_le_bytes());
            out.extend(v.as_bytes());
        }
        let mut o = b"GGUF".to_vec();
        o.extend(3u32.to_le_bytes());
        o.extend(0u64.to_le_bytes());
        o.extend(8u64.to_le_bytes());
        s(&mut o, "general.architecture");
        o.extend(8u32.to_le_bytes());
        s(&mut o, "llama");
        s(&mut o, "general.name");
        o.extend(8u32.to_le_bytes());
        s(&mut o, "Tiny");
        for (k, v) in [
            ("llama.block_count", layers),
            ("llama.context_length", 131_072),
            ("llama.embedding_length", 4096),
            ("llama.attention.head_count", 32),
            ("llama.attention.head_count_kv", kv_heads),
        ] {
            s(&mut o, k);
            o.extend(4u32.to_le_bytes());
            o.extend(v.to_le_bytes());
        }
        s(&mut o, "tokenizer.ggml.tokens");
        o.extend(9u32.to_le_bytes());
        o.extend(8u32.to_le_bytes());
        o.extend(2u64.to_le_bytes());
        s(&mut o, "a");
        s(&mut o, "b");
        o
    }

    /// A minimal GGUF header for tests: Llama-like architecture keys, then a tokenizer key.

    #[derive(Default)]
    pub struct State {
        pub routes: HashMap<String, (u16, Vec<u8>)>,
        pub auth: Vec<Option<String>>,
        pub paths: Vec<String>,
        /// Serve at most this many body bytes per request (to simulate a dropped connection).
        pub cut_after: Option<usize>,
    }

    pub struct MockHub {
        pub url: String,
        pub state: Arc<Mutex<State>>,
    }

    impl MockHub {
        pub fn start() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let state = Arc::new(Mutex::new(State::default()));
            let st = state.clone();
            std::thread::spawn(move || {
                for conn in listener.incoming().flatten() {
                    let st = st.clone();
                    std::thread::spawn(move || serve(conn, &st));
                }
            });
            Self { url, state }
        }

        pub fn route(&self, path: &str, status: u16, body: impl Into<Vec<u8>>) {
            self.state.lock().unwrap().routes.insert(path.to_string(), (status, body.into()));
        }
    }

    fn serve(conn: std::net::TcpStream, st: &Mutex<State>) {
        let mut r = BufReader::new(conn.try_clone().unwrap());
        let mut line = String::new();
        if r.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
        let (mut auth, mut range) = (None, None);
        loop {
            let mut h = String::new();
            if r.read_line(&mut h).unwrap_or(0) == 0 || h.trim().is_empty() {
                break;
            }
            let lower = h.to_ascii_lowercase();
            if lower.starts_with("authorization:") {
                auth = Some(h.split_once(':').unwrap().1.trim().to_string());
            }
            if let Some(v) = lower.strip_prefix("range:") {
                range =
                    v.trim().strip_prefix("bytes=").and_then(|v| v.split_once('-').map(|(a, b)| (a.parse::<usize>().ok(), b.parse::<usize>().ok())));
            }
        }
        let (route, cut) = {
            let mut s = st.lock().unwrap();
            s.auth.push(auth);
            s.paths.push(path.clone());
            (s.routes.get(&path).or_else(|| s.routes.get(path.split('?').next().unwrap_or(""))).cloned(), s.cut_after)
        };
        let mut w = conn;
        let Some((status, body)) = route else {
            let _ = w.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            return;
        };
        let (code, slice) = match range {
            Some((Some(a), end)) if status == 200 && a < body.len() => (206, &body[a..end.map_or(body.len(), |e| (e + 1).min(body.len()))]),
            _ => (status, &body[..]),
        };
        let reason = if code == 206 { "Partial Content" } else { "OK" };
        let _ = write!(w, "HTTP/1.1 {code} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", slice.len());
        let _ = w.write_all(&slice[..cut.map_or(slice.len(), |c| c.min(slice.len()))]);
    }
}

#[cfg(test)]
mod tests {
    use super::mock::MockHub;
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_search_hits_and_repo_files() {
        let hits = parse_hits(&json!([
            {"id": "a/b-GGUF", "downloads": 10, "likes": 2, "gated": false, "gguf": {"total": 8_000_000_000u64}},
            {"id": "c/d", "gated": "auto"},
            {"nothing": 1}
        ]));
        assert_eq!(hits.len(), 2);
        assert_eq!((hits[0].downloads, hits[0].params, hits[0].gated), (10, Some(8_000_000_000), false));
        assert!(hits[1].gated);
        let info = parse_info(
            "a/b",
            &json!({"gated": false, "tags": ["gguf"], "siblings": [{"rfilename": "x.gguf", "size": 5, "lfs": {"sha256": "ab"}}, {"rfilename": "README.md"}]}),
        );
        assert_eq!(info.files.len(), 2);
        assert_eq!(info.files[0].sha256.as_deref(), Some("ab"));
    }

    #[test]
    fn searches_with_an_encoded_query_and_sends_the_token_only_as_a_header() {
        let m = MockHub::start();
        m.route("/api/models?search=qwen3%200.6b&filter=mlx&sort=downloads&direction=-1&limit=3&expand[]=downloads&expand[]=likes&expand[]=gated&expand[]=gguf&expand[]=safetensors", 200, br#"[{"id":"a/b","downloads":1}]"#.to_vec());
        let hub = MlxHub::new(&m.url, Some("hf_abc".into()));
        assert_eq!(hub.search("qwen3 0.6b", "mlx", 3).unwrap()[0].id, "a/b");
        let s = m.state.lock().unwrap();
        assert_eq!(s.auth[0].as_deref(), Some("Bearer hf_abc"));
        assert!(!s.paths[0].contains("hf_abc"));
    }

    #[test]
    fn hub_errors_are_actionable_and_never_echo_the_token() {
        let m = MockHub::start();
        m.route("/api/models/a/gated?blobs=true", 401, "x");
        m.route("/api/models/a/limited?blobs=true", 429, "x");
        let err = MlxHub::new(&m.url, None).repo("a/gated").unwrap_err().to_string();
        assert!(err.contains("HF_TOKEN"), "{err}");
        let err = MlxHub::new(&m.url, Some("hf_topsecret".into())).repo("a/gated").unwrap_err().to_string();
        assert!(err.contains("accept the repo's terms") && !err.contains("topsecret"), "{err}");
        assert!(MlxHub::new(&m.url, None).repo("a/limited").unwrap_err().to_string().contains("rate limiting"));
        assert!(MlxHub::new(&m.url, None).repo("a/missing").unwrap_err().to_string().contains("not found"));
        assert!(MlxHub::new(&m.url, None).repo("../etc").unwrap_err().to_string().contains("not a repo id"));
    }

    #[test]
    fn reads_a_repo_license() {
        let hub = MockHub::start();
        hub.route("/api/models/acme/Teacher", 200, r#"{"cardData": {"license": "apache-2.0"}, "tags": ["license:apache-2.0"]}"#);
        hub.route("/api/models/acme/Bare", 200, r#"{"tags": ["gguf"]}"#);
        let h = MlxHub::new(&hub.url, None);
        assert_eq!(h.license("acme/Teacher").unwrap().as_deref(), Some("apache-2.0"));
        assert_eq!(h.license("acme/Bare").unwrap(), None);
        assert!(h.license("acme/Missing").is_err());
        assert!(h.license("not a repo").is_err());
    }

    #[test]
    fn reports_free_space() {
        assert!(free_bytes(Path::new(".")).is_some_and(|b| b > 0));
    }
}
