//! The Hugging Face Hub, for `ymirctl models`: search, a repo's files, a GGUF header and a
//! resumable, verified download with an optional access token.
//!
//! This is interim code: `weirdnet-model` is where it belongs (it has `hf::repo_files` and
//! `download`, but no search, token, split-file or header support in 0.4.3). Everything the rest
//! of `models` needs from the Hub goes through [`Hub`], so swapping it is one file.

use std::fmt;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::gguf::{self, GgufMeta};
use super::plan::RepoFile;

/// How much of a file to read when looking for its header (the architecture keys come first).
const HEADER_BYTES: u64 = 8 << 20;

/// A Hub access token. It prints as `<token>` and is only ever sent to the Hub itself.
#[derive(Clone)]
pub struct Token(String);

impl fmt::Debug for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Token(<redacted>)")
    }
}

impl Token {
    pub fn new(s: &str) -> Option<Self> {
        let s = s.trim();
        (!s.is_empty() && !s.contains(char::is_whitespace)).then(|| Self(s.to_string()))
    }
}

/// Where a token came from, for messages (never the token itself).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenSource {
    Env(&'static str),
    File(PathBuf),
}

/// `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, a file named by `YMIR_HF_TOKEN_FILE`, `<ymir home>/state/hf-token`,
/// then Hugging Face's own `<HF_HOME or ~/.cache/huggingface>/token`. Only this node is read.
pub fn find_token(state_dir: &Path) -> Option<(Token, TokenSource)> {
    for name in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Some(t) = std::env::var(name).ok().and_then(|v| Token::new(&v)) {
            return Some((t, TokenSource::Env(name)));
        }
    }
    let mut files: Vec<PathBuf> = Vec::new();
    if let Some(p) = std::env::var_os("YMIR_HF_TOKEN_FILE").filter(|p| !p.is_empty()) {
        files.push(p.into());
    }
    files.push(state_dir.join("hf-token"));
    let hf_home =
        std::env::var_os("HF_HOME").filter(|p| !p.is_empty()).map(PathBuf::from).or_else(|| dirs::cache_dir().map(|c| c.join("huggingface")));
    files.extend(hf_home.map(|h| h.join("token")));
    files.into_iter().find_map(|p| {
        let t = Token::new(&std::fs::read_to_string(&p).ok()?)?;
        Some((t, TokenSource::File(p)))
    })
}

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

pub struct Hub {
    base: String,
    token: Option<Token>,
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

impl Hub {
    pub fn new(base: &str, token: Option<Token>) -> Self {
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
            req = req.header("Authorization", format!("Bearer {}", t.0));
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

    pub fn repo(&self, repo: &str) -> Result<RepoInfo> {
        weirdnet_model::hf::check_repo(repo).map_err(|_| anyhow::anyhow!("`{repo}` is not a repo id (use owner/name)"))?;
        let mut resp = self.get(&format!("{}/api/models/{repo}?blobs=true", self.base), None)?;
        self.ok(&resp, repo)?;
        let text = resp.body_mut().read_to_string().context("reading the repo answer")?;
        Ok(parse_info(repo, &serde_json::from_str(&text).context("the Hub's repo answer is not JSON")?))
    }

    pub fn file_url(&self, repo: &str, file: &str) -> String {
        format!("{}/{repo}/resolve/main/{}", self.base, encode(file, true))
    }

    /// The attention layout in a GGUF file's header, read from its first bytes. `None` when the
    /// header can't be read (the caller falls back to an estimate).
    pub fn gguf_meta(&self, repo: &str, file: &str) -> Option<GgufMeta> {
        let mut resp = self.get(&self.file_url(repo, file), Some(format!("bytes=0-{}", HEADER_BYTES - 1))).ok()?;
        if !matches!(resp.status().as_u16(), 200 | 206) {
            return None;
        }
        let mut body = resp.body_mut().as_reader().take(HEADER_BYTES);
        gguf::read_meta(&mut body).ok().filter(|m| m.kv_bytes_per_token().is_some())
    }

    /// Downloads one file to `dest`: resumes `dest.part`, checks the size and, when known, the
    /// SHA-256, then renames into place. `progress(done, total)` is called as bytes arrive.
    /// Returns `false` when the file was already there and verified.
    pub fn download(&self, url: &str, dest: &Path, total: u64, sha256: Option<&str>, progress: &mut dyn FnMut(u64, u64)) -> Result<bool> {
        if let Some(dir) = dest.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        if total > 0 && std::fs::metadata(dest).is_ok_and(|m| m.len() == total) {
            match sha256 {
                Some(want) if !sha256_of(dest)?.eq_ignore_ascii_case(want) => {
                    bail!("{} exists with the right size but a different checksum; remove it first (`ymirctl models remove`)", dest.display())
                }
                _ => return Ok(false),
            }
        }
        if dest.exists() {
            bail!("{} already exists with a different size; remove it first (`ymirctl models remove`)", dest.display());
        }
        let part = PathBuf::from(format!("{}.part", dest.display()));
        let mut have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        if total > 0 && have > total {
            std::fs::remove_file(&part)?;
            have = 0;
        }
        let mut done = have;
        if total == 0 || have < total {
            let mut resp = self.get(url, (have > 0).then(|| format!("bytes={have}-")))?;
            self.ok(&resp, &format!("downloading {}", dest.file_name().map(|n| n.to_string_lossy()).unwrap_or_default()))?;
            if resp.status().as_u16() == 200 {
                done = 0;
            }
            let mut file = std::fs::OpenOptions::new().create(true).write(true).append(done > 0).truncate(done == 0).open(&part)?;
            let mut body = resp.body_mut().as_reader();
            let mut buf = vec![0u8; 1 << 16];
            loop {
                progress(done, total);
                let n = body.read(&mut buf).context("the download was interrupted (run the same command to resume)")?;
                if n == 0 {
                    break;
                }
                file.write_all(&buf[..n])?;
                done += n as u64;
            }
            file.flush()?;
        }
        if total > 0 && done != total {
            if done > total {
                let _ = std::fs::remove_file(&part);
            }
            bail!("the download stopped at {done} of {total} bytes (run the same command to resume)");
        }
        if let Some(want) = sha256 {
            let got = sha256_of(&part)?;
            if !got.eq_ignore_ascii_case(want) {
                let _ = std::fs::remove_file(&part);
                bail!("checksum mismatch for {}: got {got}, expected {want}; the partial file was deleted", dest.display());
            }
        }
        std::fs::rename(&part, dest)?;
        progress(done, done);
        Ok(true)
    }
}

pub fn sha256_of(path: &Path) -> Result<String> {
    let mut h = Sha256::new();
    std::io::copy(&mut std::fs::File::open(path)?, &mut h)?;
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Free bytes on the disk holding `dir` (which must exist), or `None` where that isn't known.
pub fn free_bytes(dir: &Path) -> Option<u64> {
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
        Some(u64::from(s.f_bavail).saturating_mul(u64::from(s.f_frsize)))
    }
    #[cfg(not(unix))]
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
            (s.routes.get(&path).cloned(), s.cut_after)
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

    fn data(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i % 251) as u8).collect()
    }

    fn sha(b: &[u8]) -> String {
        Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
    }

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
    fn the_token_never_prints() {
        let t = Token::new("  hf_secretvalue \n").unwrap();
        assert!(!format!("{t:?}").contains("secret"));
        assert!(Token::new("two words").is_none() && Token::new("   ").is_none());
    }

    #[test]
    fn searches_with_an_encoded_query_and_sends_the_token_only_as_a_header() {
        let m = MockHub::start();
        m.route("/api/models?search=qwen3%200.6b&filter=gguf&sort=downloads&direction=-1&limit=3&expand[]=downloads&expand[]=likes&expand[]=gated&expand[]=gguf&expand[]=safetensors", 200, br#"[{"id":"a/b","downloads":1}]"#.to_vec());
        let hub = Hub::new(&m.url, Token::new("hf_abc"));
        assert_eq!(hub.search("qwen3 0.6b", "gguf", 3).unwrap()[0].id, "a/b");
        let s = m.state.lock().unwrap();
        assert_eq!(s.auth[0].as_deref(), Some("Bearer hf_abc"));
        assert!(!s.paths[0].contains("hf_abc"));
    }

    #[test]
    fn hub_errors_are_actionable_and_never_echo_the_token() {
        let m = MockHub::start();
        m.route("/api/models/a/gated?blobs=true", 401, "x");
        m.route("/api/models/a/limited?blobs=true", 429, "x");
        let err = Hub::new(&m.url, None).repo("a/gated").unwrap_err().to_string();
        assert!(err.contains("HF_TOKEN"), "{err}");
        let err = Hub::new(&m.url, Token::new("hf_topsecret")).repo("a/gated").unwrap_err().to_string();
        assert!(err.contains("accept the repo's terms") && !err.contains("topsecret"), "{err}");
        assert!(Hub::new(&m.url, None).repo("a/limited").unwrap_err().to_string().contains("rate limiting"));
        assert!(Hub::new(&m.url, None).repo("a/missing").unwrap_err().to_string().contains("not found"));
        assert!(Hub::new(&m.url, None).repo("../etc").unwrap_err().to_string().contains("not a repo id"));
    }

    #[test]
    fn reads_the_gguf_header_with_a_range_request() {
        let m = MockHub::start();
        let mut file = crate::models::gguf::sample_header(32, 8);
        file.extend(data(1000));
        m.route("/a/b/resolve/main/sub/x.gguf", 200, file);
        let meta = Hub::new(&m.url, None).gguf_meta("a/b", "sub/x.gguf").unwrap();
        assert_eq!(meta.layers, 32);
        assert!(Hub::new(&m.url, None).gguf_meta("a/b", "missing.gguf").is_none());
    }

    #[test]
    fn downloads_verifies_and_skips_a_complete_file() {
        let m = MockHub::start();
        let body = data(300_000);
        m.route("/a/b/resolve/main/x.gguf", 200, body.clone());
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.gguf");
        let hub = Hub::new(&m.url, None);
        let url = hub.file_url("a/b", "x.gguf");
        let mut last = (0, 0);
        assert!(hub.download(&url, &dest, body.len() as u64, Some(&sha(&body)), &mut |d, t| last = (d, t)).unwrap());
        assert_eq!(last, (300_000, 300_000));
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert!(!dir.path().join("x.gguf.part").exists());
        let before = m.state.lock().unwrap().paths.len();
        assert!(!hub.download(&url, &dest, body.len() as u64, Some(&sha(&body)), &mut |_, _| {}).unwrap());
        assert_eq!(m.state.lock().unwrap().paths.len(), before, "no request for a file that is already there");
    }

    #[test]
    fn resumes_a_partial_download() {
        let m = MockHub::start();
        let body = data(200_000);
        m.route("/a/b/resolve/main/x.gguf", 200, body.clone());
        m.state.lock().unwrap().cut_after = Some(70_000);
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.gguf");
        let hub = Hub::new(&m.url, None);
        let url = hub.file_url("a/b", "x.gguf");
        assert!(hub.download(&url, &dest, 200_000, Some(&sha(&body)), &mut |_, _| {}).is_err());
        assert!(!dest.exists());
        assert_eq!(std::fs::metadata(dir.path().join("x.gguf.part")).unwrap().len(), 70_000);
        m.state.lock().unwrap().cut_after = None;
        assert!(hub.download(&url, &dest, 200_000, Some(&sha(&body)), &mut |_, _| {}).unwrap());
        assert_eq!(std::fs::read(&dest).unwrap(), body);
    }

    #[test]
    fn a_checksum_mismatch_deletes_the_partial_file() {
        let m = MockHub::start();
        m.route("/a/b/resolve/main/x.gguf", 200, data(1000));
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("x.gguf");
        let err = Hub::new(&m.url, None)
            .download(&Hub::new(&m.url, None).file_url("a/b", "x.gguf"), &dest, 1000, Some(&"0".repeat(64)), &mut |_, _| {})
            .unwrap_err();
        assert!(err.to_string().contains("checksum mismatch"));
        assert!(!dest.exists() && !dir.path().join("x.gguf.part").exists());
    }

    #[test]
    fn finds_a_token_in_a_file_and_prefers_the_environment() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("hf-token"), "hf_fromfile\n").unwrap();
        // The environment is process-wide, so only the file path is exercised here; env wins by order.
        if std::env::var_os("HF_TOKEN").is_none()
            && std::env::var_os("HUGGING_FACE_HUB_TOKEN").is_none()
            && std::env::var_os("YMIR_HF_TOKEN_FILE").is_none()
        {
            let (t, src) = find_token(dir.path()).unwrap();
            assert_eq!(src, TokenSource::File(dir.path().join("hf-token")));
            assert_eq!(t.0, "hf_fromfile");
        }
    }

    #[test]
    fn reports_free_space() {
        assert!(free_bytes(Path::new(".")).is_some_and(|b| b > 0));
    }
}
