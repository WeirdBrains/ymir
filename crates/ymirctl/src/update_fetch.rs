//! `ymirctl update fetch`: a bundle directory over HTTPS (or on disk) as an update [`Source`].

use std::io::Read;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use ymir_core::update::{DirSource, Source};

/// A bundle directory served over HTTPS: `BASE/manifest.env`, `BASE/<file path>`, ...
pub struct HttpSource {
    base: String,
    agent: ureq::Agent,
}

impl HttpSource {
    /// `https://` anywhere; `http://` only to this machine (a local mirror, or a test).
    pub fn new(base: &str) -> Result<Self> {
        let loopback = ["http://127.0.0.1", "http://localhost", "http://[::1]"]
            .iter()
            .any(|p| base.strip_prefix(p).is_some_and(|rest| rest.is_empty() || rest.starts_with([':', '/'])));
        if !base.starts_with("https://") && !loopback {
            bail!("an update source must be an https:// URL (http:// only to 127.0.0.1 or localhost), or a directory");
        }
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(30)))
            .timeout_recv_response(Some(Duration::from_secs(60)))
            .build()
            .into();
        Ok(Self { base: base.trim_end_matches('/').to_string(), agent })
    }
}

/// Path segments as URL path characters (manifest paths are already checked: relative, `/`
/// separated, no `..`).
fn encode(path: &str) -> String {
    let mut s = String::new();
    for b in path.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~/+".contains(&b) {
            s.push(b as char);
        } else {
            s.push_str(&format!("%{b:02X}"));
        }
    }
    s
}

impl Source for HttpSource {
    fn open(&self, path: &str) -> Result<Option<Box<dyn Read + '_>>> {
        let url = format!("{}/{}", self.base, encode(path));
        let resp =
            self.agent.get(&url).header("User-Agent", concat!("ymirctl/", env!("CARGO_PKG_VERSION"))).call().map_err(|e| anyhow!("{url}: {e}"))?;
        match resp.status().as_u16() {
            200 => Ok(Some(Box::new(resp.into_body().into_reader()))),
            404 | 410 => Ok(None),
            s => bail!("{url}: the server answered {s}"),
        }
    }

    fn describe(&self) -> String {
        self.base.clone()
    }
}

/// The source for `update fetch SOURCE`: a URL or a directory.
pub fn source(s: &str) -> Result<Box<dyn Source>> {
    if s.contains("://") {
        return Ok(Box::new(HttpSource::new(s)?));
    }
    let dir = PathBuf::from(s);
    if !dir.is_dir() {
        bail!("{s} is neither a URL nor a directory");
    }
    Ok(Box::new(DirSource(dir)))
}
