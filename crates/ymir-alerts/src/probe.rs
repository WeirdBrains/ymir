//! Health probes that turn into [`Observation`]s.

use std::path::Path;

use serde_json::json;

use crate::Observation;

/// Classifies `ip route get 1.1.1.1` output the way server-a/server-b's `route-state` does: the
/// default route's device on Wi-Fi is `degraded`, no device is `noroute`, else `ok`.
pub fn classify_route(ip_route_output: &str, is_wireless: impl Fn(&str) -> bool) -> Observation {
    let mut words = ip_route_output.split_whitespace();
    let dev = std::iter::from_fn(|| words.next()).skip_while(|w| *w != "dev").nth(1).map(str::to_string);
    let (state, dev) = match dev {
        None => ("noroute", "-".to_string()),
        Some(d) if d.starts_with("wl") || is_wireless(&d) => ("degraded", d),
        Some(d) => ("ok", d),
    };
    Observation::new("net.route", json!({"state": state, "dev": dev}))
}

/// Probes the default route (Linux: `ip route get 1.1.1.1`). `None` where it can't be checked.
pub async fn route() -> Option<Observation> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    let out = tokio::process::Command::new("ip").args(["route", "get", "1.1.1.1"]).output().await.ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    Some(classify_route(&text, |dev| Path::new("/sys/class/net").join(dev).join("wireless").exists()))
}

/// Free space on the filesystem holding `path`.
#[cfg(unix)]
pub fn disk(path: &Path) -> Option<Observation> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let c = CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut st = std::mem::MaybeUninit::<libc_statvfs::statvfs>::uninit();
    // SAFETY: `c` is a valid C string and `st` is writable storage of the right type.
    let rc = unsafe { libc_statvfs::statvfs(c.as_ptr(), st.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    // SAFETY: statvfs returned 0, so it filled `st`.
    let st = unsafe { st.assume_init() };
    let total = st.f_blocks as f64 * st.f_frsize as f64;
    let free = st.f_bavail as f64 * st.f_frsize as f64;
    if total <= 0.0 {
        return None;
    }
    let pct = (free / total * 1000.0).round() / 10.0;
    Some(Observation::new("disk", json!({"path": path.display().to_string(), "free_pct": pct, "free_bytes": free as u64})))
}

#[cfg(not(unix))]
pub fn disk(_path: &Path) -> Option<Observation> {
    None
}

#[cfg(unix)]
mod libc_statvfs {
    pub use libc::statvfs;
}

/// GPU temperatures from `nvidia-smi`, one observation per GPU. Empty without NVIDIA tools.
pub async fn gpus() -> Vec<Observation> {
    let Ok(out) =
        tokio::process::Command::new("nvidia-smi").args(["--query-gpu=index,temperature.gpu,name", "--format=csv,noheader,nounits"]).output().await
    else {
        return Vec::new();
    };
    parse_nvidia_smi(&String::from_utf8_lossy(&out.stdout))
}

pub fn parse_nvidia_smi(text: &str) -> Vec<Observation> {
    text.lines()
        .filter_map(|l| {
            let mut parts = l.split(',').map(str::trim);
            let index: u32 = parts.next()?.parse().ok()?;
            let temp: f64 = parts.next()?.parse().ok()?;
            let name = parts.next().unwrap_or("");
            Some(Observation::new("gpu", json!({"index": index, "temp_c": temp, "name": name})))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_states() {
        let wired = "1.1.1.1 via 192.0.2.1 dev eth0 src 192.0.2.20 uid 1000\n    cache\n";
        assert_eq!(classify_route(wired, |_| false).fields, json!({"state": "ok", "dev": "eth0"}));
        let wifi = "1.1.1.1 via 10.0.0.1 dev wlP9s9 src 10.0.0.5 uid 1000";
        assert_eq!(classify_route(wifi, |_| false).fields["state"], "degraded");
        let usb_wifi = "1.1.1.1 via 10.0.0.1 dev wlx00c0ca src 10.0.0.5";
        assert_eq!(classify_route(usb_wifi, |_| false).fields["state"], "degraded");
        let renamed = "1.1.1.1 dev radio0 src 10.0.0.5";
        assert_eq!(classify_route(renamed, |d| d == "radio0").fields["state"], "degraded");
        assert_eq!(classify_route("RTNETLINK answers: Network is unreachable", |_| false).fields, json!({"state": "noroute", "dev": "-"}));
    }

    #[test]
    fn nvidia_smi_lines() {
        let o = parse_nvidia_smi("0, 41, NVIDIA GeForce GPU\n1, 87, NVIDIA GB10\n");
        assert_eq!(o.len(), 2);
        assert_eq!(o[1].fields["temp_c"], 87.0);
    }

    #[test]
    fn disk_reads_tmp() {
        let o = disk(Path::new("/tmp")).unwrap();
        let pct = o.fields["free_pct"].as_f64().unwrap();
        assert!((0.0..=100.0).contains(&pct));
    }
}
