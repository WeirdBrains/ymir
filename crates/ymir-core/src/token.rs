//! The bridge token.
//!
//! The design keeps it in the OS keyring. M1 keeps it in `state/bridge.token` (mode 0600, inside a
//! 0700 directory) until the keyring integration lands; nothing else reads it.

use std::io::Write;

use crate::Paths;

/// Reads the token, or creates one.
pub fn load_or_create(paths: &Paths) -> std::io::Result<String> {
    if let Some(t) = load(paths)? {
        return Ok(t);
    }
    rotate(paths)
}

pub fn load(paths: &Paths) -> std::io::Result<Option<String>> {
    match std::fs::read_to_string(paths.token()) {
        Ok(t) if !t.trim().is_empty() => Ok(Some(t.trim().to_string())),
        Ok(_) => Ok(None),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// Writes a new token (a running bridge keeps the old one until restarted).
pub fn rotate(paths: &Paths) -> std::io::Result<String> {
    paths.ensure_state()?;
    let token = weirdnet_openai::new_token();
    let tmp = paths.token().with_extension("tmp");
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts.open(&tmp)?;
    f.write_all(token.as_bytes())?;
    f.sync_all()?;
    std::fs::rename(&tmp, paths.token())?;
    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_once_and_is_private() {
        let dir = tempfile::tempdir().unwrap();
        let p = Paths::new(dir.path());
        let a = load_or_create(&p).unwrap();
        assert_eq!(a.len(), 64);
        assert_eq!(load_or_create(&p).unwrap(), a);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(p.token()).unwrap().permissions().mode() & 0o777, 0o600);
        }
        assert_ne!(rotate(&p).unwrap(), a);
    }
}
