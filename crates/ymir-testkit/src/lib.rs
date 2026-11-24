//! Test support for Ymir. See the README.

/// A loopback TCP port for a test to write into a config or command line, for a server that has to
/// be told its port (rather than binding port 0 and reporting the one it got; prefer that).
///
/// Picks from 20000..32000, below the ephemeral ranges (Linux 32768+, macOS and Windows 49152+), so
/// the port isn't handed out to the many port-0 listeners and outgoing connections of parallel tests
/// between this check and the server's bind; a port another test picked the same way is retried.
pub fn config_port() -> u16 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
    let mut x = (u64::from(std::process::id()) << 32) ^ u64::from(nanos) ^ NEXT.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
    for _ in 0..1000 {
        // splitmix64
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let port = 20000 + (z % 12000) as u16;
        if std::net::TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
    }
    panic!("no free loopback port in 20000..32000");
}
