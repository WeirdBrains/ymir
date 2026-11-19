//! Hardware detection and model sizing.
//!
//! To be extracted into `weirdnet-model` (`hw` module, tiered catalog and sizing). Generic: no Ymir
//! types here.

mod detect;
mod sizing;

pub use detect::{Accel, Gpu, Hardware, detect, parse_meminfo_total, parse_nvidia_smi};
pub use sizing::{Budget, CatalogEntry, Plan, Purpose, budget, catalog, find, plan};

/// 1 GiB.
pub const GIB: u64 = 1 << 30;

/// Parses sizes like `12GiB`, `512MiB`, `100GB`, `8G` or a plain byte count.
pub fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim();
    let split = s.find(|c: char| !(c.is_ascii_digit() || c == '.')).unwrap_or(s.len());
    let (num, unit) = s.split_at(split);
    let num: f64 = num.parse().ok()?;
    let mult: f64 = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "k" | "kib" => 1024.0,
        "kb" => 1e3,
        "m" | "mib" => 1024.0 * 1024.0,
        "mb" => 1e6,
        "g" | "gib" => GIB as f64,
        "gb" => 1e9,
        "t" | "tib" => GIB as f64 * 1024.0,
        "tb" => 1e12,
        _ => return None,
    };
    Some((num * mult) as u64)
}

/// `11.8 GiB` (binary, like memory budgets).
pub fn human_gib(b: u64) -> String {
    format!("{:.1} GiB", b as f64 / GIB as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(parse_size("12GiB"), Some(12 * GIB));
        assert_eq!(parse_size("100GB"), Some(100_000_000_000));
        assert_eq!(parse_size("512 MiB"), Some(512 << 20));
        assert_eq!(parse_size("1.5G"), Some(GIB + GIB / 2));
        assert_eq!(parse_size("42"), Some(42));
        assert_eq!(parse_size("lots"), None);
    }
}
