//! Split models over Ray (M7): the pieces of a `vllm-ray` group that are about Ray and vLLM
//! themselves. [`crate::group`] runs them.
//!
//! The leader starts `ray start --head --block` on its private link, waits until every member's
//! `ray start --block --address HEAD` has joined (`ray status`), then runs `vllm serve
//! --distributed-executor-backend ray` on loopback against that cluster. Each `ray start` is
//! the leader of its own process group, so stopping a group signals the whole group (raylet, GCS,
//! vLLM's actors) and leaves nothing behind.
//!
//! Ray has no authentication: anyone who reaches its ports can run code on the node. So every
//! Ray process binds the private link only, and Ymir checks that: it lists the TCP listeners of
//! each Ray process group (`/proc`, Linux only) and refuses the group when any listens on another
//! address. Ray's GCS listens on every interface whatever `--node-ip-address` says, so a real
//! Ray cluster is refused unless the node sets `[model] ray_trust_firewall = true`, which says a
//! host firewall drops Ray's ports on every interface but the private link. Then listeners on all
//! interfaces are allowed (and logged); ones on another specific address are still refused.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use crate::group::SplitModel;

/// Ray's GCS port, the head's `--port`.
pub const DEFAULT_PORT: u16 = 6379;

/// `ray start` flags Ymir sets itself (or that would open Ray up); `[model] ray_args` may not.
const OWN_FLAGS: &[&str] =
    &["--head", "--address", "--node-ip-address", "--port", "--block", "--dashboard-host", "--include-dashboard", "--disable-usage-stats"];

/// Checks `[model] ray_args`: none of the flags Ymir sets, and no address on all interfaces.
pub fn check_args(args: &[String]) -> Result<(), String> {
    for a in args {
        let flag = a.split('=').next().unwrap_or(a);
        if OWN_FLAGS.contains(&flag) {
            return Err(format!("ray_args may not set `{flag}`: Ymir sets it (Ray binds the private link only)"));
        }
        if a.contains("0.0.0.0") || a.contains("[::]") || a == "::" || a.ends_with("=::") {
            return Err(format!("ray_args `{a}` names all interfaces; Ray has no authentication, so it binds the private link only"));
        }
    }
    Ok(())
}

/// `ray start --head` arguments: GCS on `link:port`, no dashboard, no usage stats.
pub fn head_args(link: IpAddr, port: u16, extra: &[String]) -> Vec<String> {
    let mut a: Vec<String> =
        ["start", "--head", "--block", "--node-ip-address", &link.to_string(), "--port", &port.to_string()].iter().map(|s| s.to_string()).collect();
    a.extend(["--include-dashboard=false".into(), "--disable-usage-stats".into()]);
    a.extend(extra.iter().cloned());
    a
}

/// `ray start` arguments for a member that joins `head` (`IP:PORT`).
pub fn worker_args(head: &str, link: IpAddr, extra: &[String]) -> Vec<String> {
    let mut a: Vec<String> = ["start", "--block", "--address", head, "--node-ip-address", &link.to_string()].iter().map(|s| s.to_string()).collect();
    a.push("--disable-usage-stats".into());
    a.extend(extra.iter().cloned());
    a
}

/// `IP:PORT`, bracketed for IPv6.
pub fn endpoint(ip: IpAddr, port: u16) -> String {
    if ip.is_ipv6() { format!("[{ip}]:{port}") } else { format!("{ip}:{port}") }
}

/// How many nodes `ray status` lists as active (its "Active:" block: lines like ` 1 node_abc`).
pub fn active_nodes(status: &str) -> usize {
    let mut on = false;
    let mut n = 0;
    for line in status.lines() {
        let t = line.trim();
        if t == "Active:" {
            on = true;
            continue;
        }
        if !on {
            continue;
        }
        if !line.starts_with(' ') || t.is_empty() {
            break;
        }
        let mut w = t.split_whitespace();
        if let (Some(count), Some(node)) = (w.next(), w.next())
            && node.starts_with("node_")
        {
            n += count.parse::<usize>().unwrap_or(0);
        }
    }
    n
}

/// vLLM's sizing for a group: tensor parallel within a node, pipeline parallel across them.
/// By default `--tensor-parallel-size` is the fewest GPUs any member has and
/// `--pipeline-parallel-size` the number of members; either can be set in the pool policy. Their
/// product (vLLM's world size, one GPU each) must fit the group's GPUs.
pub fn sizing(gpus: &[u32], tp: Option<u32>, pp: Option<u32>) -> Result<(u32, u32), String> {
    let total: u32 = gpus.iter().sum();
    let tp = tp.unwrap_or_else(|| gpus.iter().copied().min().unwrap_or(1).max(1));
    let pp = pp.unwrap_or(gpus.len().max(1) as u32);
    if tp == 0 || pp == 0 {
        return Err("tensor_parallel and pipeline_parallel must be at least 1".into());
    }
    if tp * pp > total {
        return Err(format!("tensor_parallel {tp} x pipeline_parallel {pp} needs {} GPUs; the group has {total}", tp * pp));
    }
    Ok((tp, pp))
}

/// `vllm serve` arguments for the leader: the model on loopback, Ray as the executor, the sizing.
/// The API key goes in `VLLM_API_KEY`, not on the command line.
pub fn vllm_args(split: &SplitModel, model: &str, tp: u32, pp: u32) -> Vec<String> {
    let mut a: Vec<String> = [
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        &split.serve_port.to_string(),
        "--served-model-name",
        &split.id,
        "--distributed-executor-backend",
        "ray",
        "--tensor-parallel-size",
        &tp.to_string(),
        "--pipeline-parallel-size",
        &pp.to_string(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    if let Some(c) = split.context {
        a.extend(["--max-model-len".into(), c.to_string()]);
    }
    if let Some(s) = split.slots {
        a.extend(["--max-num-seqs".into(), s.to_string()]);
    }
    a.extend(split.args.iter().cloned());
    a
}

/// One TCP listener from `/proc/net/tcp{,6}`: address, port, socket inode.
#[derive(Debug, Clone, PartialEq)]
pub struct Listener {
    pub ip: IpAddr,
    pub port: u16,
    pub inode: u64,
}

/// The listening sockets in a `/proc/net/tcp` (or `tcp6`) table.
pub fn parse_proc_net(text: &str) -> Vec<Listener> {
    let mut out = Vec::new();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 10 || f[3] != "0A" {
            continue;
        }
        let Some((addr, port)) = f[1].split_once(':') else { continue };
        let (Ok(port), Ok(inode)) = (u16::from_str_radix(port, 16), f[9].parse::<u64>()) else { continue };
        let ip = match addr.len() {
            8 => match u32::from_str_radix(addr, 16) {
                Ok(v) => IpAddr::V4(Ipv4Addr::from(v.to_ne_bytes())),
                Err(_) => continue,
            },
            32 => {
                let mut b = [0u8; 16];
                let mut ok = true;
                for i in 0..4 {
                    match u32::from_str_radix(&addr[i * 8..i * 8 + 8], 16) {
                        Ok(v) => b[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes()),
                        Err(_) => ok = false,
                    }
                }
                if !ok {
                    continue;
                }
                let v6 = Ipv6Addr::from(b);
                match v6.to_ipv4_mapped() {
                    Some(v4) => IpAddr::V4(v4),
                    None => IpAddr::V6(v6),
                }
            }
            _ => continue,
        };
        out.push(Listener { ip, port, inode });
    }
    out
}

/// Listeners that are neither loopback nor `own` (all-interface ones included).
pub fn foreign(listeners: &[Listener], own: IpAddr) -> Vec<Listener> {
    let mut v: Vec<Listener> = listeners.iter().filter(|l| !l.ip.is_loopback() && l.ip != own).cloned().collect();
    v.sort_by_key(|l| (l.port, l.inode));
    v.dedup_by_key(|l| (l.ip, l.port));
    v
}

/// `(pid, ppid, pgrp)` of every process (`/proc/PID/stat`, Linux).
#[cfg(target_os = "linux")]
fn processes() -> Vec<(i32, i32, i32)> {
    let mut out = Vec::new();
    if let Ok(dir) = std::fs::read_dir("/proc") {
        for e in dir.flatten() {
            let name = e.file_name();
            let Some(pid) = name.to_str().and_then(|s| s.parse::<i32>().ok()) else { continue };
            let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else { continue };
            // `pid (comm) state ppid pgrp ...`; comm may hold spaces and parentheses.
            let Some(rest) = stat.rsplit_once(')').map(|(_, r)| r) else { continue };
            let mut f = rest.split_whitespace().skip(1);
            if let (Some(Ok(ppid)), Some(Ok(pgrp))) = (f.next().map(str::parse::<i32>), f.next().map(str::parse::<i32>)) {
                out.push((pid, ppid, pgrp));
            }
        }
    }
    out
}

/// Which of `procs` (`(pid, ppid, pgrp)`) belong to the process Ymir started as `root` (the leader of
/// its own process group): the members of that group and, transitively, every child of one of them,
/// whatever process group or session it moved to. Ray starts its workers (and so vLLM's workers,
/// NCCL and torch.distributed) in groups of their own, so the group alone misses them.
pub fn tree_in(procs: &[(i32, i32, i32)], root: i32) -> Vec<i32> {
    let mut set: std::collections::HashSet<i32> = procs.iter().filter(|p| p.0 == root || p.2 == root).map(|p| p.0).collect();
    loop {
        let before = set.len();
        for p in procs {
            if set.contains(&p.1) {
                set.insert(p.0);
            }
        }
        if set.len() == before {
            break;
        }
    }
    let mut v: Vec<i32> = set.into_iter().collect();
    v.sort_unstable();
    v
}

/// [`tree_in`] over the running processes (Linux; empty elsewhere).
pub fn tree_of(root: i32) -> Vec<i32> {
    #[cfg(target_os = "linux")]
    {
        tree_in(&processes(), root)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = root;
        Vec::new()
    }
}

/// `(pid, start time)` of those of `pids` still running (`/proc/PID/stat` field 22; Linux), so a
/// later kill can tell a process from one that reused its pid.
pub fn started(pids: &[i32]) -> Vec<(i32, u64)> {
    pids.iter()
        .filter_map(|&pid| {
            let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
            let rest = stat.rsplit_once(')')?.1;
            Some((pid, rest.split_whitespace().nth(19)?.parse().ok()?))
        })
        .collect()
}

/// The TCP listeners of the process Ymir started as `root` and of everything under it
/// ([`tree_of`]; Linux).
#[cfg(target_os = "linux")]
pub fn listeners_of_tree(root: i32) -> Vec<Listener> {
    let mut inodes = std::collections::HashSet::new();
    for pid in tree_of(root) {
        let Ok(fds) = std::fs::read_dir(format!("/proc/{pid}/fd")) else { continue };
        for fd in fds.flatten() {
            if let Ok(t) = std::fs::read_link(fd.path())
                && let Some(i) = t.to_str().and_then(|s| s.strip_prefix("socket:[")).and_then(|s| s.strip_suffix(']'))
                && let Ok(i) = i.parse::<u64>()
            {
                inodes.insert(i);
            }
        }
    }
    let mut all = Vec::new();
    for table in ["/proc/net/tcp", "/proc/net/tcp6"] {
        if let Ok(t) = std::fs::read_to_string(table) {
            all.extend(parse_proc_net(&t));
        }
    }
    all.into_iter().filter(|l| inodes.contains(&l.inode)).collect()
}

#[cfg(not(target_os = "linux"))]
pub fn listeners_of_tree(_root: i32) -> Vec<Listener> {
    Vec::new()
}

/// The network interface that holds `ip` (`lo` for loopback), for pinning NCCL and Gloo to it.
pub fn iface_of(ip: IpAddr) -> Option<String> {
    if_addrs::get_if_addrs().ok()?.into_iter().find(|i| i.ip() == ip).map(|i| i.name)
}

/// Environment that keeps a Ray node's workers on the private link: vLLM's own address and the
/// interface NCCL and Gloo (torch.distributed) bind and connect on. Without it they pick an
/// interface themselves, such as the LAN or Tailscale.
pub fn link_env(ip: IpAddr, iface: Option<&str>) -> Vec<(String, String)> {
    let mut v = vec![("VLLM_HOST_IP".to_string(), ip.to_string())];
    if let Some(i) = iface {
        v.push(("NCCL_SOCKET_IFNAME".into(), format!("={i}")));
        v.push(("GLOO_SOCKET_IFNAME".into(), i.to_string()));
    }
    v
}

/// Whether the listeners of a Ray process tree are acceptable: `Err` names the ones that are
/// not. With `trust_firewall`, all-interface listeners pass (the caller logs them).
pub fn check_listeners(listeners: &[Listener], own: IpAddr, trust_firewall: bool) -> Result<Vec<Listener>, String> {
    let bad = foreign(listeners, own);
    let (wild, other): (Vec<Listener>, Vec<Listener>) = bad.into_iter().partition(|l| l.ip.is_unspecified());
    let refused: Vec<&Listener> = if trust_firewall { other.iter().collect() } else { wild.iter().chain(other.iter()).collect() };
    if refused.is_empty() {
        return Ok(wild);
    }
    let list = refused.iter().map(|l| endpoint(l.ip, l.port)).collect::<Vec<_>>().join(", ");
    let hint = if !trust_firewall && refused.iter().any(|l| l.ip.is_unspecified()) {
        " (Ray's GCS ignores --node-ip-address; with a host firewall that drops Ray's ports on every other interface, set [model] ray_trust_firewall = true)"
    } else {
        ""
    };
    Err(format!("Ray listens on {list}, but only {own} (the private link) and loopback are allowed: Ray has no authentication{hint}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_args_may_not_open_ray_up() {
        assert!(check_args(&["--num-gpus=2".into(), "--object-store-memory".into(), "1000000".into()]).is_ok());
        for bad in ["--head", "--port=1", "--node-ip-address", "--dashboard-host=127.0.0.1", "--block", "0.0.0.0", "--foo=[::]:1", "::"] {
            assert!(check_args(&[bad.to_string()]).is_err(), "{bad}");
        }
    }

    #[test]
    fn head_and_worker_bind_the_link() {
        let ip: IpAddr = "10.0.0.2".parse().unwrap();
        let h = head_args(ip, 6380, &["--num-gpus=1".into()]);
        assert_eq!(&h[..7], ["start", "--head", "--block", "--node-ip-address", "10.0.0.2", "--port", "6380"]);
        assert!(h.contains(&"--include-dashboard=false".to_string()) && h.ends_with(&["--num-gpus=1".to_string()]));
        let w = worker_args("10.0.0.1:6380", ip, &[]);
        assert_eq!(&w[..6], ["start", "--block", "--address", "10.0.0.1:6380", "--node-ip-address", "10.0.0.2"]);
        assert_eq!(endpoint("fd00::1".parse().unwrap(), 6379), "[fd00::1]:6379");
    }

    #[test]
    fn counts_active_nodes_in_ray_status() {
        let s = "======== Autoscaler status: 2026-11-26 ========\nNode status\n----------------\nActive:\n 1 node_aaa\n 2 node_bbb\nPending:\n (no pending nodes)\nRecent failures:\n (no failures)\n";
        assert_eq!(active_nodes(s), 3);
        assert_eq!(active_nodes("Active:\n (no active nodes)\nPending:\n"), 0);
        assert_eq!(active_nodes("garbage"), 0);
    }

    #[test]
    fn sizing_is_gpus_per_node_by_nodes() {
        assert_eq!(sizing(&[4, 4], None, None), Ok((4, 2)));
        assert_eq!(sizing(&[8, 2, 4], None, None), Ok((2, 3)));
        assert_eq!(sizing(&[1, 1], None, None), Ok((1, 2)));
        assert_eq!(sizing(&[4, 4], Some(8), Some(1)), Ok((8, 1)));
        assert!(sizing(&[4, 4], Some(4), Some(3)).is_err());
        assert!(sizing(&[4, 4], Some(0), None).is_err());
    }

    #[test]
    fn vllm_serves_loopback_with_ray_and_the_sizing() {
        let split = SplitModel {
            id: "big".into(),
            file: None,
            engine: Some("vllm".into()),
            serve_port: 18090,
            context: Some(8192),
            slots: Some(4),
            gpu_layers: None,
            args: vec!["--enforce-eager".into()],
        };
        let a = vllm_args(&split, "org/Big", 4, 2);
        let at = |f: &str| a.iter().position(|x| x == f).map(|i| a[i + 1].clone());
        assert_eq!(&a[..2], ["serve", "org/Big"]);
        assert_eq!(at("--host").as_deref(), Some("127.0.0.1"));
        assert_eq!(at("--distributed-executor-backend").as_deref(), Some("ray"));
        assert_eq!((at("--tensor-parallel-size").as_deref(), at("--pipeline-parallel-size").as_deref()), (Some("4"), Some("2")));
        assert_eq!((at("--max-model-len").as_deref(), at("--max-num-seqs").as_deref()), (Some("8192"), Some("4")));
        assert_eq!(a.last().map(String::as_str), Some("--enforce-eager"));
        assert!(!a.iter().any(|x| x.contains("api-key")), "the key stays out of argv");
    }

    #[test]
    fn reads_proc_net_listeners() {
        let v4 = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n\
   0: 0100007F:18EB 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 111 1 0 100 0 0 10 0\n\
   1: 00000000:18EC 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 222 1 0 100 0 0 10 0\n\
   2: 0200000A:18EB 0100000A:9999 01 00000000:00000000 00:00000000 00000000  1000        0 333 1 0 100 0 0 10 0\n\
   3: 0200000A:18ED 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 444 1 0 100 0 0 10 0\n";
        let l = parse_proc_net(v4);
        assert_eq!(
            l.iter().map(|l| (l.ip.to_string(), l.port, l.inode)).collect::<Vec<_>>(),
            [("127.0.0.1".into(), 6379, 111), ("0.0.0.0".into(), 6380, 222), ("10.0.0.2".into(), 6381, 444)]
        );
        let v6 = "  sl  local_address                         remote_address                        st\n\
   0: 00000000000000000000000000000000:18EB 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 555 1\n\
   1: 0000000000000000FFFF00000200000A:18EE 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 666 1\n";
        let l6 = parse_proc_net(v6);
        assert_eq!(l6[0].ip, "::".parse::<IpAddr>().unwrap());
        assert_eq!(l6[1].ip, "10.0.0.2".parse::<IpAddr>().unwrap(), "v4-mapped");
    }

    #[test]
    fn only_the_link_and_loopback_may_listen() {
        let own: IpAddr = "10.0.0.2".parse().unwrap();
        let l = |ip: &str, port| Listener { ip: ip.parse().unwrap(), port, inode: port as u64 };
        assert_eq!(check_listeners(&[l("10.0.0.2", 6379), l("127.0.0.1", 1)], own, false), Ok(vec![]));
        let e = check_listeners(&[l("10.0.0.2", 6379), l("0.0.0.0", 6380)], own, false).unwrap_err();
        assert!(e.contains("0.0.0.0:6380") && e.contains("ray_trust_firewall"), "{e}");
        assert_eq!(check_listeners(&[l("::", 6380)], own, true), Ok(vec![l("::", 6380)]), "a firewalled node allows all-interface listeners");
        let e = check_listeners(&[l("192.168.1.5", 6381)], own, true).unwrap_err();
        assert!(e.contains("192.168.1.5:6381"), "another specific address is still refused: {e}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_tree_follows_children_into_other_groups() {
        // (pid, ppid, pgrp): `ray start` 100 leads group 100; raylet 101 is in it; a worker 200
        // moved to its own group and its child 201 to another; 300 is unrelated, 400 is in 100's
        // group though its parent is elsewhere.
        let procs = [(1, 0, 1), (100, 50, 100), (101, 100, 100), (200, 101, 200), (201, 200, 201), (300, 1, 300), (400, 1, 100)];
        assert_eq!(tree_in(&procs, 100), vec![100, 101, 200, 201, 400]);
        assert_eq!(tree_in(&procs, 200), vec![200, 201]);
        assert!(tree_in(&procs, 999).is_empty());
    }

    #[test]
    fn link_env_pins_nccl_and_gloo_to_the_link() {
        let ip: IpAddr = "10.100.40.1".parse().unwrap();
        let env = link_env(ip, Some("enp1s0f1np1"));
        assert!(env.contains(&("VLLM_HOST_IP".into(), "10.100.40.1".into())));
        assert!(env.contains(&("NCCL_SOCKET_IFNAME".into(), "=enp1s0f1np1".into())));
        assert!(env.contains(&("GLOO_SOCKET_IFNAME".into(), "enp1s0f1np1".into())));
        assert_eq!(link_env(ip, None).len(), 1);
        assert_eq!(iface_of("127.0.0.1".parse().unwrap()).as_deref(), Some("lo"));
    }

    #[test]
    fn finds_the_listeners_of_a_process_group() {
        use std::os::unix::process::CommandExt;
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        // Our own process group holds `l`.
        let me = unsafe { libc::getpgrp() };
        assert!(listeners_of_tree(me).iter().any(|x| x.port == port));
        let mut child = std::process::Command::new("sleep").arg("30").process_group(0).spawn().unwrap();
        assert!(!listeners_of_tree(child.id() as i32).iter().any(|x| x.port == port), "another group does not hold it");
        let _ = child.kill();
        let _ = child.wait();
    }
}
