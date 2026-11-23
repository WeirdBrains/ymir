use serde_json::json;
use weirdnet_backends::{EngineKind, ModelLaunch};
use weirdnet_model::hw::Budget;

use super::*;

const GIB: u64 = 1 << 30;

enum V<'a> {
    U32(u32),
    Str(&'a str),
    F32(f32),
    U32s(Vec<u32>),
    Bools(Vec<bool>),
    Strs(Vec<&'a str>),
}

/// GGUF header bytes (version 3, no tensors) with these keys.
fn header(kv: &[(&str, V)]) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.extend((kv.len() as u64).to_le_bytes());
    let s = |out: &mut Vec<u8>, s: &str| {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    };
    for (k, v) in kv {
        s(&mut out, k);
        match v {
            V::U32(n) => {
                out.extend(4u32.to_le_bytes());
                out.extend(n.to_le_bytes());
            }
            V::Str(t) => {
                out.extend(8u32.to_le_bytes());
                s(&mut out, t);
            }
            V::F32(f) => {
                out.extend(6u32.to_le_bytes());
                out.extend(f.to_le_bytes());
            }
            V::U32s(items) => {
                out.extend(9u32.to_le_bytes());
                out.extend(4u32.to_le_bytes());
                out.extend((items.len() as u64).to_le_bytes());
                items.iter().for_each(|i| out.extend(i.to_le_bytes()));
            }
            V::Bools(items) => {
                out.extend(9u32.to_le_bytes());
                out.extend(7u32.to_le_bytes());
                out.extend((items.len() as u64).to_le_bytes());
                items.iter().for_each(|b| out.push(u8::from(*b)));
            }
            V::Strs(items) => {
                out.extend(9u32.to_le_bytes());
                out.extend(8u32.to_le_bytes());
                out.extend((items.len() as u64).to_le_bytes());
                items.iter().for_each(|i| s(&mut out, i));
            }
        }
    }
    out
}

/// Inkling 2 Mini's header keys, as in ggml-org/Inkling-2-Mini-GGUF.
fn inkling_header() -> Vec<u8> {
    let a = "inkling2_loop";
    let k = |s: &str| format!("{a}.{s}");
    let pattern: Vec<bool> = (0..24).map(|i| i % 6 != 5).collect();
    let keys = [
        k("block_count"),
        k("context_length"),
        k("embedding_length"),
        k("attention.head_count"),
        k("attention.head_count_kv"),
        k("attention.key_length"),
        k("attention.value_length"),
        k("attention.key_length_swa"),
        k("attention.value_length_swa"),
        k("attention.sliding_window"),
        k("attention.sliding_window_pattern"),
        k("attention.layer_norm_rms_epsilon"),
        k("loop.start"),
        k("loop.end"),
        k("loop.count"),
    ];
    header(&[
        ("general.architecture", V::Str(a)),
        (&keys[0], V::U32(24)),
        (&keys[1], V::U32(1_048_576)),
        (&keys[2], V::U32(2304)),
        (&keys[3], V::U32s(vec![24; 24])),
        (&keys[4], V::U32s(vec![8; 24])),
        (&keys[5], V::U32(128)),
        (&keys[6], V::U32(128)),
        (&keys[7], V::U32(128)),
        (&keys[8], V::U32(128)),
        (&keys[9], V::U32(1024)),
        (&keys[10], V::Bools(pattern)),
        (&keys[11], V::F32(1e-6)),
        (&keys[12], V::U32(6)),
        (&keys[13], V::U32(18)),
        (&keys[14], V::U32(3)),
        ("tokenizer.ggml.tokens", V::Strs(vec!["a", "b", "c"])),
    ])
}

/// Inkling 2 Mini's `config.json`, trimmed to what sizing reads.
fn inkling_config() -> serde_json::Value {
    json!({
        "model_type": "inkling2_loop",
        "text_config": {
            "model_max_length": 1_048_576, "hidden_size": 2304, "num_hidden_layers": 24,
            "num_attention_heads": 24, "num_key_value_heads": 8, "head_dim": 128,
            "local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22],
            "swa_head_dim": 128, "swa_num_key_value_heads": 8, "sliding_window_size": 1024,
            "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3}
        },
        "mtp_config": {
            "num_hidden_layers": 3, "num_attention_heads": 24, "num_key_value_heads": 8, "head_dim": 128,
            "max_position_embeddings": 1_048_576, "local_layer_ids": [0, 2], "sliding_window_size": 1024
        }
    })
}

fn spark_budget() -> Budget {
    use weirdnet_model::hw::{Accel, Gpu, Hardware};
    let hw = Hardware {
        os: "linux".into(),
        arch: "aarch64".into(),
        cpu_threads: 20,
        ram_bytes: 128 * GIB,
        gpus: vec![Gpu {
            accel: Accel::Nvidia,
            name: "NVIDIA GB10".into(),
            memory_bytes: 128 * GIB,
            unified: true,
            compute_capability: Some((12, 1)),
            driver: None,
        }],
    };
    weirdnet_model::hw::budget(&hw, None, None)
}

#[test]
fn the_inkling_header_counts_loop_passes_and_sliding_layers() {
    let s = gguf_shape(&inkling_header()[..]).unwrap();
    // Global layers 5, 11, 17, 23; 11 and 17 are in the loop (6..18, three passes): 8 in all.
    assert_eq!((s.full_layers, s.swa_layers), (8, 40));
    assert_eq!(s.full_elems, 8 * 8 * 256);
    assert_eq!(s.swa_elems, 40 * 8 * 256);
    assert_eq!((s.window, s.trained), (1024, Some(1_048_576)));
    // f16: 32 KiB per token in the global layers, so 32 GiB at the full 1M; the sliding layers
    // hold only their window.
    assert_eq!(s.kv_bytes(1_048_576, 1, CacheType::F16), 32 * GIB + 40 * 8 * 256 * 2 * (1024 + 512));
}

#[test]
fn a_header_without_a_window_pattern_counts_every_layer_as_global() {
    let h = header(&[
        ("general.architecture", V::Str("gemma3")),
        ("gemma3.block_count", V::U32(6)),
        ("gemma3.context_length", V::U32(131_072)),
        ("gemma3.embedding_length", V::U32(1024)),
        ("gemma3.attention.head_count", V::U32(8)),
        ("gemma3.attention.head_count_kv", V::U32(2)),
        ("gemma3.attention.sliding_window", V::U32(512)),
    ]);
    let s = gguf_shape(&h[..]).unwrap();
    assert_eq!((s.full_layers, s.swa_layers, s.full_elems), (6, 0, 6 * 2 * 256));
    // With llama.cpp's integer pattern (every 3rd layer global).
    let h = header(&[
        ("general.architecture", V::Str("x")),
        ("x.block_count", V::U32(6)),
        ("x.attention.head_count", V::U32(8)),
        ("x.attention.key_length", V::U32(64)),
        ("x.attention.sliding_window", V::U32(512)),
        ("x.attention.sliding_window_pattern", V::U32(3)),
    ]);
    let s = gguf_shape(&h[..]).unwrap();
    assert_eq!((s.full_layers, s.swa_layers, s.trained), (2, 4, None));
    assert!(gguf_shape(&b"NOPE"[..]).is_err());
}

#[test]
fn the_inkling_config_matches_its_gguf_and_adds_the_mtp_head() {
    let c = config_shape(&inkling_config()).unwrap();
    let g = gguf_shape(&inkling_header()[..]).unwrap();
    assert_eq!(c.trained, g.trained);
    assert_eq!(c.swa_layers, g.swa_layers + 2);
    assert_eq!(c.full_layers, g.full_layers + 1);
    assert_eq!(c.full_elems, g.full_elems + 8 * 256);
}

#[test]
fn configs_follow_rope_scaling_and_layer_types_like_vllm() {
    let yarn = json!({"num_hidden_layers": 4, "num_attention_heads": 8, "hidden_size": 512, "max_position_embeddings": 40960,
        "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}});
    assert_eq!(config_shape(&yarn).unwrap().trained, Some(131_072));
    let llama3 = json!({"num_hidden_layers": 4, "num_attention_heads": 8, "hidden_size": 512, "max_position_embeddings": 131072,
        "rope_scaling": {"rope_type": "llama3", "factor": 8.0}});
    assert_eq!(config_shape(&llama3).unwrap().trained, Some(131_072));
    let hybrid = json!({"num_hidden_layers": 4, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64, "max_position_embeddings": 262144,
        "sliding_window": 128, "layer_types": ["linear_attention", "sliding_attention", "linear_attention", "full_attention"]});
    let s = config_shape(&hybrid).unwrap();
    assert_eq!((s.full_layers, s.swa_layers, s.window), (1, 1, 128));
    // `use_sliding_window: false` (Qwen2) turns the window off.
    let off = json!({"num_hidden_layers": 2, "num_attention_heads": 4, "hidden_size": 256, "sliding_window": 4096, "use_sliding_window": false});
    assert_eq!(config_shape(&off).unwrap().window, 0);
}

#[test]
fn cache_types_and_flags_come_from_args() {
    let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    assert_eq!(cache_type(EngineKind::Llamacpp, &args(&["-fa", "on"])), CacheType::F16);
    assert_eq!(cache_type(EngineKind::Llamacpp, &args(&["-ctk", "q8_0", "--cache-type-v=q4_0"])).name, "q8_0");
    assert_eq!(cache_type(EngineKind::Vllm, &args(&["--kv-cache-dtype", "fp8_e4m3"])).name, "fp8");
    assert_eq!(CacheType::llama("q8_0").unwrap().bytes(32), 34);
    assert_eq!(arg_value(&args(&["--ctx-size", "8192", "-c=4096"]), context_flags(EngineKind::Llamacpp)), Some("4096"));
    assert_eq!(
        strip_arg(&args(&["--jinja", "--max-model-len", "262144", "--max_model_len=9", "-x"]), context_flags(EngineKind::Vllm)),
        args(&["--jinja", "-x"])
    );
}

#[test]
fn inkling_gets_its_full_context_on_a_128_gib_spark_and_is_cut_on_a_small_gpu() {
    let s = gguf_shape(&inkling_header()[..]).unwrap();
    let b = spark_budget();
    let weights = 27 * GIB;
    let kv_budget = (b.usable_bytes - footprint(weights, 0)) / 11 * 10;
    assert_eq!(fit_context(&s, kv_budget, 1, CacheType::F16), (1_048_576, false));
    // 8 GiB for the cache: cut, on a 1024 step, to the most that fits.
    let (c, cut) = fit_context(&s, 8 * GIB, 1, CacheType::F16);
    assert!(cut);
    assert_eq!(u64::from(c) % STEP, 0);
    assert!(s.kv_bytes(u64::from(c), 1, CacheType::F16) <= 8 * GIB);
    assert!(s.kv_bytes(u64::from(c) + STEP, 1, CacheType::F16) > 8 * GIB);
    assert_eq!(c, 253_952);
    // Nothing fits: the floor.
    assert_eq!(fit_context(&s, 0, 1, CacheType::F16), (MIN_CONTEXT, true));
    // Unknown trained length: the fallback.
    let unknown = KvShape { full_elems: 1, ..Default::default() };
    assert_eq!(fit_context(&unknown, u64::MAX, 1, CacheType::F16), (FALLBACK_CONTEXT, false));
}

/// A sparse GGUF of `size` bytes with `header` at the front.
fn sparse(dir: &Path, name: &str, header: &[u8], size: u64) -> String {
    let p = dir.join(name);
    std::fs::write(&p, header).unwrap();
    std::fs::OpenOptions::new().write(true).open(&p).unwrap().set_len(size).unwrap();
    p.to_string_lossy().into_owned()
}

#[test]
fn plans_share_memory_keep_explicit_contexts_and_skip_wraps_and_mlx() {
    let dir = tempfile::tempdir().unwrap();
    let ink = sparse(dir.path(), "ink-Q8_0.gguf", &inkling_header(), 27 * GIB);
    let budget = Budget { pool_bytes: 128 * GIB, reserve_bytes: 16 * GIB, usable_bytes: 95 * GIB };

    // Alone: the full trained length.
    let mut one = vec![ModelLaunch::new("ink", EngineKind::Llamacpp, &ink)];
    let p = plan_launches(&mut one, &budget, dir.path());
    assert_eq!(one[0].context, Some(1_048_576));
    assert!(!p[0].clamped && !p[0].explicit && p[0].note.contains("trained length is 1048576"), "{p:?}");

    // Beside a 40 GiB model in another group, both automatic: they split what is left.
    let big = sparse(dir.path(), "big.gguf", &inkling_header(), 40 * GIB);
    let mut two = vec![ModelLaunch::new("ink", EngineKind::Llamacpp, &ink), ModelLaunch::new("big", EngineKind::Llamacpp, &big)];
    let p = plan_launches(&mut two, &budget, dir.path());
    assert!(p[0].clamped && p[1].clamped, "{p:?}");
    assert!(p[0].note.contains("cut from 1048576") && p[0].note.contains("shared with 1 other"), "{}", p[0].note);
    let used: u64 =
        two.iter().map(|l| footprint(0, gguf_shape(&inkling_header()[..]).unwrap().kv_bytes(u64::from(l.context.unwrap()), 1, CacheType::F16))).sum();
    assert!(footprint(27 * GIB, 0) + footprint(40 * GIB, 0) + used <= budget.usable_bytes);

    // The same two in one swap group are never resident together: each gets its own fit.
    let mut grouped = two.clone();
    for l in &mut grouped {
        l.context = None;
        l.swap_group = Some("main".into());
    }
    plan_launches(&mut grouped, &budget, dir.path());
    assert_eq!(grouped[0].context, Some(1_048_576));

    // An explicit 32768 is kept, and the plan says the node could take more.
    let mut explicit = vec![ModelLaunch::new("ink", EngineKind::Llamacpp, &ink)];
    explicit[0].context = Some(OLD_PULL_DEFAULT);
    let p = plan_launches(&mut explicit, &budget, dir.path());
    assert_eq!(explicit[0].context, Some(OLD_PULL_DEFAULT));
    assert_eq!((p[0].explicit, p[0].auto_would_be), (true, Some(1_048_576)));

    // q8_0 KV halves the cache, so a cut context grows.
    let mut q8 = vec![ModelLaunch::new("ink", EngineKind::Llamacpp, &ink), ModelLaunch::new("big", EngineKind::Llamacpp, &big)];
    q8[0].extra_args = ["-ctk", "q8_0", "-ctv", "q8_0"].map(String::from).to_vec();
    plan_launches(&mut q8, &budget, dir.path());
    assert!(q8[0].context > two[0].context);

    // A missing file falls back with a warning; wraps and MLX are left alone.
    let mut rest = vec![
        ModelLaunch::new("gone", EngineKind::Llamacpp, dir.path().join("gone.gguf").to_string_lossy()),
        ModelLaunch::new("mlx", EngineKind::Mlx, "x"),
        ModelLaunch::new("wrapped", EngineKind::Vllm, "org/m"),
    ];
    rest[2].wrap = Some(weirdnet_backends::Wrap { url: "http://127.0.0.1:8000".into(), api_key: None });
    let p = plan_launches(&mut rest, &budget, dir.path());
    assert_eq!(rest[0].context, Some(FALLBACK_CONTEXT));
    assert!(p[0].warn && p.len() == 1, "{p:?}");
    assert_eq!((rest[1].context, rest[2].context), (None, None));
}

#[test]
fn a_vllm_hub_model_is_sized_from_the_config_pull_saved() {
    let dir = tempfile::tempdir().unwrap();
    let budget = spark_budget();
    let mut l = vec![ModelLaunch::new("inkling-2-mini", EngineKind::Vllm, "thinkingmachines/Inkling-2-Mini")];
    // Nothing saved and not in the cache: the fallback, with a warning.
    unsafe { std::env::set_var("HF_HUB_CACHE", dir.path().join("no-cache")) };
    let p = plan_launches(&mut l.clone(), &budget, dir.path());
    assert!(p[0].warn && p[0].context == Some(FALLBACK_CONTEXT), "{p:?}");
    save_vllm_meta(dir.path(), "thinkingmachines/Inkling-2-Mini", inkling_config().to_string().as_bytes(), 20 * GIB).unwrap();
    let p = plan_launches(&mut l, &budget, dir.path());
    assert_eq!(l[0].context, Some(1_048_576), "{p:?}");
    // A low --gpu-memory-utilization holds vLLM to its own pool.
    let mut low = vec![ModelLaunch::new("inkling-2-mini", EngineKind::Vllm, "thinkingmachines/Inkling-2-Mini")];
    low[0].extra_args = ["--gpu-memory-utilization", "0.25"].map(String::from).to_vec();
    plan_launches(&mut low, &budget, dir.path());
    assert!(low[0].context.unwrap() < 1_048_576);
}
