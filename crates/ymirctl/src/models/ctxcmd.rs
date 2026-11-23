//! `ymirctl models context`: the context each `[[model.local]]` runs with on this node and why
//! (the same plan ymird makes at start, `ymir_core::kvfit`), and `--auto` / `--old-defaults` to
//! put entries back on the automatic context.

use std::io::Write;

use anyhow::{Context, Result, bail};
use toml_edit::{Array, DocumentMut, Item, Value};
use weirdnet_backends::EngineKind;
use weirdnet_model::hw::Budget;
use ymir_core::NodeConfig;
use ymir_core::config::LocalModel;
use ymir_core::kvfit::{self, OLD_INKLING_SPARK_DEFAULT, OLD_PULL_DEFAULT, Plan};

use super::{ContextArgs, Ctx};

/// This node's model memory, as ymird computes it.
fn budget(ctx: &Ctx, cfg: &NodeConfig) -> Budget {
    weirdnet_model::hw::budget(
        &ctx.hw,
        weirdnet_model::hw::parse_size(&cfg.model.memory_budget),
        cfg.model.reserve.as_deref().and_then(weirdnet_model::hw::parse_size),
    )
}

fn plans(ctx: &Ctx, cfg: &NodeConfig) -> Vec<Plan> {
    let mut launches = cfg.launches();
    kvfit::plan_launches(&mut launches, &budget(ctx, cfg), ctx.dir())
}

/// What ymird would run `entry` with on this node, beside the models already in node.toml.
pub fn planned(ctx: &Ctx, entry: &LocalModel) -> Option<Plan> {
    let mut cfg = ctx.cfg.clone().unwrap_or_default();
    cfg.model.local.retain(|m| m.id != entry.id);
    cfg.model.local.push(entry.clone());
    plans(ctx, &cfg).into_iter().find(|p| p.id == entry.id)
}

fn engine(m: &LocalModel) -> EngineKind {
    match m.backend.as_deref() {
        Some("vllm") => EngineKind::Vllm,
        Some("mlx") => EngineKind::Mlx,
        _ => EngineKind::Llamacpp,
    }
}

/// The context `m` sets: `context`, or a context flag in `args` (which the engine would honor).
fn fixed(m: &LocalModel) -> Option<u32> {
    kvfit::arg_value(&m.args, kvfit::context_flags(engine(m))).and_then(|v| v.parse().ok()).or(m.context)
}

/// Whether `m` still carries a context an older `ymirctl models pull` wrote by default.
pub fn old_default(m: &LocalModel) -> bool {
    if m.wrap.is_some() || engine(m) == EngineKind::Mlx {
        return false;
    }
    let inkling = ymir_core::vllm::is_inkling(m.file.as_deref().unwrap_or(&m.id));
    match fixed(m) {
        Some(OLD_PULL_DEFAULT) => true,
        Some(OLD_INKLING_SPARK_DEFAULT) => inkling && engine(m) == EngineKind::Vllm,
        _ => false,
    }
}

/// `text` with `context` and the context flags removed from the entries in `ids`.
fn lift(text: &str, ids: &[String]) -> Result<String> {
    let mut doc: DocumentMut = text.parse().context("parsing node.toml")?;
    let Some(locals) = doc.get_mut("model").and_then(|m| m.get_mut("local")).and_then(Item::as_array_of_tables_mut) else {
        bail!("node.toml has no [[model.local]] entries");
    };
    for t in locals.iter_mut() {
        let Some(id) = t.get("id").and_then(Item::as_str).map(str::to_string) else { continue };
        if !ids.contains(&id) {
            continue;
        }
        t.remove("context");
        let backend = t.get("backend").and_then(Item::as_str).unwrap_or("llamacpp");
        let flags = kvfit::context_flags(if backend == "vllm" { EngineKind::Vllm } else { EngineKind::Llamacpp });
        if let Some(args) = t.get_mut("args").and_then(Item::as_array_mut) {
            let old: Vec<String> = args.iter().filter_map(Value::as_str).map(str::to_string).collect();
            let kept = kvfit::strip_arg(&old, flags);
            if kept.len() != old.len() {
                let mut a = Array::new();
                kept.iter().for_each(|s| a.push(s.as_str()));
                *args = a;
            }
        }
    }
    Ok(doc.to_string())
}

/// Saves a vLLM Hub model's config.json and weight size for ymird, when they are not here yet.
fn fetch_vllm_meta(ctx: &Ctx, out: &mut dyn Write, m: &LocalModel) -> Result<()> {
    let repo = m.file.clone().unwrap_or_else(|| m.id.clone());
    if engine(m) != EngineKind::Vllm || std::path::Path::new(&repo).is_dir() || kvfit::vllm_meta_dir(ctx.dir(), &repo).join("config.json").is_file() {
        return Ok(());
    }
    let fetched = ctx.mlx.file(&repo, "config.json").and_then(|c| {
        let size: u64 = ctx.mlx.repo(&repo)?.files.iter().filter(|f| f.name.ends_with(".safetensors")).map(|f| f.size).sum();
        kvfit::save_vllm_meta(ctx.dir(), &repo, &c, size)
    });
    if let Err(e) = fetched {
        writeln!(
            out,
            "Note: could not fetch {repo}'s config.json ({e:#}); ymird will look in the Hugging Face cache, else run it at {}",
            kvfit::FALLBACK_CONTEXT
        )?;
    }
    Ok(())
}

pub fn run(ctx: &Ctx, out: &mut dyn Write, a: &ContextArgs) -> Result<()> {
    let Some(cfg) = ctx.cfg.clone() else { bail!("no node.toml at {} (run `ymirctl setup` first)", ctx.config.display()) };
    if a.auto || a.old_defaults {
        let mut ids: Vec<String> = Vec::new();
        for id in &a.ids {
            if !cfg.model.local.iter().any(|m| &m.id == id) {
                bail!("node.toml has no [[model.local]] with id `{id}`");
            }
            ids.push(id.clone());
        }
        if a.old_defaults {
            let more: Vec<String> = cfg.model.local.iter().filter(|m| old_default(m) && !ids.contains(&m.id)).map(|m| m.id.clone()).collect();
            ids.extend(more);
        }
        if ids.is_empty() {
            if a.old_defaults {
                writeln!(out, "No model in {} has an old default context.", ctx.config.display())?;
            } else {
                bail!("name the models to put on the automatic context, or pass --old-defaults");
            }
        } else {
            for m in cfg.model.local.iter().filter(|m| ids.contains(&m.id)) {
                fetch_vllm_meta(ctx, out, m)?;
            }
            let text = std::fs::read_to_string(&ctx.config).with_context(|| format!("reading {}", ctx.config.display()))?;
            let new = lift(&text, &ids)?;
            let check: NodeConfig = toml::from_str(&new).context("the edited node.toml does not parse")?;
            check.validate()?;
            std::fs::write(ctx.config.with_extension("toml.bak"), &text)?;
            let tmp = ctx.config.with_extension("toml.tmp");
            std::fs::write(&tmp, new)?;
            std::fs::rename(&tmp, &ctx.config)?;
            writeln!(
                out,
                "Put {} on the automatic context in {} (the old file is {}.bak). Restart ymird to apply.\n",
                ids.iter().map(|i| format!("`{i}`")).collect::<Vec<_>>().join(", "),
                ctx.config.display(),
                ctx.config.display()
            )?;
            return list(ctx, out, &check);
        }
    }
    list(ctx, out, &cfg)
}

fn list(ctx: &Ctx, out: &mut dyn Write, cfg: &NodeConfig) -> Result<()> {
    let b = budget(ctx, cfg);
    writeln!(out, "This node: {:.1} GiB for models. Contexts ymird runs each model with:", b.usable_bytes as f64 / (1u64 << 30) as f64)?;
    let plans = plans(ctx, cfg);
    let mut old = Vec::new();
    for m in &cfg.model.local {
        let setting = match fixed(m) {
            Some(c) => format!("context = {c}"),
            None => "auto".to_string(),
        };
        let line = match plans.iter().find(|p| p.id == m.id) {
            Some(p) => p.note.clone(),
            None if m.wrap.is_some() => "a wrapped server: its own context".to_string(),
            None => "MLX-LM sizes its cache as it goes".to_string(),
        };
        writeln!(out, "  {:<28} {:<16} {line}", m.id, setting)?;
        if old_default(m) && plans.iter().any(|p| p.id == m.id && p.auto_would_be.is_some()) {
            old.push(m.id.clone());
        }
    }
    if !old.is_empty() {
        writeln!(
            out,
            "\n{} still {} a context an older `ymirctl models pull` wrote by default. `ymirctl models context --old-defaults` lifts {}.",
            old.join(", "),
            if old.len() == 1 { "has" } else { "have" },
            if old.len() == 1 { "it" } else { "them" }
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lift_removes_the_context_and_its_flags_and_keeps_the_rest() {
        let text = r#"# my node
[node]
name = "n"

[[model.local]]
id = "ink"
backend = "vllm"
file = "thinkingmachines/Inkling-2-Mini"
context = 262144 # recipe cap
args = ["--reasoning-parser", "inkling", "--max-model-len", "262144"]

[[model.local]]
id = "keep"
context = 8192
args = ["-c", "8192", "--jinja"]
"#;
        let new = lift(text, &["ink".to_string()]).unwrap();
        assert!(new.starts_with("# my node"), "{new}");
        let cfg: NodeConfig = toml::from_str(&new).unwrap();
        assert_eq!(cfg.model.local[0].context, None);
        assert_eq!(cfg.model.local[0].args, ["--reasoning-parser", "inkling"]);
        assert_eq!(cfg.model.local[1].context, Some(8192));
        assert_eq!(cfg.model.local[1].args, ["-c", "8192", "--jinja"]);
        assert!(old_default(&LocalModel {
            id: "ink".into(),
            backend: Some("vllm".into()),
            file: Some("thinkingmachines/Inkling-2-Mini".into()),
            args: vec!["--max-model-len".into(), "262144".into()],
            ..Default::default()
        }));
        assert!(old_default(&LocalModel { id: "q".into(), context: Some(32_768), ..Default::default() }));
        assert!(!old_default(&LocalModel { id: "q".into(), context: Some(65_536), ..Default::default() }));
        assert!(!old_default(&LocalModel { id: "m".into(), backend: Some("mlx".into()), context: Some(32_768), ..Default::default() }));
    }
}
