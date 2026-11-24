//! `ymirctl models adapter add | remove | list`: LoRA adapters on a `[[model.local]]` base in
//! this node's `node.toml` (`[[model.local.lora]]`, training plan P4a). Each adapter is served as
//! its own model id on the base's engine; ymird picks the change up on restart. To fetch an
//! adapter from the Hub and serve it without a restart, use `ymirctl models pull REPO --node NAME
//! --adapter-for BASE --serve`.
//!
//! `serve`, `unserve` and `scale` (with `--node`) change what a running node serves, through its
//! ymird (models/1), without restarting the base where the engine allows: vLLM loads and unloads
//! adapters at run time; llama-server rescales per request and re-enables a file it already has,
//! but a new adapter file needs the base's next load. They are in memory only; node.toml is not
//! edited.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};
use toml_edit::{ArrayOfTables, DocumentMut, Item, Table, value};
use ymir_core::NodeConfig;
use ymir_core::config::LocalAdapter;

use super::Ctx;

#[derive(Subcommand, Debug)]
pub enum AdapterCmd {
    /// Add a LoRA adapter to a model in node.toml (a copy is kept as node.toml.bak). Restart ymird to serve it.
    Add(AddArgs),
    /// Remove an adapter from node.toml (its file stays). Restart ymird to stop serving it.
    Remove(RemoveAdapterArgs),
    /// List the adapters in node.toml, with their base and file.
    List,
    /// Start a vLLM model with LoRA on, so adapters can be served on it later without a restart
    /// (`lora_slots` in node.toml). Restart ymird to apply.
    Slots(SlotsArgs),
    /// Serve an adapter already on a node's disk (`ymirctl models list --node NAME`) on a model it
    /// serves, without restarting that model where the engine allows. In memory only.
    Serve(ServeArgs),
    /// Stop serving an adapter on a node; its files stay. In memory only.
    Unserve(UnserveArgs),
    /// Change how strongly an adapter applies on a node (llama.cpp), from its next request, with
    /// no restart. In memory only; a request can also set its own (`ymirctl chat --adapter-scale`).
    Scale(ScaleArgs),
}

#[derive(Args, Debug)]
pub struct SlotsArgs {
    /// The vLLM model's id (a `[[model.local]]` in node.toml).
    pub base: String,
    /// How many adapters it can apply at once (vLLM `--max-loras`).
    #[arg(required_unless_present = "off")]
    pub slots: Option<u32>,
    /// Turn LoRA off again for it (unless it lists adapters).
    #[arg(long, conflicts_with = "slots")]
    pub off: bool,
}

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// The stored adapter's id (as `ymirctl models list --node NAME` shows it); it serves under it.
    pub id: String,
    /// The model it runs on, served by that node.
    #[arg(long)]
    pub base: String,
    /// How strongly it applies (llama.cpp; default 1.0).
    #[arg(long)]
    pub scale: Option<f32>,
    #[command(flatten)]
    pub node: super::NodeArgs,
}

#[derive(Args, Debug)]
pub struct UnserveArgs {
    /// The adapter's served id.
    pub id: String,
    #[command(flatten)]
    pub node: super::NodeArgs,
}

#[derive(Args, Debug)]
pub struct ScaleArgs {
    /// The adapter's served id.
    pub id: String,
    /// The new scale (for example 0.5); leave it out with --reset for 1.0.
    #[arg(required_unless_present = "reset", allow_negative_numbers = true)]
    pub scale: Option<f32>,
    /// Back to the adapter's default (1.0).
    #[arg(long, conflicts_with = "scale")]
    pub reset: bool,
    #[command(flatten)]
    pub node: super::NodeArgs,
}

impl AdapterCmd {
    /// The node a live change goes to (`--node`); `None` for the node.toml commands.
    pub fn node(&self) -> Option<String> {
        match self {
            AdapterCmd::Serve(a) => a.node.node.clone(),
            AdapterCmd::Unserve(a) => a.node.node.clone(),
            AdapterCmd::Scale(a) => a.node.node.clone(),
            AdapterCmd::Add(_) | AdapterCmd::Remove(_) | AdapterCmd::List | AdapterCmd::Slots(_) => None,
        }
    }
}

#[derive(Args, Debug)]
pub struct AddArgs {
    /// The base model's id (a `[[model.local]]` in node.toml).
    pub base: String,
    /// The adapter: a LoRA GGUF for llama.cpp (absolute, or relative to the models directory), or
    /// a PEFT directory or Hub id for vLLM.
    pub file: String,
    /// The model id it serves as (default `BASE+NAME`, NAME being the file name without `.gguf`).
    #[arg(long)]
    pub id: Option<String>,
    /// How strongly it applies (llama.cpp, per request; default 1.0).
    #[arg(long)]
    pub scale: Option<f32>,
    /// The adapter's revision (a training run id or hash), so the answer cache never mixes two
    /// versions under one id.
    #[arg(long)]
    pub revision: Option<String>,
    /// The base's revision it was trained on (default: the base's `revision` now, when it has
    /// one). If the base's revision changes later, the adapter is not served until retrained or
    /// this is updated.
    #[arg(long)]
    pub base_revision: Option<String>,
    /// The base's quantisation it was trained on (default: the base's `quant` now, when set).
    #[arg(long)]
    pub base_quant: Option<String>,
}

#[derive(Args, Debug)]
pub struct RemoveAdapterArgs {
    /// The adapter's id.
    pub id: String,
}

/// The default id for an adapter: `base+name`.
pub fn default_id(base: &str, file: &str) -> String {
    let name = Path::new(file.trim_end_matches('/')).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| file.to_string());
    let name = name.strip_suffix(".gguf").unwrap_or(&name).to_ascii_lowercase();
    format!("{base}+{name}")
}

pub fn run(ctx: &Ctx, out: &mut dyn Write, cmd: AdapterCmd) -> Result<()> {
    let config = &ctx.config;
    let text = std::fs::read_to_string(config).with_context(|| format!("reading {} (run `ymirctl setup` first)", config.display()))?;
    let cfg: NodeConfig = toml::from_str(&text).with_context(|| format!("parsing {}", config.display()))?;
    match cmd {
        AdapterCmd::Serve(_) | AdapterCmd::Unserve(_) | AdapterCmd::Scale(_) => {
            bail!(
                "this changes what a running node serves: name it with --node (this machine's name works too); to change node.toml instead, use `ymirctl models adapter add` or `remove`"
            )
        }
        AdapterCmd::List => list(out, &cfg),
        AdapterCmd::Slots(a) => {
            let slots = if a.off { None } else { a.slots };
            let new = set_slots(&text, &a.base, slots)?;
            toml::from_str::<NodeConfig>(&new).context("the edited node.toml does not parse")?.validate()?;
            save(config, &text, &new)?;
            match slots {
                Some(n) => writeln!(
                    out,
                    "`{}` starts with LoRA on (up to {n} adapters at once). Restart ymird to apply; then `ymirctl models adapter serve ID --base {} --node NAME` serves an adapter on it without a restart.",
                    a.base, a.base
                )?,
                None => writeln!(out, "`{}` no longer starts with LoRA on unless it lists adapters. Restart ymird to apply.", a.base)?,
            }
            Ok(())
        }
        AdapterCmd::Add(a) => {
            let id = a.id.clone().unwrap_or_else(|| default_id(&a.base, &a.file));
            let base = cfg.model.local.iter().find(|m| m.id == a.base);
            let adapter = LocalAdapter {
                id: id.clone(),
                file: a.file.clone(),
                scale: a.scale,
                revision: a.revision.clone(),
                base_revision: a.base_revision.clone().or_else(|| base.and_then(|m| m.revision.clone())),
                base_quant: a.base_quant.clone().or_else(|| base.and_then(|m| m.quant.clone())),
            };
            let new = add(&text, &a.base, &adapter)?;
            let check = validated(&new, &a.base, &id, ctx)?;
            let path = check
                .model
                .local
                .iter()
                .find(|m| m.id == a.base)
                .and_then(|m| m.lora.iter().find(|x| x.id == id))
                .map(|x| x.path(&check.models_dir()));
            save(config, &text, &new)?;
            writeln!(out, "Added LoRA adapter `{id}` on `{}` ({}).", a.base, path.unwrap_or_default())?;
            let trained = [
                adapter.base_revision.as_deref().map(|r| format!("revision {r}")),
                adapter.base_quant.as_deref().map(|q| format!("quantisation {q}")),
            ];
            let trained: Vec<String> = trained.into_iter().flatten().collect();
            if !trained.is_empty() {
                writeln!(
                    out,
                    "Recorded that it was trained on {} of `{}`: if the base changes, ymird stops serving it until it is retrained.",
                    trained.join(" and "),
                    a.base
                )?;
            }
            writeln!(out, "Restart ymird to serve it; requests for model `{id}` then run on `{}` with the adapter applied.", a.base)?;
            Ok(())
        }
        AdapterCmd::Remove(a) => {
            let (new, base) = remove(&text, &a.id)?;
            toml::from_str::<NodeConfig>(&new).context("the edited node.toml does not parse")?.validate()?;
            save(config, &text, &new)?;
            writeln!(out, "Removed adapter `{}` from `{base}` in node.toml (its file is kept). Restart ymird to stop serving it.", a.id)?;
            Ok(())
        }
    }
}

fn list(out: &mut dyn Write, cfg: &NodeConfig) -> Result<()> {
    let dir = cfg.models_dir();
    let mut any = false;
    for m in &cfg.model.local {
        if let Some(n) = m.lora_slots {
            any = true;
            writeln!(out, "{:<32} LoRA on, up to {n} adapters at once (lora_slots), served live with `adapter serve`", m.id)?;
        }
        for a in &m.lora {
            any = true;
            let p = a.path(&dir);
            let here = if Path::new(&p).exists() {
                ""
            } else if a.file.contains('/') && !Path::new(&a.file).is_absolute() && !a.file.ends_with(".gguf") {
                " (Hub id)"
            } else {
                " (missing)"
            };
            let mut extra = String::new();
            if let Some(s) = a.scale {
                extra.push_str(&format!(", scale {s}"));
            }
            if let Some(r) = &a.revision {
                extra.push_str(&format!(", revision {r}"));
            }
            if let Some(r) = &a.base_revision {
                extra.push_str(&format!(", trained on base revision {r}"));
            }
            if let Some(q) = &a.base_quant {
                extra.push_str(&format!(", trained on {q}"));
            }
            if let Some(why) = a.base_mismatch(m) {
                extra.push_str(&format!("\n    NOT SERVED: {why}"));
            }
            writeln!(out, "{:<32} on {:<24} {p}{here}{extra}", a.id, m.id)?;
        }
    }
    if !any {
        writeln!(out, "No LoRA adapters in node.toml. Add one with `ymirctl models adapter add BASE FILE`.")?;
    }
    Ok(())
}

/// `text` with `adapter` added under the `[[model.local]]` whose id is `base`.
pub fn add(text: &str, base: &str, adapter: &LocalAdapter) -> Result<String> {
    let mut doc: DocumentMut = text.parse().context("parsing node.toml")?;
    let Some(locals) = doc.get_mut("model").and_then(|m| m.get_mut("local")).and_then(Item::as_array_of_tables_mut) else {
        bail!("node.toml has no [[model.local]] entries");
    };
    let Some(t) = locals.iter_mut().find(|t| t.get("id").and_then(Item::as_str) == Some(base)) else {
        bail!("node.toml has no [[model.local]] with id `{base}`");
    };
    let mut row = Table::new();
    row["id"] = value(adapter.id.as_str());
    row["file"] = value(adapter.file.as_str());
    if let Some(s) = adapter.scale {
        row["scale"] = value(f64::from(s));
    }
    if let Some(r) = &adapter.revision {
        row["revision"] = value(r.as_str());
    }
    if let Some(r) = &adapter.base_revision {
        row["base_revision"] = value(r.as_str());
    }
    if let Some(q) = &adapter.base_quant {
        row["base_quant"] = value(q.as_str());
    }
    match t.get_mut("lora").and_then(Item::as_array_of_tables_mut) {
        Some(list) => list.push(row),
        None => {
            if t.contains_key("lora") {
                bail!("`{base}` has a `lora` key that is not a list of [[model.local.lora]] tables");
            }
            let mut list = ArrayOfTables::new();
            list.push(row);
            t.insert("lora", Item::ArrayOfTables(list));
        }
    }
    Ok(doc.to_string())
}

/// `text` with `lora_slots` set (or removed, for `None`) on the `[[model.local]]` whose id is `base`.
pub fn set_slots(text: &str, base: &str, slots: Option<u32>) -> Result<String> {
    let mut doc: DocumentMut = text.parse().context("parsing node.toml")?;
    let Some(locals) = doc.get_mut("model").and_then(|m| m.get_mut("local")).and_then(Item::as_array_of_tables_mut) else {
        bail!("node.toml has no [[model.local]] entries");
    };
    let Some(t) = locals.iter_mut().find(|t| t.get("id").and_then(Item::as_str) == Some(base)) else {
        bail!("node.toml has no [[model.local]] with id `{base}`");
    };
    match slots {
        Some(n) => t["lora_slots"] = value(i64::from(n)),
        None => {
            t.remove("lora_slots");
        }
    }
    Ok(doc.to_string())
}

/// `text` without adapter `id`, and the base it was on.
pub fn remove(text: &str, id: &str) -> Result<(String, String)> {
    let mut doc: DocumentMut = text.parse().context("parsing node.toml")?;
    let mut found = None;
    if let Some(locals) = doc.get_mut("model").and_then(|m| m.get_mut("local")).and_then(Item::as_array_of_tables_mut) {
        for t in locals.iter_mut() {
            let base = t.get("id").and_then(Item::as_str).unwrap_or_default().to_string();
            let Some(list) = t.get_mut("lora").and_then(Item::as_array_of_tables_mut) else { continue };
            let Some(i) = list.iter().position(|a| a.get("id").and_then(Item::as_str) == Some(id)) else { continue };
            list.remove(i);
            if list.is_empty() {
                t.remove("lora");
            }
            found = Some(base);
            break;
        }
    }
    if let Some(base) = found {
        return Ok((doc.to_string(), base));
    }
    bail!("node.toml has no LoRA adapter `{id}` (see `ymirctl models adapter list`)")
}

/// The edited config, refused unless it loads and validates, and (for a llama.cpp base whose file
/// is here) unless the adapter is a LoRA GGUF for the base's architecture.
pub(super) fn validated(new: &str, base: &str, id: &str, ctx: &Ctx) -> Result<NodeConfig> {
    let mut cfg: NodeConfig = toml::from_str(new).context("the edited node.toml does not parse")?;
    if ctx.cfg.as_ref().and_then(|c| c.model.models_dir.as_ref()).is_none() && cfg.model.models_dir.is_none() {
        cfg.model.models_dir = Some(ctx.dir().to_path_buf());
    }
    cfg.validate()?;
    if let Some(m) = cfg.model.local.iter().find(|m| m.id == base)
        && let Some(why) = m.lora.iter().find(|a| a.id == id).and_then(|a| a.base_mismatch(m))
    {
        bail!("`{id}` would not be served on `{base}`: {why}");
    }
    let launch = cfg.launches().into_iter().find(|l| l.id == base).context("the base is gone")?;
    if launch.engine == weirdnet_backends::EngineKind::Llamacpp && !Path::new(&launch.source).is_file() {
        bail!("{base}'s model file {} is not on this machine, so the adapter can't be checked against it", launch.source);
    }
    launch.check_adapters().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(cfg)
}

pub(crate) fn save(config: &Path, old: &str, new: &str) -> Result<()> {
    std::fs::write(config.with_extension("toml.bak"), old)?;
    let tmp = config.with_extension("toml.tmp");
    std::fs::write(&tmp, new)?;
    std::fs::rename(&tmp, config)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOML: &str = r#"# my node
[[model.local]]
id = "a"
file = "a.gguf" # keep me

[[model.local]]
id = "b"
file = "b.gguf"

[inference]
standin_after = "10s"
"#;

    fn adapter(id: &str) -> LocalAdapter {
        LocalAdapter { id: id.into(), file: format!("adapters/{id}.gguf"), scale: Some(0.5), base_revision: Some("r1".into()), ..Default::default() }
    }

    #[test]
    fn add_goes_under_its_base_and_keeps_the_rest() {
        let one = add(TOML, "a", &adapter("a+x")).unwrap();
        let two = add(&one, "a", &adapter("a+y")).unwrap();
        assert!(two.contains("# my node") && two.contains("# keep me"));
        let cfg: NodeConfig = toml::from_str(&two).unwrap();
        let ids: Vec<&str> = cfg.model.local[0].lora.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, ["a+x", "a+y"]);
        assert!(cfg.model.local[1].lora.is_empty(), "b keeps none");
        assert_eq!(cfg.model.local[0].lora[0].scale, Some(0.5));
        assert_eq!(cfg.model.local[0].lora[0].base_revision.as_deref(), Some("r1"));
        assert_eq!(cfg.inference.standin_after, "10s");
        assert!(add(TOML, "zz", &adapter("x")).unwrap_err().to_string().contains("no [[model.local]] with id `zz`"));

        let (back, base) = remove(&two, "a+x").unwrap();
        assert_eq!(base, "a");
        let (back, _) = remove(&back, "a+y").unwrap();
        let cfg: NodeConfig = toml::from_str(&back).unwrap();
        assert!(cfg.model.local[0].lora.is_empty());
        assert!(!back.contains("lora"), "the empty list goes too: {back}");
        assert!(remove(&back, "a+y").unwrap_err().to_string().contains("no LoRA adapter `a+y`"));
    }

    #[test]
    fn slots_are_set_and_cleared_in_place() {
        let on = set_slots(TOML, "b", Some(4)).unwrap();
        assert!(on.contains("# keep me"));
        let cfg: NodeConfig = toml::from_str(&on).unwrap();
        assert_eq!((cfg.model.local[0].lora_slots, cfg.model.local[1].lora_slots), (None, Some(4)));
        let off = set_slots(&on, "b", None).unwrap();
        assert!(!off.contains("lora_slots"), "{off}");
        assert!(set_slots(TOML, "zz", Some(1)).is_err());
    }

    #[test]
    fn default_ids() {
        assert_eq!(default_id("qwen3-8b", "adapters/Support-V2.gguf"), "qwen3-8b+support-v2");
        assert_eq!(default_id("big", "/srv/peft/tune/"), "big+tune");
        assert_eq!(default_id("big", "org/tune-lora"), "big+tune-lora");
    }
}
