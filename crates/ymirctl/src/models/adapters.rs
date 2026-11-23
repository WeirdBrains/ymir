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
            AdapterCmd::Add(_) | AdapterCmd::Remove(_) | AdapterCmd::List => None,
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
        AdapterCmd::Add(a) => {
            let id = a.id.clone().unwrap_or_else(|| default_id(&a.base, &a.file));
            let adapter = LocalAdapter { id: id.clone(), file: a.file.clone(), scale: a.scale, revision: a.revision.clone() };
            let new = add(&text, &a.base, &adapter)?;
            let check = validated(&new, &a.base, ctx)?;
            let path = check
                .model
                .local
                .iter()
                .find(|m| m.id == a.base)
                .and_then(|m| m.lora.iter().find(|x| x.id == id))
                .map(|x| x.path(&check.models_dir()));
            save(config, &text, &new)?;
            writeln!(out, "Added LoRA adapter `{id}` on `{}` ({}).", a.base, path.unwrap_or_default())?;
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
fn validated(new: &str, base: &str, ctx: &Ctx) -> Result<NodeConfig> {
    let mut cfg: NodeConfig = toml::from_str(new).context("the edited node.toml does not parse")?;
    if ctx.cfg.as_ref().and_then(|c| c.model.models_dir.as_ref()).is_none() && cfg.model.models_dir.is_none() {
        cfg.model.models_dir = Some(ctx.dir().to_path_buf());
    }
    cfg.validate()?;
    let launch = cfg.launches().into_iter().find(|l| l.id == base).context("the base is gone")?;
    if launch.engine == weirdnet_backends::EngineKind::Llamacpp && !Path::new(&launch.source).is_file() {
        bail!("{base}'s model file {} is not on this machine, so the adapter can't be checked against it", launch.source);
    }
    launch.check_adapters().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(cfg)
}

fn save(config: &Path, old: &str, new: &str) -> Result<()> {
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
        LocalAdapter { id: id.into(), file: format!("adapters/{id}.gguf"), scale: Some(0.5), revision: None }
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
    fn default_ids() {
        assert_eq!(default_id("qwen3-8b", "adapters/Support-V2.gguf"), "qwen3-8b+support-v2");
        assert_eq!(default_id("big", "/srv/peft/tune/"), "big+tune");
        assert_eq!(default_id("big", "org/tune-lora"), "big+tune-lora");
    }
}
