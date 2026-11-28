//! Edits to node.toml that keep the operator's comments and layout: a `[[model.local]]` entry
//! appended, a `[[model.local.lora]]` added under its base. Each edit is checked to load and
//! validate before it is written; the old file is kept as `node.toml.bak` and the new one moved
//! into place. Used by `ymirctl models` on the node and by an upload over Weirdnet that is saved
//! into the receiving node's config (`crate::upload_net`).

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use toml_edit::{ArrayOfTables, DocumentMut, Item, Table, value};

use crate::NodeConfig;
use crate::config::{LocalAdapter, LocalModel};

#[derive(Serialize)]
struct Stanza<'a> {
    model: StanzaModel<'a>,
}

#[derive(Serialize)]
struct StanzaModel<'a> {
    local: [&'a LocalModel; 1],
}

/// A `[[model.local]]` entry, in the shape Ymir's own `setup` writes.
pub fn stanza(m: &LocalModel) -> String {
    toml::to_string(&Stanza { model: StanzaModel { local: [m] } }).unwrap_or_default()
}

/// `context`: a fixed context the operator asked for; `None` leaves it to ymird.
pub fn local_model(id: &str, backend: &str, role: &str, file: &str, context: Option<u64>, accelerated: bool) -> LocalModel {
    let llama = backend == "llamacpp";
    LocalModel {
        id: id.to_string(),
        backend: Some(backend.to_string()),
        role: Some(role.to_string()),
        file: Some(file.to_string()),
        context: context.map(|c| c.min(u64::from(u32::MAX)) as u32),
        gpu_layers: (llama && accelerated).then_some(99),
        idle_ttl: Some("1800s".into()),
        args: if llama { ["-fa", "on", "--jinja"].map(String::from).to_vec() } else { Vec::new() },
        ..Default::default()
    }
}

/// Appends `stanza` (a `[[model.local]]` entry for `id`) to `config` under the comment `note`.
pub fn append_model(config: &Path, id: &str, stanza: &str, note: &str) -> Result<()> {
    let text = std::fs::read_to_string(config).with_context(|| format!("reading {} (run `ymirctl setup` first)", config.display()))?;
    let cfg: NodeConfig = toml::from_str(&text).with_context(|| format!("parsing {}", config.display()))?;
    if cfg.model.local.iter().any(|m| m.id == id) {
        bail!("{} already has a model with id `{id}`; pass --id to pick another", config.display());
    }
    let sep = if text.ends_with('\n') { "\n" } else { "\n\n" };
    let new = format!("{text}{sep}{note}\n{stanza}");
    let check: NodeConfig = toml::from_str(&new).context("the new entry does not parse")?;
    check.validate()?;
    save(config, &text, &new)
}

/// `text` with `adapter` added under the `[[model.local]]` whose id is `base`.
pub fn add_adapter(text: &str, base: &str, adapter: &LocalAdapter) -> Result<String> {
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

/// `text` with the `[[model.local]]` whose id is `base` given `mmproj = path` (a vision projector).
pub fn set_mmproj(text: &str, base: &str, path: &str) -> Result<String> {
    let mut doc: DocumentMut = text.parse().context("parsing node.toml")?;
    let Some(locals) = doc.get_mut("model").and_then(|m| m.get_mut("local")).and_then(Item::as_array_of_tables_mut) else {
        bail!("node.toml has no [[model.local]] entries");
    };
    let Some(t) = locals.iter_mut().find(|t| t.get("id").and_then(Item::as_str) == Some(base)) else {
        bail!("node.toml has no [[model.local]] with id `{base}`");
    };
    t["mmproj"] = value(path);
    Ok(doc.to_string())
}

/// Writes `new` over `config`, keeping `old` as `node.toml.bak`.
pub fn save(config: &Path, old: &str, new: &str) -> Result<()> {
    std::fs::write(config.with_extension("toml.bak"), old)?;
    let tmp = config.with_extension("toml.tmp");
    std::fs::write(&tmp, new)?;
    std::fs::rename(&tmp, config)?;
    Ok(())
}
