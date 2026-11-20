//! Prints a standalone HTML reference of every `ymirctl` command, subcommand, argument and flag,
//! read from the clap definitions, so it cannot drift from the CLI.
//!
//! `cargo run -p ymirctl --example cli_reference > reference.html` (the Pages workflow does this).

use clap::{Command, CommandFactory};

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn usage(cmd: &Command, path: &str) -> String {
    let mut c = cmd.clone().disable_help_flag(true).disable_version_flag(true).bin_name(path);
    c.render_usage().to_string().trim_start_matches("Usage: ").to_string()
}

fn arg_label(a: &clap::Arg) -> String {
    let mut parts = Vec::new();
    if let Some(s) = a.get_short() {
        parts.push(format!("-{s}"));
    }
    if let Some(l) = a.get_long() {
        parts.push(format!("--{l}"));
    }
    if parts.is_empty() {
        return format!("&lt;{}&gt;", esc(&a.get_id().as_str().to_uppercase()));
    }
    let mut s = parts.join(", ");
    if a.get_action().takes_values() {
        let v = a.get_value_names().and_then(|v| v.first().map(|n| n.to_string())).unwrap_or_else(|| a.get_id().as_str().to_uppercase());
        s.push_str(&format!(" &lt;{}&gt;", esc(&v)));
    }
    s
}

fn section(out: &mut String, cmd: &Command, path: &str, depth: usize) {
    let level = (depth + 2).min(6);
    let id = path.replace(' ', "-");
    out.push_str(&format!("<h{level} id=\"{id}\"><code>{}</code></h{level}>\n", esc(path)));
    if let Some(about) = cmd.get_long_about().or_else(|| cmd.get_about()) {
        out.push_str(&format!("<p>{}</p>\n", esc(&about.to_string())));
    }
    out.push_str(&format!("<pre><code>{}</code></pre>\n", esc(&usage(cmd, path))));
    let args: Vec<_> = cmd.get_arguments().filter(|a| !matches!(a.get_id().as_str(), "help" | "version")).collect();
    if !args.is_empty() {
        out.push_str("<table>\n<thead><tr><th>Argument</th><th>What it does</th></tr></thead>\n<tbody>\n");
        for a in args {
            let mut help = a.get_help().map(|h| esc(&h.to_string())).unwrap_or_default();
            let defaults: Vec<String> = a.get_default_values().iter().map(|d| d.to_string_lossy().into_owned()).collect();
            if !defaults.is_empty() {
                help.push_str(&format!(" (default: <code>{}</code>)", esc(&defaults.join(", "))));
            }
            if let Some(env) = a.get_env() {
                help.push_str(&format!(" (environment: <code>{}</code>)", esc(&env.to_string_lossy())));
            }
            if a.is_required_set() {
                help.push_str(" (required)");
            }
            out.push_str(&format!("<tr><td><code>{}</code></td><td>{help}</td></tr>\n", arg_label(a)));
        }
        out.push_str("</tbody></table>\n");
    }
    for sub in cmd.get_subcommands().filter(|s| s.get_name() != "help") {
        section(out, sub, &format!("{path} {}", sub.get_name()), depth + 1);
    }
}

fn contents(out: &mut String, cmd: &Command, path: &str) {
    out.push_str("<ul>\n");
    for sub in cmd.get_subcommands().filter(|s| s.get_name() != "help") {
        let p = format!("{path} {}", sub.get_name());
        let about = sub.get_about().map(|a| esc(&a.to_string())).unwrap_or_default();
        out.push_str(&format!("<li><a href=\"#{}\"><code>{}</code></a>: {about}", p.replace(' ', "-"), esc(&p)));
        if sub.has_subcommands() {
            contents(out, sub, &p);
        }
        out.push_str("</li>\n");
    }
    out.push_str("</ul>\n");
}

fn main() {
    let cmd = ymirctl::Cli::command();
    let name = cmd.get_name().to_string();
    let mut body = String::new();
    contents(&mut body, &cmd, &name);
    section(&mut body, &cmd, &name, 0);
    println!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
<title>ymirctl command reference · Ymir</title>\n<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:56rem;margin:3rem auto;padding:0 1rem}}\
code,pre{{font-family:ui-monospace,monospace;font-size:.92em}}pre{{background:#f3f3f5;padding:.7rem 1rem;overflow:auto;border-radius:6px}}\
table{{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}}td,th{{border-bottom:1px solid #ddd;padding:.35rem .6rem;text-align:left;vertical-align:top}}\
h3,h4{{margin-top:2.2rem}}@media(prefers-color-scheme:dark){{body{{background:#15151a;color:#e8e8ee}}pre{{background:#22222a}}td,th{{border-color:#333}}a{{color:#9ab8ff}}}}</style>\n</head>\n<body>\n\
<h1>ymirctl command reference</h1>\n<p>Generated from the <code>ymirctl</code> command definitions (version {version}) every time the site is built, so it always matches the code. \
For worked examples grouped by task, read the <a href=\"../guides/ymirctl/\">ymirctl guide</a>.</p>\n{body}\
<p><a href=\"../\">Back to the documentation</a>. Ymir is a <a href=\"https://weirdbrains.com\">Weird Brains</a> project, open source under the BSD-3-Clause license; copyright Weird Brains LLC.</p>\n</body>\n</html>",
        version = env!("CARGO_PKG_VERSION"),
        body = body
    );
}
