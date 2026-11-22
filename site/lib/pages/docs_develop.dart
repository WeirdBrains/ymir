import 'package:jaspr/jaspr.dart';

import '../components/layout.dart';
import '../site.dart';

const _html = r'''
<p>Ymir is a thin app on top of the <a href="https://net.weirdbrains.com">Weirdnet</a> crates, by
<a href="https://weirdbrains.com">Weird Brains</a>. This page is for people changing it. The full design is in
<a href="${repo}/blob/main/docs/DESIGN.md">docs/DESIGN.md</a>, which is kept current in the same change as the code.</p>

<h2 id="layout">Repository layout</h2>
<table>
<thead><tr><th>Path</th><th>What</th></tr></thead>
<tbody>
<tr><td><code>crates/ymir-core</code></td><td>The node coordinator, role wiring, config, setup, pool membership, control protocol and service files.</td></tr>
<tr><td><code>crates/ymird</code></td><td>The node daemon.</td></tr>
<tr><td><code>crates/ymirctl</code></td><td>The command-line front end (<code>setup</code>, <code>service</code>, <code>pool</code>, <code>status</code>, <code>chat</code>, <code>bridge</code>, …).</td></tr>
<tr><td><code>crates/ymir-testkit</code></td><td>Helpers for tests, including a mock engine.</td></tr>
<tr><td><code>site/</code>, <a href="${repo}/blob/main/roadmap.toml"><code>roadmap.toml</code></a>, <a href="${repo}/blob/main/ROADMAP.md"><code>ROADMAP.md</code></a></td><td>This website (Jaspr) and the roadmap it is generated from.</td></tr>
<tr><td><code>packaging/</code>, <code>scripts/package.py</code></td><td>The systemd unit, the Arch recipe, the Homebrew template and the packager.</td></tr>
<tr><td><code>brand/</code>, <a href="${repo}/blob/main/BRAND.md"><code>BRAND.md</code></a></td><td>Logo, icons and usage rules.</td></tr>
</tbody>
</table>
<p>Generic pieces (networking, auth, routing, backends, the OpenAI bridge, the answer cache, hardware detection) live in
<code>weirdnet-*</code> crates that other apps share. Ymir never creates or edits one; it uses the published versions. If
something is reusable, it is proposed to the Weirdnet crates, not built here. The generated
<a href="${api}">API reference</a> covers the Ymir crates themselves.</p>

<h2 id="roles">Architecture in short</h2>
<p>One <code>ymird</code> runs up to three roles, each an independent toggle in <a href="${repo}/blob/main/docs/DESIGN.md#config-format-one-file-per-node"><code>node.toml</code></a>:</p>
<ul>
<li><strong>Model</strong> nodes run engines (llama.cpp, vLLM or a wrapped stock server), advertise their models, and load or unload on command.</li>
<li><strong>Inference</strong> nodes keep one catalog, balance load, place models and fail over mid-answer.</li>
<li><strong>App</strong> nodes ask for models: the loopback OpenAI bridge, <code>ymirctl</code>, and Bragi.</li>
</ul>
<p>The protocols are the same whether a peer is local or remote; the in-process path is only a fast path. Apps never talk
to model nodes directly. Devices are linked to a Weirdnet identity (<code>ymirctl pool</code>), and what a linked device
may do is decided by signed grants in the device roster. Protocol ids are <code>/ymir/…</code>, and the link scheme is
<code>weirdnet-ymir://</code>. Read <a href="${repo}/blob/main/docs/DESIGN.md">DESIGN.md</a> for the details.</p>

<h2 id="build">Build and test</h2>
<p>You need Rust 1.99. Point the build output at a scratch directory and clean it afterwards if disk is tight.</p>
<pre><code>git clone https://github.com/WeirdBrains/ymir && cd ymir
cargo build --release -p ymird -p ymirctl
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo doc --workspace --no-deps          # the API reference, in target/doc</code></pre>
<p>Tests and manual runs set <code>YMIR_HOME</code> (and <code>WEIRDNET_MODELS_DIR</code>) to a temporary directory. Tests
never download a model, start a GPU backend or touch the real network; tests that need a real model are marked
<code>#[ignore]</code>. Nothing in the tests opens a window.</p>
<p>The site: <code>cd site &amp;&amp; dart pub get &amp;&amp; jaspr serve</code>. After editing <code>roadmap.toml</code> run
<code>python3 site/tool/gen_roadmap.py</code>; CI checks it with <code>--check</code>.</p>

<h2 id="contribute">Contributing</h2>
<ul>
<li>Follow the shared conventions in the shared Weirdnet conventions, then <a href="${repo}/blob/main/AGENTS.md">the repository's own <code>AGENTS.md</code></a>.</li>
<li>Work on <code>main</code>, in small logical commits, after <code>git pull --rebase</code>. Don't force-push or skip hooks.</li>
<li>Keep <a href="${repo}/blob/main/docs/DESIGN.md">DESIGN.md</a> current in the same change as the code.</li>
<li>No secrets in configs, logs, docs or commits. Config refers to secrets with <code>env:</code> or <code>file:</code>.</li>
<li>Public content (docs, site, comments, fixtures) uses placeholder hosts and addresses such as <code>server-a</code> and <code>192.0.2.x</code>, never real machines.</li>
<li>Ymir is BSD-3-Clause, copyright <a href="https://weirdbrains.com">Weird Brains</a> LLC; every crate sets <code>license = "BSD-3-Clause"</code>.</li>
</ul>

<h2 id="release">Releasing</h2>
<p>Releases use the shared release script vendored as <code>scripts/weirdnet_release.py</code> and configured in
<a href="${repo}/blob/main/release.toml"><code>release.toml</code></a>; never edit the vendored copy and don't write a per-project release flow. In order:</p>
<ol>
<li>Dry run: <code>python3 scripts/weirdnet_release.py --dry-run</code>. It refuses a dirty tree, the wrong branch, being behind origin or an existing tag, and runs the checks listed in <code>release.toml</code>.</li>
<li>Release for real: the script bumps the versions, updates the changelog, commits <code>release: vX.Y.Z</code>, tags <code>vX.Y.Z</code> and pushes main and the tag in one non-forced push.</li>
<li>Build the platform files (<code>scripts/package.py</code> for Linux; the release workflow for all platforms) and upload them with the script's <code>github</code> and <code>assets</code> steps.</li>
<li>Publish the Arch recipe with the script's <code>taur</code> step (try <code>--dry-run</code> first).</li>
</ol>
<p>Versions follow semantic versioning; before 1.0 a new feature bumps the minor version and a fix or docs-only change
bumps the patch version. Releases are made by the maintainers.</p>
''';

class DevelopPage extends StatelessComponent {
  const DevelopPage({super.key});

  @override
  Component build(BuildContext context) {
    return Component.fragment([
      pageHero(
        eyebrow: 'Docs · Develop',
        title: 'Developer guide',
        lede: 'Build, test, understand and contribute to Ymir.',
      ),
      prose(
        _html.replaceAll(r'${repo}', repoUrl).replaceAll(r'${api}', siteLink('/docs/api/')),
      ),
    ]);
  }
}
