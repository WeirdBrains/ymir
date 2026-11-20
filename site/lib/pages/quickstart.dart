import 'package:jaspr/jaspr.dart';

import '../components/layout.dart';
import '../site.dart';

const _html = r'''
<h2 id="build">Build it</h2>
<p>Nothing is released yet, so Ymir builds from source. You need Rust 1.99 and a <code>llama-server</code> from
<a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a>. On an NVIDIA PC, take the release's
<code>ubuntu-cuda-12.8-x64</code> build and unpack its matching <code>cudart-</code> archive into the same directory
(no CUDA toolkit needed). llama.cpp offloads every layer that fits, so <code>node.toml</code> needs only
<code>llamacpp = "system:/path/to/llama-server"</code>.</p>
<pre><code>git clone https://github.com/WeirdBrains/ymir && cd ymir
cargo build --release -p ymird -p ymirctl</code></pre>
<h2 id="run">Set up and run</h2>
<pre><code>ymirctl hw                                   # what this machine has and the model budget
ymirctl setup --llama-server /path/to/llama-server --model /path/to/model.gguf
#   or: ymirctl setup --download             # catalog models that fit, asks before downloading
ymird &amp;                                      # model, inference and app roles in one process
ymirctl chat "Hello"                         # streams through the loopback bridge
ymirctl status                               # roles, models and their state
ymirctl load &lt;model&gt; / ymirctl unload &lt;model&gt;
ymirctl bridge token                         # API key for Jan: base URL http://127.0.0.1:8484/v1
ymirctl service install                      # or <code>setup --service</code>: a systemd user unit / launchd agent</code></pre>
<p>The bridge serves <code>/v1/models</code>, <code>/v1/chat/completions</code>, <code>/v1/completions</code> and
<code>/v1/embeddings</code>, on <code>127.0.0.1</code> only. While a big model loads, a stand-in can answer, flagged with
the <code>x-weirdnet-standin</code> header. Send <code>x-wait: true</code>, or use <code>ymirctl chat --wait</code>,
to wait for the real model.</p>
<h2 id="config">Some settings</h2>
<p>One file per node, <code>~/.config/ymir/node.toml</code> (move it with <code>YMIR_HOME</code>; model files move with
<code>WEIRDNET_MODELS_DIR</code>). Secrets are <code>env:</code> or <code>file:</code> references, never inline.</p>
<pre><code>[[model.local]]
id = "qwen3-8b"
swap_group = "main"
idle_ttl = "10m"                      # unload when idle

[[model.local]]
id = "smollm2-360m"
role = "standin"                      # answers while the main model cold-starts

[[model.local]]
id = "qwen2.5-7b"
backend = "vllm"
wrap = "http://127.0.0.1:8000/v1"     # a stock server Ymir didn't start (loopback only)
wrap_api_key = "env:VLLM_API_KEY"</code></pre>
<p>A chat GGUF used for embeddings needs <code>args = ["--pooling", "mean"]</code> on llama-server.</p>
<h2 id="more">More machines</h2>
<p>Machines join over Weirdnet by being linked to your own identity: <code>ymirctl pool link</code> on a node prints a
code, and you open it on the other device (Morph, or <code>ymirctl pool add</code>). Both sides show the same six
digits to compare. Roles over Weirdnet work today, but they have so far only been tested on one computer. Several model
nodes in one pool (M4) is in progress; see the <a href="${roadmapUrl}">roadmap</a>.</p>
''';

class QuickstartPage extends StatelessComponent {
  const QuickstartPage({super.key});

  @override
  Component build(BuildContext context) {
    return Component.fragment([
      pageHero(
        eyebrow: 'Quickstart',
        title: 'Set up one box',
        lede: 'From the README: build from source, set up, and point Jan at the loopback bridge.',
      ),
      prose(_html.replaceAll(r'${roadmapUrl}', siteLink('/roadmap'))),
    ]);
  }
}
