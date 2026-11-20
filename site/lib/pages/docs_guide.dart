import 'package:jaspr/jaspr.dart';

import '../components/layout.dart';
import '../site.dart';

const _html = r'''
<p>This guide takes you from nothing to a working private model cloud: install Ymir, add a second device, connect the
Bragi chat app, and point Jan at it. Commands are exactly those <code>ymirctl</code> offers; run
<code>ymirctl --help</code> or <code>ymirctl &lt;command&gt; --help</code> for the full list.</p>
<p><strong>What is released.</strong> Nothing in Ymir or Bragi is released yet. Ymir's release files (step 1) are
built by its release workflow but not published, so for now you build them yourself from source, which takes one more
command. The bridge token and the Weirdnet identity are kept in files readable only by you (mode 0600); storing them in
the operating system's keyring comes later.</p>
<p>Ymir is a <a href="https://weirdbrains.com">Weird Brains</a> project. Devices find each other over
<a href="https://net.weirdbrains.com">Weirdnet</a>, so a pool needs no open ports and no API keys.</p>

<h2 id="install">1. Install Ymir on a server or desktop</h2>
<p>Ymir is one download with <a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a> built in: no separate
llama.cpp, CUDA toolkit, Python or Rust to install. Pick the build for the machine:</p>
<ul>
<li><strong><code>ymir</code></strong>: llama.cpp for the CPU and for any GPU through Vulkan (NVIDIA, AMD, Intel).</li>
<li><strong><code>ymir-cuda</code></strong>: llama.cpp with CUDA 13 and its runtime libraries, for NVIDIA GPUs from
Turing (RTX 20 series) through Blackwell, DGX Spark (GB10) included. It needs only the NVIDIA driver (580 or newer).
Linux x86_64 and ARM64.</li>
</ul>
<p>Each comes as a <code>tar.gz</code> or <code>zip</code> you unpack anywhere (no root needed), a <code>.deb</code>, an
<code>.rpm</code> and an Arch package (<code>ymir-bin</code>, <code>ymir-cuda-bin</code>), and there is a Nix flake.
A server install is three steps: unpack, set up, start.</p>
<pre><code># 1. Unpack (user level), or install the package: sudo apt install ./ymir-cuda_VERSION-1_arm64.deb
tar xzf ymir-cuda-VERSION-aarch64-unknown-linux-gnu.tar.gz -C ~/.local/opt
ln -s ~/.local/opt/ymir-cuda-VERSION-aarch64-unknown-linux-gnu/bin/ymir* ~/.local/bin/

# 2 and 3. Pick models that fit, download them, and start ymird as a service
ymirctl setup --download --yes --service
loginctl enable-linger $USER                 # keep it running after you log out</code></pre>
<p>Setup says which engine it found, such as <code>llama.cpp: llama.cpp b12590 (cuda: cpu, cuda), shipped with
Ymir</code>, and tells you when the machine has an NVIDIA GPU but the <code>ymir</code> build is installed (it then runs
on the GPU through Vulkan, which is slower than CUDA). The engine runs as its own process under <code>ymird</code>, so a
model that crashes is restarted without taking <code>ymird</code> down.</p>
<p><strong>Until the first release: build the files yourself.</strong> You need Rust 1.99 and Python 3.11. The package
script downloads the pinned llama.cpp build (checking its SHA-256) and writes the same files a release has into
<code>dist/</code>:</p>
<pre><code>git clone https://github.com/WeirdBrains/ymir && cd ymir
cargo build --release -p ymird -p ymirctl
python3 scripts/package.py                   # dist/ymir-…; add --variant cuda for dist/ymir-cuda-…</code></pre>
<p><strong>Your own engine.</strong> <code>ymirctl setup --llama-server /path/to/llama-server</code> uses a llama.cpp
build you provide instead of the bundled one. vLLM, for NVIDIA servers that want it, stays a separate install:
<code>ymird</code> finds <code>vllm</code> on <code>PATH</code> or wraps a <code>vllm serve</code> that is already
running.</p>
<p><strong>Look at the machine.</strong> <code>ymirctl hw</code> shows the hardware and the memory budget for models
(<code>--json</code> for scripts).</p>
<p><strong>Set it up.</strong> <code>ymirctl setup</code> detects the hardware, picks models that fit and writes
<code>~/.config/ymir/node.toml</code> (move the directory with <code>YMIR_HOME</code> or <code>--home</code>). Choose
one of:</p>
<pre><code>ymirctl setup --dry-run                      # show the proposal, write nothing
ymirctl setup --download                     # catalog models that fit; asks before downloading (--yes skips the question)
ymirctl setup --model /path/to/model.gguf    # your own GGUF; repeat --model, the first is the main model</code></pre>
<p>Useful options: <code>--name</code> (this node's name), <code>--memory-budget 10GiB</code>,
<code>--reserve 12GiB</code> (memory kept for the OS), <code>--models-dir</code>, <code>--port</code> (bridge port on
127.0.0.1, default 8484) and <code>--force</code> to replace an existing <code>node.toml</code>.</p>
<p><strong>Roles.</strong> One <code>ymird</code> can be three things, and <code>--roles</code> picks which (default
<code>model,inference,app</code>):</p>
<ul>
<li><code>model</code>: runs the models and loads or unloads them on command.</li>
<li><code>inference</code>: coordinates the pool (one catalog, load balancing, placement, failover). Run more than one for redundancy.</li>
<li><code>app</code>: the loopback bridge and <code>ymirctl chat</code>, for asking the pool questions.</li>
</ul>
<p>A single box keeps all three. For a GPU server that only runs models, use <code>ymirctl setup --roles model</code>
and let another device run the inference role.</p>
<p><strong>Run it as a service.</strong> <code>ymirctl service install</code> writes a user service (systemd
<code>--user</code> on Linux, a launchd agent on macOS), then enables and starts it. <code>setup --service</code> does
setup and this in one go. <code>ymirctl service status</code> says whether it is running and
<code>ymirctl service uninstall</code> removes it. Add <code>--no-start</code> to write the unit without starting it,
and <code>--dry-run</code> to see what would happen. To keep a systemd user service running after you log out, run
<code>loginctl enable-linger $USER</code>.</p>
<p><strong>Check.</strong></p>
<pre><code>ymirctl status          # roles, models and their state
ymirctl models          # the pool's merged model list
ymirctl chat "Hello"    # streams an answer (add --wait to wait for a model that is still loading)</code></pre>
<p>While a large model loads, a small stand-in model can answer, flagged with the <code>x-weirdnet-standin</code>
header. <code>ymirctl load &lt;model&gt;</code> and <code>ymirctl unload &lt;model&gt;</code> control models by hand.</p>

<h2 id="pool">2. Start a pool and add a device</h2>
<p>A pool is the set of devices linked to one Weirdnet identity. Only those devices can use each other's models.</p>
<p><strong>The first device.</strong> If you already have a Weirdnet identity (in Morph, for example), skip to the next
paragraph and link this node to it. Otherwise start a new identity of your own on this node:</p>
<pre><code>ymirctl pool init --standalone     # a new identity, with this node as its first device
systemctl --user restart ymird     # restart ymird so it joins Weirdnet</code></pre>
<p>Without <code>--standalone</code> the command refuses, because it would create an identity separate from any you
already hold.</p>
<p><strong>Add another device.</strong> On the <em>new</em> device, install Ymir (step 1), then:</p>
<pre><code>ymirctl pool link</code></pre>
<p>It prints a <code>weirdnet-ymir://link/…</code> code that is good for two minutes. On a device that is
<em>already</em> in the pool, paste that code:</p>
<pre><code>ymirctl pool add 'weirdnet-ymir://link/…' [--profile-name NAME]</code></pre>
<p>Both screens now show the same six digits. Compare them: if they match, answer <code>y</code> on both devices. If they
differ, answer <code>N</code>; someone else may be in the middle. When the link finishes, restart <code>ymird</code>
on the new device so it joins Weirdnet. A code from Morph (<code>weirdnet-morph://link/…</code>) works with
<code>pool add</code> too, and Morph accepts Ymir's codes (Identity, Add device).</p>
<p><strong>Look and tidy.</strong></p>
<pre><code>ymirctl pool devices            # the identity and every linked device (this node, linked, revoked)
ymirctl pool revoke PEER_ID     # cut a device off everywhere the roster reaches
ymirctl nodes                   # the model nodes the pool routes to
ymirctl drain NODE              # stop sending a node new requests (--resume undoes it)</code></pre>
<p><code>pool link</code> refuses on a node that is already linked. Devices that join over Weirdnet find each other on
your local network (mDNS) or through the default <a href="https://weirdbrains.com">Weird Brains</a> relay, so they
can be anywhere.</p>

<h2 id="bragi">3. Connect Bragi</h2>
<p><a href="${bragiSite}">Bragi</a> is the chat app, for Linux desktops and Android phones. It does not talk to
<code>ymird</code> directly: it joins your Weirdnet identity like any other device, finds your inference nodes and
streams answers from their models. You can link it either way round, and both ways end with the same six-digit check.</p>
<p><strong>Bragi shows the code</strong> (the usual way for a new phone or desktop):</p>
<ol>
<li>Open Bragi on the new device. On first start it says <em>Link Bragi to your Weirdnet</em>. Tap <strong>Show a link code</strong>.</li>
<li>Bragi shows a <code>weirdnet-bragi://link/…</code> code, as text and as a QR code, for two minutes. On the phone you can copy it.</li>
<li>On a device that is already in the pool, run <code>ymirctl pool add 'weirdnet-bragi://link/…'</code> (or open it in another linked Bragi).</li>
<li><code>ymirctl</code> prints <em>Pairing with …: the code is</em> six digits, and Bragi shows <em>Do both screens show this code?</em> with the same six digits. Compare them. If they match, answer <code>y</code> on <code>ymirctl</code> and tap <strong>They match</strong> in Bragi; otherwise answer <code>N</code> and tap <strong>They don't</strong>.</li>
<li>Bragi goes online and lists the models of every inference node in the pool. Pick one and chat.</li>
</ol>
<p><strong>The pool shows the code</strong> (to add a Ymir node from a Bragi that is already linked): run
<code>ymirctl pool link</code> on the new Ymir node, open its <code>weirdnet-ymir://link/…</code> code in Bragi, compare
the six digits, and confirm on both. Restart <code>ymird</code> on the new node afterwards.</p>
<p>Ymir and Bragi read each other's link codes, so a code from either can be opened in the other.
Bragi's pairing is covered by automated tests, and by an end-to-end test of a headless Bragi and real <code>ymird</code>
nodes on one computer, in both directions; it has not yet been run across separate real machines, and both apps are
unreleased. Bragi has no QR scanner yet, so on a phone the code is pasted rather than scanned.</p>
<p>You need at least one device with the <code>inference</code> role in the pool for Bragi to have models to show. Phones
are reachable only while the app is open.</p>

<h2 id="jan">4. Use Jan through the local bridge</h2>
<p><code>ymird</code>'s <code>app</code> role runs an OpenAI-compatible bridge on <code>127.0.0.1</code> (default port
8484). It serves <code>/v1/models</code>, <code>/v1/chat/completions</code>, <code>/v1/completions</code> and
<code>/v1/embeddings</code>, needs a bearer token on every request, and answers only for the host names
<code>127.0.0.1</code> and <code>localhost</code>, so <a href="https://jan.ai">Jan</a> must run on the same machine as
the bridge.</p>
<ol>
<li>Get the key: <code>ymirctl bridge token</code> prints it. It lives in <code>~/.config/ymir/state/bridge.token</code>. <code>ymirctl bridge token --rotate</code> makes a new one; restart <code>ymird</code> to use it. There is one shared token per bridge. All of <code>ymirctl</code> is covered in the <a href="${ymirctl}#bridge-token">ymirctl guide</a>, including running it against a service that uses another account.</li>
<li>List the model ids: <code>ymirctl models</code>.</li>
<li>In Jan, open <strong>Settings → Model Providers</strong> and add an OpenAI-compatible provider. Menu names can differ a little between Jan versions.</li>
<li>Set the base URL to <code>http://127.0.0.1:8484/v1</code> and paste the token as the API key.</li>
<li>If Jan does not list the models by itself, add them by id, exactly as <code>ymirctl models</code> shows them (the embedding model too, if you want embeddings).</li>
</ol>
<p>Test it without Jan: <code>curl -H "Authorization: Bearer $(ymirctl bridge token)" http://127.0.0.1:8484/v1/models</code>.
Send the header <code>x-wait: true</code> to wait for a model that is still loading instead of taking a stand-in's answer.</p>

<h2 id="https">5. Optional: an HTTPS address on a private network</h2>
<p>The bridge is loopback-only on purpose, and it needs no identity to use. If you also want clients on your private
network (a phone browser, a script on another computer) that cannot join Weirdnet to reach it, put a reverse proxy in
front of it on the machine that runs the bridge. Keep it on a private network such as a Tailscale tailnet; do not put it
on the open internet. Clients log in with the bridge token, so anyone holding it can use your models.</p>
<p>Example with <a href="https://caddyserver.com">Caddy</a> and a Tailscale certificate, for a machine named
<code>server-a.example.ts.net</code> whose tailnet address is <code>192.0.2.10</code> (use your own):</p>
<pre><code># 1. Turn on HTTPS certificates for your tailnet in the Tailscale admin console, then:
sudo tailscale cert server-a.example.ts.net      # writes server-a.example.ts.net.crt and .key

# 2. Caddyfile
https://server-a.example.ts.net:9443 {
    bind 192.0.2.10
    tls /etc/caddy/tls/server-a.example.ts.net.crt /etc/caddy/tls/server-a.example.ts.net.key
    reverse_proxy 127.0.0.1:8484 {
        header_up Host 127.0.0.1:8484     # the bridge refuses any other Host header
        flush_interval -1                 # stream tokens as they arrive
        transport http {
            response_header_timeout 15m   # a cold model load can take minutes
            read_timeout 15m
        }
    }
}</code></pre>
<p>Clients then use <code>https://server-a.example.ts.net:9443/v1</code> as the base URL and the bridge token as the API
key. Use the host name, not the address, because the certificate covers the name. Tailscale certificates are renewed
by re-running <code>tailscale cert</code> (a daily timer works) and reloading Caddy. Any reverse proxy that rewrites the
<code>Host</code> header and does not buffer the response works the same way.</p>

<h2 id="faq">Questions</h2>
<p><strong>Do I need to install Ymir on every machine?</strong> No, only on the machines that serve: every machine that
runs models (the <code>model</code> role), coordinates the pool (the <code>inference</code> role) or hosts the
OpenAI-compatible bridge (the <code>app</code> role). One <code>ymird</code> can be all three, and a pool needs at least
one inference node somewhere, which can be the same machine as a model node. Machines that only ask questions don't
need Ymir:</p>
<ul>
<li><a href="${bragiSite}">Bragi</a> joins your Weirdnet identity on its own and talks to your inference nodes (step 3).</li>
<li>Apps built on <code>weirdnet-agent</code> do the same.</li>
<li>Plain OpenAI clients, such as Jan or a script, use the bridge: on the machine that runs it at
<code>http://127.0.0.1:8484/v1</code> (step 4), or from another device through an HTTPS address on a private network,
with the bridge token as the API key (step 5).</li>
</ul>
<p><strong>What about a machine that only runs models?</strong> Install Ymir and set it up with
<code>ymirctl setup --roles model</code>; it then runs models for the pool and leaves routing to the inference nodes.
It still needs Ymir, because <code>ymird</code> is what joins it to the pool and starts and stops its models.</p>
<p><strong>Do I need llama.cpp or CUDA installed?</strong> No. Every Ymir download includes llama.cpp; the
<code>ymir-cuda</code> build also includes the CUDA runtime, so an NVIDIA machine needs only its driver. Only vLLM is a
separate install, and it is optional.</p>
<p><strong>What happens when a model crashes?</strong> Each model runs in its own engine process. If one dies,
<code>ymird</code> keeps running and starts the model again (a request it had not started answering is retried once);
a model that keeps crashing is marked failed instead of being restarted forever. With more than one model node, the inference
node sends requests to another node that has the model.</p>

<h2 id="next">Where next</h2>
<p>See the <a href="${docs}">documentation index</a>, the <a href="${roadmap}">roadmap</a>, and
<a href="https://github.com/WeirdBrains/ymir/blob/main/docs/DESIGN.md">DESIGN.md</a> for the full design.</p>
''';

class DeploymentGuidePage extends StatelessComponent {
  const DeploymentGuidePage({super.key});

  @override
  Component build(BuildContext context) {
    return Component.fragment([
      pageHero(
        eyebrow: 'Docs · Guides',
        title: 'Deployment guide',
        lede: 'Install Ymir, add a device, connect Bragi, use Jan, and common questions.',
      ),
      prose(
        _html
            .replaceAll(r'${bragiSite}', bragiSiteUrl)
            .replaceAll(r'${ymirctl}', siteLink('/docs/guides/ymirctl'))
            .replaceAll(r'${docs}', siteLink('/docs'))
            .replaceAll(r'${roadmap}', siteLink('/roadmap')),
      ),
    ]);
  }
}
