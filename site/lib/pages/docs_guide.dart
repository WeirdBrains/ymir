import 'package:jaspr/jaspr.dart';

import '../components/layout.dart';
import '../site.dart';

const _html = r'''
<p>This guide takes you from nothing to a working private model cloud: install Ymir, add a second device, connect the
Bragi chat app, and point Jan at it. Commands are exactly those <code>ymirctl</code> offers; run
<code>ymirctl --help</code> or <code>ymirctl &lt;command&gt; --help</code> for the full list.</p>
<p><strong>What is released.</strong> Nothing in Ymir or Bragi is released yet, so there are no downloads to install:
you build Ymir from source (step 1). The Linux packages the repository builds (tar.gz, zip, deb, an Arch package) are
unsigned and unpublished, and there is no macOS or Windows installer. The bridge token and the Weirdnet identity are
kept in files readable only by you (mode 0600); storing them in the operating system's keyring comes later. Linking
Bragi to a Ymir pool is covered honestly in step 3.</p>
<p>Ymir is a <a href="https://weirdbrains.com">Weird Brains</a> project. Devices find each other over
<a href="https://net.weirdbrains.com">Weirdnet</a>, so a pool needs no open ports and no API keys.</p>

<h2 id="install">1. Install Ymir on a server or desktop</h2>
<p>You need Rust 1.99, a C toolchain, and a <code>llama-server</code> from
<a href="https://github.com/ggml-org/llama.cpp">llama.cpp</a> (Ymir can also wrap a vLLM server on NVIDIA hardware).
On an NVIDIA PC, take llama.cpp's <code>ubuntu-cuda-12.8-x64</code> build and unpack its matching
<code>cudart-</code> archive into the same directory; no CUDA toolkit is needed.</p>
<pre><code>git clone https://github.com/WeirdBrains/ymir && cd ymir
cargo build --release -p ymird -p ymirctl
# put target/release/ymird and target/release/ymirctl on your PATH, side by side</code></pre>
<p><strong>Look at the machine.</strong> <code>ymirctl hw</code> shows the hardware and the memory budget for models
(<code>--json</code> for scripts).</p>
<p><strong>Set it up.</strong> <code>ymirctl setup</code> detects the hardware, picks models that fit and writes
<code>~/.config/ymir/node.toml</code> (move the directory with <code>YMIR_HOME</code> or <code>--home</code>). Choose
one of:</p>
<pre><code>ymirctl setup --dry-run                      # show the proposal, write nothing
ymirctl setup --download                     # catalog models that fit; asks before downloading (--yes skips the question)
ymirctl setup --llama-server /path/to/llama-server --model /path/to/model.gguf
                                             # your own GGUF; repeat --model, the first is the main model</code></pre>
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
streams answers from their models.</p>
<ol>
<li>Open Bragi on the new device. On first start it says <em>Link Bragi to your Weirdnet</em>. Tap <strong>Show a link code</strong>.</li>
<li>Bragi shows a code, as text and as a QR code, for two minutes. On the phone you can copy it.</li>
<li>Open that code on a device that is already linked, as the device that adds it.</li>
<li>Bragi shows <em>Do both screens show this code?</em> with six digits. Compare them with the other device. If they match, tap <strong>They match</strong>; otherwise tap <strong>They don't</strong>.</li>
<li>Bragi goes online and lists the models of every inference node in the pool. Pick one and chat.</li>
</ol>
<p>Bragi's codes start with <code>weirdnet-bragi://link/</code>. <code>ymirctl pool add</code> reads
<code>weirdnet-ymir://</code> and <code>weirdnet-morph://</code> codes today, so a Bragi code has to be opened on a
device that accepts that scheme, such as an already-linked Bragi. Bragi's pairing has been exercised against a built-in
demo inference node and not yet against a Ymir pool on separate real machines, and the Android app has no QR scanner yet,
so codes are pasted. Both apps are unreleased.</p>
<p>You need at least one device with the <code>inference</code> role in the pool for Bragi to have models to show. Phones
are reachable only while the app is open.</p>

<h2 id="jan">4. Use Jan through the local bridge</h2>
<p><code>ymird</code>'s <code>app</code> role runs an OpenAI-compatible bridge on <code>127.0.0.1</code> (default port
8484). It serves <code>/v1/models</code>, <code>/v1/chat/completions</code>, <code>/v1/completions</code> and
<code>/v1/embeddings</code>, needs a bearer token on every request, and answers only for the host names
<code>127.0.0.1</code> and <code>localhost</code>, so <a href="https://jan.ai">Jan</a> must run on the same machine as
the bridge.</p>
<ol>
<li>Get the key: <code>ymirctl bridge token</code> prints it. It lives in <code>~/.config/ymir/state/bridge.token</code>. <code>ymirctl bridge token --rotate</code> makes a new one; restart <code>ymird</code> to use it.</li>
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
        lede: 'Install Ymir, add a device, connect Bragi, and use Jan.',
      ),
      prose(
        _html
            .replaceAll(r'${bragiSite}', bragiSiteUrl)
            .replaceAll(r'${docs}', siteLink('/docs'))
            .replaceAll(r'${roadmap}', siteLink('/roadmap')),
      ),
    ]);
  }
}
