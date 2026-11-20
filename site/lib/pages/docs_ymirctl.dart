import 'package:jaspr/jaspr.dart';

import '../components/layout.dart';
import '../site.dart';

const _html = r'''
<p><code>ymirctl</code> is the command line for Ymir. It sets a machine up, shows what the pool is doing, loads and unloads
models, links devices, prints the API key, and lets you chat from a terminal. This guide covers every command, grouped by
what you want to do. Each flag here comes from the command definitions in the code; for the exhaustive, always current
list, see the <a href="${reference}">generated command reference</a>. <code>ymirctl --help</code> and
<code>ymirctl &lt;command&gt; --help</code> show the same thing offline.</p>
<p>Ymir is a <a href="https://weirdbrains.com">Weird Brains</a> project. The examples use
<code>server-a.example</code> and <code>192.0.2.10</code> as stand-ins for your own machines.</p>

<h2 id="how">How ymirctl works</h2>
<ul>
<li><strong>It talks to the running daemon.</strong> <code>status</code>, <code>nodes</code>, <code>load</code>,
<code>unload</code>, <code>drain</code>, <code>group</code>, <code>context</code>, <code>batch</code> and
<code>chat</code> go through the control socket <code>ymird</code> opens in its state folder, so <code>ymird</code> must be
running. <code>models</code> and <code>chat</code> also use the local bridge and its token. <code>hw</code>,
<code>setup</code>, <code>update</code>, <code>service</code> and <code>bridge token</code> work on files and need no daemon.</li>
<li><strong>Where it looks.</strong> Config and state live in one folder, <code>~/.config/ymir</code> by default. Change it
with the global <code>--home DIR</code> flag or the <code>YMIR_HOME</code> environment variable; the flag wins. Inside it:
<code>node.toml</code> (the config), and <code>state/</code> with <code>ymird.sock</code>, <code>bridge.token</code>,
<code>pool/</code> (this device's identity and linked devices) and <code>logs/</code>.</li>
<li><strong>Global flags.</strong> Only <code>--home</code>, plus <code>--help</code> and <code>--version</code>.</li>
</ul>

<h2 id="node">Set up a node</h2>
<h3 id="hw"><code>ymirctl hw</code></h3>
<p>Shows the CPU, memory, accelerators and the memory budget available to models. <code>--json</code> prints the same as
JSON for scripts.</p>
<pre><code>ymirctl hw
ymirctl hw --json</code></pre>

<h3 id="setup"><code>ymirctl setup</code></h3>
<p>Detects the hardware, picks a backend and models that fit, and writes <code>node.toml</code>. It also creates the bridge
token. It refuses to overwrite an existing <code>node.toml</code> unless you pass <code>--force</code>. Start with
<code>--dry-run</code> to see the proposal and the file it would write.</p>
<table>
<thead><tr><th>Flag</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--name NAME</code></td><td>The node's name in the pool.</td></tr>
<tr><td><code>--llama-server PATH</code></td><td>Use this external <code>llama-server</code> instead of the llama.cpp shipped with Ymir (the default).</td></tr>
<tr><td><code>--backend ENGINE</code></td><td>Engine for chat models: <code>llamacpp</code> or <code>mlx</code> (Apple Silicon only). Default: <code>mlx</code> on Apple Silicon Macs, <code>llamacpp</code> elsewhere. Models that take images and the embedding model always use llama.cpp.</td></tr>
<tr><td><code>--model PATH</code></td><td>Serve this GGUF file instead of catalog models. Repeat for more; the first is the main model.</td></tr>
<tr><td><code>--catalog</code></td><td>With <code>--model</code>, also propose catalog models.</td></tr>
<tr><td><code>--memory-budget SIZE</code></td><td>Memory models may use, such as <code>10GiB</code> (default: automatic).</td></tr>
<tr><td><code>--reserve SIZE</code></td><td>Unified memory kept for the operating system, such as <code>12GiB</code>.</td></tr>
<tr><td><code>--models-dir DIR</code></td><td>Where GGUF files live (default: <code>$WEIRDNET_MODELS_DIR</code>, else the shared Weirdnet models folder).</td></tr>
<tr><td><code>--port PORT</code></td><td>Bridge port on 127.0.0.1 (default 8484).</td></tr>
<tr><td><code>--roles LIST</code></td><td>Roles to turn on, comma separated (default <code>model,inference,app</code>).</td></tr>
<tr><td><code>--download</code></td><td>Download the proposed catalog models, and the vision projector of those that take images; asks first. On Apple Silicon it also installs MLX-LM and downloads the MLX weights.</td></tr>
<tr><td><code>--yes</code></td><td>Answer yes to the download prompts.</td></tr>
<tr><td><code>--force</code></td><td>Replace an existing <code>node.toml</code>.</td></tr>
<tr><td><code>--dry-run</code></td><td>Print the proposal and write nothing.</td></tr>
<tr><td><code>--service</code></td><td>Also install and start <code>ymird</code> as a user service: systemd on Linux, launchd on macOS, a logon task on Windows (see <a href="#services">Services</a>). With <code>--dry-run</code>, also print the service it would write.</td></tr>
<tr><td><code>--unit-dir DIR</code>, <code>--no-start</code>, <code>--ymird PATH</code></td><td>Only with <code>--service</code>: write the unit file into DIR and never start it; write it but do not start it; use this <code>ymird</code> binary.</td></tr>
</tbody></table>
<pre><code># a workstation with a GPU: see the plan, then do it and start the service
ymirctl setup --dry-run
ymirctl setup --download --yes --service

# a model node with a file you already have, bridge off the default port
ymirctl setup --name server-a --model /srv/models/gemma.gguf --port 8585 --roles model,inference,app</code></pre>

<h3 id="status"><code>ymirctl status</code></h3>
<p>Shows the running node: its name and version, roles, the bridge address, the main model, each model's state (loading,
ready, failed), the model nodes, alerts, and whether it is on Weirdnet. <code>--json</code> prints the full report.</p>
<pre><code>ymirctl status
ymirctl status --json | jq .models</code></pre>

<h2 id="models">Manage models</h2>
<h3 id="models-cmd"><code>ymirctl models</code></h3>
<p>Lists the pool's models (the merged catalog of every model node): id, purpose, and whether it is ready, busy, loading or
only on disk. Use these ids with every other command and in apps like Jan.</p>
<pre><code>ymirctl models</code></pre>
<h3 id="load"><code>ymirctl load MODEL [--node NODE]</code></h3>
<p>Loads a model on a model node. <code>--node</code> picks the node by name (see <code>ymirctl nodes</code>).</p>
<pre><code>ymirctl load gemma4-e4b
ymirctl load qwen3.6-35b-a3b --node server-a</code></pre>
<h3 id="unload"><code>ymirctl unload MODEL</code></h3>
<p>Unloads a model everywhere it is loaded, freeing its memory.</p>
<h3 id="nodes"><code>ymirctl nodes</code></h3>
<p>Lists the model nodes the pool routes to: serving or draining, how many of their models are ready, and requests in flight.</p>
<h3 id="drain"><code>ymirctl drain NODE [--resume]</code></h3>
<p>Stops sending a node new requests; it finishes what it has. Use it before maintenance. <code>--resume</code> undoes it.</p>
<pre><code>ymirctl drain server-a      # then reboot it
ymirctl drain server-a --resume</code></pre>

<h2 id="split">Split models</h2>
<p>A model too big for one machine can be served by a group of nodes over a private link. The model must be marked
<code>split</code> in the pool policy first (see the <a href="https://github.com/WeirdBrains/ymir/blob/main/docs/DESIGN.md">design document</a>).</p>
<ul>
<li><code>ymirctl group up MODEL</code>: plan and start the group for MODEL.</li>
<li><code>ymirctl group down MODEL</code>: stop it; its members return to normal placement.</li>
<li><code>ymirctl group status</code>: the groups that are up (model, leader, members) and why recent ones went down.</li>
</ul>
<pre><code>ymirctl group up big-model-70b
ymirctl group status</code></pre>

<h2 id="chat">Chat, conversations and batches</h2>
<h3 id="chat-cmd"><code>ymirctl chat PROMPT...</code></h3>
<p>Asks the pool something and streams the answer. The words after the options are joined into the prompt; <code>-</code>
reads the prompt from standard input.</p>
<table>
<thead><tr><th>Flag</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>-m, --model ID</code></td><td>Model to use (default: the main model).</td></tr>
<tr><td><code>--system TEXT</code></td><td>A system prompt.</td></tr>
<tr><td><code>--temperature N</code></td><td>Sampling temperature.</td></tr>
<tr><td><code>--max-tokens N</code></td><td>Cap the answer's length.</td></tr>
<tr><td><code>--wait</code></td><td>Wait for the model to load instead of taking a stand-in's answer.</td></tr>
</tbody></table>
<pre><code>ymirctl chat "Explain RAID levels in two sentences"
ymirctl chat -m qwen3.6-35b-a3b --system "Answer in French" --max-tokens 200 "What is a mutex?"
git diff | ymirctl chat --system "Write a commit message" -</code></pre>
<h3 id="context"><code>ymirctl context</code></h3>
<p>Conversations the inference and app nodes keep and sync with each other.</p>
<ul>
<li><code>context show [CONVERSATION]</code>: list the conversations, or show one.</li>
<li><code>context add CONVERSATION TEXT [--role ROLE]</code>: add a message (role defaults to <code>user</code>).</li>
<li><code>context ask CONVERSATION TEXT [-m MODEL]</code>: add a user message and have the pool answer; the answer is saved
and syncs to the other nodes. It also prints the model, the node that answered and the route taken.</li>
<li><code>context delete CONVERSATION</code>: delete it on every node.</li>
</ul>
<pre><code>ymirctl context ask notes "Summarize what we decided about backups"
ymirctl context show notes</code></pre>
<h3 id="batch"><code>ymirctl batch FILE</code></h3>
<p>Spreads many prompts (one per line of FILE, or standard input with <code>-</code>; blank lines are skipped) over the
pool at batch priority, so interactive chat stays fast. Prints each answer in order as <em>index, node, text</em>, with a
summary at the end.</p>
<table>
<thead><tr><th>Flag</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>-m, --model ID</code></td><td>Model to use.</td></tr>
<tr><td><code>--concurrency N</code></td><td>Tasks in flight at once.</td></tr>
<tr><td><code>--retries N</code></td><td>Extra tries for a task that fails in a way another try could fix.</td></tr>
</tbody></table>
<pre><code>ymirctl batch prompts.txt --concurrency 4 --retries 2 &gt; answers.tsv</code></pre>

<h2 id="pool">Pool and devices</h2>
<p>These commands manage this node's place in your Weirdnet: the identity shared by your linked devices. They need
<code>node.toml</code>. Linking moves your identity to the new device with a code and a six-digit check, so no recovery
words are typed. Link codes work between Ymir, Morph and Bragi: <code>pool add</code> accepts
<code>weirdnet-ymir://</code>, <code>weirdnet-morph://</code> and <code>weirdnet-bragi://</code> codes.</p>
<h3 id="pool-link"><code>ymirctl pool link</code></h3>
<p>Run on the <em>new</em> device. It shows a link code (good for two minutes) to open on a device already in your
Weirdnet. Both screens show a short code; confirm they match. It refuses if this node is already linked. Restart
<code>ymird</code> afterwards to put the node on Weirdnet.</p>
<h3 id="pool-add"><code>ymirctl pool add LINK</code></h3>
<p>Run on a device that is <em>already</em> in the pool, to link another device: paste the code the other device shows.</p>
<table>
<thead><tr><th>Flag</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--profile-name NAME</code></td><td>The name the other device gets for this user (default <code>Ymir</code>).</td></tr>
<tr><td><code>--yes</code></td><td>Skip the "same code?" prompt. Meant for tests; the codes still have to match.</td></tr>
</tbody></table>
<pre><code># on the new server
ymirctl pool link
# on a device already linked
ymirctl pool add 'weirdnet-ymir://link/...'
# Pairing with Ymir on server-a: the code is 123456
# Does the other device show the same code? [y/N] y</code></pre>
<h3 id="pool-devices"><code>ymirctl pool devices</code></h3>
<p>Lists the identity and each device: <em>this node</em>, <em>linked</em> or <em>revoked</em>, with its peer id.</p>
<h3 id="pool-addresses"><code>ymirctl pool addresses</code></h3>
<p>Prints where this node can be reached (<code>/ip4/…/tcp/4848/p2p/PEER</code>, from the running <code>ymird</code> or the
configured <code>[network] listen</code>), followed by the configured <code>[network] peers</code> that are linked devices.
A device that can't discover your nodes on its own, such as Bragi on another network than nodes with discovery off, needs
one: paste it into Bragi's <strong>Link a device</strong> box. <code>pool add</code> sends the same list to the device it
links, and <code>pool link</code> prints the list it received when discovery is off here.</p>
<h3 id="pool-revoke"><code>ymirctl pool revoke PEER</code></h3>
<p>Revokes a device everywhere the roster reaches. Use the peer id from <code>pool devices</code>. If <code>ymird</code> is
running on Weirdnet the change goes through it.</p>
<h3 id="pool-init"><code>ymirctl pool init --standalone</code></h3>
<p>Starts a brand-new identity on this node, separate from the one Morph holds, for a cloud of its own. It refuses without
<code>--standalone</code>. To join your existing devices use <code>pool link</code> instead.</p>

<h2 id="bridge">Bridge and API keys</h2>
<p>The app role runs a bridge: an OpenAI-compatible server on <code>127.0.0.1</code> (port 8484 by default) for apps like
<a href="https://jan.ai">Jan</a>. It listens only on the local machine and requires a token as the API key.</p>
<h3 id="bridge-token"><code>ymirctl bridge token [--rotate]</code></h3>
<ul>
<li><code>ymirctl bridge token</code> prints the token. It is stored in <code>state/bridge.token</code> inside the Ymir
folder, readable only by its owner. If there is none yet, one is created.</li>
<li><code>ymirctl bridge token --rotate</code> makes a new token and prints it. <strong>Restart <code>ymird</code> to
use it</strong>; until then the old one still works. Update every client afterwards.</li>
<li>There is one token per bridge. Every client shares it, so you cannot give each device its own key or revoke just
one; rotate it to cut everyone off.</li>
</ul>
<pre><code>ymirctl bridge token
curl -H "Authorization: Bearer $(ymirctl bridge token)" http://127.0.0.1:8484/v1/models

ymirctl bridge token --rotate
systemctl --user restart ymird</code></pre>
<p>In Jan, add an OpenAI-compatible provider with base URL <code>http://127.0.0.1:8484/v1</code> and the token as the key.
The <a href="${deployment}#jan">deployment guide</a> walks through it, including how to reach the bridge from other
machines over HTTPS.</p>

<h2 id="services">Run as a service</h2>
<p><code>ymirctl service</code> runs <code>ymird</code> in the background, the same way on every OS. By default it is a
<em>user-level</em> service that runs as you, with your <code>YMIR_HOME</code>, and needs no root or admin rights:</p>
<table>
<tr><th>OS</th><th>What <code>service install</code> sets up</th><th>Log</th></tr>
<tr><td>Linux</td><td>a systemd user unit, <code>~/.config/systemd/user/ymird.service</code> (restarts on failure)</td><td><code>journalctl --user -u ymird</code></td></tr>
<tr><td>macOS</td><td>a launchd agent, <code>~/Library/LaunchAgents/com.weirdbrains.ymird.plist</code> (starts at login, restarted if it fails)</td><td><code>state/logs/ymird.log</code> in the node folder</td></tr>
<tr><td>Windows</td><td>a Task Scheduler task, <code>\Ymir\ymird</code>, that starts <code>ymird</code> at logon with no console window and restarts it after a failure</td><td><code>state\logs\ymird.log</code> in the node folder</td></tr>
</table>
<ul>
<li><code>service install</code>: write the service, then enable and start it. <code>ymirctl setup</code> must have run first.
Flags: <code>--ymird PATH</code> (default: next to <code>ymirctl</code>, else on <code>PATH</code>),
<code>--no-start</code> (write it only), <code>--managed-updates</code> (start through <code>ymirctl update boot</code>,
so a staged update gets a trial and a rollback), <code>--unit-dir DIR</code> (write the file there and never register or
start it), <code>--dry-run</code> (print the file and the commands, change nothing).</li>
<li><code>service start</code> and <code>service stop</code>: start it, or stop it cleanly with its engines. A stopped
service starts again at the next login or boot; <code>uninstall</code> removes it for good.</li>
<li><code>service status</code>: is it installed, what does the service manager say, and does <code>ymird</code> answer?</li>
<li><code>service uninstall</code>: stop, disable and remove it.</li>
</ul>
<p>All of them take <code>--dry-run</code>, <code>--unit-dir DIR</code> and <code>--system</code>.</p>
<pre><code>ymirctl service install --dry-run
ymirctl service install --managed-updates
ymirctl service stop &amp;&amp; ymirctl service start
ymirctl service status</code></pre>
<p><strong>Linux:</strong> run <code>loginctl enable-linger $USER</code> once, or the user service stops when you log out
and only starts when you log in. <strong>macOS:</strong> the agent runs in your login session; to run without anyone
logged in, use <code>--system</code>. <strong>Windows:</strong> the task runs while you are logged on; for a machine that
should serve from boot, use <code>--system</code>.</p>

<h3 id="system-service">A machine service (<code>--system</code>)</h3>
<p>With <code>--system</code>, run as root (<code>sudo</code>) or from an elevated prompt on Windows, <code>ymirctl service</code>
manages the machine's service instead. Its node folder defaults to <code>/var/lib/ymir</code> (Linux),
<code>/Library/Application Support/Ymir</code> (macOS) or <code>%ProgramData%\Ymir</code> (Windows); pass
<code>--home</code> for another.</p>
<ul>
<li><strong>Linux:</strong> a system unit, <code>/etc/systemd/system/ymird.service</code>, with <code>User=</code> set to
<code>--run-as USER</code> (default: the user who ran <code>sudo</code>; never root). The node folder is handed to that account.</li>
<li><strong>macOS:</strong> a launchd daemon, <code>/Library/LaunchDaemons/com.weirdbrains.ymird.plist</code>, with
<code>UserName</code> set the same way, so it starts at boot.</li>
<li><strong>Windows:</strong> the Windows service <code>ymird</code> (display name "Ymir node"), started at boot, running as its
own virtual account <code>NT SERVICE\ymird</code> (no password). The node folder gets a protected ACL for SYSTEM,
Administrators and that account only. Stopping it (<code>service stop --system</code>, <code>sc stop ymird</code> or the
Services app) shuts <code>ymird</code> and its engines down cleanly; after a crash it restarts in 60 seconds. The
Windows installer registers the same service.</li>
</ul>
<pre><code># Linux or macOS
sudo ymirctl --home /var/lib/ymir setup --download --yes
sudo ymirctl service install --system --run-as ymir

# Windows, in an elevated PowerShell
ymirctl --home "$env:ProgramData\Ymir" setup --download --yes
ymirctl service install --system
ymirctl service status --system</code></pre>

<h3 id="other-account">Using ymirctl with a service that runs as another account</h3>
<p>A system service (<code>--system</code>, or your own system unit) runs <code>ymird</code> under a dedicated account,
for example <code>ymir</code>, with its folder at <code>/var/lib/ymir</code>.
Start the daemon with <code>ymird --home /var/lib/ymir</code> (or <code>YMIR_HOME=/var/lib/ymir</code> in the unit).
Then every <code>ymirctl</code> command must point at the same folder <em>and</em> run as that account, because
<code>state/</code> is private to it (mode 0700) and holds the control socket and token:</p>
<pre><code>sudo -u ymir ymirctl --home /var/lib/ymir status
sudo -u ymir env YMIR_HOME=/var/lib/ymir ymirctl bridge token
sudo -u ymir ymirctl --home /var/lib/ymir load gemma4-e4b</code></pre>
<p><strong>Mind the account.</strong> Run as yourself without these, <code>ymirctl</code> looks in
<code>~/.config/ymir</code>, finds no daemon, and <code>bridge token</code> quietly creates a new, unrelated token there.
To avoid typing it each time, set an alias such as
<code>alias yctl='sudo -u ymir env YMIR_HOME=/var/lib/ymir ymirctl'</code>. Root can read the folder too, but then files
it creates (a new token, a new <code>node.toml</code>) would be owned by root, so use the service account.</p>
<p><strong>Windows:</strong> <code>ymirctl</code> reaches <code>ymird</code> over a local named pipe derived from the node
folder, so pass the same <code>--home</code>. For the Windows service, run it from an elevated prompt: only the service
account, SYSTEM and Administrators can use its pipe and read its token.</p>
<pre><code>ymirctl --home "$env:ProgramData\Ymir" status
ymirctl --home "$env:ProgramData\Ymir" bridge token</code></pre>

<h2 id="updates">Updates</h2>
<p><code>ymirctl update</code> installs signed releases with a trial period and automatic rollback, without a package
manager. You pin the organization's signing root once, stage a bundle, and the next start runs it on trial; if it does not
become healthy, the previous good release comes back.</p>
<ul>
<li><code>update trust --pin FINGERPRINT --root FILE</code>: pin the organization root that updates must chain to (the hex
fingerprint and the first signed root file).</li>
<li><code>update stage BUNDLE_DIR</code>: verify a bundle directory and stage it for the next start.</li>
<li><code>update status</code>: whether a root is pinned, the good release, the health of a trial under way, and releases marked bad.</li>
<li><code>update boot [--exec] [--installed PATH] [-- ARGS...]</code>: choose what to start (the good release, a candidate on
trial, or a rollback) and print its path. With <code>--exec</code> it runs it, passing the arguments after <code>--</code>;
<code>--installed</code> names the daemon to use when no release is installed (default <code>ymird</code> on
<code>PATH</code>). This is the service's start command; <code>service install --managed-updates</code> sets it up.</li>
</ul>
<pre><code>ymirctl update trust --pin 3f9a...c1 --root root.signed
ymirctl update stage ./ymir-0.2.0-bundle
sudo systemctl restart ymird        # or restart the user service
ymirctl update status</code></pre>
<p>Nothing from Ymir is released yet, so there are no public bundles to stage today.</p>

<h2 id="recipes">Common tasks at a glance</h2>
<table>
<thead><tr><th>I want to…</th><th>Run</th></tr></thead>
<tbody>
<tr><td>see what a machine can run</td><td><code>ymirctl hw</code></td></tr>
<tr><td>configure and start a node</td><td><code>ymirctl setup --download --service</code></td></tr>
<tr><td>check on the pool</td><td><code>ymirctl status</code>, <code>ymirctl nodes</code>, <code>ymirctl models</code></td></tr>
<tr><td>take a node out for maintenance</td><td><code>ymirctl drain server-a</code> … <code>--resume</code></td></tr>
<tr><td>add a device</td><td><code>ymirctl pool link</code> on it, <code>ymirctl pool add CODE</code> on a linked one</td></tr>
<tr><td>get or change the API key</td><td><code>ymirctl bridge token</code>, <code>--rotate</code>, then restart <code>ymird</code></td></tr>
<tr><td>run many prompts</td><td><code>ymirctl batch prompts.txt</code></td></tr>
</tbody></table>

<h2 id="next">Where next</h2>
<p>The <a href="${reference}">generated command reference</a> lists every flag exactly as the program defines it. See also the
<a href="${deployment}">deployment guide</a>, the <a href="${docs}">documentation index</a> and the
<a href="https://github.com/WeirdBrains/ymir/blob/main/docs/DESIGN.md">design document</a>.</p>
''';

class YmirctlGuidePage extends StatelessComponent {
  const YmirctlGuidePage({super.key});

  @override
  Component build(BuildContext context) {
    return Component.fragment([
      pageHero(
        eyebrow: 'Docs · Guides',
        title: 'ymirctl guide',
        lede: 'Everything you can do with ymirctl, grouped by task.',
      ),
      prose(
        _html
            .replaceAll(r'${reference}', siteLink('/docs/reference/ymirctl/'))
            .replaceAll(r'${deployment}', siteLink('/docs/guides/deployment/'))
            .replaceAll(r'${docs}', siteLink('/docs')),
      ),
    ]);
  }
}
