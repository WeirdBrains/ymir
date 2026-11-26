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
ready, failed) and what it takes (images, audio), the model nodes, alerts, and whether it is on Weirdnet.
<code>--json</code> prints the full report.</p>
<pre><code>ymirctl status
ymirctl status --json | jq .models</code></pre>

<h2 id="models">Manage models</h2>
<h3 id="models-cmd"><code>ymirctl models</code></h3>
<p>Lists the pool's models (the merged catalog of every model node): id, purpose, whether it is ready, busy, loading or
only on disk, and, when the node knows, where it came from (<code>hub</code>, <code>upload</code> or <code>train</code>)
and its license. Use these ids with every other command and in apps like Jan.</p>
<pre><code>ymirctl models</code></pre>
<h3 id="models-search"><code>ymirctl models search QUERY...</code></h3>
<p>Searches the Hugging Face Hub for GGUF models, plus MLX models on Apple Silicon and safetensors/vLLM
repos on Linux NVIDIA nodes (or with <code>--format mlx</code>, <code>--format vllm</code> or <code>--format all</code>), most
downloaded first. For each repo it shows the best quant for <em>this</em> node and whether it fits: the file size plus the KV
cache at the context length a quant is picked at (default 32768, or less if the model was trained for less; change it with <code>--context</code>)
plus about 0.5 GiB of overhead, against this node's GPU or unified memory. A verdict is <em>fits</em>, <em>partial</em> (only
by spilling layers to RAM, which is slower) or <em>too big</em>. <code>--limit N</code> sets how many repos to show.</p>
<pre><code>ymirctl models search qwen3 8b
ymirctl models search --format all --limit 3 gemma</code></pre>
<h3 id="models-pull"><code>ymirctl models pull REPO</code></h3>
<p>Downloads a model into this node's model directory (<code>models_dir</code> in <code>node.toml</code>, else the shared
Weirdnet models folder; <code>--dir</code> overrides it). For a GGUF repo it takes the best quant that fits, or the one you name with
<code>--quant Q4_K_M</code> or <code>--file NAME.gguf</code>. Split GGUFs download every shard. A <code>mmproj-*.gguf</code>
in the same repo is also fetched (preferring Q8_0) and set as <code>mmproj</code> so image input works. Downloads resume if
interrupted (run the same command again) and are checked against the Hub's SHA-256 before they are renamed into place.
<code>--dry-run</code> shows the plan and downloads nothing; <code>--force</code> takes the smallest quant even if none fits.</p>
<p>A safetensors / transformers repo (for example <code>thinkingmachines/Inkling-2-Mini</code>) is registered for
<code>backend = "vllm"</code> as a Hub id: the weights are not copied into <code>models_dir</code>; vLLM fetches them into
the Hugging Face cache on first load. Inkling 2 Mini also gets the <code>inkling</code> reasoning and tool-call parsers,
DSpark and the <code>vllm/vllm-openai:inkling-2-mini</code> image; its <code>config.json</code> is saved so
<code>ymird</code> can size the context.
That needs vLLM 0.35.0 or newer, or that image (support is not in 0.34.0). You can also start the image yourself and set
<code>wrap = "http://127.0.0.1:8000/v1"</code>, or set <code>[model] vllm = "image:vllm/vllm-openai:inkling-2-mini"</code>.
MLX repos for Inkling are refused until mlx-lm#2086 is released.</p>
<p>When it finishes it prints a ready <code>[[model.local]]</code> entry. <code>node.toml</code> is <strong>not</strong>
changed unless you add <code>--register</code>, which appends the entry (keeping a copy as <code>node.toml.bak</code>). Restart
<code>ymird</code> afterwards to load it. <code>--id</code>, <code>--role</code> and <code>--context</code> shape the entry; the
role defaults to <code>main</code> only when the node has no main model yet.</p>
<p>Gated repos need a token. Ymir reads it on this node only, from <code>HF_TOKEN</code>, <code>HUGGING_FACE_HUB_TOKEN</code>, the
file named by <code>YMIR_HF_TOKEN_FILE</code>, <code>state/hf-token</code> in the Ymir home, or Hugging Face's own
<code>~/.cache/huggingface/token</code>. It is only sent to the Hub, and never printed or logged.</p>
<pre><code>ymirctl models pull Qwen/Qwen3-8B-GGUF
ymirctl models pull bartowski/Qwen3-8B-GGUF --quant Q5_K_M --register
ymirctl models pull thinkingmachines/Inkling-2-Mini --register
ymirctl models pull ggml-org/Inkling-2-Mini-GGUF --quant IQ4_NL --register</code></pre>
<p>The entry gets no fixed <code>context</code> unless you pass <code>--context N</code>. <code>ymird</code> then runs each
model with its full trained context (from the GGUF header, or <code>config.json</code> for vLLM), and cuts it only where the
KV cache and attention buffers would not fit beside the weights and the models running next to it, keeping 20% of the
free memory free. From 0.3.1 the plan counts both f32 score matrices llama.cpp keeps for attention with a score
bias (Inkling 2 Mini, T5), so it no longer under-counts and runs a node out of memory. When a model is asked for, idle models (no request for 30 seconds) are unloaded first if it would
otherwise be cut; busy ones never are. The log says which and why, and the catalog shows the context it runs with. Pull
prints the context the model will get on this node.</p>
<h3 id="models-context"><code>ymirctl models context [ID...]</code></h3>
<p>Shows the context <code>ymird</code> starts each model with on this node, before it looks at what else runs, and why: <code>auto</code> entries get their full
trained length or what fits, and a <code>context</code> you set is kept. <code>--auto ID...</code> removes
<code>context</code> (and any <code>--ctx-size</code> or <code>--max-model-len</code> in <code>args</code>) from those
entries; <code>--old-defaults</code> does that for every entry that still has the 32768 an older pull wrote (or 262144 for
Inkling 2 Mini on vLLM). Both keep a copy as <code>node.toml.bak</code>; restart <code>ymird</code> to apply.</p>
<pre><code>ymirctl models context
ymirctl models context --old-defaults</code></pre>
<h3 id="models-list"><code>ymirctl models list</code></h3>
<p>Lists the model files on this node (GGUF, split GGUF as one entry, MLX folders and unfinished downloads) and whether a
<code>[[model.local]]</code> in <code>node.toml</code> uses each.</p>
<h3 id="models-remove"><code>ymirctl models remove ID</code></h3>
<p>Deletes a downloaded model's files after asking (<code>--yes</code> skips the question). It never edits
<code>node.toml</code>; if the model is registered, remove its entry yourself.</p>
<p>Without <code>--node</code> these commands act on the machine they run on and need no daemon.</p>
<h3 id="models-node"><code>--node</code>: manage models on your other devices</h3>
<p><code>search</code>, <code>pull</code>, <code>list</code> and <code>remove</code> take
<code>--node &lt;name|peer-id|auto|all&gt;</code> and then run on your model nodes over Weirdnet instead. The request goes
through the <code>ymird</code> on this machine (it must be running and linked to your devices, see
<a href="#pool"><code>ymirctl pool</code></a>), which asks each node over
<code>/weirdnet-backends/models/1</code>. A node answers only devices linked to its owner (you); any other device gets a refusal
and nothing is downloaded. The Hugging Face token stays on the node that downloads and is never sent.</p>
<ul>
<li><code>--node NAME</code>: a device name from <code>ymirctl pool list</code> (any case), a peer id, or the start of one.
<code>all</code> means every model node.</li>
<li><code>pull REPO --node auto</code>: asks every model node how the repo fits its memory and takes the one that can hold the
biggest model (the one with more memory on a tie), and says which it picked. <code>--quant</code> or <code>--file</code> narrow
the model first.</li>
<li><code>pull ... --serve</code>: the node's running <code>ymird</code> starts serving the model as soon as the download ends, with no
restart (<code>--load</code> loads it too). That is in memory only: <code>node.toml</code> on another node is never edited. The pull prints the
<code>[[model.local]]</code> entry to add there to keep it after a restart.</li>
<li>A pull streams its progress, one line per node. <code>remove --node</code> stops serving the model and deletes its files.</li>
<li><code>upload FILE --node NAME</code>: sends a GGUF model or LoRA adapter on this machine (<code>--adapter-for BASE</code>, or
<code>--from-run RUN_DIR</code> for a training run's adapter) to that node. It is checked there as it arrives, resumes if it
is cut off, must leave <code>[model] upload_keep_free</code> free (by default the larger of 10% and 20 GiB), is refused by a node
with <code>[model] upload = false</code> (which still pulls and removes), and is served
and saved in that node's <code>node.toml</code> unless you pass <code>--no-persist</code>. A llama.cpp adapter is used from its
base's next start. With <code>--rollout NAME</code> it becomes the next version of that rollout here, named
<code>ID-rN</code>, at weight 0; <code>--canary PCT</code> gives it exactly that percentage of new conversations, any of
0 to 100 (such as 20 or 50), the other versions sharing the rest. With <code>--eval ROWS</code> the share is set only if it
does at least as well as the version that has the most traffic. <code>--inference NODE</code> adds it to that inference
node's rollout instead, saved in its <code>node.toml</code> unless <code>--no-persist</code>.</li>
<li><code>upload FILE --node all</code> (every model node but this one) or <code>--node A,B</code>: the file is sent once, to
the node with the most free space (or <code>--stage NODE</code>), and the others copy it from that node, node to node. A node
without room after what it keeps free is left out before anything is sent; one that can't copy from the other node
gets the file from this machine instead. Each one saves it in its own <code>node.toml</code> unless <code>--no-persist</code>.</li>
<li><code>upload DIR --node NAME</code> also sends a directory of safetensors (to a vLLM node) or MLX weights, or a PEFT LoRA
adapter directory with <code>--adapter-for</code> a vLLM model. <code>upload mmproj.gguf --node NAME --projector-for MODEL</code>
sends a vision projector for a llama.cpp model, saved as its <code>mmproj</code>. Pickles and code are refused. A node takes full
models for one engine, <code>[model] upload_engine</code> (by default llama.cpp, or vLLM when <code>llamacpp = "off"</code>).</li>
<li><code>upload DIR</code> or <code>upload mmproj.gguf --projector-for MODEL</code> without <code>--node</code>: on a machine
that serves models it is placed in this machine's models directory and saved in its <code>node.toml</code> (it serves after
a restart); on one that serves none it goes to the model node with the most free space after what it keeps free.
<code>--node auto</code> picks that node for any upload.</li>
<li><code>search --node all</code> shows each node's own fit verdict for every repo.</li>
</ul>
<p>A node running an older <code>ymird</code> answers that it does not take model management: upgrade it, or run the command on
that machine. Set <code>[model] remote_manage = false</code> in a node's <code>node.toml</code> to turn this off there.</p>
<pre><code>ymirctl models search qwen3 8b --node all
ymirctl models pull Qwen/Qwen3-8B-GGUF --node auto --serve
ymirctl models list --node gpu-box
ymirctl models remove qwen3-8b-q4_k_m --node gpu-box --yes
ymirctl models upload ./my-tune-Q4_K_M.gguf --node gpu-box --license apache-2.0
ymirctl models upload ./my-tune-Q4_K_M.gguf --node all
ymirctl models upload ./my-tune-Q4_K_M.gguf --node gpu-box --rollout support --canary 20
ymirctl models upload ./my-tune-Q4_K_M.gguf --node gpu-box --rollout support --inference inference-box --canary 20
ymirctl models upload ./Tune-8B --node auto
ymirctl models upload ./mmproj-F16.gguf --projector-for my-vision-model
ymirctl models upload ./my-tune-Q4_K_M.gguf --node gpu-box --rollout support --canary 50 --eval support-rows.jsonl</code></pre>
<h3 id="load"><code>ymirctl load MODEL [--node NODE]</code></h3>
<p>Loads a model on a model node. <code>--node</code> picks the node by name (see <code>ymirctl nodes</code>).</p>
<pre><code>ymirctl load gemma4-e4b
ymirctl load qwen3.6-35b-a3b --node server-a</code></pre>
<h3 id="unload"><code>ymirctl unload MODEL</code></h3>
<p>Unloads a model everywhere it is loaded, freeing its memory.</p>
<h3 id="move"><code>ymirctl move MODEL --to NODE [--from NODE] [--grace 60]</code></h3>
<p>Moves a loaded model to another model node without cutting anyone off: it loads on <code>--to</code> (which needs its
own copy of the model; <code>ymirctl models pull MODEL --node NODE</code>), new requests go there at once, and the old node
unloads it once the answers it is streaming finish (at most <code>--grace</code> seconds). If the load on the new node
fails, nothing changes on the old one.</p>
<pre><code>ymirctl move qwen3.6-35b-a3b --to server-b</code></pre>
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
<h3 id="batch-jsonl"><code>ymirctl batch FILE --jsonl</code>: generate a dataset</h3>
<p>With <code>--jsonl</code>, FILE holds one JSON row per line: <code>prompt</code> or <code>messages</code>, and optionally
<code>id</code>, <code>system</code>, <code>temperature</code>, <code>top_p</code>, <code>top_k</code>, <code>max_tokens</code>,
<code>seed</code> and <code>stop</code>. The teacher model (<code>--model</code>, required) answers every row, waiting for
the model to load rather than taking a stand-in. Each answer is written to <code>--out</code> as a JSON row with the
messages sent, the completion, the model and node that answered, and <code>usable</code>: false when the row failed,
came from the answer cache, came from a stand-in or was answered by another model. A manifest,
<code>OUT.manifest.json</code>, records the teacher, the license you gave for it, the input's hash and the counts.
No dataset is built in.</p>
<table>
<thead><tr><th>Flag</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--out FILE</code></td><td>The output (JSONL). It must not exist unless <code>--resume</code>.</td></tr>
<tr><td><code>--license ID</code></td><td>The teacher's license, such as <code>apache-2.0</code>. Some licenses forbid training on a model's outputs. Without it, the license comes from <code>--license-from OWNER/NAME</code> (that Hub repo's model card) or from the pool's catalog; with neither, the run is refused. <code>unknown</code> needs <code>--allow-unknown-license</code>.</td></tr>
<tr><td><code>--resume</code></td><td>Continue an interrupted run: rows already answered are skipped, failed rows are tried again.</td></tr>
<tr><td><code>--system TEXT</code></td><td>System prompt for rows without one.</td></tr>
<tr><td><code>--temperature</code>, <code>--top-p</code>, <code>--max-tokens</code>, <code>--seed</code></td><td>Defaults for rows that don't set their own.</td></tr>
<tr><td><code>--sources FILE</code></td><td>A sources manifest (JSON: a <code>topic</code>, and each source's <code>id</code>, <code>title</code>, <code>url</code> and <code>license</code>). Every row then names its <code>source</code>, and the dataset's manifest records the sources and rows per source.</td></tr>
<tr><td><code>--cache bypass|refresh|use</code></td><td><code>bypass</code> (default) asks for fresh answers; <code>refresh</code> also lets them refresh the cache; <code>use</code> lets the cache answer, and rows it answers are marked unusable.</td></tr>
</tbody></table>
<pre><code>ymirctl batch prompts.jsonl --jsonl --model teacher-model --license apache-2.0 --out data.jsonl --temperature 0.7</code></pre>
<h3 id="eval"><code>ymirctl eval FILE</code>: score models</h3>
<p>Scores one or more models (<code>--model</code>, repeatable) on a held-out JSONL file. Rows are like
<code>batch --jsonl</code> rows plus <code>expect</code>, checked with the same scorers as <code>ymirctl bench</code>
(<code>exact</code>, <code>contains</code>, <code>contains_all</code>, <code>number</code>, <code>json</code> or
<code>lines</code>), and/or <code>rubric</code>, graded PASS or FAIL by <code>--judge MODEL</code>. Answers are at
temperature 0 with <code>--seed</code> (default 0) unless a row says otherwise. Prints checks and judge scores
separately per model, with errors, cached and stand-in answers; <code>--out FILE</code> writes every answer and verdict.</p>
<pre><code>ymirctl eval held-out.jsonl --model base-model --model fine-tuned --judge judge-model --out eval.jsonl</code></pre>
<h3 id="train"><code>ymirctl train</code>: fine-tune on this node</h3>
<p><code>train run</code> trains a LoRA adapter for a llama.cpp model on the node it runs on, with the trainer named in
<code>[train]</code> in <code>node.toml</code>: upstream llama.cpp's full fine-tuning, then a step that extracts the
adapter, pinned to one llama.cpp build (<code>build</code>). A trainer that reports another build is refused;
<code>train check</code> shows what it reports. The dataset must have the manifest <code>batch --jsonl</code> or
<code>context export --training</code> writes next to it, so the run can record its source and license. The node's
GPU is held for the run. A running <code>ymird</code> unloads its models and loads none until the run ends, and
refuses when the node is the only one serving a loaded model unless you pass <code>--force</code>. A
<code>ymird</code> that starts meanwhile loads nothing. The adapter goes to <code>adapters/ID.gguf</code> in the
models directory, and the run's record (<code>run.json</code>) to the run's work directory. <code>--add</code> puts
the adapter in <code>node.toml</code>. Ctrl-C cancels the run. Every run on a node that serves your pool takes it out
of the pool while it trains.</p>
<table><thead><tr><th>Command</th><th>What it does</th></tr></thead><tbody>
<tr><td><code>train run --base ID --dataset FILE [--id ADAPTER] [--epochs N] [--lr X] [--rank N] [--add]</code></td><td>One run. Also <code>--context</code>, <code>--batch</code>, <code>--seed</code>, <code>--from FILE</code> (train from an F32 copy of a quantised base: llama.cpp trains F32 models only), <code>--keep-full</code>, <code>--allow-unknown-license</code> and <code>--force</code>.</td></tr>
<tr><td><code>train run ... --node NAME</code></td><td>Run on another of your model nodes instead, from any linked device, over Weirdnet through this device's <code>ymird</code>. The node needs <code>[train] remote = true</code> (and, to narrow who may, <code>callers = ["PEER_ID"]</code>). The dataset goes with the run (up to 24 MiB); <code>--from</code> names a GGUF in the node's models directory; <code>--add</code> is refused, since <code>node.toml</code> is never edited over the network: serve the adapter with <code>models adapter serve ID --base BASE --node NAME</code>. Ctrl-C cancels the run there.</td></tr>
<tr><td><code>train reserve [--node NAME] [--force]</code>, <code>train release [--node NAME]</code></td><td>Hold a node's GPU by hand and free it. Without <code>--node</code>, this box, also across a <code>ymird</code> restart.</td></tr>
<tr><td><code>train run ... --job</code>, <code>train resume JOB</code></td><td>A resumable job: a checkpoint after each epoch (trainer interface 2). If the run fails or stops, <code>train resume JOB</code> carries it on from the last checkpoint, checked against the job's dataset, settings, base revision and trainer build. Ctrl-C pauses a job. <code>train jobs</code> lists jobs; <code>train job pause|unpause|cancel JOB</code> manages your own. With <code>[train] resume = true</code> a node carries on other nodes' jobs (once the job record is synced).</td></tr>
<tr><td><code>train seed questions --sources FILE --texts DIR --out FILE</code>, <code>train seed prompts ANSWERS --out FILE</code></td><td>Seed prompts from source texts you fetched (<code>DIR/ID.txt</code> or <code>.md</code> per source): excerpts become rows asking the teacher for questions; its answers become prompt rows that keep their source. Also <code>--chunk-chars</code>, <code>--questions</code>, <code>--max-per-source</code>.</td></tr>
<tr><td><code>train status [--node NODE] [--json]</code></td><td>A model node's runs and jobs, read over Weirdnet from any of your linked devices (what Bragi's training view shows). Without <code>--node</code>, this node's.</td></tr>
<tr><td><code>train check</code>, <code>train list</code></td><td>The trainer against the pin; past runs with their status, loss and adapter.</td></tr>
</tbody></table>
<p>The trainer is <code>scripts/trainer.py</code> in Ymir's repository, at the same llama.cpp build as the engine
Ymir ships. Upstream's release files don't include <code>llama-finetune</code>, so build it with
<code>python3 scripts/engine.py build --trainer --out DIR</code>, and install the trainer's Python packages in a venv
(<code>pip install -r scripts/trainer-requirements.txt</code>). It renders each conversation with the base's chat
template and trains one sequence of <code>--context</code> tokens per step (<code>--batch</code> is not used). Set
<code>YMIR_TRAINER_NGL=0</code> under <code>[train.env]</code> to train on the CPU.</p>
<pre><code>[train]
trainer = "/opt/ymir/scripts/trainer.py"
build = "b12628"
[train.env]
YMIR_TRAINER_PYTHON = "/opt/ymir-trainer/bin/python"
YMIR_LLAMA_FINETUNE = "/opt/ymir-trainer/llama-finetune"

ymirctl train run --base qwen3-8b --dataset data.jsonl --id qwen3-8b+support --epochs 2 --add</code></pre>
<h3 id="rollout"><code>ymirctl models rollout</code>: versions, canaries and rollback</h3>
<p>A rollout is one name your apps ask for, with versions behind it, each a model in the pool with a traffic weight.
It lives in this inference node's <code>node.toml</code> (<code>[[inference.rollout]]</code>). Each conversation stays
on one version; <code>NAME@N</code> asks for a version whatever its weight. Changes apply to the running
<code>ymird</code> at once. <code>--canary PCT</code> on <code>add</code> or <code>weight</code> gives a version exactly
that percentage of new conversations (any of 0 to 100), the others sharing the rest. A raw weight that gives it more
than 10% needs <code>--eval FILE</code> (an <code>ymirctl eval --out</code> file where it does at least as well as the
version with the most traffic on the same rows) or <code>--force</code>.</p>
<table><thead><tr><th>Command</th><th>What it does</th></tr></thead><tbody>
<tr><td><code>models rollout list</code></td><td>Rollouts, versions, weights and shares (live when <code>ymird</code> runs).</td></tr>
<tr><td><code>models rollout add NAME@N --model ID [--weight W | --canary PCT]</code></td><td>Add a version (weight 0 by default: pinnable, no traffic yet; <code>--canary 20</code>: 20% of new conversations).</td></tr>
<tr><td><code>models rollout weight NAME@N W [--eval FILE] [--force]</code></td><td>Change a version's weight.</td></tr>
<tr><td><code>models rollout weight NAME@N --canary PCT</code></td><td>Give it exactly PCT percent of new conversations (0 to 100); the other versions share the rest.</td></tr>
<tr><td><code>models rollout rollback NAME</code></td><td>Move the newest version's traffic back to the version before it.</td></tr>
<tr><td><code>models rollout retire NAME@N [--force]</code></td><td>Drop a version with no traffic; its model stays loaded until you unload it.</td></tr>
</tbody></table>
<pre><code>ymirctl models rollout add support@3 --model qwen3-8b+support-r3 --weight 5
ymirctl eval held-out.jsonl --model qwen3-8b+support-r2 --model qwen3-8b+support-r3 --out eval.jsonl
ymirctl models rollout weight support@3 100 --eval eval.jsonl
ymirctl models rollout weight support@2 0
ymirctl models rollout rollback support</code></pre>
<p>Add <code>--inference NODE</code> to any of these to read or change another inference node&rsquo;s rollouts over
Weirdnet, by that node&rsquo;s rules (<code>--eval FILE</code> sends the scored rows with the request). The change applies
there at once and is saved in its <code>node.toml</code>; <code>--no-persist</code> changes only its live rollouts. Any of
your linked devices may, unless that node narrows them with <code>[inference] rollout_callers</code> (peer ids);
<code>[inference] remote_rollouts = false</code> turns it off.</p>
<pre><code>ymirctl models rollout add support@4 --model qwen3-8b+support-r4 --inference NODE
ymirctl models rollout rollback support --inference NODE</code></pre>

<h2 id="pool">Pool and devices</h2>
<p>These commands manage this node's place in your Weirdnet: the identity shared by your linked devices. They need
<code>node.toml</code>. Linking moves your identity to the new device with a code and a six-digit check, so no recovery
words are typed. Link codes work between Ymir, Morph and Bragi: <code>pool add</code> accepts
<code>weirdnet-ymir://</code>, <code>weirdnet-morph://</code> and <code>weirdnet-bragi://</code> codes.</p>
<h3 id="pool-link"><code>ymirctl pool link</code></h3>
<p>Run on the <em>new</em> device. It shows a link code (good for two minutes) to open on a device already in your
Weirdnet. Both screens show a short code; confirm they match. It refuses if this node is already linked. Restart
<code>ymird</code> afterwards to put the node on Weirdnet.</p>
<p>With <code>--cert-only</code> it takes only a cert-only link (sent with <code>pool add --service</code>): this server gets a
certificate and a signed service grant, never your identity seed. If the other device sends the seed anyway, nothing is
stored. A cert-only node serves and syncs like any other, but cannot link or revoke devices, and revoking it never means
starting a new identity.</p>
<h3 id="pool-add"><code>ymirctl pool add LINK</code></h3>
<p>Run on a device that is <em>already</em> in the pool, to link another device: paste the code the other device shows.</p>
<table>
<thead><tr><th>Flag</th><th>What it does</th></tr></thead>
<tbody>
<tr><td><code>--profile-name NAME</code></td><td>The name the other device gets for this user (default <code>Ymir</code>).</td></tr>
<tr><td><code>--yes</code></td><td>Skip the "same code?" prompt. Meant for tests; the codes still have to match.</td></tr>
<tr><td><code>--service inference|model</code></td><td>Link a server cert-only, without the seed. <code>inference</code> also gets the keys for the pool's conversations; <code>model</code> gets no data keys at all. Use it for every server.</td></tr>
</tbody></table>
<pre><code># on the new server
ymirctl pool link --cert-only
# on a device already linked that holds your identity
ymirctl pool add 'weirdnet-ymir://link/...' --service inference
# Pairing with Ymir on server-a: the code is 123456
# Does the other device show the same code? [y/N] y</code></pre>
<h3 id="pool-devices"><code>ymirctl pool devices</code></h3>
<p>Lists the identity and each device: <em>this node</em>, <em>linked</em> or <em>revoked</em>, with its peer id, and
<em>cert-only</em> with its role for a server linked without the seed.</p>
<h3 id="pool-cert-only"><code>ymirctl pool cert-only</code></h3>
<p>Moves a server that was linked with the seed to cert-only. Stop <code>ymird</code> first. It shows a link code for this
node's own key; on a device that holds your identity run <code>ymirctl pool add CODE --service inference</code> (or
<code>model</code>). Afterwards the node's <code>seed</code> file is overwritten and deleted; its peer id stays the same, so
nothing else needs relinking. Snapshots and backups made earlier still hold the old seed.</p>
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

<h2 id="mcp">MCP server for Cursor, Claude and VS Code</h2>
<p><code>ymirctl mcp</code> turns this node into a <a href="https://modelcontextprotocol.io">Model Context Protocol</a> server,
so an AI tool can look at your pool and use its models. It talks to the same <code>ymird</code> control socket and bridge as
the other commands, so <code>ymird</code> must be running.</p>
<h3 id="mcp-run"><code>ymirctl mcp [--http [ADDR]] [--read-only] [--allow-remove]</code></h3>
<ul>
<li><strong>stdio (default).</strong> The client starts <code>ymirctl mcp</code> itself and speaks JSON-RPC on stdin and stdout.</li>
<li><strong><code>--http [ADDR]</code>.</strong> Serves streamable HTTP at <code>http://127.0.0.1:8485/mcp</code> instead. Only loopback
addresses are accepted, requests must carry the bridge token (<code>ymirctl bridge token</code>) as a bearer token, and a request
addressed to another host name or sent from a web page is refused.</li>
<li><strong><code>--read-only</code>.</strong> Hides every tool that changes anything.</li>
<li><strong><code>--allow-remove</code>.</strong> Also offers <code>ymir_remove_model</code>. Off by default.</li>
</ul>
<h3 id="mcp-tools">Tools</h3>
<p>Reading: <code>ymir_status</code>, <code>ymir_nodes</code>, <code>ymir_devices</code>, <code>ymir_models</code> (with what each
model takes: images, audio), <code>ymir_chat</code> (text, images up to 5&nbsp;MB and audio up to 10&nbsp;MB, as files or content
parts), <code>ymir_complete</code>, <code>ymir_embed</code>, <code>ymir_service_status</code>, <code>ymir_models_search</code> and
<code>ymir_models_local</code>.</p>
<p>Resources (read-only, for clients that show them): <code>ymir://status</code>, <code>ymir://nodes</code>,
<code>ymir://models</code> and <code>ymir://models/local</code>.</p>
<p>Changing, marked <strong>MUTATES</strong> in their description, with <code>readOnlyHint: false</code>, and doing nothing unless
the call says <code>confirm: true</code>: <code>ymir_load_model</code>, <code>ymir_unload_model</code>,
<code>ymir_move_model</code> (as <code>ymirctl move</code>) and <code>ymir_pull_model</code> (without <code>confirm</code> it only shows what would be downloaded). <code>ymir_remove_model</code>
deletes downloaded files and needs <code>--allow-remove</code> as well as <code>confirm</code>. No tool edits
<code>node.toml</code>. <code>ymirctl bench</code> is not a tool yet.</p>
<h3 id="mcp-clients">Add it to your tools</h3>
<p><strong>Cursor</strong> (<code>~/.cursor/mcp.json</code>) and <strong>Claude Desktop</strong>
(<code>claude_desktop_config.json</code>):</p>
<pre><code>{
  "mcpServers": {
    "ymir": { "command": "ymirctl", "args": ["mcp"] }
  }
}</code></pre>
<p><strong>VS Code</strong> (<code>.vscode/mcp.json</code>, or the <em>MCP: Add Server</em> command):</p>
<pre><code>{
  "servers": {
    "ymir": { "type": "stdio", "command": "ymirctl", "args": ["mcp"] }
  }
}</code></pre>
<p><strong>Claude Code:</strong> <code>claude mcp add ymir -- ymirctl mcp</code>. Use the full path to <code>ymirctl</code> if your tool
does not see your shell's <code>PATH</code>, add <code>"--home", "DIR"</code> to the arguments for a node folder that is not the default, and
add <code>"--read-only"</code> to keep a tool from loading or pulling models. For the HTTP form, start
<code>ymirctl mcp --http</code> and point the client at <code>http://127.0.0.1:8485/mcp</code> with the header
<code>Authorization: Bearer TOKEN</code>.</p>

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
<li><code>update fetch SOURCE [--announcement FILE]</code>: download a bundle and stage it. <code>SOURCE</code> is the
bundle directory's <code>https://</code> URL (<code>http://</code> only to this machine) or a directory. The manifest is
checked against the pinned root before anything else is downloaded, and each file is read only up to its signed size.
With <code>--announcement</code>, a signed announcement is checked first, and the source must serve the manifest it
names. Without <code>SOURCE</code>, the announcement's first <code>https://</code> source is used.</li>
<li><code>update watch --relay ADDR [--fetch] [--once] [--wait-secs N]</code>: hear update announcements from a Weirdnet
relay that carries them (<code>…/p2p/ID</code>), and check each against the pinned root. With <code>--fetch</code>, it
downloads and stages a new one. It connects from a fresh key that doesn't listen and isn't linked to your pool.</li>
<li><code>update status</code>: whether a root is pinned, the good release, the health of a trial under way, and releases marked bad.</li>
<li><code>update boot [--exec] [--installed PATH] [-- ARGS...]</code>: choose what to start (the good release, a candidate on
trial, or a rollback) and print its path. With <code>--exec</code> it runs it, passing the arguments after <code>--</code>;
<code>--installed</code> names the daemon to use when no release is installed (default <code>ymird</code> on
<code>PATH</code>). This is the service's start command; <code>service install --managed-updates</code> sets it up.</li>
</ul>
<pre><code>ymirctl update trust --pin 3f9a...c1 --root root.signed
ymirctl update stage ./ymir-0.3.1-bundle
sudo systemctl restart ymird        # or restart the user service
ymirctl update status</code></pre>
<p>Stage a bundle directory you built or received from the release signer, or fetch one from where it is published. A
published bundle is the same directory over HTTPS (<code>manifest.env</code>, an optional <code>freshness.env</code>,
root rotations as <code>roots/1.env</code>, <code>roots/2.env</code> and so on, and the files at their paths). A relay
can withhold an announcement, but it can't make an update acceptable: everything is checked against your pinned root.</p>

<h2 id="recipes">Common tasks at a glance</h2>
<table>
<thead><tr><th>I want to…</th><th>Run</th></tr></thead>
<tbody>
<tr><td>see what a machine can run</td><td><code>ymirctl hw</code></td></tr>
<tr><td>configure and start a node</td><td><code>ymirctl setup --download --service</code></td></tr>
<tr><td>check on the pool</td><td><code>ymirctl status</code>, <code>ymirctl nodes</code>, <code>ymirctl models</code></td></tr>
<tr><td>take a node out for maintenance</td><td><code>ymirctl drain server-a</code> … <code>--resume</code></td></tr>
<tr><td>add a device</td><td><code>ymirctl pool link</code> on it, <code>ymirctl pool add CODE</code> on a linked one</td></tr>
<tr><td>add a server without the seed</td><td><code>ymirctl pool link --cert-only</code> on it, <code>ymirctl pool add CODE --service inference</code> (or <code>model</code>) on a device with the seed</td></tr>
<tr><td>get or change the API key</td><td><code>ymirctl bridge token</code>, <code>--rotate</code>, then restart <code>ymird</code></td></tr>
<tr><td>use the pool from Cursor, Claude or VS Code</td><td><a href="#mcp"><code>ymirctl mcp</code></a></td></tr>
<tr><td>run many prompts</td><td><code>ymirctl batch prompts.txt</code></td></tr>
<tr><td>generate a dataset</td><td><code>ymirctl batch rows.jsonl --jsonl --model M --license ID --out data.jsonl</code></td></tr>
<tr><td>score models</td><td><code>ymirctl eval held-out.jsonl --model M</code></td></tr>
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
