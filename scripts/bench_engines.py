#!/usr/bin/env python3
"""llama.cpp vs mistral.rs on the same GGUF: load time, TTFT, prompt and decode speed, memory, and a
vision request. Scratch directory and ports only; every server it starts is stopped again.

Used for the engine evaluation in docs/DESIGN.md ("Engines per platform") on m1 (GB10):
  YMIR_BENCH_DIR=~/ymir-bench (holds the unpacked mistral.rs release: mistralrs + lib/)
  python3 bench_engines.py mistral.rs:gemma4-e4b llama.cpp:gemma4-e4b ...
An engine suffix `-nopa` turns mistral.rs paged attention off; `-text` skips the projector.
The model table and llama-server path at the top are m1's; change them for another machine."""
import json, os, signal, subprocess, sys, time, urllib.request, statistics

HOME = os.path.expanduser(os.environ.get("YMIR_BENCH_DIR", "~/ymir-bench"))
LLAMA = os.environ.get("YMIR_BENCH_LLAMA", "/opt/llm/llama.cpp/build/bin/llama-server")
MRS = f"{HOME}/mistralrs"
CTX = 8192
GEN = 256
RUNS = 3
WORDS = ("The northern rail line crosses three rivers and climbs through pine forest before reaching the "
         "old mining town, where the station clock has been stopped at a quarter past four since the flood. ")
PROMPT = "Read the passage and then write a long, detailed story that continues it.\n\n" + WORDS * 40

def red_png(n=64):
    import zlib, struct, base64
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * n for _ in range(n))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode()

def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":"); d[k] = int(v.split()[0]) * 1024
    return d["MemAvailable"]

def tree_rss(pid):
    pids, todo = set(), [pid]
    while todo:
        p = todo.pop(); pids.add(p)
        try:
            for t in os.listdir(f"/proc/{p}/task"):
                for c in open(f"/proc/{p}/task/{t}/children").read().split():
                    todo.append(int(c))
        except OSError: pass
    rss = 0
    for p in pids:
        try:
            for line in open(f"/proc/{p}/status"):
                if line.startswith("VmRSS"): rss += int(line.split()[1]) * 1024
        except OSError: pass
    return rss

def gpu_mem(pid):
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            p, m = [x.strip() for x in line.split(",")]
            if p == str(pid) and m.isdigit(): return int(m) * 2**20
    except Exception: pass
    return None

def get(url, timeout=2):
    with urllib.request.urlopen(url, timeout=timeout) as r: return r.status, r.read()

def chat(port, model, prompt, max_tokens, stream=True):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": stream, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.monotonic(); first = None; n = 0; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        if not stream:
            return time.monotonic() - t0, json.loads(r.read())
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            ev = json.loads(data)
            if ev.get("usage"): usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    n += 1
                    if first is None: first = time.monotonic()
    end = time.monotonic()
    toks = (usage or {}).get("completion_tokens") or n
    ptoks = (usage or {}).get("prompt_tokens")
    return {"ttft": first - t0, "total": end - t0, "gen_tokens": toks, "chunks": n, "prompt_tokens": ptoks,
            "decode_tps": (toks - 1) / (end - first) if toks > 1 and end > first else None,
            "prefill_tps": ptoks / (first - t0) if ptoks else None}

def start(engine, gguf, mmproj, port, alias, tok=None, nopa=False):
    log = open(f"{HOME}/{engine}-{alias}.log", "w")
    env = dict(os.environ)
    if engine == "llama.cpp":
        cmd = [LLAMA, "--model", gguf, "--host", "127.0.0.1", "--port", str(port), "--alias", alias, "--no-webui",
               "--ctx-size", str(CTX), "--parallel", "1", "--n-gpu-layers", "99", "-fa", "on", "--jinja"]
        if mmproj: cmd += ["--mmproj", mmproj]
    else:
        env["LD_LIBRARY_PATH"] = f"{HOME}/lib"
        d, f = os.path.split(gguf)
        cmd = [MRS, "serve", "--host", "127.0.0.1", "-p", str(port), "--no-ui", "--max-seqs", "1",
               "--kv-cache", f"{CTX}tokens", "--disable-metrics", "--token-source", "none",
               "-m", d, "-f", f, "--max-model-len", str(CTX)]
        if mmproj: cmd += ["--mmproj", os.path.basename(mmproj)]
        if tok: cmd += ["--tok-model-id", tok]
        if nopa:
            i = cmd.index("--kv-cache"); del cmd[i:i+2]
            cmd += ["--paged-attn", "off"]
    cmd = ["nice", "-n", "10", "ionice", "-c2", "-n7"] + cmd
    print("+", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)

def stop(p):
    try: os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError: return
    try: p.wait(30)
    except subprocess.TimeoutExpired: os.killpg(p.pid, signal.SIGKILL); p.wait()

def engine_pid(p):
    # nice -> ionice -> exec'd engine: same pid after exec.
    return p.pid

def bench(engine, alias, gguf, mmproj, port, tok=None):
    nopa = engine.endswith("-nopa"); engine = engine.removesuffix("-nopa")
    before = meminfo()
    t0 = time.monotonic()
    p = start(engine, gguf, mmproj, port, alias, tok, nopa)
    res = {"engine": engine + (" (paged-attn off)" if nopa else ""), "model": alias}
    log_name = f"{HOME}/{engine}-{alias}.log"
    try:
        while True:
            if p.poll() is not None:
                res["error"] = f"exited {p.returncode}: " + open(log_name).read()[-1500:]
                return res
            try:
                st, body = get(f"http://127.0.0.1:{port}/v1/models")
                if st == 200:
                    alias = json.loads(body)["data"][0]["id"]
                    chat(port, alias, "Say hi.", 1, stream=False); break
            except Exception: pass
            if time.monotonic() - t0 > 900: res["error"] = "not ready after 900s"; return res
            time.sleep(0.5)
        res["load_s"] = time.monotonic() - t0
        chat(port, alias, "Warm-up. " + PROMPT, 16)
        runs = [chat(port, alias, f"Session {i}-{time.time_ns()}. " + PROMPT, GEN) for i in range(RUNS)]
        res["runs"] = runs
        for k in ("ttft", "decode_tps", "prefill_tps", "gen_tokens", "prompt_tokens"):
            vals = [r[k] for r in runs if r[k] is not None]
            res[k] = statistics.median(vals) if vals else None
        res["rss"] = tree_rss(engine_pid(p))
        res["gpu_mem"] = gpu_mem(engine_pid(p))
        res["mem_delta"] = before - meminfo()
        if mmproj:
            try:
                img = "data:image/png;base64," + red_png()
                body = {"model": alias, "max_tokens": 32, "temperature": 0, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "What colour is this image?"}, {"type": "image_url", "image_url": {"url": img}}]}]}
                req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    m = json.loads(r.read())["choices"][0]["message"]
                    res["vision"] = "ok: " + ((m.get("content") or m.get("reasoning_content") or "")[:80]).replace("\n", " ")
            except Exception as e:
                res["vision"] = f"failed: {e}"
        return res
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}; log: " + open(log_name).read()[-1500:]
        return res
    finally:
        stop(p)
        time.sleep(3)

if __name__ == "__main__":
    models = {
        "gemma4-e4b": ("/srv/models/gguf/gemma4-e4b-q4_K_M.gguf", "/srv/models/gguf/gemma4-e4b-mmproj-f16.gguf", None),
        "qwen3.6-35b-a3b": ("/srv/models/gguf/qwen3.6-35b-a3b-q4_K_M.gguf", "/srv/models/gguf/qwen3.6-35b-a3b-mmproj-q8_0.gguf", "Qwen/Qwen3.6-35B-A3B"),
    }
    sel = sys.argv[1:] or ["llama.cpp:gemma4-e4b", "mistral.rs:gemma4-e4b", "llama.cpp:qwen3.6-35b-a3b", "mistral.rs:qwen3.6-35b-a3b"]
    out = []
    for i, s in enumerate(sel):
        engine, alias = s.split(":")
        gguf, mmproj, tok = models[alias]
        if engine.endswith("-text"): engine, mmproj = engine[:-5], None
        r = bench(engine, alias, gguf, mmproj, 18511 + i, tok)
        print(json.dumps(r), flush=True)
        out.append(r)
    json.dump(out, open(f"{HOME}/results-{int(time.time())}.json", "w"), indent=1)
