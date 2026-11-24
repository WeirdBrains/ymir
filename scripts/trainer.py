#!/usr/bin/env python3
"""Ymir's llama.cpp trainer: the program `[train] trainer` points at (training plan P3a).

It speaks Ymir's trainer interface 1 (crates/ymir-core/src/train.rs) around upstream llama.cpp at
the build pinned in packaging/engines/llama.cpp.toml (the one scripts/engine.py ships):

    trainer.py version
    trainer.py train --base BASE.gguf --data DATA.jsonl --config CONFIG.json --out FULL.gguf
    trainer.py extract-lora --base BASE.gguf --tuned FULL.gguf --rank N --out ADAPTER.gguf

`train` is a full fine-tune with upstream `llama-finetune`: every weight of an F32 GGUF base (the
only kind upstream trains), on the dataset's conversations rendered with the base's own chat
template. `extract-lora` takes each weight's difference between the tuned model and the base and
keeps its top `rank` singular directions (a truncated SVD), written as a GGUF LoRA adapter that
llama.cpp loads on the base, any quantization of it included (`--lora`, training plan P4a).

Upstream's release archives don't carry `llama-finetune`; build it at the pin with
`python3 scripts/engine.py build --trainer --out DIR`. This program finds it in
$YMIR_LLAMA_FINETUNE, next to itself, in ../lib/ymir/llama.cpp, or on PATH, and refuses any build
but the pinned one, or one without Ymir's patches (REQUIRED_PATCHES, applied by `build --trainer`
from packaging/engines/patches/; the patched program names them with `--ymir-patches`).

Needs Python 3.11+, numpy and gguf 0.19.0 (llama.cpp's gguf-py at the pin), and jinja2 for chat
templates: `pip install -r scripts/trainer-requirements.txt`. With them in a venv, set
YMIR_TRAINER_PYTHON (under [train.env] in node.toml) to its python and this program runs itself
there.

Environment: YMIR_TRAINER_NGL (llama-finetune -ngl, default all; 0 trains on the CPU),
YMIR_TRAINER_ARGS (extra llama-finetune arguments, shell-quoted).
"""

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path

INTERFACE = 1
ENGINE = "llama.cpp"
HERE = Path(__file__).resolve().parent
PIN_FILE = HERE.parent / "packaging/engines/llama.cpp.toml"
# The pin when this program is installed without the repository; test_trainer.py checks it
# against PIN_FILE.
PINNED_BUILD = "b12628"
# Ymir's patches the pinned llama-finetune must carry (scripts/engine.py TRAINER_PATCHES; the
# patched program prints these ids for `--ymir-patches`). parse-special: the training text is
# tokenized with parse_special = true, so chat-template special tokens train as special tokens
# (upstream b12628 tokenizes them as text). Drop it when upstream releases the fix (tracked: draft
# ggml-org/llama.cpp#31258) and the pin moves past it.
REQUIRED_PATCHES = ("parse-special",)


def pinned_build():
    if PIN_FILE.exists():
        import tomllib

        return tomllib.loads(PIN_FILE.read_text())["build"]
    return PINNED_BUILD


# ---------------------------------------------------------------------------------------------
# Events (stdout, one JSON object per line)


def emit(event, **kw):
    print(json.dumps({"event": event, **kw}), flush=True)


def log(message):
    emit("log", message=message)


class Refusal(Exception):
    pass


# ---------------------------------------------------------------------------------------------
# Dependencies


def need_deps():
    """Imports numpy and gguf, re-running under $YMIR_TRAINER_PYTHON once if they are missing."""
    try:
        import gguf  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as e:
        py = os.environ.get("YMIR_TRAINER_PYTHON")
        # Not resolved: a venv's python is a symlink to the system one, with other packages.
        if py and os.environ.get("_YMIR_TRAINER_REEXEC") != "1" and os.path.abspath(py) != os.path.abspath(sys.executable):
            os.environ["_YMIR_TRAINER_REEXEC"] = "1"
            os.execv(py, [py, str(Path(__file__).resolve()), *sys.argv[1:]])
        raise Refusal(f"{e.name} is not installed for {sys.executable}: pip install -r scripts/trainer-requirements.txt, or set YMIR_TRAINER_PYTHON to a python that has it") from e


# ---------------------------------------------------------------------------------------------
# llama-finetune


def find_finetune():
    name = "llama-finetune.exe" if os.name == "nt" else "llama-finetune"
    env = os.environ.get("YMIR_LLAMA_FINETUNE")
    if env:
        return Path(env)
    for d in (HERE, HERE.parent / "lib/ymir/llama.cpp"):
        if (d / name).exists():
            return d / name
    found = shutil.which(name)
    if found:
        return Path(found)
    raise Refusal("no llama-finetune: build it at the pin (python3 scripts/engine.py build --trainer --out DIR) and set YMIR_LLAMA_FINETUNE to it")


def finetune_build(exe):
    """The build `llama-finetune --version` reports (`version: 12628 (abc1234)` -> b12628)."""
    try:
        r = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60)
    except OSError as e:
        raise Refusal(f"running {exe} --version: {e}") from e
    m = re.search(r"version:\s*(\d+)", r.stdout + r.stderr)
    if not m:
        raise Refusal(f"{exe} --version printed no build ({(r.stdout + r.stderr).strip()[:200]!r})")
    return f"b{m.group(1)}"


def finetune_patches(exe):
    """The Ymir patches `llama-finetune --ymir-patches` names; none for an unpatched build (upstream
    refuses the unknown argument)."""
    try:
        r = subprocess.run([str(exe), "--ymir-patches"], capture_output=True, text=True, timeout=60)
    except OSError as e:
        raise Refusal(f"running {exe} --ymir-patches: {e}") from e
    if r.returncode != 0:
        return []
    return [line.strip() for line in r.stdout.splitlines() if re.fullmatch(r"[a-z0-9-]+", line.strip())]


def check_pin(exe):
    build, pin = finetune_build(exe), pinned_build()
    if build != pin:
        raise Refusal(f"{exe} is llama.cpp {build}, and this trainer is pinned to {pin} (packaging/engines/llama.cpp.toml); build the pinned one")
    missing = [p for p in REQUIRED_PATCHES if p not in finetune_patches(exe)]
    if missing:
        raise Refusal(
            f"{exe} is llama.cpp {build} without Ymir's {', '.join(missing)} patch (packaging/engines/patches); "
            "build it with python3 scripts/engine.py build --trainer"
        )
    return build


# Upstream's progress bar (ggml-opt.cpp ggml_opt_epoch_callback_progress_bar), one per batch:
# `train: [=====>    ] data=0000012/0000100 loss=2.34567±0.01234 acc=...`, ended by \r.
PROGRESS = re.compile(r"^\s*(\w+)\s*:?\s*\[.*?\]\s*data=(\d+)/(\d+)\s+loss=([-+0-9.eEnaif]+)")


class Progress:
    """Turns llama-finetune's progress bars into interface progress events over all epochs."""

    def __init__(self, epochs):
        self.epochs = max(1, epochs)
        self.epoch = 0
        self.last = None
        self.val_loss = None

    def feed(self, chunk):
        m = PROGRESS.match(chunk)
        if not m:
            return None
        kind, idata, imax, loss = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
        if kind.startswith("val"):
            self.val_loss = loss
            return None
        if imax <= 0:
            return None
        step = min(self.epoch, self.epochs - 1) * imax + idata
        if idata >= imax:
            self.epoch += 1
        if step == self.last:
            return None
        self.last = step
        return {"step": step, "steps": self.epochs * imax, "loss": loss}


def run_finetune(cmd, progress):
    """Runs llama-finetune, relaying progress; SIGTERM stops it. Returns its exit code."""
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def stop(signum, _frame):
        if proc.poll() is None:
            proc.terminate()

    signal.signal(signal.SIGTERM, stop)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, stop)
    buf = b""
    tail = []
    while True:
        data = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
        if not data:
            break
        buf += data
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()
        for raw in parts:
            line = raw.decode("utf-8", "replace").rstrip()
            if not line:
                continue
            ev = progress.feed(line)
            if ev:
                emit("progress", **ev)
            elif not PROGRESS.match(line):
                print(line, flush=True)  # not an event: goes to the run's log
                tail = (tail + [line])[-20:]
    if buf.strip():
        print(buf.decode("utf-8", "replace"), flush=True)
    code = proc.wait()
    return code, tail


# ---------------------------------------------------------------------------------------------
# GGUF helpers


def field_str(reader, key):
    f = reader.fields.get(key)
    if f is None or not f.data:
        return None
    return bytes(f.parts[f.data[0]]).decode("utf-8")


def field_int(reader, key):
    f = reader.fields.get(key)
    if f is None or not f.data:
        return None
    return int(f.parts[f.data[0]][0])


def token_text(reader, tid):
    f = reader.fields.get("tokenizer.ggml.tokens")
    if tid is None or f is None or tid >= len(f.data):
        return ""
    return bytes(f.parts[f.data[tid]]).decode("utf-8", "replace")


def check_trainable(reader, path):
    """Upstream trains F32 models only: every matrix of the base must be F32."""
    from gguf import GGMLQuantizationType as T

    bad = sorted({t.tensor_type.name for t in reader.tensors if len(t.shape) >= 2 and t.tensor_type != T.F32})
    if bad:
        raise Refusal(f"{path} has {', '.join(bad)} weights, and llama-finetune ({pinned_build()}) trains F32 models only; convert the base with convert_hf_to_gguf.py --outtype f32 and pass it with --from (the adapter still serves on the quantized base)")


# ---------------------------------------------------------------------------------------------
# The dataset as training text


def render(rows, template, bos, eos):
    """One text: each conversation rendered with the base's chat template (or plain turns)."""
    if template:
        try:
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError:
            template = None
            log("jinja2 is not installed: conversations are rendered as plain `role: text` turns, not the model's chat template")
    if not template:
        return "\n\n".join("\n".join(f"{m['role']}: {m['content']}" for m in r["messages"]) + eos for r in rows), False

    def raise_exception(msg):
        raise ValueError(msg)

    def strftime_now(fmt):
        import datetime

        return datetime.datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals.update(raise_exception=raise_exception, strftime_now=strftime_now)
    t = env.from_string(template)
    # bos is left out: llama-finetune adds one BOS to the whole text.
    return "".join(t.render(messages=r["messages"], add_generation_prompt=False, bos_token="", eos_token=eos) for r in rows), True


def read_rows(path):
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        msgs = r.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise Refusal(f"{path}:{n}: no messages")
        rows.append({"messages": [{"role": str(m["role"]), "content": str(m["content"])} for m in msgs]})
    if not rows:
        raise Refusal(f"{path} has no conversations")
    return rows


# ---------------------------------------------------------------------------------------------
# Commands


def cmd_version(_a):
    exe = find_finetune()
    print(json.dumps({"interface": INTERFACE, "engine": ENGINE, "build": finetune_build(exe), "patches": finetune_patches(exe)}), flush=True)
    return 0


def cmd_train(a):
    need_deps()
    from gguf import GGUFReader

    exe = find_finetune()
    build = check_pin(exe)
    hyper = json.loads(Path(a.config).read_text())
    epochs, ctx = int(hyper["epochs"]), int(hyper["context"])
    reader = GGUFReader(a.base)
    check_trainable(reader, a.base)
    rows = read_rows(a.data)
    eos = token_text(reader, field_int(reader, "tokenizer.ggml.eos_token_id"))
    text, templated = render(rows, field_str(reader, "tokenizer.chat_template"), "", eos)
    del reader
    if not templated:
        log("the base has no chat template: conversations are rendered as plain `role: text` turns")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    corpus = out.with_name(out.name + ".train.txt")
    fd = os.open(corpus, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    if int(hyper.get("batch", 1)) != 1:
        log(f"llama-finetune {build} trains one {ctx}-token sequence per step; batch {hyper['batch']} is not used")
    cmd = [
        str(exe), "-m", str(a.base), "-f", str(corpus), "-o", str(out),
        "-c", str(ctx), "-b", str(ctx), "-ub", str(ctx),
        "--epochs", str(epochs), "-lr", repr(float(hyper["learning_rate"])),
        "--seed", str(int(hyper.get("seed", 0))),
        "-ngl", os.environ.get("YMIR_TRAINER_NGL", "all"), "-fa", "off",
        *shlex.split(os.environ.get("YMIR_TRAINER_ARGS", "")),
    ]
    log(f"llama-finetune {build}: {len(rows)} conversations, {len(text)} characters, {epochs} epoch(s), context {ctx}")
    progress = Progress(epochs)
    try:
        code, tail = run_finetune(cmd, progress)
    finally:
        corpus.unlink(missing_ok=True)
    if code != 0 or not out.exists():
        raise Refusal(f"llama-finetune exited with {code}" + (f": {tail[-1]}" if tail else ""))
    if progress.val_loss is not None:
        log(f"validation loss {progress.val_loss:.5f}")
    emit("done")
    return 0


def lora_tensors(base, tuned, rank):
    """(name, lora_a, lora_b, kept) for each matrix the fine-tune changed; kept is the share of
    the change's energy the rank keeps."""
    import numpy as np
    from gguf.quants import dequantize

    tuned_by_name = {t.name: t for t in tuned.tensors}
    for t in base.tensors:
        if len(t.shape) != 2 or not t.name.startswith("blk."):
            continue  # norms, biases, embeddings and the output head stay as the base has them
        u = tuned_by_name.get(t.name)
        if u is None:
            raise Refusal(f"the tuned model has no {t.name}")
        b = dequantize(t.data, t.tensor_type).astype(np.float32)
        w = dequantize(u.data, u.tensor_type).astype(np.float32)
        if w.shape != b.shape:
            raise Refusal(f"{t.name} is {list(w.shape)} in the tuned model and {list(b.shape)} in the base")
        delta = w - b
        if not np.any(delta):
            yield t.name, None, None, 1.0
            continue
        U, S, Vt = np.linalg.svd(delta.astype(np.float64), full_matrices=False)
        r = min(rank, S.shape[0])
        energy = float((S**2).sum())
        kept = float((S[:r] ** 2).sum()) / energy if energy else 1.0
        lora_b = np.zeros((delta.shape[0], rank), np.float32)
        lora_a = np.zeros((rank, delta.shape[1]), np.float32)
        lora_b[:, :r] = U[:, :r] * S[:r]
        lora_a[:r] = Vt[:r]
        yield t.name, lora_a, lora_b, kept


def cmd_extract(a):
    need_deps()
    import gguf
    from gguf import GGUFReader

    check_pin(find_finetune())
    if a.rank < 1:
        raise Refusal("--rank must be at least 1")
    base, tuned = GGUFReader(a.base), GGUFReader(a.tuned)
    arch = field_str(base, "general.architecture")
    if not arch:
        raise Refusal(f"{a.base} has no general.architecture")
    if field_str(tuned, "general.architecture") != arch:
        raise Refusal(f"{a.tuned} is not a fine-tune of {a.base} (architecture {field_str(tuned, 'general.architecture')} vs {arch})")
    total = sum(1 for t in base.tensors if len(t.shape) == 2 and t.name.startswith("blk."))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(out), arch)
    w.add_type(gguf.GGUFType.ADAPTER)
    w.add_string(gguf.Keys.Adapter.TYPE, "lora")
    # llama.cpp scales an adapter by alpha / rank: alpha = rank applies the difference as it is.
    w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, float(a.rank))
    name = field_str(base, "general.name")
    if name:
        w.add_name(f"{name} LoRA")
    done = changed = 0
    kept_sum = 0.0
    for tname, la, lb, kept in lora_tensors(base, tuned, a.rank):
        done += 1
        if la is not None:
            w.add_tensor(f"{tname}.lora_a", la)
            w.add_tensor(f"{tname}.lora_b", lb)
            changed += 1
            kept_sum += kept
        emit("progress", step=done, steps=total)
    if not changed:
        raise Refusal(f"{a.tuned} has the same weights as {a.base}: nothing to extract")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    log(f"rank {a.rank} adapter for {changed} of {total} matrices; it keeps {100 * kept_sum / changed:.1f}% of the fine-tune's change on average")
    emit("done")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version")
    p = sub.add_parser("train")
    for k in ("--base", "--data", "--config", "--out"):
        p.add_argument(k, required=True)
    p = sub.add_parser("extract-lora")
    for k in ("--base", "--tuned", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--rank", type=int, required=True)
    a = ap.parse_args(argv)
    try:
        return {"version": cmd_version, "train": cmd_train, "extract-lora": cmd_extract}[a.cmd](a)
    except Refusal as e:
        if a.cmd == "version":
            print(str(e), file=sys.stderr)
        else:
            emit("error", message=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
