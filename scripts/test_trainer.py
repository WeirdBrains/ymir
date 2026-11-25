"""Tests for scripts/trainer.py against a fake llama-finetune (scripts/testdata). No model is
trained. Needs the trainer's requirements (scripts/trainer-requirements.txt); CI runs it on Linux:

    python3 scripts/test_trainer.py
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
import unittest
from pathlib import Path

import numpy as np
import gguf
from gguf import GGUFReader

HERE = Path(__file__).resolve().parent
TRAINER = HERE / "trainer.py"
FAKE = HERE / "testdata/fake_llama_finetune.py"
FAKE_TOKENIZE = HERE / "testdata/fake_llama_tokenize.py"
CHATML = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"

sys.path.insert(0, str(HERE))
import trainer  # noqa: E402


def write_model(path, dtype=np.float32, template=CHATML, arch="llama"):
    rng = np.random.default_rng(1)
    w = gguf.GGUFWriter(str(path), arch)
    w.add_name("Tiny")
    if template is not None:
        w.add_chat_template(template)
    # Control tokens (type 3) are what llama-finetune with Ymir's parse-special patch tokenizes as specials.
    w.add_token_list(["<s>", "</s>", "<|im_start|>", "<|im_end|>", "a"])
    w.add_token_types([3, 3, 3, 3, 1])
    w.add_bos_token_id(0)
    w.add_eos_token_id(3)
    w.add_tensor("token_embd.weight", rng.standard_normal((5, 6)).astype(np.float32))
    w.add_tensor("blk.0.attn_norm.weight", np.ones(6, np.float32))
    for name, shape in (("attn_q", (6, 6)), ("attn_k", (3, 6)), ("ffn_up", (10, 6)), ("ffn_down", (6, 10))):
        w.add_tensor(f"blk.0.{name}.weight", rng.standard_normal(shape).astype(dtype))
    w.add_tensor("output.weight", rng.standard_normal((5, 6)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


class TrainerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ymir-trainer-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        exe = self.tmp / "llama-finetune"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
        exe.chmod(0o755)
        self.record = self.tmp / "record.json"
        self.env = {**os.environ, "YMIR_LLAMA_FINETUNE": str(exe), "FAKE_FINETUNE_RECORD": str(self.record)}
        self.base = self.tmp / "base.gguf"
        write_model(self.base)
        self.data = self.tmp / "data.jsonl"
        rows = [
            {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
            {"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]},
        ]
        self.data.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.config = self.tmp / "config.json"
        self.config.write_text(json.dumps({"epochs": 2, "learning_rate": 1e-5, "context": 256, "batch": 4, "seed": 9, "rank": 4}))

    def run_trainer(self, *args, env=None):
        r = subprocess.run([sys.executable, str(TRAINER), *args], capture_output=True, text=True, env=env or self.env, timeout=120)
        events = [json.loads(line) for line in r.stdout.splitlines() if line.startswith("{")]
        return r, events

    def train(self, out=None, env=None):
        out = out or self.tmp / "full.gguf"
        return self.run_trainer("train", "--base", str(self.base), "--data", str(self.data), "--config", str(self.config), "--out", str(out), env=env)

    def test_pin_matches_engine_pin(self):
        pin = tomllib.loads((HERE.parent / "packaging/engines/llama.cpp.toml").read_text())["build"]
        self.assertEqual(trainer.PINNED_BUILD, pin, "update PINNED_BUILD in scripts/trainer.py with the engine pin")

    def test_required_patches_match_engine(self):
        import engine

        self.assertEqual(set(trainer.REQUIRED_PATCHES), set(engine.TRAINER_PATCHES), "the trainer requires exactly what build --trainer applies")
        for pid, path in zip(engine.TRAINER_PATCHES, engine.trainer_patches()):
            text = path.read_text()
            self.assertIn(f'std::printf("{pid}\\n");', text, f"{path.name} names itself as {pid} for --ymir-patches")
        self.assertIn("+    std::vector<llama_token> tokens  = common_tokenize(ctx, params.prompt, true, /*parse_special =*/ true);", (engine.PATCHES / engine.TRAINER_PATCHES["parse-special"]).read_text())

    def test_parse_build(self):
        self.assertEqual(trainer.parse_build("version: 1.0.0-dev (build 12628, commit 8a97fa9)\nbuilt with Clang"), "b12628")
        self.assertEqual(trainer.parse_build("version: 12628 (8a97fa9)"), "b12628", "the format before llama.cpp 1.0")
        self.assertEqual(trainer.parse_build("version: 1.0.0-dev (build 1, commit 8a97fa9)"), "b1", "a shallow clone without LLAMA_BUILD_NUMBER")
        self.assertIsNone(trainer.parse_build("usage: llama-finetune ..."))

    def test_unpatched_build_is_refused(self):
        unpatched = {**self.env, "FAKE_FINETUNE_PATCHES": ""}
        r, _ = self.run_trainer("version", env=unpatched)
        self.assertEqual(json.loads(r.stdout)["patches"], [], "version reports what the program carries")
        r, ev = self.train(env=unpatched)
        self.assertEqual(r.returncode, 1)
        self.assertIn("without Ymir's parse-special patch", ev[-1]["message"])
        self.assertIn("engine.py build --trainer", ev[-1]["message"])
        self.assertFalse(self.record.exists(), "nothing was trained")
        r, ev = self.run_trainer("extract-lora", "--base", str(self.base), "--tuned", str(self.base), "--rank", "2", "--out", str(self.tmp / "a.gguf"), env=unpatched)
        self.assertEqual(r.returncode, 1)
        self.assertIn("parse-special", ev[-1]["message"])
        r, ev = self.train(env={**self.env, "FAKE_FINETUNE_PATCHES": "something-else"})
        self.assertEqual(r.returncode, 1)
        self.assertIn("parse-special", ev[-1]["message"])

    def test_special_tokens_train_as_specials(self):
        r, _ = self.train()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        rec = json.loads(self.record.read_text())
        self.assertEqual(rec["special_tokens"], ["<|im_start|>", "<|im_end|>"] * 5, "each turn's markers are single special tokens")
        self.assertNotIn("<s>", rec["special_tokens"])

    def test_version(self):
        r, _ = self.run_trainer("version")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout), {"interface": 1, "engine": "llama.cpp", "build": "b12628", "patches": ["parse-special"]})
        r, _ = self.run_trainer("version", env={**self.env, "FAKE_FINETUNE_BUILD": "13000"})
        self.assertEqual(json.loads(r.stdout)["build"], "b13000", "version reports the real build, so Ymir's pin check sees it")
        env = {k: v for k, v in self.env.items() if k != "YMIR_LLAMA_FINETUNE"}
        env["PATH"] = str(self.tmp / "empty")
        r = subprocess.run([sys.executable, str(TRAINER), "version"], capture_output=True, text=True, env=env)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("engine.py build --trainer", r.stderr)

    def test_train(self):
        r, ev = self.train()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(ev[-1], {"event": "done"})
        prog = [e for e in ev if e["event"] == "progress"]
        self.assertEqual([p["step"] for p in prog], list(range(1, 9)))
        self.assertTrue(all(p["steps"] == 8 for p in prog))
        self.assertAlmostEqual(prog[-1]["loss"], 0.25)
        logs = " ".join(e["message"] for e in ev if e["event"] == "log")
        self.assertIn("batch 4 is not used", logs)
        self.assertIn("validation loss 1.50000", logs)
        self.assertIn("llama_model_loader", r.stdout, "llama-finetune's own lines go to the run log")
        rec = json.loads(self.record.read_text())
        a = rec["args"]
        for k, v in (("-c", "256"), ("-b", "256"), ("-ub", "256"), ("--epochs", "2"), ("-lr", "1e-05"), ("--seed", "9"), ("-ngl", "all"), ("-fa", "off")):
            self.assertEqual(a[a.index(k) + 1], v, k)
        self.assertEqual(rec["text_mode"], "0o600")
        self.assertEqual(rec["text"], "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n")
        self.assertTrue((self.tmp / "full.gguf").exists())
        self.assertFalse((self.tmp / "full.gguf.train.txt").exists(), "the training text (private data) is removed")

    def test_train_env_args(self):
        r, _ = self.train(env={**self.env, "YMIR_TRAINER_NGL": "0", "YMIR_TRAINER_ARGS": "--threads 3"})
        self.assertEqual(r.returncode, 0, r.stdout)
        a = json.loads(self.record.read_text())["args"]
        self.assertEqual(a[a.index("-ngl") + 1], "0")
        self.assertEqual(a[-2:], ["--threads", "3"])

    def test_train_without_template(self):
        write_model(self.base, template=None)
        r, ev = self.train()
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("no chat template", " ".join(e.get("message", "") for e in ev))
        self.assertTrue(json.loads(self.record.read_text())["text"].startswith("user: hi\nassistant: hello<|im_end|>"))

    def test_train_refusals(self):
        write_model(self.base, dtype=np.float16)
        r, ev = self.train()
        self.assertEqual(r.returncode, 1)
        self.assertIn("F32 models only", ev[-1]["message"])
        write_model(self.base)
        r, ev = self.train(env={**self.env, "FAKE_FINETUNE_BUILD": "13000"})
        self.assertEqual(r.returncode, 1)
        self.assertIn("pinned to b12628", ev[-1]["message"])
        r, ev = self.train(env={**self.env, "FAKE_FINETUNE_FAIL": "1"})
        self.assertEqual(r.returncode, 1)
        self.assertEqual(ev[-1]["event"], "error")
        self.assertIn("exited with 3: GGML_ASSERT failed: out of memory", ev[-1]["message"])
        self.assertFalse((self.tmp / "full.gguf.train.txt").exists())
        self.data.write_text("\n")
        r, ev = self.train()
        self.assertIn("no conversations", ev[-1]["message"])

    def add_tokenize(self):
        """Puts the fake llama-tokenize next to the fake llama-finetune, as build --trainer does."""
        exe = self.tmp / "llama-tokenize"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_TOKENIZE}" "$@"\n')
        exe.chmod(0o755)
        self.tok_record = self.tmp / "tokenize.json"
        return {**self.env, "FAKE_TOKENIZE_RECORD": str(self.tok_record)}

    def test_min_tokens(self):
        self.assertEqual([trainer.trained_context(c) for c in (1, 64, 256, 257, 512, 4096)], [256, 256, 256, 512, 512, 4096])
        self.assertEqual([trainer.min_tokens(c) for c in (64, 256, 257, 512)], [513, 513, 1025, 1025])

    def test_short_text_is_refused_before_training(self):
        env = self.add_tokenize()
        r, ev = self.train(env={**env, "FAKE_TOKENIZE_COUNT": "512"})
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        msg = ev[-1]["message"]
        self.assertEqual(ev[-1]["event"], "error")
        for part in ("renders to 512 tokens", "at least 513 (2 x context + 1)", "about 1 more tokens"):
            self.assertIn(part, msg)
        self.assertNotIn("lower --context", msg, "256 is as low as llama.cpp goes")
        self.assertFalse(self.record.exists(), "llama-finetune is not started")
        self.assertFalse((self.tmp / "full.gguf.train.txt").exists())
        rec = json.loads(self.tok_record.read_text())
        self.assertEqual(rec["args"], ["-m", str(self.base), "--stdin", "--ids", "--log-disable"])
        self.assertTrue(rec["text"].endswith("4<|im_end|>"), "one trailing newline is dropped, as llama-finetune -f does")

    def test_enough_text_trains(self):
        env = self.add_tokenize()
        r, ev = self.train(env={**env, "FAKE_TOKENIZE_COUNT": "513"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(ev[-1], {"event": "done"})
        self.assertIn("513 tokens, 2 windows of 256", " ".join(e.get("message", "") for e in ev))

    def test_context_is_rounded_up_to_256(self):
        env = self.add_tokenize()
        self.config.write_text(json.dumps({"epochs": 1, "learning_rate": 1e-5, "context": 300, "batch": 1}))
        r, ev = self.train(env={**env, "FAKE_TOKENIZE_COUNT": "700"})
        self.assertEqual(r.returncode, 1)
        msg = ev[-1]["message"]
        for part in ("renders to 700 tokens", "at least 1025", "context 300 trains as 512", "lower --context (down to 256, which needs 513)"):
            self.assertIn(part, msg)

    def test_tokenize_failure_is_refused(self):
        env = self.add_tokenize()
        r, ev = self.train(env={**env, "FAKE_TOKENIZE_FAIL": "1"})
        self.assertEqual(r.returncode, 1)
        self.assertIn("could not count the training text's tokens", ev[-1]["message"])
        self.assertFalse(self.record.exists())

    def test_short_text_crash_without_tokenize(self):
        """A build without llama-tokenize: upstream's crash on a short text becomes the same refusal."""
        r, ev = self.train(env={**self.env, "FAKE_FINETUNE_SHORT": "1"})
        self.assertEqual(r.returncode, 1)
        logs = " ".join(e.get("message", "") for e in ev)
        self.assertIn("no llama-tokenize next to", logs)
        msg = ev[-1]["message"]
        self.assertIn("the dataset is too short", msg)
        self.assertIn("at least 513 (2 x context + 1)", msg)
        self.assertFalse((self.tmp / "full.gguf.train.txt").exists())

    def test_cancel(self):
        out = self.tmp / "full.gguf"
        p = subprocess.Popen([sys.executable, str(TRAINER), "train", "--base", str(self.base), "--data", str(self.data), "--config", str(self.config), "--out", str(out)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={**self.env, "FAKE_FINETUNE_SLEEP": "5"})
        line = p.stdout.readline()
        while '"progress"' not in line:
            line = p.stdout.readline()
            self.assertTrue(line, "no progress before the end")
        p.send_signal(signal.SIGTERM)
        t = time.monotonic()
        code = p.wait(timeout=20)
        self.assertLess(time.monotonic() - t, 10, "cancel stops llama-finetune within CANCEL_GRACE")
        self.assertEqual(code, 1)
        self.assertIn('"error"', p.stdout.read())
        p.stdout.close()
        p.stderr.close()
        self.assertFalse(out.exists())
        self.assertFalse((self.tmp / "full.gguf.train.txt").exists())

    def test_extract_lora(self):
        r, _ = self.train()
        self.assertEqual(r.returncode, 0, r.stdout)
        tuned, adapter = self.tmp / "full.gguf", self.tmp / "adapter.gguf"
        r, ev = self.run_trainer("extract-lora", "--base", str(self.base), "--tuned", str(tuned), "--rank", "4", "--out", str(adapter))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(ev[-1], {"event": "done"})
        self.assertEqual([e["step"] for e in ev if e["event"] == "progress"], [1, 2, 3, 4])
        self.assertIn("3 of 4 matrices; it keeps 100.0%", " ".join(e.get("message", "") for e in ev))
        a = GGUFReader(adapter)
        self.assertEqual(trainer.field_str(a, "general.type"), "adapter")
        self.assertEqual(trainer.field_str(a, "adapter.type"), "lora")
        self.assertEqual(trainer.field_str(a, "general.architecture"), "llama")
        self.assertEqual(float(a.fields["adapter.lora.alpha"].parts[-1][0]), 4.0)
        names = sorted(t.name for t in a.tensors)
        self.assertEqual(names, sorted(f"blk.0.{n}.weight.lora_{x}" for n in ("attn_q", "ffn_up", "ffn_down") for x in "ab"))
        by = {t.name: np.array(t.data) for t in a.tensors}
        base = {t.name: np.array(t.data) for t in GGUFReader(self.base).tensors}
        tun = {t.name: np.array(t.data) for t in GGUFReader(tuned).tensors}
        for n in ("attn_q", "ffn_up", "ffn_down"):
            k = f"blk.0.{n}.weight"
            la, lb = by[k + ".lora_a"], by[k + ".lora_b"]
            out_dim, in_dim = base[k].shape
            # llama.cpp's layout (convert_lora_to_gguf.py): lora_a is [rank, in], lora_b [out, rank].
            self.assertEqual(la.shape, (4, in_dim))
            self.assertEqual(lb.shape, (out_dim, 4))
            np.testing.assert_allclose(base[k] + lb @ la, tun[k], atol=1e-5)

    def test_extract_truncates_and_pads(self):
        self.train()
        tuned, adapter = self.tmp / "full.gguf", self.tmp / "adapter.gguf"
        r, ev = self.run_trainer("extract-lora", "--base", str(self.base), "--tuned", str(tuned), "--rank", "1", "--out", str(adapter))
        self.assertEqual(r.returncode, 0, r.stdout)
        kept = [e["message"] for e in ev if e["event"] == "log"][-1]
        self.assertNotIn("100.0%", kept, "rank 1 can't keep a rank-2 change")
        r, ev = self.run_trainer("extract-lora", "--base", str(self.base), "--tuned", str(tuned), "--rank", "8", "--out", str(adapter))
        self.assertEqual(r.returncode, 0, r.stdout)
        t = {x.name: np.array(x.data) for x in GGUFReader(adapter).tensors}
        self.assertEqual(t["blk.0.attn_q.weight.lora_a"].shape, (8, 6), "a matrix smaller than the rank is padded, so alpha / rank holds")

    def test_extract_refusals(self):
        r, ev = self.run_trainer("extract-lora", "--base", str(self.base), "--tuned", str(self.base), "--rank", "4", "--out", str(self.tmp / "a.gguf"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("nothing to extract", ev[-1]["message"])
        other = self.tmp / "other.gguf"
        write_model(other, arch="qwen2")
        r, ev = self.run_trainer("extract-lora", "--base", str(self.base), "--tuned", str(other), "--rank", "4", "--out", str(self.tmp / "a.gguf"))
        self.assertIn("is not a fine-tune of", ev[-1]["message"])


if __name__ == "__main__":
    unittest.main()
