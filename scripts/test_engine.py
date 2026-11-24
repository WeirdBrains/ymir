"""Tests for scripts/engine.py's trainer build: which steps `build --trainer` runs, and that Ymir's
llama-finetune patches apply to the pinned llama.cpp. Standard library only; nothing is compiled.
The apply test reads the pinned build's examples/training/finetune.cpp from GitHub and skips when it
can't (or with YMIR_TEST_OFFLINE=1):

    python3 scripts/test_engine.py
"""

import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from contextlib import redirect_stderr
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engine  # noqa: E402

FINETUNE = "examples/training/finetune.cpp"


def args(trainer):
    return argparse.Namespace(trainer=trainer, backend=["vulkan"], jobs=2, no_cpu_variants=True, variant="default")


class TrainerBuildTest(unittest.TestCase):
    def test_trainer_build_applies_the_patches_first(self):
        steps = engine.build_plan(args(True), Path("/src"), Path("/build"))
        patches = [s for k, s in steps if k == "patch"]
        self.assertEqual(patches, engine.trainer_patches())
        self.assertTrue(patches and all(p.exists() for p in patches))
        self.assertEqual([k for k, _ in steps], ["patch"] * len(patches) + ["run", "run"], "patches go in before cmake runs")
        configure, build = (s for k, s in steps if k == "run")
        self.assertIn("-DLLAMA_BUILD_EXAMPLES=ON", configure)
        self.assertIn("llama-finetune", build)

    def test_serving_build_is_unpatched(self):
        steps = engine.build_plan(args(False), Path("/src"), Path("/build"))
        self.assertEqual([k for k, _ in steps], ["run", "run"])
        self.assertNotIn("llama-finetune", steps[-1][1])

    def test_patches_touch_only_llama_finetune(self):
        for p in engine.trainer_patches():
            files = set(re.findall(r"^(?:---|\+\+\+) [ab]/(\S+)", p.read_text(), re.M))
            self.assertEqual(files, {FINETUNE}, f"{p.name} must leave llama-server and rpc-server as upstream built them")

    def test_engine_info_names_the_patches(self):
        out = Path(tempfile.mkdtemp(prefix="ymir-engine-test-"))
        self.addCleanup(shutil.rmtree, out, True)
        engine.write_info(out, "b1", "default", ["cpu"], "src", list(engine.TRAINER_PATCHES))
        self.assertIn('trainer_patches = ["parse-special"]', (out / "engine.toml").read_text())
        engine.write_info(out, "b1", "default", ["cpu"], "src")
        self.assertNotIn("trainer_patches", (out / "engine.toml").read_text())

    def test_patches_apply_at_the_pin(self):
        if os.environ.get("YMIR_TEST_OFFLINE") == "1" or not shutil.which("git"):
            self.skipTest("offline, or no git")
        build = engine.load_pin()["build"]
        url = f"https://raw.githubusercontent.com/{engine.UPSTREAM}/{build}/{FINETUNE}"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                upstream = r.read()
        except OSError as e:
            self.skipTest(f"can't read {url}: {e}")
        src = Path(tempfile.mkdtemp(prefix="ymir-engine-src-test-"))
        self.addCleanup(shutil.rmtree, src, True)
        subprocess.run(["git", "init", "-q", str(src)], check=True)
        (src / FINETUNE).parent.mkdir(parents=True)
        (src / FINETUNE).write_bytes(upstream)
        self.assertIn(b"common_tokenize(ctx, params.prompt, true);", upstream, f"upstream {build} still tokenizes specials as text")
        for p in engine.trainer_patches():
            with redirect_stderr(io.StringIO()) as err:
                engine.apply_patch(src, p)
            self.assertIn("applied", err.getvalue())
        patched = (src / FINETUNE).read_text()
        self.assertIn("common_tokenize(ctx, params.prompt, true, /*parse_special =*/ true);", patched)
        self.assertIn('"--ymir-patches"', patched)
        with redirect_stderr(io.StringIO()) as err:
            engine.apply_patch(src, engine.trainer_patches()[0])
        self.assertIn("already applied", err.getvalue(), "a reused --work tree is left as it is")


if __name__ == "__main__":
    unittest.main()
