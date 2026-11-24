"""A stand-in for llama.cpp's `llama-finetune` (b12628) for scripts/test_trainer.py: the same
arguments, version line and progress bars, no training. It writes the base back with a rank-2
change to every blk.* matrix, so the extracted adapter can be checked exactly.

FAKE_FINETUNE_BUILD (default 12628), FAKE_FINETUNE_RECORD (write the arguments and the training
text there as JSON), FAKE_FINETUNE_FAIL (exit 3 after the first batch), FAKE_FINETUNE_SLEEP
(seconds per batch)."""

import json
import os
import sys
import time

import numpy as np
import gguf
from gguf import GGUFReader

args = sys.argv[1:]
if args == ["--version"]:
    print(f"version: {os.environ.get('FAKE_FINETUNE_BUILD', '12628')} (deadbee)", file=sys.stderr)
    print("built with fake for test", file=sys.stderr)
    sys.exit(0)


def opt(*names):
    for n in names:
        if n in args:
            return args[args.index(n) + 1]
    return None


model, text_file, out = opt("-m", "--model"), opt("-f", "--file"), opt("-o", "--output")
epochs = int(opt("--epochs", "-epochs") or 2)
text = open(text_file, encoding="utf-8").read()
if os.environ.get("FAKE_FINETUNE_RECORD"):
    with open(os.environ["FAKE_FINETUNE_RECORD"], "w") as f:
        json.dump({"args": args, "text": text, "text_mode": oct(os.stat(text_file).st_mode & 0o777)}, f)
print("llama_model_loader: loaded meta data", file=sys.stderr)
batches = 4
for e in range(epochs):
    for i in range(1, batches + 1):
        sys.stderr.write(f"train: [\u2588\u2588   ] data={i:07d}/{batches:07d} loss={2.0 / (e * batches + i):.5f}\u00b10.01000 acc=10.00\u00b11.00% t=00:00:01 ETA=00:00:01 \r")
        sys.stderr.flush()
        if os.environ.get("FAKE_FINETUNE_FAIL") and i == 1:
            sys.stderr.write("\nGGML_ASSERT failed: out of memory\n")
            sys.exit(3)
        time.sleep(float(os.environ.get("FAKE_FINETUNE_SLEEP", "0")))
    sys.stderr.write("\n")
    sys.stderr.write(f"val:   [\u2588\u2588\u2588\u2588] data=0000001/0000001 loss=1.50000\u00b10.00000 acc=0.00\u00b10.00% t=00:00:00 ETA=00:00:00 \r\n")

r = GGUFReader(model)
arch = bytes(r.fields["general.architecture"].parts[r.fields["general.architecture"].data[0]]).decode()
w = gguf.GGUFWriter(out, arch)
rng = np.random.default_rng(7)
for t in r.tensors:
    a = np.array(t.data, dtype=np.float32)
    if a.ndim == 2 and t.name.startswith("blk.") and not t.name.endswith("attn_k.weight"):
        a = a + (rng.standard_normal((a.shape[0], 2)) @ rng.standard_normal((2, a.shape[1]))).astype(np.float32) * 0.01
    w.add_tensor(t.name, a)
for k in ("general.name",):
    if k in r.fields:
        w.add_name(bytes(r.fields[k].parts[r.fields[k].data[0]]).decode())
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
