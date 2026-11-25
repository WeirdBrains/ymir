"""A stand-in for llama.cpp's `llama-tokenize` (b12628) for scripts/test_trainer.py, as the trainer runs
it: `-m MODEL --stdin --ids --log-disable`. One token per character of standard input, or
FAKE_TOKENIZE_COUNT tokens; FAKE_TOKENIZE_RECORD (write the arguments and the input there as JSON),
FAKE_TOKENIZE_FAIL (exit 1 as when the model does not load)."""

import json
import os
import sys

args = sys.argv[1:]
text = sys.stdin.buffer.read().decode("utf-8")
if os.environ.get("FAKE_TOKENIZE_RECORD"):
    with open(os.environ["FAKE_TOKENIZE_RECORD"], "w") as f:
        json.dump({"args": args, "text": text}, f)
if os.environ.get("FAKE_TOKENIZE_FAIL"):
    print("error: unable to load model", file=sys.stderr)
    sys.exit(1)
count = int(os.environ.get("FAKE_TOKENIZE_COUNT", len(text)))
print("[" + ", ".join(str(i % 5) for i in range(count)) + "]")
