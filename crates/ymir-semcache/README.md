# ymir-semcache

**To be extracted into `weirdnet-semcache`.** In M1 this is mostly a trait seam:

- `AnswerCache`: what the inference node asks (`get`, `put`) before and after routing a prompt.
- `ExactCache`: an embedded, bounded, in-memory cache that only answers byte-identical
  deterministic requests (temperature 0, no tools) for the same model. It is a placeholder.

Not here yet: semantic matching with an embedding model (the design's threshold 0.12), the rules
from server-a/server-b's `rvl-app.py`, the storage trait, the optional Redis backend and pool-mode sharding.

License: BSD-3-Clause.
