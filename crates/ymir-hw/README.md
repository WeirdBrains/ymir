# ymir-hw

**To be extracted into `weirdnet-model` (a `hw` module plus the tiered catalog and sizing
function).** This crate is generic: nothing in it knows about Ymir. It lives here only until the
Weirdnet lead moves it; Ymir code uses it through `ymir-hw` so the move is a rename.

- `detect()`: NVIDIA (`nvidia-smi` query; NVML through `nvml-wrapper` is the target), AMD (sysfs
  `mem_info_vram_total`), Apple (`sysctl hw.memsize`, unified), CPU (`/proc/meminfo`). GB10 / any
  NVIDIA device that reports no dedicated memory is treated as unified memory.
- `budget()`: the usable memory budget (85% of VRAM, or of unified memory after the OS reserve).
- `plan()`: picks a main chat model, a stand-in and an embedding model from a small tiered catalog
  whose entries are `weirdnet_model::ModelSpec`s, so they download with `weirdnet-model`'s consent
  gate.

License: BSD-3-Clause.
