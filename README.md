# Ymir

**A cloud of LLM models: a private, self-hosted model cloud.** Start with one machine. Add
more over Weirdnet. The pool acts as one endpoint, and the models are spread across it.

> Status: design only (2026-11-19). Nothing here runs yet. Read [docs/DESIGN.md](docs/DESIGN.md).
> "Ymir" is a working name; see the naming options in the design doc.

Ymir turns a computer with a decent GPU (a DGX Spark, a gaming PC, a Mac, a workstation, or a
CPU-only box) into an LLM serving node with one command. Nodes join each other over
[Weirdnet](https://net.weirdbrains.com) peer to peer, with no open HTTPS port, no reverse proxy
and no shared API keys. Your Weirdnet identity and device roster decide who can use the pool.
A local OpenAI-compatible bridge lets Jan, editors, scripts and agents use the whole pool as if
it were one server on `localhost`.

- Single node first: hardware detection, a backend that fits (llama.cpp everywhere, vLLM where it
  is supported) and models sized to your VRAM or unified memory.
- Grow when you want: invite a node or link one of your own devices. The pool balances requests
  by load, places and replicates models, splits batch and agent jobs, and shares a semantic cache.
- Models too big for one machine can be split across nodes (big mode, generalized).
- Redis, Caddy, Tailscale and a second node are all optional.

Generic pieces live in `weirdnet-*` crates; this repo holds only the Ymir-specific app.

## License

BSD-3-Clause (proposed; see the open questions in the design doc).
