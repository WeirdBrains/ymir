# Ymir

**A cloud of LLM models: a private, self-hosted model cloud.** Start with one machine. Add
more over Weirdnet. The pool acts as one endpoint, and the models are spread across it.

> Status: design only (2026-11-19). Nothing here runs yet. Read [docs/DESIGN.md](docs/DESIGN.md).

Ymir turns a computer with a decent GPU (a DGX Spark, a gaming PC, a Mac, a workstation, or a
CPU-only box) into part of your own model cloud with one command. Machines join over
[Weirdnet](https://net.weirdbrains.com) peer to peer by being linked to your Weirdnet identity:
no open HTTPS port, no reverse proxy, no API keys. Only your own linked devices can use the pool.

Three roles, placed freely (all on one box, or spread over any number of devices):

- **Model nodes** run the models (llama.cpp everywhere, vLLM where supported, or a stock server
  wrapped), advertise what they have, and load or unload on command.
- **Inference nodes** coordinate: one model catalog, load balancing, placement and cloud-wide model
  swapping, automatic mid-answer failover, and conversation context synced with `weirdnet-sync`.
  Run more than one for redundancy.
- **App nodes** ask for models and send prompts: Bragi, `ymirctl`, the loopback-only OpenAI bridge
  for Jan and scripts on `127.0.0.1`, or any `weirdnet-agent` app.

Redis, Tailscale and a second machine are all optional. Models too big for one machine can be split
across several (big mode, generalized).

The chat client, **Bragi**, has its own repo. Its desktop (GPUI) and phone (Flutter) apps work exactly
the same way: both get their models from the Ymir cloud and send prompts across it, through one
shared Rust core. Only the UI shell differs.

Generic pieces live in `weirdnet-*` crates (owned by the Weirdnet lead); this repo holds only the
Ymir-specific app.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
