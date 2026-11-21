# Handoff

A running log of finished work, newest first. Each entry says what changed, the commits, the checks run and what was not tested, what is left, and what is next.

## 2026-11-21: A new node joined the pool

**What changed**

- Installed Ymir (built-in backends binary, user-level service) on a new GPU-equipped Linux machine with the [documented setup](docs/DESIGN.md): `ymirctl setup --roles model --download --service`. It serves a 14B chat model, a small stand-in model and an embedding model.
- Paired it with the existing Weirdnet using a matched link code (`ymirctl pool link` on the new node, `ymirctl pool add` on an existing one). The first pairing did not route: the inference node only dials the peers in its `[network] peers` list and the mDNS/relay paths, and it never learns the address of a node paired after it started. The node was revoked, its pool state reset, and it was paired again; routing worked once the new node was given the inference node's address in its own `[network] peers`, so it dials in.

**Checks run:** the pool's node list shows the new node serving; the bridge `/v1/models` lists its models; a small chat completion for the 14B model was served by the new node (about 40 tokens/s, so on the GPU or a fast path) and the stand-in answered while the model loaded.

**Not tested:** the machine's NVIDIA driver and library versions mismatch until it reboots, so GPU use is inferred from speed only.

**What's left:** pairing does not share the new node's address with the other nodes, a running node does not hot-reload the roster, and nodes do not discover each other on their own. All three are future work; until then a new node must dial an existing one through `[network] peers`.

**What's next:** restrict `[model] callers` to the inference nodes, and fix the three gaps above.

## 2026-11-20: Site and docs cleanup (status copy and clickable references)

**What changed**

- Site copy and READMEs state plainly what Ymir does. Hedging lines ("so far tested on one computer", "not yet tested", "nothing is released yet" on the home page, and similar) are gone from the [home page](https://ymir.weirdbrains.com/), [quickstart](https://ymir.weirdbrains.com/quickstart), [hardware](https://ymir.weirdbrains.com/hardware), the [deployment guide](https://ymir.weirdbrains.com/docs/guides/deployment/) and the [README](README.md).
- [roadmap.toml](roadmap.toml) no longer carries per-milestone gap lists, and the [roadmap page](https://ymir.weirdbrains.com/roadmap) never showed them. Statuses: M8 (move the existing servers onto Ymir) planned → done. M6, M7 and M9 stay in progress, M10 stays planned; their summaries now say what is built and what is next, and M9 now lists the Windows service as built. [ROADMAP.md](ROADMAP.md) and the site data are regenerated. The M8 line in [docs/DESIGN.md](docs/DESIGN.md) says done.
- Platform claims: the hardware page says Ymir builds for Linux, macOS and Windows (BSD packages planned); the deployment guide describes Bragi for desktops and phones on all five platforms and links Bragi 0.1.0.
- Every reference to another document is a link, for example "Read [DESIGN.md](docs/DESIGN.md) for the details" in the [developer guide](https://ymir.weirdbrains.com/docs/develop/).
- New CI check [site/tool/check_links.py](site/tool/check_links.py), run by the [Pages workflow](.github/workflows/pages.yml): it fails on a `.md` file name or a bare URL that is not a link in the page sources, the README, [ROADMAP.md](ROADMAP.md) and [HANDOFF.md](HANDOFF.md).

**Commits:** `028298c` on main.

**Checks run:** `site/tool/gen_roadmap.py --check`, `site/tool/check_links.py`, the live pages fetched and searched for the removed phrases. The Pages workflow builds the site and runs both checks on the push.

**Not tested:** no local `dart analyze` or site build (the local Dart tooling could not resolve the site's packages); CI is the check for those. [docs/DESIGN.md](docs/DESIGN.md) was not swept for caveat wording beyond the M8 line.

**What's left:** the status sections in [DESIGN.md](docs/DESIGN.md) still use "partly" and "with gaps" in a few headings.

**What's next:** M6, M7 and M9 work, then wiring up M10 once its Weirdnet crates are verified.
