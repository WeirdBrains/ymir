# Sharing a pool with another Weirdnet identity

Status: built 2026-11-27 on `weirdnet-agent` 0.5.9 (mocks and in-process transport only; not yet over real networks or
relays). Michael (11-27, t66u) asked for it after Morph's m3talsmith profile (identity `5e452f1b…`) could not reach the Ymir
pool (identity `08e2b657…`): until now Ymir admitted only devices on its own identity. Read DESIGN.md, "Security and auth",
first; this document only adds guests.

## Decisions (2026-11-27)

- **Grants, not a second roster.** The pool owner records a *guest grant* for another identity's user id. The guest's devices
  keep their own identity and present their own `NodeCert`; nothing about the guest is copied into the owner's roster, and the
  guest never gets a seed, an epoch key or a sync grant.
- **Catalog and inference only.** A guest gets `/weirdnet-agent/catalog/1` and `/weirdnet-agent/infer/2` (and infer/1): chat,
  embeddings, transcription. Never `Input::Context` (the owner's `ymir.context` conversations), never wake cards, `task/1`,
  admin or control, `models`, uploads, rollouts or training. Those protocols keep `RosterAuth` and are never handed the guest
  check.
- **The synced grant is the only authority.** The share link carries dial information only (no secrets); anyone can hold one.
  Who gets in is decided per request by the grant list every inference node syncs.
- **Revocation is immediate.** Every infer/2 request and catalog query asks the policy again; live catalog subscriptions are
  re-checked at least every `catalog::RECHECK_EVERY` (5 s). Responses already streaming finish.

## How it fits together

```
owner device:  ymirctl pool share <guest user id>  →  ymir.pool record guest/<user id>   (synced to every node)
                                                    →  prints wnet-pool1:… / weirdnet-pool://share/…
guest device:  PoolShare::parse(link) → connect_shared(node, &share, &user_key, &cert)
               presents {"guest":1, key, cert} on catalog/1 and infer/2
inference node: SharedPoolAuth = RosterAuth (own devices) + GuestAuth (crypto checks) + guests::policy (live grants)
```

### Guest grants (`crates/ymir-core/src/guests.rs`)

A grant is a JSON record in the synced `ymir.pool` type with record id `guest/<user_id>` (unknown fields ignored, so newer
grants stay readable):

| field | meaning |
|---|---|
| `user_id` | the guest identity (32 hex characters, `weirdnet_net::user::user_id` of its key) |
| `label` | the owner's name for them; shown only, never checked |
| `peers` | optional pinned devices (peer ids); empty means any device the guest's user certified |
| `models` | optional allowlist, matched against the requested name or the routed model; empty means every model |
| `scopes` | `catalog` and/or `infer`; empty means both |
| `expires_at` | optional Unix ms after which the grant admits no one |
| `granted_at`, `revoked_at` | when it was given, and when `unshare` revoked it |

Records are overwritten, never deleted: `unshare` writes the grant back with `revoked_at` so the revocation syncs like any
other record. Grants hold no secrets, so the type stays unsealed; only the owner's roster devices sync it. A grant for the
pool's own identity is refused (link the device instead). Cert-only nodes read grants but `pool share` runs on a device that
holds the identity.

### What a node checks

1. `GuestAuth::verify` (in `weirdnet-agent`): the presented key hashes to the cert's `user_id`; it is not the owner; the cert
   is for the Noise-authenticated peer; the cert and any revocation verify; it is not revoked; `issued_at` is not more than
   10 minutes in the future.
2. `guests::policy`: reads the grant for that user id live and refuses `not-shared`, `unshared`, `share-expired`,
   `guest-peer-not-pinned` or `guest-scope`.
3. `InferenceHost` (`host.rs`), for a caller `identify` reports as a guest: `Input::Context` is refused (`guest-context`), and
   a model outside the allowlist is refused (`guest-model`) for chat, infer/1 turns and embeddings.

`remote::guests(m)` builds the `SharedPoolAuth`; `remote::serve` uses it for infer and (through `scoped(Catalog)`) for the
rechecking `CatalogFeed`. `roster_auth` and `role_auth`, which every other protocol uses, are unchanged and refuse guests.

The catalog a guest sees is the whole pool's; the model allowlist is enforced at request time. Peer pinning also covers the
one thing the owner cannot see: if the guest's own user revokes one of its devices, the owner's nodes only learn that once the
revocation reaches them, so pin specific devices for anything sensitive.

### Share links

`weirdnet-agent`'s `PoolShare` (`remote::share`): `{v, owner, name?, nodes: [{peer_id, addrs}], expires_at?}` as
`wnet-pool1:<base64url JSON>` or `weirdnet-pool://share/<same>` (16 KiB / 64 nodes cap). `ymirctl pool share` fills `nodes`
with this node (when it has the inference role), the cert-only inference nodes in the roster and any `--node`, each with the
addresses known for it (`pool addresses`). The link's `expires_at` follows the grant's; it is a hint for the guest's app only.

### Commands

- `ymirctl pool share USER_ID [--label L] [--peer PEER]… [--models a,b] [--scopes catalog,infer] [--expires 7d] [--name N]
  [--node PEER_OR_ADDR]…`: records the grant (through the running `ymird` when it is on Weirdnet, so it lands in the live
  roster state and syncs) and prints the token and link.
- `ymirctl pool unshare USER_ID`: revokes it.
- `ymirctl pool guests [--all]`: lists active grants (or revoked and expired ones too).

### Clients

- **Morph** (another identity on the same person): paste the link; Morph calls `connect_shared` with its profile's user key
  and its device cert. No bookmarks or history move; the Ymir pool simply answers it.
- **Bragi** on another identity: the same, then the usual catalog and chat screens. Pool admin screens (models, uploads,
  training) stay unavailable to a guest by design.
- **Anyone else** with a Weirdnet identity: send them the link and grant their user id; `unshare` cuts them off on every node.

## Tests

- Unit (`guests.rs`): serialization and unknown fields, record parsing, scope/expiry/revocation/peer-pin matching, model
  allowlist, validation, `PoolShare` generation, expiry parsing.
- Integration ([`crates/ymir-core/tests/pool_sharing.rs`](../../crates/ymir-core/tests/pool_sharing.rs)): an owner inference node
  and a device of a second identity over `connect_shared`; refused before the grant; catalog and chat after; `Input::Context`
  refused; `roster_auth`/`role_auth` (models, train, tasks, wake, admin) refuse the guest credential; model allowlist; a second
  device of the guest identity refused when the grant pins the first; `unshare` refuses the next chat and closes a live
  catalog subscription within `RECHECK_EVERY`; an expired grant admits no one.

## Not done yet

- A real-network test (two identities on different networks, through a relay).
- Per-guest rate limits and a per-guest catalog filtered to the allowlist.
- A Bragi and Morph "open share link" screen (their side).
