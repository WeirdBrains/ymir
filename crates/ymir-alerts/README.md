# ymir-alerts

Node health alerts for Ymir. **To be extracted into `weirdnet-alerts`** (Weirdnet lead); nothing
here is Ymir-specific.

- Observations (`engine.crash_loop`, `net.route`, `disk`, `gpu`, ...) are matched against
  [`weirdnet-rules`](https://crates.io/crates/weirdnet-rules) `RuleSpec`s whose then-actions are
  `fail`, `ok` or `event`.
- Dedupe per key: a failing key alerts at most once per repeat interval (default an hour), and a
  recovery notice goes out once when it clears.
- Sinks: SMTP (lettre, STARTTLS/TLS; the password comes from `env:` or `file:`, never from the
  config itself), JSON webhooks, push (off until `weirdnet-push` has a Weirdnet front end), the
  log, and an in-memory sink for tests. Every sink has a dry-run mode that records the message it
  would send.
- Probes: default route on Wi-Fi or missing (the server-a/server-b `route-state` rule), free disk, GPU
  temperature.
