# Nerdit

Nerdit runs your apps on your own machine the way a PaaS would: build from a
folder, run as a supervised service, get an HTTPS URL, restart on crash, roll
back on a bad deploy. When an app needs an AI model, Nerdit serves one locally
on your GPU or proxies an external API, and the app sees the same OpenAI-style
endpoint either way.

Your code and your data stay on your hardware.

```bash
nerdit serve llama3.1:8b --gpus 1   # local model, OpenAI-compatible endpoint
nerdit deploy ./my-app              # build, run, HTTPS URL, model wired in
```

## How an app declares what it needs

`nerdit.toml` in the app folder:

```toml
[deploy]
name = "my-app"
port = 8000

[ai.default]
provider = "ollama"      # or "api" for an external endpoint
model = "llama3.1:8b"
```

The app starts with `OPENAI_BASE_URL` and `OPENAI_API_KEY` set, is reachable at
`https://<host>/my-app`, and comes back after a reboot. Switching the model
from local to an external API is a config change; the app code does not move.

## Install

```bash
curl -fsSL https://get.nerdit.ai | sh
```

One signed archive with the CLI, the daemon and a pinned Caddy build. No
Python needed. It installs a service unit and starts the daemon.

User guides and API documentation are maintained separately for the
[Nerdit website](https://nerdit.ai/). This repository contains the source,
tests and contributor instructions.

From source:

```bash
git clone https://github.com/nerdit-ai/nerdit && cd nerdit
python -m venv venv && venv/bin/pip install -e ".[dev,mcp]"
(cd src/nerdit/daemon/web && npm ci && npm run build)   # dashboard, required
venv/bin/nerdit init
```

## What you get

- **Deploy** from a folder, a Git URL or a template. Dockerfile, Python and
  Node buildpacks. Versioned images, `--rollback`, `--dry-run`, `--wait`.
- **Services** with restart policy, backoff, HTTP and TCP health checks, stable
  ports, and zero-downtime cutover on redeploy.
- **URLs and TLS** from an embedded Caddy: path or subdomain routing, your own
  domains, mDNS on the LAN, an internal CA you can pin.
- **Models** via Ollama or vLLM on your GPU, or an external API. Same injected
  env vars in both cases.
- **Databases**: managed Postgres and Redis, credentials minted for you and
  bound into apps through `[db.*]`.
- **Secrets**: write-only, encrypted at rest, rotatable, with a shared scope.
- **Agents**: an MCP server with 49 tools, scoped tokens, idempotent writes,
  structured errors and an audit log. A coding agent can deploy, wire and
  diagnose apps without shell access.
- **Backups** of the control plane, with an offline restore — plus, for a
  managed database, a file-level volume tar and a logical,
  application-consistent dump (`nerdit db dump`) you can restore live.

## No telemetry

The daemon sends nothing anywhere. The one analytics hook, `[posthog]`, is off
by default and points at your own PostHog project if you turn it on.

## Platforms

Linux (Ubuntu 22.04 or 24.04, systemd), macOS on Apple Silicon, Windows through
WSL2. NVIDIA GPUs need Linux or WSL2; macOS runs models on CPU. AMD support is
experimental and behind a flag.

## Nerdit Cloud

Everything above is free and local. [app.nerdit.ai](https://app.nerdit.ai) is a
separate paid service that adds remote access to your node and hosted URLs for
apps you choose to share. It gates nothing local: a daemon that never links
keeps deploying, proxying and serving models.

## License

Apache-2.0, see [LICENSE](LICENSE). "Nerdit" and the logo are trademarks and
are not covered by the code license, see [NOTICE](NOTICE).

## Contributing

Issues and pull requests are welcome. [CONTRIBUTING.md](CONTRIBUTING.md)
explains how contributions land in releases.
