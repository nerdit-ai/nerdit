# Nerdit

Nerdit helps independent builders and small teams deploy and update the tools
they create with AI, on their own infrastructure. Build from a folder or Git
repository, run the app as a supervised service, open its URL and connect it
to persistent databases or local models when needed.

**All current features are free during the public beta**, with up to **five
linked machines per account**. You provide the machines and pay for any
external services you choose. Apps depend on their host staying available.

```bash
nerdit serve llama3.1:8b --gpus 1   # local model, OpenAI-compatible endpoint
nerdit deploy ./my-app --wait       # build, run and wait for health
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

The app starts with `OPENAI_BASE_URL` and `OPENAI_API_KEY` set. Enable the
[HTTPS proxy](https://docs.nerdit.ai/engine/networking) to serve it at
`https://<host>/my-app`; otherwise use its local loopback endpoint. Switching the model
from local to an external API is a config change; the app code does not move.

## Install

```bash
curl -fsSL https://get.nerdit.ai | sh
```

One signed archive with the CLI, the daemon and a pinned Caddy build. No
Python needed. It installs a service unit and starts the daemon.

Follow the [getting-started guide](https://docs.nerdit.ai/engine/quickstart)
for prerequisites, deployment and opening your app. This repository contains
the source, tests and contributor instructions.

### With pip

Prefer a Python package to the shell installer? With **Python 3.11+** and
Docker installed and running on Linux, macOS, or Windows through WSL2,
create a dedicated environment. Native Windows is not supported:

```bash
python3 -m venv ~/.venvs/nerdit
source ~/.venvs/nerdit/bin/activate
python -m pip install nerdit
nerdit init
nerdit doctor
```

The PyPI package includes the CLI, daemon and built dashboard. Open
`http://127.0.0.1:9321/` after initialization. Add MCP support with
`python -m pip install "nerdit[mcp]"` in the same environment. For LAN mDNS,
version 0.6.4 includes support in the base package. With 0.6.3, use
`python -m pip install "nerdit[mdns]"`, or `"nerdit[mcp,mdns]"` for both.

Unlike the signed installer, pip does not bundle Caddy or register a service
for startup after reboot. `nerdit init` starts a background daemon; run it
again after a reboot, or manage `nerditd` with your service manager. The
0.6.4 release enables HTTPS on port 8443 and mDNS in new configurations;
install Caddy separately for HTTPS with pip. Existing configurations are preserved.
On 0.6.3 or an existing configuration with the proxy disabled, run
`nerdit config set proxy enabled=true https_port=8443` and
`nerdit daemon restart` after installing Caddy.
See [installation and upgrades](https://docs.nerdit.ai/engine/installation#install-with-pip)
for lifecycle, networking and update details.

### From source

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
- **Projects and variables**: several services in one `nerdit.toml`
  (`nerdit apply`), with plain or secret variables at project or service
  scope. Secrets are write-only; every value is encrypted at rest and
  rotatable, with a machine-wide shared scope.
- **Agents**: an MCP server with 57 tools, scoped tokens, idempotent writes,
  structured errors and an audit log. A coding agent can deploy, wire and
  diagnose apps without shell access.
- **Backups** of the control plane, with an offline restore — plus, for a
  managed database, a file-level volume tar and a logical,
  application-consistent dump (`nerdit db dump`) you can restore live.

## No telemetry

The daemon has no product telemetry. Optional cloud links, public relay traffic
and external AI APIs send the data needed for those features. The `[posthog]`
dashboard analytics hook is off by default and uses your own project if enabled.

## Platforms

Linux (Ubuntu 22.04 or 24.04, systemd), macOS on Apple Silicon, Windows through
WSL2. NVIDIA GPUs need Linux or WSL2; macOS runs models on CPU. AMD support is
experimental and behind a flag.

## Nerdit Cloud

[app.nerdit.ai](https://app.nerdit.ai) adds remote access, GitHub deployment
and hosted URLs through an outbound connection from your machine. These
features are included in the free public beta, for active accounts with up
to five linked machines; no subscription is required.

Hosted shares are private to the signed-in owner by default. Public access
requires explicit publication and is subject to the
[Terms](https://nerdit.ai/terms) and abuse enforcement. An unlinked daemon
keeps deploying, proxying and serving models locally.

Future fleet/cluster management and a separate Managed infrastructure offer
are not available as part of this delivery.

Questions or feedback: [feedback@nerdit.ai](mailto:feedback@nerdit.ai).

## License

Apache-2.0, see [LICENSE](LICENSE). "Nerdit" and the logo are trademarks and
are not covered by the code license, see [NOTICE](NOTICE).

## Contributing

Issues and pull requests are welcome. [CONTRIBUTING.md](CONTRIBUTING.md)
explains how contributions land in releases.
