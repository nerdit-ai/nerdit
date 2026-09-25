# Nerdit

**A self-hosted PaaS for your own machine, built so an AI agent can run it.**

Point Nerdit at a folder or a Git repository: it builds the app, runs it as a
supervised service, gives it a URL, and wires it to a database or a local
model when it needs one. You drive it from the `nerdit` CLI, the local
dashboard, or an AI agent through its MCP server. The agent gets the same
operations as the CLI, behind scoped tokens and an audit log.

```bash
nerdit deploy ./my-app --wait       # build, run, wait until healthy
nerdit serve llama3.1:8b --gpus 1   # local model behind an OpenAI-compatible API
```

Your apps, data and models stay on your hardware. The optional
[Nerdit Cloud](#nerdit-cloud) account adds remote access and hosted URLs, but
it doesn't run your apps. An unlinked machine keeps working on its own.

**Free during the public beta**, with up to five linked machines per account.

## How it works

One daemon, `nerditd`, runs on your machine next to Docker:

```
  nerdit CLI ───┐                 ┌─ apps (Docker, health-checked, restarted)
  dashboard ────┼─ REST ► nerditd ┼─ models (Ollama / vLLM, or an external API)
  AI agent (MCP)┘                 ├─ databases (Postgres, Redis)
                                  └─ Caddy: URLs + TLS
```

An app says what it needs in a `nerdit.toml` next to its code:

```toml
[deploy]
name = "my-app"
port = 8000

[ai.default]
provider = "ollama"      # or "api" for an external endpoint
model = "llama3.1:8b"
```

The app starts with `OPENAI_BASE_URL` and `OPENAI_API_KEY` set. To move from a
local model to a hosted API, you change the config. The app code stays the
same.

## Install

```bash
curl -fsSL https://get.nerdit.ai | sh
nerdit doctor
```

This installs one signed archive: the CLI, the daemon and a pinned Caddy. You
don't need Python. It also registers a service and starts the daemon. You need
Docker installed and running. The dashboard is at `http://127.0.0.1:9321/`.

For your first deployment, see the
[quickstart](https://docs.nerdit.ai/engine/quickstart).
[Other install methods](#other-install-methods) are listed below.

## Use it with an AI agent

Nerdit ships an MCP server with 59 tools. Your agent can deploy, wire and
diagnose apps without a shell.

### Claude Code, Cursor and other local clients (stdio)

On the machine where `nerdit` is installed:

```bash
claude mcp add nerdit -- nerdit mcp
```

Other clients take the equivalent entry:

```json
{
  "mcpServers": {
    "nerdit": { "command": "nerdit", "args": ["mcp"] }
  }
}
```

If your client doesn't inherit your shell's `PATH`, use an absolute path to
`nerdit`.

The server acts with the credentials of your `nerdit` CLI. Give an agent a
narrow token rather than your admin one:

```bash
nerdit token create --role submitter --scope my-app --expires-in 24h
```

The `readonly` role can list and read but not write. A token scoped to a
service can only change that service. See
[Security](https://docs.nerdit.ai/engine/security) for how the CLI selects a
token.

### Claude on the web or desktop (remote, through Nerdit Cloud)

Link the machine with `nerdit link`. Then open **Connect** in
[app.nerdit.ai](https://app.nerdit.ai) and add the connection URL it shows as a
custom connector in Claude. You approve which machines or projects the
assistant can reach. No token leaves your machine. See the
[guide](https://docs.nerdit.ai/app/connect-assistant).

### Over HTTP

Set `[mcp].http_enabled = true` to serve the same tools at
`http://127.0.0.1:9321/api/mcp`, behind a bearer token. See
[MCP setup](https://docs.nerdit.ai/engine/mcp).

### What to ask

> Deploy the app in `./todo-api`, give it a Postgres database, and tell me the
> URL once it's healthy.

> `blog` is crash-looping. Find out why and fix it.

> Serve `qwen2.5:7b` and point `my-chatbot` at it instead of the OpenAI API.

A typical run: `capabilities` → `deploy` (with `dry_run` first) →
`wait_for_service` → `diagnose_service` and `service_logs` if something fails.
An agent that has no files on the machine can use `write_app_files`, then
`deploy_app`. It can also use `deploy_git`.

The API is built for agents to retry safely:

- Every write is **idempotent**, so a retry never runs an action twice.
- Every write is **audited**.
- Errors carry a stable `code` and a recovery `hint`.
- Reads are **bounded**, so no single call floods the context window.
- Secrets are **write-only**: an agent can set them but never read them back.

## What you get

- **Deploy** from a folder, a Git URL or a template, using a Dockerfile or the
  Python or Node buildpacks. Images are versioned, and deploys support
  `--dry-run`, `--wait` and `--rollback`.
- **Services** restart with backoff and have HTTP and TCP health checks and
  stable ports. A redeploy cuts over with no downtime.
- **URLs and TLS** come from an embedded Caddy. It routes by path or
  subdomain, serves your own domains, advertises over mDNS on the LAN, and uses
  an internal CA that you can pin.
- **Models** run through Ollama or vLLM on your GPU, or through an external
  API. Apps get the same environment variables either way.
- **Databases**: managed Postgres and Redis. Nerdit creates the credentials
  and binds them into apps through `[db.*]`.
- **Projects** group several services in one `nerdit.toml` (`nerdit apply`).
  Variables are plain or secret. Secrets are encrypted at rest and can be
  rotated.
- **Backups** cover the control plane, with an offline restore. A managed
  database also gets a logical dump (`nerdit db dump`) that you can restore
  while it runs.

## Other install methods

**pip.** Needs Python 3.11+ and Docker, on Linux, macOS, or Windows through
WSL2:

```bash
python3 -m venv ~/.venvs/nerdit && source ~/.venvs/nerdit/bin/activate
python -m pip install nerdit
nerdit init && nerdit doctor
```

The CLI, the daemon, the dashboard, MCP and mDNS are all included. The pip
package doesn't bundle Caddy or register a startup service. Install Caddy
yourself if you want HTTPS, and run `nerdit init` again after a reboot. See
[installation](https://docs.nerdit.ai/engine/installation#install-with-pip).

**From source.**

```bash
git clone https://github.com/nerdit-ai/nerdit && cd nerdit
python -m venv venv && venv/bin/pip install -e ".[dev,mcp]"
(cd src/nerdit/daemon/web && npm ci && npm run build)   # dashboard, required
venv/bin/nerdit init
```

## Platforms

- **Linux**: Ubuntu 22.04 or 24.04, with systemd.
- **macOS**: Apple Silicon. Models run on the CPU.
- **Windows**: through WSL2 only.
- **GPUs**: NVIDIA needs Linux or WSL2. AMD support is experimental and behind a
  flag.

## No telemetry

The daemon collects no product telemetry. Linking to Nerdit Cloud, public relay
traffic and external AI APIs send only the data their feature needs. The
optional `[posthog]` dashboard analytics hook is off by default and reports to
your own PostHog project.

## Nerdit Cloud

[app.nerdit.ai](https://app.nerdit.ai) adds:

- remote access and assistant connections
- deploys from GitHub
- hosted URLs

Your machine connects out to the cloud, so you open no inbound port. Hosted
shares are private to you by default. Public access needs an explicit
publication and follows the [Terms](https://nerdit.ai/terms).

Questions or feedback: [feedback@nerdit.ai](mailto:feedback@nerdit.ai).

## License

Apache-2.0, see [LICENSE](LICENSE). "Nerdit" and the logo are trademarks and
are not covered by the code license, see [NOTICE](NOTICE).

## Contributing

Issues and pull requests are welcome. [CONTRIBUTING.md](CONTRIBUTING.md)
explains how contributions land in releases.
