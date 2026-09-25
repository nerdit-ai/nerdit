# Changelog

All notable changes to Nerdit are recorded here.

## Unreleased

## 0.7.0 (2026-09-24)

MCP in every installation, stable project IDs for agents, and a hardening pass
that closes token-less LAN access. **Read "Before upgrading": some changes can
break existing setups.**

### Agents and MCP

- **MCP is included in every installation.** `pip install nerdit` now ships the
  MCP server alongside the prebuilt dashboard and mDNS, so `nerdit mcp` works
  with no extra. `nerdit[mcp]` still installs. HTTP MCP still needs explicit
  configuration (or linking) and a daemon authentication token.
- **Project logs and diagnosis.** The new MCP tools `project_logs` and
  `diagnose_project` take a project and a service (`web` by default), so an agent
  no longer needs to look up the service's internal name first.
- **Address projects by stable ID.** Project reads, apply and variable operations,
  over REST and MCP, accept the immutable `prj_…` ID as well as the name. The ID
  keeps pointing at the same project when its display name changes. A deleted ID
  is refused, even if a new project reuses the name.

### Projects

- **Rename a project's display label** with `nerdit projects rename <project-id>
  <name>`. Duplicate labels are allowed. The original namespace, service names,
  public slugs and granted access don't change.
- Project-scoped tokens can manage service-scope variables on the services
  inside their project. `/secrets/{service}` stays label-only.
- A Git project apply clones the repository once and builds every service from
  that same commit.
- **Delegated access from Nerdit Cloud.** A Cloud assistant connection can be
  approved for a single project instead of the whole machine. It can then read
  the project, its logs and diagnosis, and write variables, but can't deploy.
  Deploying needs machine access. Cloud forwards variable values to the daemon
  without storing them or reading them back. Older daemons refuse project-only
  access instead of treating it as machine access.

### Security and reliability

- New `[containers].pids_limit` (default 4096) caps the processes and threads of
  every service, run, model and database container.
- Webhook deliveries add `X-Nerdit-Timestamp` and a replay-resistant
  `X-Nerdit-Signature-V2`, an HMAC over `<timestamp>.<delivery>.<body>`.
  `X-Nerdit-Signature` is unchanged. Targets now drain concurrently, so a dead
  endpoint no longer delays the others.
- `POST /api/deploy` and `POST /api/projects/{project}/apply` require a
  `Content-Length`, so a chunked upload gets `411 length_required`. Both are
  bounded at `[daemon].max_upload_bytes` + 1 MiB before parsing
  (`413 payload_too_large`).
- Failed authentications (a missing, invalid or stale bearer token) are audited
  as `auth.denied`. Filter them with `GET /api/audit?action_prefix=auth`. An
  `X-Request-Id` outside `[A-Za-z0-9._:-]{1,128}` is replaced by a new ID
  instead of being echoed.
- `nerdit connect` can read the token from a hidden prompt or one line of stdin,
  which keeps it off argv (`--token` is still accepted). It warns when a
  non-loopback host uses plain HTTP. `connect` and `nerdit init` write
  `~/.nerdit/config.toml` atomically with mode 0600.
- `nerdit link nk_…` with a positional key exits 2 and points to `--key-stdin`:
  a pre-auth key never goes on argv.
- Deleting a service with `purge=data` no longer removes the data of a same-name
  service created after the delete committed. The response reports
  `purged.data: false`.
- The workspace cleanup no longer removes a project's workspace.
- Under `[proxy].dashboard_apex` in path mode, new services named `api` or
  `assets` are refused (`422 service.reserved_name`), and `nerdit doctor` warns
  about existing ones.

### Before upgrading

- **Token-less LAN access is refused.** A daemon without `[daemon].auth_token`
  now answers non-public requests only when the `Host` is `localhost` or a
  loopback IP. Other hosts get `421 invalid_host`. If you reach a token-less
  daemon from another machine (a `0.0.0.0` bind, or a token-less
  `dashboard_apex`), configure a token first. Public paths (`/health`,
  `/api/proxy/ca`, the dashboard shell, `/assets`) stay reachable. Daemons with
  a token behave as before.
- **`.env` files are no longer uploaded.** `nerdit deploy`, `apply`, `dev` and
  the MCP `deploy` tool skip `.env*` files and symlinks that resolve outside the
  app folder. If an image relied on a baked-in `.env`, move those values to
  `nerdit secrets set` or `nerdit vars set`.
- **Git redirects are no longer followed.** Clones and push-to-deploy polling
  fail with `400 deploy.git_clone_failed` for a renamed or moved repository.
  Redeploy it with its new URL. A `repo_url` containing whitespace or control
  characters gets `422 deploy.git_url_invalid`, including on redeploys of an
  existing source.
- **Containers get a 4096-task cap** on the first restart after the upgrade.
  Raise `[containers].pids_limit` if a workload needs more. The default memory
  limit is unchanged.
- Adding Cloud destinations or changing an assistant's approved scope
  invalidates earlier OAuth credentials: reconnect the assistant.

## 0.6.4 (2026-09-21)

Deploy a web frontend, API and worker as one project, with shared variables,
secrets ready before the first deploy, and a dashboard that brings them together.

### Installation

- **Install with pip as an alternative to the shell installer.** The installation
  guide now covers `pip install nerdit` in a dedicated Python 3.11+ environment,
  optional MCP and mDNS support, upgrades and removal. The package includes the dashboard;
  Caddy and startup after reboot are configured separately for pip installs.
- **LAN discovery is included.** `zeroconf` ships in both the base Python package
  and signed bundles. Fresh installations enable HTTPS on port 8443 and mDNS
  for the machine’s `.local` name. Existing configuration is preserved;
  mDNS requires a multicast-capable LAN, and HTTPS uses the local CA.

### Projects and deployment

- **Declare several services in one `nerdit.toml`.** Add a `[project]` section
  and a `[services.<name>]` table for each service, then run
  `nerdit apply --dry-run` to validate the plan or `nerdit apply --wait` to
  deploy it. Apply works from a local folder or Git repository; agents can use
  the `write_project_files` and `apply_project` MCP tools with their workspace.
- **Manage the project as a whole.** Use `nerdit projects list|create|show|delete`
  to inspect services, addresses and referenced models and databases, or remove
  the project and its services. Service commands accept `project/service`, for
  example `nerdit logs my-app/api`.
- **Existing apps become projects automatically.** Their service names and
  addresses stay unchanged, and the existing single-app deployment workflow
  remains available.

### Variables and secrets

- **Share configuration across services or override it per service.**
  `nerdit vars set|unset|list|resolve` manages plain and secret variables at
  project or service scope. Both are encrypted at rest; secret values are
  never returned by the variable APIs. `resolve` shows which scope supplies
  each key without exposing its value.
- **Provide secrets before the first deploy.** Set project variables or use
  `nerdit secrets set` before a service exists. Required variables declared in
  `[vars] required` are checked before deployment: if any are missing, apply
  lists their names and the command to set them, then exits without deploying.
  Use `--secret --prompt KEY` to enter a secret without putting it in shell
  history or command-line arguments. Existing `nerdit secrets` commands remain
  supported.

### Dashboard

- **One project, one place to look.** The Apps list groups services by project.
  The project page shows service status, addresses, referenced models and
  databases, and a variable editor for owners and admins. Individual service
  pages and existing links remain available.
- **More reliable editing.** Temporary network errors offer a retry instead
  of sending you to another page. Background refreshes preserve drafts;
  switching projects clears unsaved secret inputs.

### Security and reliability

- Retained secrets stay associated with their owner when a service is deleted,
  even during concurrent writes or database creation. A later deployment by
  another owner cannot inherit them, and a delayed purge cannot delete a new
  owner's secrets.
- Variable reads remain safe during concurrent changes between plain and
  secret values. Workspace deployments enforce the workspace owner's project
  and secret permissions, including when initiated by an admin.
- `nerdit apply --wait` tracks the deployment it started for each service and
  reports when another deployment supersedes it.

### API and agent compatibility

`GET /api/capabilities` exposes `features.projects`, `features.variables`,
`features.project_apply` and `features.secrets_before_deploy` for feature detection.

| Surface | HTTP endpoints | MCP tools |
| --- | --- | --- |
| Projects | `GET /api/projects`, `POST /api/projects`, `GET /api/projects/{project}`, `DELETE /api/projects/{project}` | `create_project`, `list_projects`, `get_project`, `delete_project` |
| Variables | `GET /api/projects/{project}/variables`, `PUT /api/projects/{project}/variables`, `GET /api/projects/{project}/variables/resolve`, `DELETE /api/projects/{project}/variables/{key}` | `set_variable`, `resolve_variables` |
| Apply | `POST /api/projects/{project}/apply` | `write_project_files`, `apply_project` |

- Apply accepts `workspace=true` for the caller's workspace or `repo_url` for
  Git. Missing required variables return `waiting_for_variables` (CLI exit 4).
  Partial failures return `project.apply_incomplete`; single-app deployment
  endpoints reject declarations with HTTP 422 `deploy.use_apply`.
  `--wait` tracks the applied `build_version`.
- Ownership conflicts use HTTP 409 `project.owned`, `service.name_claimed`
  or `secret.orphaned_scope`. Incomplete project deletion returns HTTP 409
  `project.delete_incomplete`.
- API/MCP variable writes default to secret. Launch precedence is
  `[deploy].env` < project < service < injected bindings. `${vars.KEY}` and
  `${vars.shared.KEY}` references work in `[ai.*].api_key`, `[db.*].password`,
  `[deploy.edge_auth].password` and Git `token_ref`; daemon notification targets
  continue to use `${secrets.shared.…}`. `nerdit vars set --machine` retains
  the `nerdit secrets set --shared` behavior.

### Behavior to know before upgrading

- A project remains after its last service is removed and keeps its name
  reserved. Use `nerdit projects delete <name>` to remove the project itself;
  retained secrets can also keep a service name reserved.
- Apply deploys services sequentially, with **no project-wide rollback**. If a
  service fails, earlier services stay deployed. Fix the failure and apply
  again; services already applied are redeployed too. Removing a service from
  `nerdit.toml` does not delete it: use `nerdit services rm <project>/<service>`.
- Declared services are updated by running `nerdit apply` again. Push-to-deploy
  (`auto_deploy`) is not supported for them and is disabled when apply converts
  an existing Git-deployed service.
- Projects currently use the production environment only; staging and preview
  environments are not included in this release.

## 0.6.3 (2026-09-17)

A security and correctness hardening release, drawn from two focused audits over the agent/MCP surface and the daemon. No new features and nothing breaking — upgrading is drop-in. Highlights:

- A managed database binding is authorized against the database it names: an app can only bind a managed database its own account created, or one an admin binds for it, checked at both the deploy and app-config paths.
- Build files the daemon generates into an app's build context are written without following symlinks, so a crafted source tree cannot redirect a daemon-side write out of the context.
- Over the remote MCP transport, a path-based `deploy` is refused with a clear error instead of reading the daemon host's own filesystem; content-bearing deploys (`deploy_app`, ZIP, git) are unaffected.
- Configuration-change audit entries record only which keys changed, never the submitted values — on every path, including dry-run and validation failure.
- A service-logs read without an explicit tail comes back as a bounded page with a continuation cursor instead of the full history.
- Two overlapping redeploys of one app can no longer land on the same version or leave a superseded build context behind on disk.
- Changes under `[containers]`, `[nerdit]` and `[monitor]` correctly report that they need a daemon restart.
- An app declaring the maximum number of volumes deploys and launches instead of being stranded, and per-service disk accounting walks each data directory once.
- A bearer token carrying non-ASCII characters is rejected as a normal 403.
- Agent path: MCP validation errors report sanitized field locations without echoing the submitted input, `wait_for_service` floors a negative timeout, the CLI brackets IPv6 daemon hosts when composing the daemon URL, `dump_database` surfaces its disk numbers at the top level, and a failed MCP-over-HTTP startup shuts its session manager down cleanly.

## 0.6.2 (2026-09-14)

Build-time variables for browser apps, secrets that never touch the command line, and an MCP surface an agent can drive from the schema alone. Highlights:

- Build-time public variables: `[deploy.build_settings.public_env]` values are passed to the build and embedded in browser assets, shown in clear in the dry-run plan; secret references and reserved names are refused. Advertised as `deploy.build_settings.public_env` in `GET /api/capabilities`.
- `nerdit secrets set NAME --prompt KEY` reads a secret value without echo (repeatable), so a value never has to sit on argv or in a transcript.
- `PUT /api/services/{name}/share` accepts `preserve_existing: true`: create a missing share as requested, but never downgrade an existing one, so opening your own app privately cannot un-publish it.
- An invalid local `nerdit.toml` makes `nerdit deploy`, `dev` and `serve` exit with one value-free line naming the field instead of a traceback.
- Agent path: the deploy tools and `get_app_config` describe `release` (pre-cutover command), `cutover` and the secret-reference contract (`env` is literal; refs only in `ai`/`db`/`edge_auth`/`token_ref`); a submitter hitting a private repository is told to ask the node owner instead of being pointed at admin-only remedies.
- Agent path: every MCP tool argument now carries its own one-sentence rule in the tool's input schema (default when omitted, what null means, bounds, reference-not-value), so an agent that reads only the schema gets the contract; tool descriptions shed the sentences that only restated an argument. The cloud gateway's `nerdit_list_tools` gains `compact` (name plus one line) and `prefix` so one node's tool list fits a tool-result cap.
- Build detection: a `pyproject.toml` that only carries tool configuration (`[tool.ruff]`, `[tool.black]`) no longer turns a Node app into a Python build; `[project]`, `[build-system]`, `[tool.poetry]`, `setup.py`/`setup.cfg` and `requirements.txt` keep Python precedence. Docs: the Vite `preview` + `base` redirect loop behind a path-mode proxy, and `submitted_via` is always `cli` today.

## 0.6.1

- Build Node.js and Next.js apps with automatic framework detection and selectable presets.
- Preview and save build commands, package manager, Node version and monorepo root; reset individual overrides when needed.
- Improve deployment validation, credential checks and recovery from incompatible saved presets.

The first public release starts this file's history. Nerdit was developed in a
private repository before it was opened; that pre-open-source history is not
reproduced here, and each release's notes on the releases page summarize what
the tag carries.

Entries land under this heading as changes merge, and move under a version
heading when that version is tagged.
