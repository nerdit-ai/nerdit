# Changelog

All notable changes to Nerdit are recorded here.

## Unreleased

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
