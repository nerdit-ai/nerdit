# CLAUDE.md

Guidance for Claude Code (or any coding agent) working in this repository.

## Concise code documentation

- Use concise Google-style Python docstrings: one purpose sentence, then
  `Args`, `Returns`, `Yields`, `Raises` or `Attributes` only when useful. Avoid
  repeating type hints or adding boilerplate to obvious helpers.
- Keep comments focused on why code exists and its non-obvious constraints.
  Remove repeated explanations and phase history; check claims against current
  code. Preserve security, ownership, cancellation, recovery, ordering, units,
  side effects and known limitations.
- Keep CLI command docstrings useful as user-facing help. Preserve required
  examples, warnings and caller-visible contracts when shortening text.
- **MCP exception:** exclude MCP modules, tool descriptions, related config/help
  and golden fixtures from general prose cleanup. Change them only when the
  task explicitly calls for an MCP change, and verify their contract tests.
- For documentation-only edits, preserve executable code, API field strings,
  tooling directives and test assertions, including explicitly tested wording.
  Verify relevant help/docs and tests; finish edits before measuring coverage.
- Prefer existing helpers and standard-library features over new abstractions.
  Simplify for clarity, never to meet a line-count target.

## What Nerdit is

Nerdit is a local AI PaaS: point it at a folder and it builds, runs, and
gives an app an HTTPS URL, wiring it to a local or remote AI model on
request. Two processes — a stateless CLI (`nerdit`) talking over HTTP to a
long-running daemon (`nerditd`) — cooperate to deploy, supervise, proxy,
and operate apps, models and databases on the user's own hardware.

## Invariants (do not violate these)

1. **The OpenAI API is the model contract.** A `ModelBackend` Protocol
   (Ollama, vLLM) and an app's `[ai.*]` binding (`provider = "ollama" |
   "api"`) both resolve to the same injected env
   (`OPENAI_BASE_URL`/`OPENAI_API_KEY`). Backend and provider are
   swappable; the app never sees the difference.
2. **The proxy is route-agnostic.** A route is *data*
   (`service_endpoints.route`), never a hardcoded shape. Path-based and
   subdomain-based routing are both first-class, selected by config, not
   by separate code paths.
3. **Agent-grade by construction.** Every write endpoint is scoped-token
   authorized, idempotent (`Idempotency-Key`), returns a structured
   machine-readable error on failure, and is audit-logged; reads are
   bounded/paginated. The MCP server is a thin projection of this same
   REST API — it opens no shortcut around auth, idempotency or audit.

## Repo layout

```
src/nerdit/
├── cli/      # Typer commands
├── daemon/   # FastAPI server, routes, auth middleware, web/ (React dashboard)
├── core/     # GPU discovery, WorkloadManager/ServiceController, container
│             # runtime abstraction, proxy, builder, models/, data/, gitsource
├── db/       # SQLite (aiosqlite) — database, models, queries
├── config/   # daemon settings, per-app project config, app templates
└── utils/    # ids, logging
tests/        # unit + integration; frontend e2e under daemon/web/tests/e2e
docker/       # base container image
```

Core has no runtime import of daemon or CLI — that layering is load-bearing,
keep it that way.

User guides and API documentation are maintained separately for the
[Nerdit website](https://nerdit.ai/).

## Build, test, lint

```bash
pip install -e ".[dev,mcp]"
cd src/nerdit/daemon/web && npm ci && npm run build   # dashboard (required
                                                        # before running the
                                                        # daemon from source)

pytest tests/ -v --asyncio-mode=auto        # full suite
pytest tests/test_services_reconcile.py -v  # one file
ruff check . && ruff format .
mypy src/nerdit
```

Some sandboxes hang running the full suite through Starlette's `TestClient`
combined with aiosqlite. If a run hangs or times out, fall back to focused
test files (`pytest tests/<file> -x -q --asyncio-mode=auto` with an explicit
timeout) rather than assuming green.

## Contribution model

This is a private-primary project: development and full hardware-gated CI
happen internally, and this public repo is the release channel. An accepted
PR here is imported internally (authorship preserved) and ships in the next
release — see `CONTRIBUTING.md` for the full flow before proposing a large
change.
