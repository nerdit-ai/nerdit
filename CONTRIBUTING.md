# Contributing to Nerdit

Thanks for considering it. Read this first — the model here is a little
different from a typical GitHub project, and it's better to know that going
in.

## Open source, not open development

Development happens in a private repository, whose CI includes live runs
against real GPU hardware and Docker/Caddy that can't run as public,
untrusted-fork CI. This public repository is the source of truth for
**releases** — one commit per version, tagged, built by an internal
publish workflow whenever a release ships.

Issues and pull requests are welcome here. An accepted PR is imported into
the internal repository with your authorship preserved (`git am`, so you
stay the commit author), run through the full internal test/review gates,
and ships in the next release. At that point your PR is closed with a
comment naming the release that carries your change. It isn't merged in
the GitHub sense — the code lands via the release sync — but the credit
and the history are yours.

If that loop ever becomes the bottleneck (high contribution volume), we'll
revisit going public-primary. Until then, this is the honest description
of how it works, not a euphemism for "we ignore PRs."

## What CI runs where

- **Public (this repo, every PR):** lint (`ruff`), type-check (`mypy`), and
  the unit-test subset that needs no Docker or GPU. Fast signal for
  contributors, no secrets involved.
- **Internal (after import):** the full test suite (over 6,000 tests), live
  agentic CLI runs against a real daemon on real hardware, and the
  hardware-gated checks that can't run on a public fork's CI. This is what
  actually gates a release.

## Developer Certificate of Origin

Every commit must be signed off:

```bash
git commit -s
```

This adds a `Signed-off-by: Your Name <email>` trailer certifying you wrote
the change or otherwise have the right to submit it under the project's
license (Apache-2.0). The DCO check on the PR will tell you if a commit is
missing it — fix with `git commit --amend -s` (or an interactive rebase for
older commits), don't open a new PR.

## Dev setup

```bash
git clone https://github.com/nerdit-ai/nerdit && cd nerdit
python -m venv venv && venv/bin/pip install -e ".[dev,mcp]"
cd src/nerdit/daemon/web && npm ci && npm run build && cd -
```

```bash
venv/bin/pytest tests/ -v --asyncio-mode=auto      # full suite
venv/bin/pytest tests/test_services_reconcile.py   # a single file
venv/bin/ruff check . && venv/bin/ruff format .
venv/bin/mypy src/nerdit
```

> Some environments hang inside Starlette's `TestClient` combined with
> aiosqlite when running the entire suite at once. If that happens, run
> focused test files instead of the whole tree and it should behave
> normally.

## Before you send a large change

For anything that touches more than a handful of files — a new subsystem,
a rework of an existing one, a change to a wire contract — open an issue
first and describe the approach. It's much cheaper to redirect a plan than
a finished diff, and the import model above means a rejected large PR is a
larger loss of your time than in a normal repo.

Small, focused fixes (a bug, a docs correction, a missing test) don't need
this — just open the PR.

## Documentation

User guides and API documentation are maintained separately for the
[Nerdit website](https://nerdit.ai/). Report documentation issues here with
the page URL and your Nerdit version. Source comments, tests and the root
contributor documents stay in this repository.
