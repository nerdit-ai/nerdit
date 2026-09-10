# `packaging/` — how a Nerdit node is built and shipped

P30. A customer never sees a Python environment:
they get one tarball per platform holding two frozen executables, the pinned
Caddy binary, the service-manager templates and the installer.

| File | Owner | What it does |
|---|---|---|
| `nerdit.spec` | WP-1 | PyInstaller spec — one onedir bundle, both entry points, one shared `_internal/` |
| `entry_nerdit.py` / `entry_nerditd.py` | WP-1 | The launchers PyInstaller freezes (it freezes scripts, not console-script entry points) |
| `assemble.sh` | WP-1 | **The single owner of the tarball inner layout.** CI and local builds both call it |
| `smoke.sh` | WP-1 | The acceptance bar: a real deploy loop against a real Docker on a machine with no Python |
| `caddy.pin` | WP-2 | The pinned Caddy version + the sha256 of each upstream asset (D-P30-5) |
| `install.sh` | WP-3 | What `curl -fsSL https://get.nerdit.ai \| sh` runs |
| `units/` | WP-4 | systemd (system + `--user`) and launchd templates |

Release engineering — the release procedure, CI secrets, signing key custody,
and the `get.nerdit.ai` hosting setup — lives in the private development
repository and is not part of this public tree.

Release builds happen in CI: a `v*` tag push assembles and validates the public
source tree once. Each platform builds from that same archive, then CI calls
`assemble.sh`, signs `SHA256SUMS` and publishes the matching source and binaries.
Release notes link the exact public commit; reruns refuse to replace binaries
when the public tag contains a different tree. Source publication is required.

The installer downloads these signed binaries; users do not need Python or a
compiler. The local build below is for development and build verification.

---

## Tarball layout

`assemble.sh` produces `nerdit-<version>-<os>-<arch>.tar.gz` around a single
top-level directory:

```
nerdit-<version>/
  nerdit                        frozen CLI executable
  nerditd                       frozen daemon executable
  _internal/                    shared PyInstaller payload (stdlib, deps, data files)
  caddy                         pinned static Caddy binary, 0755
  units/nerdit.service          systemd system unit template
  units/nerdit-user.service     systemd --user unit template
  units/ai.nerdit.daemon.plist  launchd agent template
  install.sh                    byte-identical copy of packaging/install.sh
  VERSION                       "<version>\n"
```

`<version>` is bare (`0.5.0`); the git tag is v-prefixed (`v0.5.0`). `<os>` is
`linux` or `macos`, `<arch>` is `x86_64` or `arm64`. Three artifacts ship:
linux/x86_64, linux/arm64, macos/arm64 — **macOS x86_64 is not built** (D-P30-2)
and `assemble.sh` refuses that combination.

`install.sh` rides inside the tarball because an installed node carries its own
installer: `nerdit update` re-execs `<ROOT>/current/install.sh`.

`--os`/`--arch` name the artifact; they do **not** verify that `--dist` was
actually built for that platform (PyInstaller can only build for the host, so
the CI matrix — one runner per cell — is what makes the label true). Mislabel
a local build and you get a correctly-named tarball full of the wrong binaries.

Change the layout in `assemble.sh` and nowhere else.

---

## Building locally

Start from the public release tag you want to build. Prerequisites: Python,
Node.js/npm, and Docker for the smoke. Build the dashboard from its lockfile;
the bundle is not committed. The spec asserts `src/nerdit/daemon/web/dist/index.html`
exists and fails the build by name if it does not:

```bash
(cd src/nerdit/daemon/web && npm ci && npm run build)
```

PyInstaller is deliberately **not** a project dependency — it is a build tool,
not a runtime one. CI uses Python 3.12, Node.js 20 and PyInstaller 6.16.0.
Use a clean build environment with the same runtime extra:

```bash
python3.12 -m venv /tmp/nerdit-build-venv
/tmp/nerdit-build-venv/bin/pip install '.[mcp]' pyinstaller==6.16.0
/tmp/nerdit-build-venv/bin/pyinstaller --noconfirm --clean packaging/nerdit.spec
# -> dist/nerdit/{nerdit,nerditd,_internal}
```

> **The build environment IS the artifact manifest.** PyInstaller ships
> whatever it can reach from the entry points in the environment it runs in, so
> a *dev* venv leaks its extras into a customer tarball. Building this repo's
> `venv/` (which has `[dev]`, `[mcp]` and `[mdns]`) produced a bundle carrying
> `mcp`, `zeroconf` and — via pydantic's own mypy plugin module — the whole of
> mypy. **Release builds use a clean environment with `pip install '.[mcp]'` and
> the pinned PyInstaller build tool**; the spec additionally
> excludes `pydantic.mypy` / `pydantic.v1.mypy` so the mypy chain cannot come
> back through the side door. Expect a locally built bundle to be larger than a
> CI one, and never treat a local build as representative of what ships.

Then fetch the pinned Caddy for your platform (CI does this from `caddy.pin`
with a checksum gate; by hand, read the version out of the pin file):

```bash
V=$(grep '^CADDY_VERSION=' packaging/caddy.pin | cut -d= -f2)
# linux/x86_64 -> caddy_${V}_linux_amd64.tar.gz
# linux/arm64  -> caddy_${V}_linux_arm64.tar.gz
# macos/arm64  -> caddy_${V}_mac_arm64.tar.gz
curl -fsSLO "https://github.com/caddyserver/caddy/releases/download/v$V/caddy_${V}_mac_arm64.tar.gz"
shasum -a 256 "caddy_${V}_mac_arm64.tar.gz"   # compare to CADDY_SHA256_MACOS_ARM64
tar -xzf "caddy_${V}_mac_arm64.tar.gz" caddy
```

and assemble:

```bash
sh packaging/assemble.sh \
    --version 0.5.0 --os macos --arch arm64 \
    --dist dist/nerdit --caddy ./caddy --out ./artifacts
# the last stdout line is the artifact path
```

A local build is fine for a smoke run; it is **not** what a release ships. The
WP-2 "done when" clause requires the smoke to pass against a *downloaded,
signature-verified* artifact.

---

## Running the smoke

```bash
sh packaging/smoke.sh artifacts/nerdit-0.5.0-macos-arm64.tar.gz
```

Seven legs, each printing `SMOKE <n> PASS`:

1. extract; `nerdit --version` matches the `VERSION` file, every layout member present
2. the frozen `nerditd` boots under a scratch `HOME` and answers `/health`
3. `nerdit doctor` renders its table; the `docker` **and** `proxy` checks are `ok`
   (the proxy check being `ok` is what proves the *bundled* Caddy started)
4. `nerdit store deploy node-starter` converges to `running`
5. the app answers **200** on its `public_url` through the bundled Caddy, over
   TLS validated against the internal CA fetched from `/proxy/ca`
6. `POST /api/daemon/restart` → the daemon drains, re-execs, `uptime_s` resets,
   the app is still `running` and still serving
7. teardown: the service (with `--purge secrets,data,images`), the daemon
   (killed by the PID captured at spawn — never `pkill`), the scratch dirs, the
   built images

Run it on a clean **Linux x86_64** VM and a clean **macOS arm64** machine, both
with Docker running. That two-platform pass is the WP-1 acceptance criterion.

Host prerequisites: `docker`, `curl`, `tar`, and **`git`** — the app-template
store clones server-side using the host's git; the bundle does not carry one.
Network access to `github.com` is needed for leg 4.

Knobs (all optional): `NERDIT_SMOKE_PORT` (9333), `NERDIT_SMOKE_HTTPS_PORT`
(9443), `NERDIT_SMOKE_APP` (`smoke-app`), `NERDIT_SMOKE_BOOT_TIMEOUT` (60),
`NERDIT_SMOKE_DEPLOY_BUDGET` (900).

### Why the smoke sets `DOCKER_CONFIG`

The docker CLI resolves its BuildKit plugin from `$DOCKER_CONFIG/cli-plugins`,
defaulting to `$HOME/.docker`. The smoke runs the daemon under a scratch `HOME`,
which hides a per-user buildx and makes **every** build fail with
`BuildPlatformError`. So it exports `DOCKER_CONFIG` pointing at the invoking
user's real docker config — exactly the accommodation the daemon's own doctor
hint asks a service unit to make. Anything that runs `nerditd` under a
service-account `HOME` (the D-P30-8 units included) has the same problem.

---

## Notes and deliberate exclusions

- **The `mdns` (`zeroconf`) extra is not installed in the release build
  environment**, so it does not ship. `[proxy].mdns` degrades exactly as it does
  on a source install without the extra — the advertiser stays off and
  `doctor` reports `mdns: skipped`.
- **The `mcp` extra ships since 0.5.3** (`release.yml` installs `.[mcp]`; the
  spec collects the package). The v1 note that "the MCP server remains
  reachable over `[mcp].http_enabled` without the extra" was wrong: that flag
  hard-errors without the extra (`appfactory.build_app`), which the first clean
  E2E found when the remote connector forwarded to `/api/mcp` and got 405
  (2026-08-23). `smoke.sh` leg 3 now boots with the transport on and asserts an
  MCP `initialize`, so a bundle without it cannot be published.
- **`tkinter` is excluded** from the freeze; nothing imports it.
- **No `packaging/hooks/` directory exists**, deliberately: the spike needed no
  custom PyInstaller hook or runtime hook. The spec picks the directory up
  automatically if one is ever added, so a future hook is a file drop, not a
  spec edit.
- **No UPX.** Compression saves tens of megabytes but is a reliable source of
  false-positive malware verdicts and of bootloader breakage on macOS.
- **macOS signing/notarization is deferred** (D-P30-3): the `curl | sh` path
  does not set the quarantine xattr, so Gatekeeper does not intercept. A
  browser-downloaded tarball would — the docs state the curl path as the only
  supported install.
- **The bundle packages the public engine source**, its Python runtime and
  dependencies for convenient installation. Matching source does not imply
  byte-for-byte reproducible archives: Python dependencies and host build
  environments can still differ.
