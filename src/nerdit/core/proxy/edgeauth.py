"""Validate edge-auth specs and derive Caddy-only credential material.

Re-read persisted `[deploy].edge_auth` with tri-state validation. The manager
resolves password references at route-build time; only bcrypt hashes reach
Caddy, never plaintext. One fingerprint function compares desired and live
auth handlers. This module performs no database, secret-store, or Caddy I/O.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import bcrypt

from nerdit.config.project import _EDGE_AUTH_USER_RE, SECRET_REF_RE

# bcrypt's hard input bound: bcrypt >= 4 raises on longer inputs rather than
# silently truncating (older bcrypts truncated — a credential the operator
# thinks is 100 chars strong would verify on its first 72 bytes). ONE constant
# + ONE predicate, shared by `hash_password` and the diagnose classifier
# (`routes/service_diagnose.py::_classify_edge_auth`), so the route plane and
# diagnose can never disagree about whether a resolved secret is servable.
BCRYPT_MAX_PASSWORD_BYTES = 72

# The fix for an over-length password is a SHORTER SECRET, never the
# declaration — the default EdgeAuthInvalid sentence would send the operator
# to `nerdit config app set`, which cannot help here.
BCRYPT_LIMIT_HINT = (
    "the resolved password exceeds bcrypt's "
    f"{BCRYPT_MAX_PASSWORD_BYTES}-byte input limit; set a shorter secret "
    "with 'nerdit secrets set <app> <KEY>=…' (the declaration itself is "
    "fine)."
)


def password_exceeds_bcrypt_limit(plaintext: str) -> bool:
    """True when `plaintext` cannot be bcrypt-hashed (input > 72 bytes)."""
    return len(plaintext.encode("utf-8")) > BCRYPT_MAX_PASSWORD_BYTES


@dataclass(frozen=True)
class EdgeAuthSpec:
    """A well-formed persisted `edge_auth` declaration.

    `password_ref` is the `${secrets.KEY}` reference **verbatim**, never a
    resolved value: resolution happens once per route build so rotating the
    secret takes effect without a redeploy.
    """

    user: str
    password_ref: str


class EdgeAuthInvalid(Exception):  # noqa: N818 — public name, kept stable
    """A persisted `edge_auth` blob is present but unusable.

    Carries the offending FIELD NAMES only (`user` / `password`), never
    the blob's values — it reaches a diagnose payload and a log line.

    `hint` overrides the default repair-the-declaration sentence for the
    one case where the declaration is fine and the SECRET is the problem
    (the bcrypt length refusal below), so the log points at the right fix.
    """

    def __init__(self, fields: tuple[str, ...], hint: str | None = None) -> None:
        self.fields = fields
        super().__init__(
            "Invalid [deploy] edge_auth: missing or malformed "
            f"{', '.join(fields)} (values not shown) — "
            + (
                hint
                or "repair the declaration with 'nerdit config app set <app> deploy edge_auth=…'."
            )
        )


@dataclass(frozen=True)
class EdgeAuthMaterial:
    """A resolved, hashed credential ready to ride the Caddy admin API."""

    user: str
    bcrypt_hash: str


def load_edge_auth(raw: object) -> EdgeAuthSpec | None:
    """Tolerantly re-read a persisted edge_auth blob, like `HealthCheck`.

    TRI-STATE, and the three states are NOT interchangeable:

    * absent / `raw is None`  → `None`: no edge auth, the route serves
      openly;
    * declared and well-formed → the `EdgeAuthSpec`;
    * declared but MALFORMED (blank/missing `user`, missing `password`,
      or a `password` that does not match `SECRET_REF_RE`) → **raise**
      `EdgeAuthInvalid`, which every caller routes into the SAME
      fail-closed path as an unresolvable secret (excluded from
      `desired_ids` ⇒ pruned; `register` refuses).

    Returning `None` for the third case would collapse "declared but
    broken" into "no auth" and publish a route its owner asked to protect.
    Unknown keys are still ignored — tolerance covers EXTRA keys, never a
    missing or ungrammatical credential. The grammar re-check is the backstop
    against a write path that bypassed `EdgeAuthConfig`.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        # A non-table blob is declared-but-unusable, not "no auth": same
        # fail-closed path. Name the section, the only field name there is.
        raise EdgeAuthInvalid(("edge_auth",))

    bad: list[str] = []

    raw_user = raw.get("user")
    # The user grammar is re-checked, not just its presence: a control
    # character smuggled past `EdgeAuthConfig` would ride the Caddy account
    # object and every log line that mentions it (the `[deploy].release`
    # rationale), and a ':' silently reshapes the RFC 7617 credential.
    user = raw_user if isinstance(raw_user, str) else ""
    # `fullmatch` (not `match`) here too: `$` matches before a trailing
    # newline, and `"admin\n"` is exactly the control-char vector this
    # backstop exists to stop.
    if not _EDGE_AUTH_USER_RE.fullmatch(user):
        bad.append("user")

    raw_password = raw.get("password")
    # `fullmatch`, not `match`: `SECRET_REF_RE`'s trailing `$` also
    # matches before a final newline, so `match` would accept
    # `"${secrets.K}\n<literal>"`-shaped smuggling on this backstop path.
    password = raw_password if isinstance(raw_password, str) else ""
    if not SECRET_REF_RE.fullmatch(password):
        bad.append("password")

    if bad:
        raise EdgeAuthInvalid(tuple(bad))

    return EdgeAuthSpec(user=user, password_ref=password)


def auth_fingerprint(user: str, bcrypt_hash: str) -> str:
    """Short digest identifying a materialized edge-auth handler.

    ONE function computes both the desired and the live fingerprint, over the
    same two inputs, so drift detection cannot disagree with itself. The
    inputs are a username and a bcrypt hash — neither is a plaintext
    credential — but the digest is what travels, so nothing else needs to.
    """
    return hashlib.sha256(f"{user}\x00{bcrypt_hash}".encode()).hexdigest()[:16]


def hash_password(plaintext: str) -> str:
    """Bcrypt-hash a resolved edge-auth password at the default cost.

    A **module-level function, always called through the module** (never
    imported by value into `manager.py`) so tests can patch it: bcrypt salts
    per call, so its output is never byte-stable and a golden snapshot needs a
    deterministic stand-in. That same per-call salting is why
    `ProxyManager._auth_cache` exists; after a restart each authed service
    still costs one (convergent) upsert, since the cache is memory-only.

    Raises `EdgeAuthInvalid` naming `password` — never the value —
    when the resolved plaintext exceeds bcrypt's 72-byte input limit (bcrypt
    >= 4 refuses rather than silently truncating, and truncating a credential
    on the user's behalf is worse than refusing to serve it). The caller
    already routes `EdgeAuthInvalid` into the fail-closed path; the
    hint names the SECRET as the fix — the declaration is fine here — and
    `password_exceeds_bcrypt_limit` is the single source of the bound
    so the diagnose classifier can mirror it exactly.
    """
    if password_exceeds_bcrypt_limit(plaintext):
        raise EdgeAuthInvalid(("password",), hint=BCRYPT_LIMIT_HINT)
    return bcrypt.hashpw(plaintext.encode("utf-8"), bcrypt.gensalt()).decode("ascii")
