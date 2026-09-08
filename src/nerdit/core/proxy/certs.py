"""Read ACME certificate state from Caddy's storage tree.

The admin API exposes intent, not live certificate state. Read the exact
`certificates/<directory-storage-key>/<domain>/<domain>.crt` path for the
configured directory; globbing could mistake staging certificates for
production ones. `acme_storage_key` mirrors certmagic's `KeyBuilder.Safe`,
checked against live issuance by the Pebble smoke test.

`pending` means no usable leaf on disk, including both issuance in progress and
failure. Diagnose stuck issuance through listener checks and `caddy.log`.
No TLS handshake or admin call is needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from cryptography import x509

#: The certificate vocabulary, shared by `/proxy/status`, the domain views and
#: the CLI. `internal` = this node's own CA (an `acme=0` row: the default and
#: not a defect); `disabled` = the row asks for a public certificate but
#: `[proxy.acme].enabled` is false; `pending` = no usable leaf on disk yet;
#: `issued` / `expired` = a leaf exists, valid or not.
CertState = Literal["internal", "disabled", "pending", "issued", "expired"]


@dataclass(frozen=True, slots=True)
class CertStatus:
    """One domain's certificate fact. `not_after` is set for issued/expired only."""

    state: CertState
    not_after: datetime | None = None


# certmagic's `KeyBuilder.Safe` replacement table, in its order. Applied as a
# chain rather than in one pass (Go's `strings.NewReplacer`): no replacement
# here emits a character another pattern matches, so the two agree — verified
# against the measured pins in `tests/test_proxy_certs.py`.
_SAFE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (" ", "_"),
    ("+", "_plus_"),
    ("*", "wildcard_"),
    (":", "-"),
    ("..", ""),  # certmagic's directory-traversal guard; single dots survive
    ("/", "-"),
)

#: certmagic's final filter. `re.ASCII` on purpose: Go's `\w` is
#: `[0-9A-Za-z_]`, so a non-ASCII character is dropped there and must be
#: dropped here too.
_SAFE_STRIP_RE = re.compile(r"[^\w@.-]", re.ASCII)


def acme_storage_key(directory: str) -> str:
    """The storage folder name certmagic derives from an ACME directory URL.

    `host + path` of the URL (the scheme is not part of the key), sanitised by
    the `KeyBuilder.Safe` table above. Measured pins:

    * `https://127.0.0.1:14000/dir` → `127.0.0.1-14000-dir`
    * `https://acme-v02.api.letsencrypt.org/directory`
      → `acme-v02.api.letsencrypt.org-directory`

    Userinfo is stripped rather than sanitised into the key — `[proxy.acme]`
    refuses a directory that carries any, and a credential must not reach a
    directory name on disk even if that refusal is ever relaxed.
    """
    parts = urlsplit(directory)
    # `netloc` is host[:port] with optional `user:pass@` — Go's `URL.Host`
    # is the part after the `@`. Brackets around an IPv6 literal are kept
    # exactly as Go keeps them; the final filter drops them either way.
    host = parts.netloc.rpartition("@")[2]
    key = f"{host}{parts.path}".lower().strip()
    for needle, replacement in _SAFE_REPLACEMENTS:
        key = key.replace(needle, replacement)
    return _SAFE_STRIP_RE.sub("", key)


def acme_cert_path(storage_root: Path, directory: str, domain: str) -> Path:
    """Where Caddy stores the issued leaf+chain for `domain` under `directory`.

    `storage_root` is the Caddy storage root this daemon pins in the bootstrap
    config (`<data_dir>/caddy`), so the path is deterministic on every
    platform — Caddy's XDG default never applies here.
    """
    return storage_root / "certificates" / acme_storage_key(directory) / domain / f"{domain}.crt"


def inspect_cert(path: Path, *, now: datetime | None = None) -> CertStatus:
    """Classify the leaf at `path`: `issued`, `expired` or `pending`.

    Absent, unreadable and unparsable all collapse to `pending` — the honest
    reading of every one of them is "there is no usable public certificate
    here", and none of them is a reason to raise on an observability path that
    runs per domain row on a list request.

    The file Caddy writes is a full chain; `load_pem_x509_certificate` reads
    the first certificate in it, which is the leaf — the one whose expiry is the
    fact we want.
    """
    try:
        pem = path.read_bytes()
    except OSError:
        return CertStatus("pending")
    try:
        cert = x509.load_pem_x509_certificate(pem)
    except (ValueError, TypeError):
        return CertStatus("pending")
    moment = datetime.now(UTC) if now is None else now
    if moment.tzinfo is None:
        # A naive caller means UTC here; comparing it to an aware `not_after`
        # would raise TypeError and turn a status read into a 500.
        moment = moment.replace(tzinfo=UTC)
    not_after = cert.not_valid_after_utc
    if not_after <= moment:
        return CertStatus("expired", not_after)
    return CertStatus("issued", not_after)
