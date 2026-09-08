"""``core/proxy/certs.py`` — the derived certificate facts (P26 WP2 / S-W2-6).

Two things are pinned here. The **storage key** mirrors a foreign
implementation detail (certmagic's ``KeyBuilder.Safe``), so the measured pins
below are the contract: get them wrong and every ACME domain reports ``pending``
forever while its certificate sits happily on disk one directory away. And
:func:`inspect_cert` must never raise — it runs per domain row on a list
request, where "no readable leaf" is an answer, not an error.

The Pebble smoke test re-proves the storage key against a **real** issuance;
these tests are the fast, offline half of the same assertion.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from nerdit.config.settings import DEFAULT_ACME_DIRECTORY, LE_STAGING_DIRECTORY
from nerdit.core.proxy.certs import (
    CertStatus,
    acme_cert_path,
    acme_storage_key,
    inspect_cert,
)

# --- helpers ------------------------------------------------------------------


def _leaf_pem(domain: str, *, not_before: dt.datetime, not_after: dt.datetime) -> bytes:
    """A self-signed leaf shaped like the one Caddy writes (chain head)."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, domain)])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(domain)]), critical=False)
        .sign(key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.PEM)


# --- acme_storage_key ---------------------------------------------------------


@pytest.mark.parametrize(
    ("directory", "expected"),
    [
        # Measured against a real Pebble issuance (explore-caddy-acme.md §5).
        ("https://127.0.0.1:14000/dir", "127.0.0.1-14000-dir"),
        (DEFAULT_ACME_DIRECTORY, "acme-v02.api.letsencrypt.org-directory"),
        (LE_STAGING_DIRECTORY, "acme-staging-v02.api.letsencrypt.org-directory"),
    ],
)
def test_storage_key_pins(directory, expected):
    assert acme_storage_key(directory) == expected


def test_storage_key_ignores_the_scheme():
    """certmagic keys on ``host + path``; http and https of one CA collide by
    design — and that is what the measured pins show."""
    assert acme_storage_key("http://127.0.0.1:14000/dir") == acme_storage_key(
        "https://127.0.0.1:14000/dir"
    )


def test_storage_key_is_lowercased():
    assert acme_storage_key("https://CA.Example.COM/Directory") == "ca.example.com-directory"


def test_storage_key_applies_the_full_safe_table():
    """The replacement table verbatim: space, plus, star, colon, ``..``, slash,
    then every remaining non-``[\\w@.-]`` character dropped.

    The query string is **not** part of the key — certmagic builds it from
    ``Host + Path`` only, so two directory URLs differing only in their query
    share a storage tree. Mirrored deliberately rather than improved on.
    """
    assert acme_storage_key("https://ca.example.com/a b+c*d..e/f?x=1") == (
        "ca.example.com-a_b_plus_cwildcard_de-f"
    )


def test_storage_key_drops_userinfo():
    """A credential must not land in a directory name on disk even if the
    ``[proxy.acme]`` refusal is ever relaxed."""
    key = acme_storage_key("https://user:hunter2@ca.example.com/directory")

    assert key == "ca.example.com-directory"
    assert "hunter2" not in key


def test_storage_key_has_no_path_separator_left():
    """The whole point of the sanitisation: one flat directory name."""
    assert "/" not in acme_storage_key("https://ca.example.com/acme/v2/directory")


# --- acme_cert_path -----------------------------------------------------------


def test_cert_path_layout(tmp_path):
    path = acme_cert_path(tmp_path, "https://127.0.0.1:14000/dir", "app.example.test")

    assert path == (
        tmp_path
        / "certificates"
        / "127.0.0.1-14000-dir"
        / "app.example.test"
        / "app.example.test.crt"
    )


def test_cert_path_is_directory_scoped(tmp_path):
    """Staging and production leaves live apart — which is exactly why the state
    is derived from the configured directory and never globbed for."""
    staging = acme_cert_path(tmp_path, LE_STAGING_DIRECTORY, "app.example.test")
    production = acme_cert_path(tmp_path, DEFAULT_ACME_DIRECTORY, "app.example.test")

    assert staging != production


# --- inspect_cert -------------------------------------------------------------


def test_absent_file_is_pending(tmp_path):
    assert inspect_cert(tmp_path / "nope.crt") == CertStatus("pending")


def test_unreadable_file_is_pending(tmp_path):
    """A directory where a file is expected raises ``IsADirectoryError`` (an
    ``OSError``) — an observability read must not turn that into a 500."""
    path = tmp_path / "app.example.test.crt"
    path.mkdir()

    assert inspect_cert(path) == CertStatus("pending")


def test_unparsable_file_is_pending(tmp_path):
    path = tmp_path / "app.example.test.crt"
    path.write_bytes(b"not a certificate\n")

    assert inspect_cert(path) == CertStatus("pending")


def test_valid_leaf_is_issued_with_its_expiry(tmp_path):
    now = dt.datetime(2026, 8, 21, 12, 0, tzinfo=dt.UTC)
    not_after = now + dt.timedelta(days=60)
    path = tmp_path / "app.example.test.crt"
    path.write_bytes(
        _leaf_pem("app.example.test", not_before=now - dt.timedelta(days=1), not_after=not_after)
    )

    status = inspect_cert(path, now=now)

    assert status.state == "issued"
    assert status.not_after is not None
    assert status.not_after.replace(microsecond=0) == not_after.replace(microsecond=0)


def test_expired_leaf_is_expired_and_still_reports_not_after(tmp_path):
    now = dt.datetime(2026, 8, 21, 12, 0, tzinfo=dt.UTC)
    not_after = now - dt.timedelta(days=1)
    path = tmp_path / "app.example.test.crt"
    path.write_bytes(
        _leaf_pem("app.example.test", not_before=now - dt.timedelta(days=90), not_after=not_after)
    )

    status = inspect_cert(path, now=now)

    assert status.state == "expired"
    assert status.not_after is not None


def test_expiry_boundary_is_inclusive(tmp_path):
    """``not_after <= now`` — a certificate is not usable in its last instant."""
    now = dt.datetime(2026, 8, 21, 12, 0, tzinfo=dt.UTC)
    path = tmp_path / "app.example.test.crt"
    path.write_bytes(
        _leaf_pem("app.example.test", not_before=now - dt.timedelta(days=1), not_after=now)
    )

    assert inspect_cert(path, now=now).state == "expired"


def test_a_naive_now_is_read_as_utc(tmp_path):
    """Comparing a naive datetime to the aware ``not_valid_after_utc`` would
    raise TypeError and turn a status read into a 500."""
    aware = dt.datetime(2026, 8, 21, 12, 0, tzinfo=dt.UTC)
    path = tmp_path / "app.example.test.crt"
    path.write_bytes(
        _leaf_pem(
            "app.example.test",
            not_before=aware - dt.timedelta(days=1),
            not_after=aware + dt.timedelta(days=1),
        )
    )

    assert inspect_cert(path, now=aware.replace(tzinfo=None)).state == "issued"


def test_the_first_certificate_of_a_chain_is_the_one_read(tmp_path):
    """Caddy writes leaf + intermediate into one ``.crt``; the leaf's expiry is
    the fact we want, and it is the first block in the file."""
    now = dt.datetime(2026, 8, 21, 12, 0, tzinfo=dt.UTC)
    leaf_after = now + dt.timedelta(days=30)
    chain = _leaf_pem(
        "app.example.test", not_before=now - dt.timedelta(days=1), not_after=leaf_after
    ) + _leaf_pem(
        "intermediate.example.test",
        not_before=now - dt.timedelta(days=365),
        not_after=now + dt.timedelta(days=3650),
    )
    path = tmp_path / "app.example.test.crt"
    path.write_bytes(chain)

    status = inspect_cert(path, now=now)

    assert status.state == "issued"
    assert status.not_after is not None
    assert status.not_after.replace(microsecond=0) == leaf_after.replace(microsecond=0)


def test_cert_status_is_frozen():
    """The status is a fact, not a mutable accumulator — projections copy it."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        CertStatus("pending").state = "issued"  # type: ignore[misc]
