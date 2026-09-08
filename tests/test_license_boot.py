"""Boot-time license loading: degrade, never abort (P17d WP-D1, D-LIC2).

The one property that matters more than any other here is the **upgrade /
``make e2e-real`` compatibility pin**: a daemon with no license file must boot
exactly as it did before P17d, at the ``[link]`` seam and everywhere else. Every
other case degrades to doctor-visible with at most one ERROR log and one durable
event — a billing opinion must never take the local control plane down.
"""

from __future__ import annotations

import ast
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nerdit.config.settings import LicenseSettings, LinkSettings, NerditSettings
from nerdit.core.eventlog import EventRecorder
from nerdit.core.license import LicenseState
from nerdit.daemon.bootstrap import build_license_state, build_link_manager
from tests.license_vectors import (
    GOLDEN_BLOB,
    GOLDEN_CLAIMS,
    TEST_TRUSTED_KEYS,
    sign_license,
)

NOW_VALID = datetime(2026, 6, 1, tzinfo=UTC)


def _settings(tmp_path: Path, **license_kwargs) -> NerditSettings:
    return NerditSettings(
        data_dir=str(tmp_path),
        license=LicenseSettings(**license_kwargs),
    )


async def _events(db):
    cur = await db.conn.execute("SELECT type, reason FROM events ORDER BY id")
    return [tuple(row) for row in await cur.fetchall()]


async def _build(settings, queries, **kwargs) -> LicenseState:
    return await build_license_state(
        settings,
        EventRecorder(queries),
        trusted_keys=TEST_TRUSTED_KEYS,
        now=lambda: NOW_VALID,
        **kwargs,
    )


# --- absent: the compatibility pin --------------------------------------------


async def test_no_license_file_is_the_ordinary_state(tmp_path, db, queries, caplog):
    """**e2e-real / upgrade compatibility pin.**

    nerdit-cloud's ``make e2e-real`` and every already-linked daemon in the
    field boot this daemon with **no license file**. That path must stay silent:
    no log line, no durable event, no exception — an unlicensed daemon is the
    free local product, not a degraded one.
    """
    with caplog.at_level("DEBUG"):
        state = await _build(_settings(tmp_path), queries)

    assert state.installed is False
    assert state.state is None
    assert state.require_entitlement("remote_link").allowed is True
    assert await _events(db) == []
    assert [r for r in caplog.records if "license" in r.getMessage().lower()] == []


async def test_licenseless_boot_still_builds_the_link_manager(tmp_path, db, queries, caplog):
    """The same pin at the seam that consumes the decision: the tunnel is built.

    ``build_link_manager`` is called with a licenseless holder in scope and
    returns a real manager; the P17d advisory is exactly that — advisory.
    """
    settings = NerditSettings(
        data_dir=str(tmp_path),
        link=LinkSettings(
            enabled=True,
            relay_url="wss://relay.example.com/link",
            node_id="00000000-0000-4000-8000-00000000000a",
            slug="pin-node",
        ),
    )
    recorder = EventRecorder(queries)
    state = await build_license_state(settings, recorder, trusted_keys=TEST_TRUSTED_KEYS)
    manager = await build_link_manager(settings, recorder)

    assert state.installed is False
    assert manager is not None
    assert state.require_entitlement("remote_link").allowed is True


# --- valid / grace / expired --------------------------------------------------


@pytest.mark.parametrize(
    ("expires_at", "expected"),
    [
        ("2027-01-01T00:00:00+00:00", "valid"),
        # 3 days before ``now`` — inside the 7-day grace.
        ("2026-05-29T00:00:00+00:00", "expired_grace"),
        ("2026-01-01T00:00:00+00:00", "expired"),
    ],
)
async def test_temporal_states_load_without_an_event(
    tmp_path, db, queries, expires_at, expected, caplog
):
    """Only *invalid* is an event. Expiry is doctor's job, not the feed's."""
    (tmp_path / "license.jws").write_text(
        sign_license(dict(GOLDEN_CLAIMS, expires_at=expires_at)) + "\n"
    )
    with caplog.at_level("ERROR"):
        state = await _build(_settings(tmp_path), queries)

    assert state.state == expected
    assert state.claims is not None
    assert await _events(db) == []
    assert caplog.records == []


async def test_valid_license_entitles_remote_link(tmp_path, queries):
    (tmp_path / "license.jws").write_text(GOLDEN_BLOB + "\n")
    state = await _build(_settings(tmp_path), queries)
    decision = state.require_entitlement("remote_link")
    assert decision.allowed is True
    assert decision.state == "valid"
    assert decision.reason is None


# --- invalid: ERROR + event + boot proceeds -----------------------------------


async def test_invalid_license_logs_a_reason_token_and_emits_the_event(
    tmp_path, db, queries, caplog
):
    (tmp_path / "license.jws").write_text(sign_license(GOLDEN_CLAIMS, kid="not-enrolled") + "\n")
    with caplog.at_level("ERROR"):
        state = await _build(_settings(tmp_path), queries)

    assert state.state == "invalid"
    assert state.reason == "unknown_kid"
    # Fail-open: a corrupt file must not disable more than no file at all.
    assert state.require_entitlement("remote_link").allowed is True
    assert await _events(db) == [("license.rejected", "unknown_kid")]

    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1
    assert "unknown_kid" in messages[0]


async def test_boot_never_leaks_the_blob_or_customer_id(tmp_path, db, queries, caplog):
    tampered = GOLDEN_BLOB[:-4] + "AAAA"
    (tmp_path / "license.jws").write_text(tampered + "\n")
    with caplog.at_level("DEBUG"):
        state = await _build(_settings(tmp_path), queries)

    assert state.reason == "bad_signature"
    blob = "".join(r.getMessage() for r in caplog.records)
    assert tampered not in blob
    assert GOLDEN_CLAIMS["customer_id"] not in blob
    # The durable feed carries the machine token and nothing else.
    cur = await db.conn.execute("SELECT type, reason, data FROM events")
    rows = list(await cur.fetchall())
    assert len(rows) == 1
    assert rows[0][1] == "bad_signature"
    assert rows[0][2] in (None, "", "{}")


async def test_unreadable_file_degrades_to_malformed_rather_than_aborting(
    tmp_path, db, queries, caplog
):
    """A symlinked license is refused by ``O_NOFOLLOW`` — and boot continues."""
    real = tmp_path / "elsewhere.jws"
    real.write_text(GOLDEN_BLOB + "\n")
    (tmp_path / "license.jws").symlink_to(real)

    with caplog.at_level("ERROR"):
        state = await _build(_settings(tmp_path), queries)

    assert (state.state, state.reason) == ("invalid", "malformed")
    assert await _events(db) == [("license.rejected", "malformed")]
    assert state.require_entitlement("remote_link").allowed is True


async def test_unreadable_file_error_log_carries_no_blob(tmp_path, db, queries, caplog):
    path = tmp_path / "license.jws"
    path.write_bytes(b"\xff" + GOLDEN_BLOB.encode())
    with caplog.at_level("ERROR"):
        state = await _build(_settings(tmp_path), queries)
    assert state.reason == "malformed"
    assert GOLDEN_BLOB not in "".join(r.getMessage() for r in caplog.records)


async def test_a_nul_in_the_configured_path_degrades_instead_of_aborting_boot(
    tmp_path, db, queries, caplog
):
    """**The never-abort-boot pin against the one non-``OSError`` failure class.**

    ``os.open`` raises ``ValueError`` — not ``OSError`` — for a path carrying an
    embedded NUL, and TOML round-trips a NUL happily. The validator now refuses
    such a value at the config PUT, but a path persisted by a pre-fix daemon
    must still degrade here: a ``ValueError`` escaping ``build_license_state``
    would propagate out of the lifespan and take the whole control plane down
    over a billing opinion.

    ``model_construct`` bypasses the (now-fixed) validator on purpose — this
    test is about the *second* belt.
    """
    settings = _settings(tmp_path)
    settings.license = LicenseSettings.model_construct(file=f"{tmp_path}/lic\x00ense.jws")

    with caplog.at_level("ERROR"):
        state = await _build(settings, queries)

    assert (state.state, state.reason) == ("invalid", "malformed")
    assert await _events(db) == [("license.rejected", "malformed")]
    # Fail-open survives the degradation, exactly as for any unreadable file.
    assert state.require_entitlement("remote_link").allowed is True
    assert len(caplog.records) == 1


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
async def test_unreadable_permissions_do_not_abort_boot(tmp_path, db, queries):
    path = tmp_path / "license.jws"
    path.write_text(GOLDEN_BLOB + "\n")
    os.chmod(path, 0o000)
    try:
        state = await _build(_settings(tmp_path), queries)
    finally:
        os.chmod(path, 0o600)
    assert (state.state, state.reason) == ("invalid", "malformed")


# --- the configured path ------------------------------------------------------


async def test_license_file_override_is_honoured(tmp_path, queries):
    elsewhere = tmp_path / "custom"
    elsewhere.mkdir()
    (elsewhere / "lic.jws").write_text(GOLDEN_BLOB + "\n")
    state = await _build(_settings(tmp_path, file=str(elsewhere / "lic.jws")), queries)
    assert state.state == "valid"
    # And the data_dir default is NOT read when an override is set.
    assert not (tmp_path / "license.jws").exists()


async def test_boot_reads_the_file_off_the_event_loop(tmp_path, queries, monkeypatch):
    """Blocking I/O + a decode on the loop is the ``load_or_create_identity``
    mistake this seam deliberately avoids."""
    import nerdit.daemon.bootstrap as bootstrap

    seen: list[object] = []
    real_to_thread = bootstrap.asyncio.to_thread

    async def spy(func, *args, **kwargs):
        seen.append(func)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(bootstrap.asyncio, "to_thread", spy)
    (tmp_path / "license.jws").write_text(GOLDEN_BLOB + "\n")
    await _build(_settings(tmp_path), queries)
    assert bootstrap.read_license_file in seen


# --- the ONE entitlement call site (D-LIC2) -----------------------------------


def _server_source() -> str:
    return Path("src/nerdit/daemon/server.py").read_text(encoding="utf-8")


def test_exactly_one_entitlement_call_site_in_the_daemon():
    """D-LIC2: one helper, one v1 call site — the ``[link]`` boot seam.

    SSO/retention will call the same helper from their own seams later; until
    then a second call site is a plan edit, not a diff.
    """
    src = Path("src/nerdit")
    call_sites = [
        path
        for path in src.rglob("*.py")
        if "require_entitlement(" in path.read_text(encoding="utf-8") and path.name != "license.py"
    ]
    assert [p.name for p in call_sites] == ["server.py"]
    assert _server_source().count("require_entitlement(") == 1


def test_the_entitlement_advisory_never_blocks_the_tunnel():
    """The WARNING is followed by ``link_manager.start()``, not by a return.

    Parsed rather than grepped so a future refactor that turns the advisory into
    an early return trips this test instead of quietly shipping fail-closed.

    The guard is keyed on ``decision.reason``, not on ``decision.allowed``: the
    D-LIC2 state matrix wants a boot WARNING for four states, two of which are
    allowed-with-advisory (``invalid``, ``expired_grace``) — see
    :func:`test_every_matrix_warning_row_carries_an_advisory_reason`.
    """
    tree = ast.parse(_server_source())
    guard = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "decision." in ast.unparse(node.test):
            guard = node
    assert guard is not None, "the [link] entitlement advisory disappeared"
    # Exactly the reason test — not ``not decision.allowed``, and not the two
    # ANDed together, which would drop the same two matrix rows.
    assert ast.unparse(guard.test) == "decision.reason is not None"
    body = ast.unparse(guard)
    assert "logger.warning" in body
    assert "return" not in body
    assert "raise" not in body


@pytest.mark.parametrize(
    ("case", "warns"),
    [
        ("absent", False),
        ("valid", False),
        ("invalid", True),
        ("expired_grace", True),
        ("expired", True),
        ("feature_absent", True),
    ],
)
async def test_every_matrix_warning_row_carries_an_advisory_reason(tmp_path, queries, case, warns):
    """D-LIC2's state matrix: four of the six rows want "starts; 1 WARNING".

    The boot seam warns on ``decision.reason is not None``, so this pins the
    reason's nullability per state — the property the seam's guard reads. Two
    of the four warning rows are ``allowed=True`` (``invalid`` and
    ``expired_grace``), which is exactly why keying the guard on ``allowed``
    silently dropped half the matrix.
    """
    if case != "absent":
        claims = dict(GOLDEN_CLAIMS)
        if case == "expired_grace":
            claims["expires_at"] = "2026-05-29T00:00:00+00:00"  # 3 days ago
        elif case == "expired":
            claims["expires_at"] = "2026-01-01T00:00:00+00:00"
        elif case == "feature_absent":
            claims["features"] = ["sso"]
        blob = (
            sign_license(claims, kid="not-enrolled") if case == "invalid" else sign_license(claims)
        )
        (tmp_path / "license.jws").write_text(blob + "\n")

    state = await _build(_settings(tmp_path), queries)
    decision = state.require_entitlement("remote_link")

    assert (decision.reason is not None) is warns, decision
    # Fail-open is unconditional: the seam starts the tunnel either way.
    assert decision.allowed is (case in {"absent", "valid", "invalid", "expired_grace"})


def test_license_is_stashed_before_the_link_manager_is_built():
    src = _server_source()
    assert src.index("app.state.license = ") < src.index("build_link_manager(settings")
