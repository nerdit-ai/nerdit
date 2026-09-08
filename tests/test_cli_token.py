"""CLI tests for the ``nerdit token`` sub-app (P1 / S9).

Verifies the bare ``nerdit token`` (frozen behavior: print the current token) is
preserved by the ``invoke_without_command`` callback, and that the new
``create``/``list``/``revoke`` sub-commands project the admin token API.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from nerdit.cli.app import app
from nerdit.config.settings import ClientSettings, DaemonSettings, NerditSettings

runner = CliRunner()


# --- bare `nerdit token` (frozen behavior, now via the callback) --------------


def test_bare_token_displays_daemon_token():
    settings = NerditSettings(daemon=DaemonSettings(auth_token="test-secret-123"))
    with patch("nerdit.cli.commands.token.load_settings", return_value=settings):
        result = runner.invoke(app, ["token"])
    assert result.exit_code == 0
    assert "test-secret-123" in result.output


def test_bare_token_displays_client_token_when_remote():
    settings = NerditSettings(
        client=ClientSettings(remote_host="10.0.0.1", auth_token="remote-token-456"),
    )
    with patch("nerdit.cli.commands.token.load_settings", return_value=settings):
        result = runner.invoke(app, ["token"])
    assert result.exit_code == 0
    assert "remote-token-456" in result.output


def test_bare_token_no_token():
    settings = NerditSettings()
    with patch("nerdit.cli.commands.token.load_settings", return_value=settings):
        result = runner.invoke(app, ["token"])
    assert result.exit_code == 0
    assert "No authentication token configured" in result.output


# --- create -------------------------------------------------------------------


def test_token_create_prints_plaintext_once_with_warning():
    fake = AsyncMock()
    fake.create_token = AsyncMock(
        return_value={"id": "tok-1", "role": "submitter", "token": "nrd_PLAINTEXT"}
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["token", "create", "ci", "--role", "submitter"])
    assert result.exit_code == 0
    assert "nrd_PLAINTEXT" in result.output
    assert "only once" in result.output
    fake.create_token.assert_awaited_once()
    assert fake.create_token.await_args.args[0] == "ci"
    assert fake.create_token.await_args.kwargs["role"] == "submitter"


def test_token_create_passes_quotas():
    fake = AsyncMock()
    fake.create_token = AsyncMock(return_value={"id": "t", "role": "submitter", "token": "nrd_x"})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(
            app, ["token", "create", "ci", "--max-gpus", "2", "--max-concurrent-jobs", "3"]
        )
    assert result.exit_code == 0
    kwargs = fake.create_token.await_args.kwargs
    assert kwargs["max_gpus"] == 2
    assert kwargs["max_concurrent_jobs"] == 3


# --- list ---------------------------------------------------------------------


def test_token_list_renders_without_secrets():
    fake = AsyncMock()
    fake.list_tokens = AsyncMock(
        return_value=[
            {"id": "tok-1", "name": "ci", "role": "submitter", "revoked": False},
            {"id": "tok-2", "name": "old", "role": "readonly", "revoked": True},
        ]
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["token", "list", "--include-revoked"])
    assert result.exit_code == 0
    assert "tok-1" in result.output
    assert "tok-2" in result.output
    assert "revoked" in result.output
    fake.list_tokens.assert_awaited_once_with(include_revoked=True)


def test_token_list_empty():
    fake = AsyncMock()
    fake.list_tokens = AsyncMock(return_value=[])
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["token", "list"])
    assert result.exit_code == 0
    assert "No tokens" in result.output


# --- revoke -------------------------------------------------------------------


def test_token_revoke():
    fake = AsyncMock()
    fake.revoke_token = AsyncMock(return_value={"id": "tok-1", "revoked": True})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["token", "revoke", "tok-1"])
    assert result.exit_code == 0
    assert "Revoked token tok-1" in result.output
    fake.revoke_token.assert_awaited_once_with("tok-1")


def test_token_revoke_error_exits_nonzero():
    fake = AsyncMock()
    fake.revoke_token = AsyncMock(side_effect=RuntimeError("boom"))
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["token", "revoke", "tok-x"])
    assert result.exit_code == 1


# --- P25: --expires-in / --scope + the expiry column --------------------------


def test_parse_duration_accepts_bare_seconds_and_suffixes():
    from nerdit.cli.commands.token import parse_duration

    assert parse_duration("3600") == 3600
    assert parse_duration("90m") == 5400
    assert parse_duration("24h") == 86400
    assert parse_duration("30d") == 2592000
    assert parse_duration("2w") == 1209600


@pytest.mark.parametrize("bad", ["", "24 hours", "-1", "1y", "abc", "3.5h", "h"])
def test_parse_duration_rejects_anything_else(bad):
    from nerdit.cli.commands.token import parse_duration

    with pytest.raises(ValueError):
        parse_duration(bad)


def test_token_create_passes_expiry_and_scope():
    fake = AsyncMock()
    fake.create_token = AsyncMock(
        return_value={
            "id": "t",
            "role": "submitter",
            "token": "nrd_x",
            "expires_at": "2030-01-01T00:00:00+00:00",
            "scope_services": ["api"],
        }
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(
            app,
            ["token", "create", "ci", "--expires-in", "24h", "--scope", "api", "--scope", "worker"],
        )
    assert result.exit_code == 0
    kwargs = fake.create_token.await_args.kwargs
    assert kwargs["expires_in_s"] == 86400
    assert kwargs["scope_services"] == ["api", "worker"]


def test_token_create_omits_expiry_and_scope_when_not_given():
    """The omitted-vs-null wire semantics: the client must not send ``null``."""
    fake = AsyncMock()
    fake.create_token = AsyncMock(return_value={"id": "t", "role": "submitter", "token": "nrd_x"})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["token", "create", "ci"])
    assert result.exit_code == 0
    kwargs = fake.create_token.await_args.kwargs
    assert kwargs["expires_in_s"] is None
    assert kwargs["scope_services"] is None


def test_token_create_rejects_a_bad_duration_without_calling_the_daemon():
    fake = AsyncMock()
    fake.create_token = AsyncMock()
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["token", "create", "ci", "--expires-in", "forever"])
    assert result.exit_code == 1
    fake.create_token.assert_not_awaited()


def test_token_list_shows_expiry_and_expired_flag():
    fake = AsyncMock()
    fake.list_tokens = AsyncMock(
        return_value=[
            {"id": "tok-1", "name": "ci", "role": "submitter", "revoked": False},
            {
                "id": "tok-2",
                "name": "old",
                "role": "submitter",
                "revoked": False,
                "expires_at": "2020-01-01T00:00:00+00:00",
            },
        ]
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["token", "list"])
    assert result.exit_code == 0
    assert "never" in result.output
    assert "expired" in result.output
    assert "active" in result.output


def test_token_list_never_renders_an_unmapped_style():
    """The P6 MarkupError lesson: a state we did not map must render bare."""
    from nerdit.cli.display import _token_expiry_state

    assert _token_expiry_state(None) == ("never", "active")
    assert _token_expiry_state("not-a-timestamp") == ("not-a-timestamp", "active")


# --- whoami / rotate (P25 WP3, D-P25-4) ---------------------------------------


def _self_view(**overrides) -> dict:
    view = {
        "id": "tok-1",
        "name": "ci",
        "role": "submitter",
        "scope_services": None,
        "expires_at": None,
        "expires_in_s": None,
        "rotatable": True,
    }
    view.update(overrides)
    return view


def test_token_whoami_renders_identity_scope_and_expiry():
    fake = AsyncMock()
    fake.get_self_token = AsyncMock(
        return_value=_self_view(
            scope_services=["api"],
            expires_at="2030-01-01T00:00:00+00:00",
            expires_in_s=99_999_999,
        )
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["token", "whoami"])
    assert result.exit_code == 0
    assert "tok-1" in result.output
    assert "submitter" in result.output
    assert "api" in result.output
    assert "2030-01-01" in result.output
    # Far from expiry: no nag.
    assert "Rotate it now" not in result.output


def test_token_whoami_warns_inside_the_seven_day_window():
    """(D-P25-2 sub-ruling) The warning IS the remediation path — after expiry
    the token cannot rotate itself."""
    fake = AsyncMock()
    fake.get_self_token = AsyncMock(
        return_value=_self_view(expires_at="2030-01-01T00:00:00+00:00", expires_in_s=2 * 86_400)
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["token", "whoami"])
    assert result.exit_code == 0
    assert "expires in 2d" in result.output
    assert "nerdit token rotate" in result.output


def test_token_whoami_of_a_sentinel_principal_says_not_rotatable():
    fake = AsyncMock()
    fake.get_self_token = AsyncMock(
        return_value=_self_view(
            id=None, name="local", role="admin", rotatable=False, expires_in_s=3600
        )
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["token", "whoami"])
    assert result.exit_code == 0
    assert "local/legacy admin" in result.output
    assert "nerdit token create" in result.output


def test_token_whoami_error_exits_nonzero():
    fake = AsyncMock()
    fake.get_self_token = AsyncMock(side_effect=RuntimeError("nope"))
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["token", "whoami"])
    assert result.exit_code == 1


def test_token_rotate_prints_the_plaintext_with_every_warning():
    fake = AsyncMock()
    fake.rotate_self_token = AsyncMock(
        return_value={"id": "tok-1", "role": "submitter", "token": "nrd_ROTATED"}
    )
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["token", "rotate"])
    assert result.exit_code == 0
    assert "nrd_ROTATED" in result.output
    # The three P25 D-P25-4 obligations: store it, update your config, and the
    # honest statement that a lost response needs an admin re-mint.
    assert "shown only once" in result.output
    assert "config.toml" in result.output
    assert "nerdit token create" in result.output
    fake.rotate_self_token.assert_awaited_once_with(extend=False, expires_in_s=None)


def test_token_rotate_extend_passes_the_flag():
    fake = AsyncMock()
    fake.rotate_self_token = AsyncMock(return_value={"id": "t", "role": "submitter", "token": "x"})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["token", "rotate", "--extend"])
    assert result.exit_code == 0
    fake.rotate_self_token.assert_awaited_once_with(extend=True, expires_in_s=None)


def test_token_rotate_expires_in_implies_extend():
    """A lifetime the daemon would ignore without ``extend`` is a silent no-op."""
    fake = AsyncMock()
    fake.rotate_self_token = AsyncMock(return_value={"id": "t", "role": "submitter", "token": "x"})
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["token", "rotate", "--expires-in", "24h"])
    assert result.exit_code == 0
    fake.rotate_self_token.assert_awaited_once_with(extend=True, expires_in_s=86400)


def test_token_rotate_rejects_a_bad_duration_without_calling_the_daemon():
    fake = AsyncMock()
    fake.rotate_self_token = AsyncMock()
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["token", "rotate", "--expires-in", "soon"])
    assert result.exit_code == 1
    fake.rotate_self_token.assert_not_awaited()


def test_token_rotate_error_exits_nonzero():
    fake = AsyncMock()
    fake.rotate_self_token = AsyncMock(side_effect=RuntimeError("403"))
    with patch("nerdit.cli.client.get_configured_client", return_value=fake):
        result = runner.invoke(app, ["token", "rotate"])
    assert result.exit_code == 1


def test_bare_token_is_still_frozen_alongside_the_new_verbs():
    """The new sub-commands must not disturb ``invoke_without_command``."""
    settings = NerditSettings(daemon=DaemonSettings(auth_token="still-here"))
    with patch("nerdit.cli.commands.token.load_settings", return_value=settings):
        result = runner.invoke(app, ["token"])
    assert result.exit_code == 0
    assert "still-here" in result.output
