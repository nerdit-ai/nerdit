"""Unit tests for the P14c backup/restore CLI command dispatch + rendering.

House pattern (``test_cli_secrets.py``): a Typer app with the command registered,
driven by ``CliRunner`` with the client patched; assertions on the rendered
custody block, the keep==0 hint, the minted Idempotency-Key, and confirm gating.

The P26 WP2 rows at the bottom drive the offline ``_run_restore`` against a real
``create_backup`` tar (the ``test_restore.py`` pattern) — the move-in of the
Caddy ``certificates``/``acme`` trees is only observable end-to-end.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import typer
from typer.testing import CliRunner

from nerdit.cli.commands import backup as backup_mod
from nerdit.cli.commands.backup import _caddy_trees, backup, restore
from nerdit.config.settings import NerditSettings
from nerdit.core.backup import create_backup
from nerdit.core.secrets import SecretManager
from nerdit.db.database import Database

runner = CliRunner()


def _backup_app() -> typer.Typer:
    app = typer.Typer()
    app.command()(backup)
    return app


def _restore_app() -> typer.Typer:
    app = typer.Typer()
    app.command()(restore)
    return app


def _stub_client(**kw) -> SimpleNamespace:
    result = {
        "path": "/data/backups/nerdit-backup-20260713T000000Z-abcdef.tar.gz",
        "size_bytes": 4096,
        "kid": "deadbeef",
    }
    result.update(kw)
    return SimpleNamespace(create_backup=AsyncMock(return_value=result))


def test_backup_yes_renders_custody_and_keep_hint(monkeypatch):
    stub = _stub_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = runner.invoke(_backup_app(), ["--yes"])
    assert res.exit_code == 0
    assert "deadbeef" in res.output  # kid rendered
    assert "secrets.key" in res.output  # custody block
    assert "backup_keep_last" in res.output  # M4 keep==0 honesty hint
    # An Idempotency-Key was minted at the command layer.
    stub.create_backup.assert_awaited_once()
    assert stub.create_backup.await_args.kwargs["idempotency_key"]


def test_backup_confirm_declined_makes_no_call(monkeypatch):
    stub = _stub_client()
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = runner.invoke(_backup_app(), input="n\n")
    assert res.exit_code == 0
    assert "Aborted" in res.output
    stub.create_backup.assert_not_awaited()


def test_backup_client_error_exits_1(monkeypatch):
    stub = SimpleNamespace(create_backup=AsyncMock(side_effect=RuntimeError("daemon down")))
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = runner.invoke(_backup_app(), ["--yes"])
    assert res.exit_code == 1


def test_volume_backup_points_at_the_logical_flavour(monkeypatch):
    """(P37, D-P37-7) The two flavours point at each other, once each.

    ``--volume`` is the PHYSICAL capture; an operator who actually wanted a
    portable, application-consistent one wanted ``nerdit db dump``, and this
    line is the only place they will find that out in time.
    """
    stub = SimpleNamespace(
        create_volume_backup=AsyncMock(
            return_value={
                "path": "/data/backups/nerdit-volumes-pg-20260907T000000Z-abcdef.tar.gz",
                "size_bytes": 8192,
            }
        )
    )
    monkeypatch.setattr("nerdit.cli.client.get_configured_client", lambda: stub)
    res = runner.invoke(_backup_app(), ["--volume", "pg", "--yes"])
    assert res.exit_code == 0
    assert "Volume backup written" in res.output
    assert "nerdit db dump pg" in res.output
    # The physical/logical distinction is stated, not merely implied by a verb.
    assert "application-consistent" in res.output
    # The volume tar's own custody + retention lines are untouched by the addition.
    assert "SCRAM password verifiers" in res.output
    assert "volume_backup_keep_last" in res.output


def test_restore_missing_tar_exits_1(tmp_path):
    res = runner.invoke(_restore_app(), [str(tmp_path / "nope.tar.gz")])
    assert res.exit_code == 1
    assert "No such backup archive" in res.output


# --------------------------------------------------------------------------- #
# P26 WP2 — restore swaps the Caddy ACME trees, and only when the tar has them
# --------------------------------------------------------------------------- #


async def _build_backup(data_dir: Path, *, with_acme: bool):
    """Populate a data dir and produce a real control-plane tar."""
    db = Database(str(data_dir / "nerdit.db"))
    await db.connect()
    await db.init_schema()
    mgr = SecretManager(data_dir / "secrets")
    mgr.set("app", {"API_KEY": "abc"})
    if with_acme:
        leaf = data_dir / "caddy" / "certificates" / "127.0.0.1-14000-dir" / "app.example.test"
        leaf.mkdir(parents=True)
        (leaf / "app.example.test.crt").write_text("TAR-LEAF")
        (leaf / "app.example.test.key").write_text("TAR-LEAF-KEY")
        account = data_dir / "caddy" / "acme" / "127.0.0.1-14000-dir" / "users" / "ops@x.test"
        account.mkdir(parents=True)
        (account / "admin.key").write_text("TAR-ACCOUNT-KEY")
    result = await create_backup(db=db, secret_manager=mgr, data_dir=data_dir)
    await db.close()
    return data_dir / "backups" / result.basename, result.manifest


def _restore_settings(data_dir: Path) -> NerditSettings:
    return NerditSettings(
        data_dir=str(data_dir),
        daemon={"pid_file": str(data_dir / "nonexistent.pid"), "port": 59998},
    )


def _patch_offline(monkeypatch, settings: NerditSettings) -> None:
    """Make ``_run_restore`` believe no daemon is running (both legs)."""
    monkeypatch.setattr(backup_mod, "load_settings", lambda: settings)
    monkeypatch.setattr(backup_mod, "_health_probe", lambda port: False)


def test_restore_replaces_caddy_certificates_and_acme(tmp_path, monkeypatch):
    """A WP2 tar owns the Caddy trees: the live ones are replaced wholesale, so
    a stale leaf left behind on the target never survives the restore."""
    source = tmp_path / "source"
    source.mkdir()
    tar_path, manifest = asyncio.run(_build_backup(source, with_acme=True))
    assert manifest["contents"]["caddy_certificates"] is True
    assert manifest["contents"]["caddy_acme"] is True

    target = tmp_path / "target"
    stale = target / "caddy" / "certificates" / "stale-dir" / "old.example.test"
    stale.mkdir(parents=True)
    (stale / "old.example.test.crt").write_text("STALE-LEAF")

    _patch_offline(monkeypatch, _restore_settings(target))
    res = runner.invoke(_restore_app(), [str(tar_path), "--yes"])
    assert res.exit_code == 0, res.output

    certs = target / "caddy" / "certificates"
    assert (
        certs / "127.0.0.1-14000-dir" / "app.example.test" / "app.example.test.crt"
    ).read_text() == "TAR-LEAF"
    assert not (certs / "stale-dir").exists()  # replaced, not merged
    account = target / "caddy" / "acme" / "127.0.0.1-14000-dir" / "users" / "ops@x.test"
    assert (account / "admin.key").read_text() == "TAR-ACCOUNT-KEY"
    # The plan block names the trees the archive carries.
    assert "certificates" in res.output and "acme" in res.output


def test_restore_from_pre_wp2_tar_leaves_live_acme_trees_alone(tmp_path, monkeypatch):
    """An archive predating WP2 carries no caddy/certificates or caddy/acme
    members. Absent must mean "this tar says nothing", never "delete them" —
    wiping them would force a re-issuance the CA's duplicate limit punishes."""
    source = tmp_path / "source"
    source.mkdir()
    tar_path, manifest = asyncio.run(_build_backup(source, with_acme=False))
    assert manifest["contents"]["caddy_certificates"] is False
    assert manifest["contents"]["caddy_acme"] is False

    target = tmp_path / "target"
    live_leaf = target / "caddy" / "certificates" / "live-dir" / "live.example.test"
    live_leaf.mkdir(parents=True)
    (live_leaf / "live.example.test.crt").write_text("LIVE-LEAF")
    live_account = target / "caddy" / "acme" / "live-dir" / "users" / "ops@live.test"
    live_account.mkdir(parents=True)
    (live_account / "admin.key").write_text("LIVE-ACCOUNT-KEY")

    _patch_offline(monkeypatch, _restore_settings(target))
    res = runner.invoke(_restore_app(), [str(tar_path), "--yes"])
    assert res.exit_code == 0, res.output

    assert (live_leaf / "live.example.test.crt").read_text() == "LIVE-LEAF"
    assert (live_account / "admin.key").read_text() == "LIVE-ACCOUNT-KEY"
    assert "caddy:        none" in res.output


def test_caddy_trees_renders_only_declared_flags():
    """The plan line is driven by the manifest, and an unknown/absent flag set
    renders honestly rather than claiming a tree the tar does not hold."""
    assert _caddy_trees({"contents": {"caddy_pki": True, "caddy_acme": True}}) == "pki, acme"
    assert _caddy_trees({"contents": {}}) == "none"
    assert _caddy_trees({}) == "none"
