"""Tests for the [posthog] dashboard-analytics config section (feat/posthog).

Covers the settings model defaults, TOML load, config-store registration /
round-trip / restart flag / non-redaction of the publishable project key, and
the `/cluster/info` gating logic (emit key+host only when enabled AND set).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from nerdit.config.settings import NerditSettings, PostHogSettings, load_settings
from nerdit.config.store import ConfigStore
from nerdit.daemon.routes.cluster import cluster_info


def test_posthog_defaults_off_and_keyless() -> None:
    ph = NerditSettings().posthog
    # Off by default (opt-in) with no bundled key: nothing phones home until an
    # operator sets BOTH enabled=true and a project_key.
    assert ph.enabled is False
    assert ph.project_key is None
    assert ph.host == "https://us.i.posthog.com"


def test_posthog_inert_when_key_cleared() -> None:
    # Clearing the key opts a deployment out even when explicitly enabled.
    for empty in ("", None):
        ph = PostHogSettings(project_key=empty, enabled=True)
        assert not (ph.enabled and bool(ph.project_key))


def test_posthog_loads_from_toml(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('[posthog]\nproject_key = "phc_abc"\nhost = "https://eu.i.posthog.com"\n')
    ph = load_settings(cfg).posthog
    assert ph.project_key == "phc_abc"
    assert ph.host == "https://eu.i.posthog.com"


def test_posthog_section_is_known_and_key_not_redacted(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('[posthog]\nproject_key = "phc_abc"\n')
    store = ConfigStore(cfg)
    assert "posthog" in store.known_sections()
    # The *project* key is a publishable client-side token: it must stay visible
    # in the config view (a masked key would break the SPA read).
    assert store.effective_section("posthog")["project_key"] == "phc_abc"


def test_posthog_null_clear_stays_inert(tmp_path: Path) -> None:
    """A `null` project_key via the config API clears it durably.

    Regression for the baked-default footgun: with no bundled key, the config
    store's null-delete reverts project_key to the ``None`` default, so an
    operator clearing the key through the API stays inert — even if `enabled`
    is still true — rather than reloading a bundled key on the next restart.
    """
    cfg = tmp_path / "config.toml"
    cfg.write_text('[posthog]\nenabled = true\nproject_key = "phc_operator"\n')
    store = ConfigStore(cfg)
    store.commit(store.stage("posthog", {"project_key": None}))
    assert store.effective_section("posthog").get("project_key") is None
    reloaded = load_settings(cfg).posthog
    assert reloaded.enabled is True
    assert not (reloaded.enabled and bool(reloaded.project_key))


def test_posthog_fields_require_restart(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.toml")
    assert store.stage("posthog", {"enabled": True}).requires_restart is True
    assert store.stage("posthog", {"project_key": "phc_x"}).requires_restart is True
    assert store.stage("posthog", {"host": "https://eu.i.posthog.com"}).requires_restart is True


def _cluster_info(settings: NerditSettings):
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(settings=settings, started_at=datetime.now(UTC)))
    )
    return asyncio.run(cluster_info(request))  # type: ignore[arg-type]


def test_cluster_info_omits_key_when_disabled_or_unset() -> None:
    # Key cleared -> inert even though enabled by default.
    settings = NerditSettings(posthog=PostHogSettings(project_key=""))
    info = _cluster_info(settings)
    assert info.posthog_key is None and info.posthog_host is None

    # Key set but disabled -> still omitted.
    settings = NerditSettings(posthog=PostHogSettings(project_key="phc_x", enabled=False))
    info = _cluster_info(settings)
    assert info.posthog_key is None and info.posthog_host is None


def test_cluster_info_emits_key_when_enabled_and_set() -> None:
    settings = NerditSettings(posthog=PostHogSettings(project_key="phc_x", enabled=True))
    info = _cluster_info(settings)
    assert info.posthog_key == "phc_x"
    assert info.posthog_host == "https://us.i.posthog.com"
