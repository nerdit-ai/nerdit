"""Upgrade tolerance for config sections this build no longer knows.

`[telemetry]` shipped in 0.5.x and was deleted before the first public
release, so a fielded ``config.toml`` still carries the block. Nothing may
fail on it: not the boot-time settings load, not the config-as-API read of
the whole daemon config, not a declarative apply that does not name it — and
the leftover block survives a write instead of being silently dropped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nerdit.config.settings import load_settings
from nerdit.config.store import ConfigError, ConfigStore

_LEGACY = """\
[daemon]
port = 9321

[telemetry]
enabled = false
project_key = "phc_legacy"

[somethingelse]
future = true
"""


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(_LEGACY)
    return path


def test_settings_load_ignores_a_removed_section(tmp_path: Path) -> None:
    settings = load_settings(_config(tmp_path))

    assert settings.daemon.port == 9321
    assert not hasattr(settings, "telemetry")


def test_removed_section_is_not_a_known_config_section(tmp_path: Path) -> None:
    store = ConfigStore(_config(tmp_path))

    assert "telemetry" not in store.known_sections()
    # Reading it explicitly is a plain 404, the same answer as any typo.
    with pytest.raises(ConfigError) as exc:
        store.view_section("telemetry")
    assert exc.value.status_code == 404
    assert exc.value.code == "config.unknown_section"


def test_whole_config_read_tolerates_the_leftover_block(tmp_path: Path) -> None:
    store = ConfigStore(_config(tmp_path))

    # What GET /config/daemon does: project every known section. The leftover
    # is simply absent from the projection, never an error.
    views = {section: store.view_section(section) for section in store.known_sections()}

    assert views["daemon"]["port"] == 9321
    assert store.current_etag()


def test_apply_not_naming_the_leftover_succeeds_and_preserves_it(tmp_path: Path) -> None:
    path = _config(tmp_path)
    store = ConfigStore(path)

    staged = store.stage_many({"daemon": {"port": 9400}})
    store.commit(staged)

    assert load_settings(path).daemon.port == 9400
    # The unknown blocks are round-tripped, not dropped: a downgrade or a later
    # build that knows them again finds its values intact.
    assert store.load_raw()["telemetry"] == {"enabled": False, "project_key": "phc_legacy"}
    assert store.load_raw()["somethingelse"] == {"future": True}
