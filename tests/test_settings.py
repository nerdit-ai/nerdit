"""Tests for host-level daemon settings."""

import pytest
from pydantic import ValidationError

from nerdit.config.settings import load_settings


def test_monitor_zml_smi_path_defaults_to_path_lookup(tmp_path):
    assert load_settings(tmp_path / "missing.toml").monitor.zml_smi_path == "zml-smi"


def test_monitor_zml_smi_path_loads_from_config(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[monitor]\nzml_smi_path = "/opt/zml/bin/zml-smi"\n')

    assert load_settings(config).monitor.zml_smi_path == "/opt/zml/bin/zml-smi"


def test_amd_settings_defaults():
    from nerdit.config.settings import NerditSettings

    settings = NerditSettings()
    assert settings.monitor.enable_amd is False


def test_amd_settings_from_toml(tmp_path):
    from nerdit.config.settings import load_settings

    config_file = tmp_path / "config.toml"
    config_file.write_text("[monitor]\nenable_amd = true\n")
    settings = load_settings(config_file)
    assert settings.monitor.enable_amd is True


# --- [services] settings (P2 / S4) -------------------------------------------


def test_services_settings_defaults():
    from nerdit.config.settings import NerditSettings

    settings = NerditSettings()
    assert settings.services.service_port_range == "9400-9499"
    assert settings.services.service_max_restarts == 3
    assert settings.services.restart_window_seconds == 300


def test_services_settings_from_toml(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("[services]\nservice_port_range = '9500-9599'\nservice_max_restarts = 5\n")
    settings = load_settings(config)
    assert settings.services.service_port_range == "9500-9599"
    assert settings.services.service_max_restarts == 5


def test_services_validator_rejects_malformed_range():
    import pytest
    from pydantic import ValidationError

    from nerdit.config.settings import ServicesSettings

    with pytest.raises(ValidationError):
        ServicesSettings(service_port_range="nope")


def test_services_validator_rejects_range_with_daemon_port():
    import pytest
    from pydantic import ValidationError

    from nerdit.config.settings import ServicesSettings

    with pytest.raises(ValidationError):
        ServicesSettings(service_port_range="9300-9400")


def test_services_validator_rejects_inverted_range():
    import pytest
    from pydantic import ValidationError

    from nerdit.config.settings import ServicesSettings

    with pytest.raises(ValidationError):
        ServicesSettings(service_port_range="9500-9400")


# --- [proxy] base_domain / hostname_override (P3.5 / D5 / S5) ----------------


def test_proxy_dns_fields_default_to_none():
    from nerdit.config.settings import ProxySettings

    settings = ProxySettings()
    assert settings.base_domain is None
    assert settings.hostname_override is None


def test_proxy_dns_fields_accept_plain_names():
    from nerdit.config.settings import ProxySettings

    settings = ProxySettings(base_domain="apps.lan.local", hostname_override="box.lan")
    assert settings.base_domain == "apps.lan.local"
    assert settings.hostname_override == "box.lan"


def test_proxy_dns_fields_accept_ipv4_literal():
    # hostname_override's contract is "name/IP" (settings.py docstring) — IPv4
    # octets pass the DNS label grammar naturally and must stay legal.
    from nerdit.config.settings import ProxySettings

    settings = ProxySettings(hostname_override="192.168.1.10")
    assert settings.hostname_override == "192.168.1.10"


def test_proxy_dns_fields_accept_single_label():
    # base_domain="localhost" (*.localhost dev setup, D6) is a single DNS label.
    from nerdit.config.settings import ProxySettings

    settings = ProxySettings(base_domain="localhost")
    assert settings.base_domain == "localhost"


@pytest.mark.parametrize(
    "bad",
    [
        "Lan.Example",  # mixed case — rejected, not silently lowercased (D5 Q3)
        "lan:8443",  # port
        "*.lan",  # wildcard
        " lan ",  # whitespace
        "lan example",  # internal whitespace
        "https://lan",  # scheme
        ".lan",  # leading dot
        "lan.",  # trailing dot
        "-lan.local",  # leading hyphen in a label
        "lan-.local",  # trailing hyphen in a label
        "lan..local",  # empty label
        "a" * 254,  # exceeds 253 chars
    ],
)
def test_proxy_base_domain_rejects_malformed(bad):
    from pydantic import ValidationError

    from nerdit.config.settings import ProxySettings

    with pytest.raises(ValidationError):
        ProxySettings(base_domain=bad)


def test_proxy_hostname_override_rejects_malformed():
    from pydantic import ValidationError

    from nerdit.config.settings import ProxySettings

    with pytest.raises(ValidationError):
        ProxySettings(hostname_override="Box.Lan")
    with pytest.raises(ValidationError):
        ProxySettings(hostname_override="box:9000")


# --- [proxy] extra_hostnames — the SAN registry (P25 WP5 / D-P25-9) -----------


def test_proxy_extra_hostnames_defaults_to_empty_list():
    from nerdit.config.settings import ProxySettings

    assert ProxySettings().extra_hostnames == []


def test_proxy_extra_hostnames_accepts_names_and_ip_literals():
    # IP literals are the motivating case: reaching the box as 192.168.1.50 with
    # a valid cert (D-P25-9).
    from nerdit.config.settings import ProxySettings

    settings = ProxySettings(extra_hostnames=["192.168.1.50", "nerd-box.local", "box"])
    assert settings.extra_hostnames == ["192.168.1.50", "nerd-box.local", "box"]


def test_proxy_extra_hostnames_dedupes_preserving_order():
    from nerdit.config.settings import ProxySettings

    settings = ProxySettings(extra_hostnames=["a.lan", "b.lan", "a.lan"])
    assert settings.extra_hostnames == ["a.lan", "b.lan"]


def test_proxy_extra_hostnames_rejects_wildcard_with_its_own_message():
    # A wildcard is a base_domain concern — including the exact subject
    # ``_tls_subjects`` derives in subdomain mode, which would only be deduped
    # away. Rejecting keeps the config self-documenting.
    from pydantic import ValidationError

    from nerdit.config.settings import ProxySettings

    with pytest.raises(ValidationError) as exc:
        ProxySettings(mode="subdomain", base_domain="lan.local", extra_hostnames=["*.lan.local"])
    assert "base_domain" in str(exc.value)
    with pytest.raises(ValidationError):
        ProxySettings(extra_hostnames=["*.other"])


@pytest.mark.parametrize(
    "bad",
    [
        "Box.Lan",  # mixed case — rejected, not silently lowercased
        "",  # empty entry
        "box:8443",  # port
        "https://box",  # scheme
        "box.lan/app",  # path
        " box ",  # whitespace
        "box lan",  # internal whitespace
        ".box",  # leading dot
        "box.",  # trailing dot
    ],
)
def test_proxy_extra_hostnames_rejects_malformed(bad):
    from pydantic import ValidationError

    from nerdit.config.settings import ProxySettings

    with pytest.raises(ValidationError):
        ProxySettings(extra_hostnames=[bad])


def test_proxy_extra_hostnames_caps_at_32_entries():
    from pydantic import ValidationError

    from nerdit.config.settings import ProxySettings

    ok = [f"n{i}.lan" for i in range(32)]
    assert len(ProxySettings(extra_hostnames=ok).extra_hostnames) == 32
    with pytest.raises(ValidationError):
        ProxySettings(extra_hostnames=[*ok, "n32.lan"])


def test_proxy_extra_hostnames_from_toml(tmp_path):
    # The registry is a hand-editable TOML array (the CLI `config set` grammar
    # is scalar-only, as for every other list-valued key such as
    # [git].allowed_hosts) — pin the load path.
    config = tmp_path / "config.toml"
    config.write_text(
        '[proxy]\nenabled = true\nextra_hostnames = ["192.168.1.50", "nerd-box.local"]\n'
    )
    settings = load_settings(config)
    assert settings.proxy.extra_hostnames == ["192.168.1.50", "nerd-box.local"]


def test_proxy_extra_hostnames_is_restart_required():
    # Every [proxy] key is bound at startup; the SAN registry is no exception
    # (live, no-restart TLS sync is deferred — D-P25-9).
    from nerdit.config.store import _RESTART_KEYS

    assert "extra_hostnames" in _RESTART_KEYS["proxy"]


# --- [retention] section (P14b) -----------------------------------------------


def test_retention_settings_defaults_when_section_absent(tmp_path):
    # No [retention] block ⇒ the model defaults are used.
    settings = load_settings(tmp_path / "missing.toml")
    r = settings.retention
    assert r.job_log_days == 14
    assert r.audit_days == 90
    assert r.audit_archive_dir == ""
    assert r.sweep_interval_seconds == 3600
    assert r.daemon_log_max_bytes == 10_485_760
    assert r.daemon_log_backups == 3
    assert r.container_log_max_size == "10m"
    assert r.container_log_max_file == 3
    assert r.backup_keep_last == 0


def test_retention_settings_from_toml(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        '[retention]\njob_log_days = 30\ncontainer_log_max_size = "50m"\nbackup_keep_last = 5\n'
    )
    r = load_settings(config).retention
    assert r.job_log_days == 30
    assert r.container_log_max_size == "50m"
    assert r.backup_keep_last == 5
    # Unset keys keep their defaults.
    assert r.audit_days == 90


def test_retention_audit_days_validator():
    from pydantic import ValidationError

    from nerdit.config.settings import RetentionSettings

    assert RetentionSettings(audit_days=0).audit_days == 0
    assert RetentionSettings(audit_days=7).audit_days == 7
    with pytest.raises(ValidationError):
        RetentionSettings(audit_days=3)
    with pytest.raises(ValidationError):
        RetentionSettings(audit_days=-1)


def test_retention_audit_archive_dir_validator():
    from pydantic import ValidationError

    from nerdit.config.settings import RetentionSettings

    assert RetentionSettings(audit_archive_dir="").audit_archive_dir == ""
    assert RetentionSettings(audit_archive_dir="/srv/archive").audit_archive_dir == "/srv/archive"
    with pytest.raises(ValidationError):
        RetentionSettings(audit_archive_dir="relative/archive")


def test_retention_container_log_max_size_validator():
    from pydantic import ValidationError

    from nerdit.config.settings import RetentionSettings

    for good in ("10m", "500k", "2g"):
        assert RetentionSettings(container_log_max_size=good).container_log_max_size == good
    # (F7) "" is the explicit opt-out: no log_config ⇒ Docker's default driver
    # is preserved. Whitespace normalizes to the same empty sentinel.
    for empty in ("", "   "):
        assert RetentionSettings(container_log_max_size=empty).container_log_max_size == ""
    for bad in ("10", "10mb", "abc", "10M"):
        with pytest.raises(ValidationError):
            RetentionSettings(container_log_max_size=bad)


def test_retention_empty_container_log_size_round_trips_from_toml(tmp_path):
    """(F7) The empty opt-out survives a TOML load unchanged."""
    from nerdit.config.settings import RetentionSettings

    config_file = tmp_path / "config.toml"
    config_file.write_text('[retention]\ncontainer_log_max_size = ""\n')

    settings = load_settings(config_file)

    assert settings.retention.container_log_max_size == ""
    # The default is unchanged (bounded by default stays the P14b decision).
    assert RetentionSettings().container_log_max_size == "10m"


def test_retention_daemon_log_backups_requires_rotation():
    """(F13) A byte cap with 0 backups never rotates ⇒ reject the combination.

    stdlib RotatingFileHandler.doRollover skips the rename chain entirely when
    backupCount == 0 and just reopens the base file in append mode, so the log
    grows without bound while looking capped.
    """
    from pydantic import ValidationError

    from nerdit.config.settings import RetentionSettings

    # File logging off ⇒ backups are moot.
    assert RetentionSettings(daemon_log_max_bytes=0, daemon_log_backups=0).daemon_log_backups == 0
    # Rotation on ⇒ at least one backup required.
    with pytest.raises(ValidationError):
        RetentionSettings(daemon_log_max_bytes=1024, daemon_log_backups=0)
    ok = RetentionSettings(daemon_log_max_bytes=1024, daemon_log_backups=1)
    assert ok.daemon_log_backups == 1
    # Defaults (10 MiB / 3) stay valid.
    assert RetentionSettings().daemon_log_backups == 3


def test_retention_numeric_bounds():
    from pydantic import ValidationError

    from nerdit.config.settings import RetentionSettings

    with pytest.raises(ValidationError):
        RetentionSettings(sweep_interval_seconds=30)
    with pytest.raises(ValidationError):
        RetentionSettings(container_log_max_file=0)
    with pytest.raises(ValidationError):
        RetentionSettings(job_log_days=-1)
    with pytest.raises(ValidationError):
        RetentionSettings(backup_keep_last=-1)


# --- [databases] section (P15) ------------------------------------------------


def test_databases_settings_defaults_when_section_absent(tmp_path):
    # No [databases] block ⇒ the model defaults are used.
    from nerdit.config.settings import (
        DEFAULT_DATABASES_BACKEND,
        DEFAULT_DB_READY_TIMEOUT_S,
        DEFAULT_DB_START_PERIOD_S,
        DEFAULT_POSTGRES_IMAGE,
        DEFAULT_REDIS_IMAGE,
    )

    d = load_settings(tmp_path / "missing.toml").databases
    assert d.default_backend == DEFAULT_DATABASES_BACKEND
    assert d.postgres_image == DEFAULT_POSTGRES_IMAGE
    assert d.redis_image == DEFAULT_REDIS_IMAGE
    assert d.start_period_s == DEFAULT_DB_START_PERIOD_S
    assert d.ready_timeout_s == DEFAULT_DB_READY_TIMEOUT_S


def test_databases_settings_from_toml(tmp_path):
    # A [databases] section in config.toml must round-trip through
    # load_settings into NerditSettings.databases (regression: the section was
    # silently dropped by load_settings — P15 C4).
    config = tmp_path / "config.toml"
    config.write_text(
        "[databases]\n"
        'default_backend = "redis"\n'
        'postgres_image = "postgres:17"\n'
        'redis_image = "redis:8"\n'
        "start_period_s = 45\n"
        "ready_timeout_s = 7\n"
    )
    d = load_settings(config).databases
    assert d.default_backend == "redis"
    assert d.postgres_image == "postgres:17"
    assert d.redis_image == "redis:8"
    assert d.start_period_s == 45
    assert d.ready_timeout_s == 7


# --- legacy [scheduler] compat (Track A / WP6) --------------------------------


def test_legacy_scheduler_section_still_loads(tmp_path):
    """A pre-Track-A ``config.toml`` (every one of which carries the ``[scheduler]``
    section ``nerdit init`` used to write) must still load cleanly — the section is
    simply ignored now that ``SchedulerSettings`` is gone."""
    config = tmp_path / "config.toml"
    config.write_text(
        "[scheduler]\n"
        "default_priority = 5\n"
        "max_concurrent_jobs = 0\n"
        "enable_amd = true\n"
        "loop_interval = 4.0\n"
        "\n"
        "[monitor]\n"
        "interval_seconds = 7\n"
    )
    settings = load_settings(config)
    assert not hasattr(settings, "scheduler")
    assert settings.monitor.interval_seconds == 7
    # The relocated keys are NOT read back out of the stale section: they now
    # live under [monitor]/[services] and fall back to their defaults.
    assert settings.monitor.enable_amd is False
    assert settings.services.loop_interval == 2.0


def test_legacy_scheduler_section_survives_a_config_apply(tmp_path):
    """R-9: a config write to a surviving section leaves the stale ``[scheduler]``
    section byte-for-byte intact, so the daemon never rewrites a user's file into
    something an older nerdit could not read."""
    from nerdit.config.store import ConfigStore

    config = tmp_path / "config.toml"
    config.write_text("[scheduler]\ndefault_priority = 5\n\n[monitor]\ninterval_seconds = 5\n")
    store = ConfigStore(config)
    store.commit(store.stage("monitor", {"interval_seconds": 9}))

    raw = store.load_raw()
    assert raw["scheduler"] == {"default_priority": 5}
    assert raw["monitor"]["interval_seconds"] == 9
    # And the applied file still loads.
    assert load_settings(config).monitor.interval_seconds == 9


def test_legacy_scheduler_section_is_not_a_writable_api_section(tmp_path):
    """The stale section is tolerated on disk but is no longer part of the
    config-as-API surface: a targeted read/write 404s as ``config.unknown_section``."""
    import pytest as _pytest

    from nerdit.config.store import ConfigError, ConfigStore

    store = ConfigStore(tmp_path / "config.toml")
    assert "scheduler" not in store.known_sections()
    with _pytest.raises(ConfigError) as exc:
        store.view_section("scheduler")
    assert exc.value.status_code == 404
    assert exc.value.code == "config.unknown_section"


# --- [proxy.acme] — public certificates (P26 WP2 / D-P26-5) -------------------


def test_proxy_acme_defaults_when_the_block_is_absent(tmp_path):
    """A pre-WP2 config has no ``[proxy.acme]`` table at all: the nested model
    must default in, off, so every reader can say ``settings.proxy.acme.enabled``
    without a getattr dance."""
    settings = load_settings(tmp_path / "missing.toml")

    assert settings.proxy.acme.enabled is False
    assert settings.proxy.acme.email is None
    assert settings.proxy.acme.http_port == 80


def test_proxy_acme_boot_refuses_a_config_that_would_not_bind(tmp_path):
    """Load-time refusal, not a Caddy bind failure: Caddy declines to start when
    ANY listener fails, so a port collision would take down the whole proxy."""
    config = tmp_path / "config.toml"
    config.write_text(
        "[proxy]\n"
        "enabled = true\n"
        "https_port = 8443\n"
        "\n"
        "[proxy.acme]\n"
        "enabled = true\n"
        'email = "ops@example.com"\n'
        "http_port = 8443\n"
    )

    with pytest.raises(ValidationError):
        load_settings(config)


def test_proxy_acme_is_restart_required():
    """The whole block is one restart key — ``ProxyManager`` captures
    ``ProxySettings`` at construction and the ``:80`` server lives in the
    bootstrap config Caddy is spawned with."""
    from nerdit.config.store import _RESTART_KEYS

    assert "acme" in _RESTART_KEYS["proxy"]


# --- [daemon].auth_token charset ---------------------------------------------


def test_non_ascii_auth_token_fails_at_load(tmp_path):
    """A hand-edited non-ASCII token is refused at boot rather than silently
    matching nothing: the Authorization header carries no charset that would
    round-trip it, so every request would 403."""
    config = tmp_path / "config.toml"
    config.write_text('[daemon]\nauth_token = "café"\n')

    with pytest.raises(ValidationError, match="ASCII"):
        load_settings(config)


def test_non_ascii_auth_token_hint_points_at_the_file_not_a_command(tmp_path):
    """The refusal must name a repair that exists.

    Every CLI verb loads this file, so a wedged token means no shipped command
    can run at all: `nerdit token` only DISPLAYS the configured value and
    `nerdit init --auth-token-only` leaves a non-empty token alone. Telling the
    operator to "generate one with 'nerdit token'" sent them in a circle.
    """
    config = tmp_path / "config.toml"
    config.write_text('[daemon]\nauth_token = "caf\u00e9"\n')

    with pytest.raises(ValidationError) as excinfo:
        load_settings(config)

    message = str(excinfo.value)
    assert "[daemon].auth_token" in message
    assert "by hand" in message
    assert "Generate one with 'nerdit token'" not in message


def test_ascii_auth_token_loads(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[daemon]\nauth_token = "nrd_abc123"\n')

    assert load_settings(config).daemon.auth_token == "nrd_abc123"
