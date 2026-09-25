"""ConfigStore tests (P1 / S8).

Covers the pure config-as-API core: validation + diagnostics, dry-run diffing
without writing, atomic writes that preserve unknown sections, secret redaction
in views and diffs, the ETag, and the ``auth_token`` write block.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nerdit.config.redaction import REDACTED, redact_section
from nerdit.config.settings import load_settings
from nerdit.config.store import ConfigError, ConfigStore


def _store(tmp_path: Path, body: str = "") -> ConfigStore:
    path = tmp_path / "config.toml"
    if body:
        path.write_text(body)
    return ConfigStore(path)


# --- redaction source ---------------------------------------------------------


def test_redact_section_masks_secrets():
    out = redact_section({"port": 1, "auth_token": "abc", "api_key": "k"})
    assert out == {"port": 1, "auth_token": REDACTED, "api_key": REDACTED}


def test_redact_section_keeps_unset_secret_none():
    # An unset secret must not be reported as if a value existed.
    assert redact_section({"auth_token": None})["auth_token"] is None


def test_redact_section_strips_embedded_url_credentials():
    url = "https://user:password@example.com/v1"
    assert redact_section({"base_url": url, "url": url}) == {
        "base_url": "https://example.com/v1",
        "url": "https://example.com/v1",
    }


# --- known sections / unknown section ----------------------------------------


def test_known_sections_are_daemon_only(tmp_path):
    sections = _store(tmp_path).known_sections()
    assert "daemon" in sections and "monitor" in sections
    # No per-app config in P1.
    assert all(not s.startswith("app") for s in sections)


def test_view_unknown_section_raises_404(tmp_path):
    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).view_section("nope")
    assert exc.value.status_code == 404
    assert exc.value.code == "config.unknown_section"


# --- effective values / defaults ---------------------------------------------


def test_effective_section_fills_defaults(tmp_path):
    store = _store(tmp_path, "[monitor]\ninterval_seconds = 9\n")
    values = store.effective_section("monitor")
    assert values["interval_seconds"] == 9
    # Defaults are filled for un-set keys.
    assert values["enable_amd"] is False


def test_view_section_redacts_secrets(tmp_path):
    store = _store(tmp_path, '[daemon]\nauth_token = "topsecret"\n')
    assert store.view_section("daemon")["auth_token"] == REDACTED


# --- validation / diagnostics ------------------------------------------------


def test_stage_invalid_value_raises_422_with_diagnostics(tmp_path):
    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).stage("daemon", {"port": "not-an-int"})
    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"
    assert exc.value.diagnostics
    assert ["port"] in [d.loc for d in exc.value.diagnostics]


def test_stage_blocks_auth_token_write(tmp_path):
    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).stage("daemon", {"auth_token": "x"})
    assert exc.value.status_code == 403
    assert exc.value.code == "config.forbidden"


def test_stage_rejects_unknown_key(tmp_path):
    """A typo'd/unsupported key must 422, not silently no-op (PR review #42)."""
    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).stage("daemon", {"prot": 9000})  # typo for 'port'
    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"
    assert ["prot"] in [d.loc for d in exc.value.diagnostics]


def test_stage_rejects_unknown_key_alongside_valid_ones(tmp_path):
    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).stage("daemon", {"port": 9000, "bogus": True})
    assert exc.value.code == "config.invalid"
    assert ["bogus"] in [d.loc for d in exc.value.diagnostics]


# --- diff + dry-run (no write) -----------------------------------------------


def test_stage_computes_diff_without_writing(tmp_path):
    store = _store(tmp_path, "[monitor]\ninterval_seconds = 5\n")
    staged = store.stage("monitor", {"interval_seconds": 9})
    assert len(staged.diff) == 1
    entry = staged.diff[0]
    assert entry.key == "monitor.interval_seconds"
    assert entry.old == 5 and entry.new == 9
    # No write happened: the file still holds the old value.
    assert store.effective_section("monitor")["interval_seconds"] == 5


def test_diff_redaction_is_defensive_via_redact_value():
    # The diff builder masks secret leaves through the single redaction source,
    # so a secret value can never surface in a diff entry.
    from nerdit.config.redaction import redact_value

    assert redact_value("auth_token", "plaintext") == REDACTED
    assert redact_value("remote_host", "10.0.0.1") == "10.0.0.1"


def test_diff_for_non_secret_change_is_clean(tmp_path):
    store = _store(tmp_path, '[client]\nremote_host = "old"\n')
    staged = store.stage("client", {"remote_host": "10.0.0.1"})
    assert [e.key for e in staged.diff] == ["client.remote_host"]
    assert all(REDACTED not in (e.old, e.new) for e in staged.diff)


def test_requires_restart_for_host_port(tmp_path):
    store = _store(tmp_path, "[daemon]\nport = 9321\n")
    assert store.stage("daemon", {"port": 9999}).requires_restart is True
    # [client] is re-read per CLI invocation, so it is honestly not restart-keyed.
    assert store.stage("client", {"remote_host": "10.0.0.1"}).requires_restart is False


@pytest.mark.parametrize(
    ("section", "body", "keys"),
    [
        # Sandbox hardening: reported inert until restart, because it is.
        ("containers", {"read_only_rootfs": True}, ["containers.read_only_rootfs"]),
        (
            "containers",
            {"drop_all_caps": False, "default_memory_limit": "256m"},
            ["containers.default_memory_limit", "containers.drop_all_caps"],
        ),
        ("nerdit", {"log_level": "debug"}, ["nerdit.log_level"]),
        ("nerdit", {"data_dir": "~/elsewhere"}, ["nerdit.data_dir"]),
        ("monitor", {"interval_seconds": 3}, ["monitor.interval_seconds"]),
        ("monitor", {"zml_smi_path": "/opt/zml-smi"}, ["monitor.zml_smi_path"]),
        ("monitor", {"gpu_temp_warning": 70}, ["monitor.gpu_temp_warning"]),
    ],
)
def test_boot_frozen_sections_report_requires_restart(tmp_path, section, body, keys):
    # Every key of [containers]/[nerdit]/[monitor] is captured once at boot, so a
    # PUT that answered requires_restart=false would tell the operator the change
    # is live when it is not.
    staged = _store(tmp_path).stage(section, body)
    assert staged.requires_restart is True
    assert sorted(staged.restart_keys) == keys
    assert all(entry.requires_restart is True for entry in staged.diff)


def test_dead_container_knobs_are_gone(tmp_path):
    # runtime/cache_dir had no consumer; deleted rather than minted a restart key.
    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).stage("containers", {"runtime": "podman"})
    assert exc.value.code == "config.invalid"


# --- atomic commit + preservation --------------------------------------------


def test_commit_persists_and_changes_etag(tmp_path):
    store = _store(tmp_path, "[monitor]\ninterval_seconds = 5\n")
    before = store.current_etag()
    new_etag = store.commit(store.stage("monitor", {"interval_seconds": 7}))
    assert new_etag != before
    assert new_etag == store.current_etag()
    assert store.effective_section("monitor")["interval_seconds"] == 7


def test_commit_preserves_unknown_sections(tmp_path):
    store = _store(tmp_path, "[future]\nfoo = 1\n[monitor]\ninterval_seconds = 5\n")
    store.commit(store.stage("monitor", {"interval_seconds": 8}))
    raw = store.load_raw()
    # An unknown future section is untouched by a daemon-section write.
    assert raw["future"] == {"foo": 1}
    assert raw["monitor"]["interval_seconds"] == 8


def test_commit_strips_none_so_toml_serializes(tmp_path):
    # A model with None-default optional fields must still serialize.
    store = _store(tmp_path)
    store.commit(store.stage("daemon", {"port": 8080}))
    raw = store.load_raw()
    assert raw["daemon"] == {"port": 8080}


def test_partial_write_preserves_sibling_keys_incl_auth_token(tmp_path):
    """H1 regression: a partial [daemon] write must not strip auth_token/host.

    ``auth_token`` cannot be written via the API; replacing the section instead
    of merging it would silently drop the persisted token, dropping the daemon
    into the unauthenticated ``token=None`` admin path on the next restart.
    """
    store = _store(
        tmp_path,
        '[daemon]\nauth_token = "super-secret"\nhost = "0.0.0.0"\nport = 9321\n',
    )
    store.commit(store.stage("daemon", {"port": 9000}))
    raw = store.load_raw()
    assert raw["daemon"]["port"] == 9000
    assert raw["daemon"]["auth_token"] == "super-secret"
    assert raw["daemon"]["host"] == "0.0.0.0"


def test_commit_temp_file_yields_owner_only_config(tmp_path):
    """L6 regression: the written config is not world/group readable."""
    store = _store(tmp_path, "[monitor]\ninterval_seconds = 5\n")
    store.commit(store.stage("monitor", {"interval_seconds": 6}))
    mode = (tmp_path / "config.toml").stat().st_mode & 0o077
    assert mode == 0, f"config.toml is group/world accessible: {oct(mode)}"


def test_etag_empty_file_is_stable(tmp_path):
    store = _store(tmp_path)
    assert store.current_etag() == store.current_etag()


# --- [services] section (P2 / S4) --------------------------------------------


def test_services_section_is_known(tmp_path):
    assert "services" in _store(tmp_path).known_sections()


def test_services_effective_defaults(tmp_path):
    values = _store(tmp_path).effective_section("services")
    assert values["service_port_range"] == "9400-9499"
    assert values["service_max_restarts"] == 3
    assert values["restart_window_seconds"] == 300


def test_services_config_api_round_trip(tmp_path):
    store = _store(tmp_path)
    store.commit(store.stage("services", {"service_port_range": "9500-9599"}))
    raw = store.load_raw()
    assert raw["services"]["service_port_range"] == "9500-9599"
    assert store.effective_section("services")["service_port_range"] == "9500-9599"


def test_services_port_range_requires_restart(tmp_path):
    staged = _store(tmp_path).stage("services", {"service_port_range": "9500-9599"})
    assert staged.requires_restart is True


def test_services_restart_policy_requires_restart(tmp_path):
    """The restart-policy pair is snapshotted by ``ServiceController.__init__``.

    It used to answer ``False`` here, which was the same untruthfulness M6
    closed for ``[containers]``: the row is rewritten, the running controller
    keeps enforcing its boot values.
    """
    staged = _store(tmp_path).stage(
        "services", {"service_max_restarts": 5, "restart_window_seconds": 120}
    )
    assert staged.requires_restart is True
    assert staged.restart_keys == [
        "services.restart_window_seconds",
        "services.service_max_restarts",
    ]


def test_services_max_concurrent_builds_requires_restart(tmp_path):
    # P13 WP8: the global build semaphore size is bound once at controller
    # construction → restart-required.
    store = _store(tmp_path)
    staged = store.stage("services", {"max_concurrent_builds": 4})
    assert staged.requires_restart is True
    store.commit(staged)
    assert store.effective_section("services")["max_concurrent_builds"] == 4


def test_services_p20_run_keys_require_restart(tmp_path):
    """(P20) The run trio is read off the settings captured at startup — the run
    route from ``app.state.settings``, the release hook from the controller's
    own copy — so a config PUT only rewrites TOML until the daemon restarts.

    Reported per key AND together, because ``stage_many``/``apply`` (P7) unions
    the flags: a single unflagged key would make a multi-key apply under-report.
    """
    store = _store(tmp_path)
    for key, value in (
        ("run_timeout_max_s", 900),
        ("release_timeout_s", 120),
        ("max_concurrent_runs", 2),
    ):
        assert store.stage("services", {key: value}).requires_restart is True, key

    staged = store.stage(
        "services",
        {"run_timeout_max_s": 900, "release_timeout_s": 120, "max_concurrent_runs": 2},
    )
    assert staged.requires_restart is True
    store.commit(staged)
    effective = store.effective_section("services")
    assert effective["run_timeout_max_s"] == 900
    assert effective["release_timeout_s"] == 120
    assert effective["max_concurrent_runs"] == 2


def test_services_p20_run_keys_round_trip_through_load_settings(tmp_path):
    """``load_settings`` copies the whole ``[services]`` dict into
    ``NerditSettings``, so the three keys are picked up with no per-key
    plumbing — and a hand-edited config.toml really boots with them."""
    (tmp_path / "config.toml").write_text(
        "[services]\nrun_timeout_max_s = 60\nrelease_timeout_s = 30\nmax_concurrent_runs = 8\n"
    )
    settings = load_settings(tmp_path / "config.toml")
    assert settings.services.run_timeout_max_s == 60
    assert settings.services.release_timeout_s == 30
    assert settings.services.max_concurrent_runs == 8


def test_services_p20_run_key_defaults(tmp_path):
    """The shipped defaults (30 min run cap, 5 min release cap, 4 runs)."""
    effective = _store(tmp_path).effective_section("services")
    assert effective["run_timeout_max_s"] == 1800
    assert effective["release_timeout_s"] == 300
    assert effective["max_concurrent_runs"] == 4


@pytest.mark.parametrize("key", ["run_timeout_max_s", "release_timeout_s", "max_concurrent_runs"])
def test_services_p20_run_keys_reject_zero(tmp_path, key):
    """All three are ``ge=1``: a zero cap would either forbid every run or make
    the timeout unsatisfiable."""
    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).stage("services", {key: 0})
    assert exc.value.status_code == 422


def test_services_reject_zero_concurrent_builds(tmp_path):
    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).stage("services", {"max_concurrent_builds": 0})
    assert exc.value.status_code == 422


def test_services_reject_malformed_range(tmp_path):
    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).stage("services", {"service_port_range": "not-a-range"})
    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"


def test_services_reject_range_containing_daemon_port(tmp_path):
    # 9321 (DEFAULT_PORT) inside [9300, 9400] must be refused.
    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).stage("services", {"service_port_range": "9300-9400"})
    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"


def test_proxy_url_shape_requires_restart(tmp_path):
    # P3: every [proxy] field is bound at startup (ProxyManager/ServiceController
    # capture settings at construction; config PUT only rewrites TOML). The
    # URL-shape fields must flag restart so the CLI/API warn accurately instead of
    # silently no-op'ing the change until restart.
    store = _store(tmp_path)
    assert store.stage("proxy", {"mode": "subdomain"}).requires_restart is True
    assert store.stage("proxy", {"base_domain": "lan.local"}).requires_restart is True
    assert store.stage("proxy", {"enabled": True}).requires_restart is True


def test_proxy_rejects_http_scheme(tmp_path):
    # P3: the embedded Caddy bootstrap is TLS-only, so scheme must be 'https'.
    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).stage("proxy", {"scheme": "http"})
    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"


def test_proxy_rejects_non_loopback_admin_addr(tmp_path):
    # P3 security: the Caddy admin API has full control over the proxy config;
    # it must never be bound to a LAN / all-interfaces address.
    store = _store(tmp_path)
    for bad in ("0.0.0.0:2019", ":2019", "192.168.1.10:2019", "2019"):
        with pytest.raises(ConfigError) as exc:
            store.stage("proxy", {"admin_addr": bad})
        assert exc.value.code == "config.invalid", bad
    # loopback forms are accepted
    for ok in ("localhost:2019", "127.0.0.1:2019", "[::1]:2019"):
        assert store.stage("proxy", {"admin_addr": ok}).section == "proxy"


def test_proxy_rejects_malformed_base_domain(tmp_path):
    # P3.5 / D5: base_domain must be a bare, lowercase DNS name — reject with a
    # hint rather than silently normalizing (mixed case included, open Q3).
    store = _store(tmp_path)
    for bad in ("Lan.Example", "lan:8443", "*.lan", " lan "):
        with pytest.raises(ConfigError) as exc:
            store.stage("proxy", {"base_domain": bad})
        assert exc.value.code == "config.invalid", bad


def test_proxy_rejects_malformed_hostname_override(tmp_path):
    store = _store(tmp_path)
    for bad in ("Box.Lan", "box:9000", "*.box"):
        with pytest.raises(ConfigError) as exc:
            store.stage("proxy", {"hostname_override": bad})
        assert exc.value.code == "config.invalid", bad


def test_proxy_subdomain_mode_with_base_domain_stages(tmp_path):
    # The official flip UX (D4): mode + base_domain staged together, flagged
    # restart-required.
    store = _store(tmp_path)
    staged = store.stage("proxy", {"mode": "subdomain", "base_domain": "apps.lan"})
    assert staged.requires_restart is True
    store.commit(staged)
    assert store.effective_section("proxy")["mode"] == "subdomain"
    assert store.effective_section("proxy")["base_domain"] == "apps.lan"


def test_proxy_subdomain_mode_without_base_domain_stages_cleanly(tmp_path):
    # D5: no model_validator rejects mode="subdomain" with base_domain unset —
    # the base_domain-or-hostname fallback (core/proxy.py) is legal behavior.
    store = _store(tmp_path)
    staged = store.stage("proxy", {"mode": "subdomain"})
    assert staged.requires_restart is True
    store.commit(staged)
    assert store.effective_section("proxy")["mode"] == "subdomain"
    assert store.effective_section("proxy")["base_domain"] is None


def test_proxy_clear_nullable_field(tmp_path):
    # P3: a nullable field set through the API must be clearable through it too
    # (TOML has no null → clearing deletes the key, reverting to the default).
    store = _store(tmp_path)
    store.commit(store.stage("proxy", {"base_domain": "lan.local"}))
    assert store.effective_section("proxy")["base_domain"] == "lan.local"

    staged = store.stage("proxy", {"base_domain": None})
    assert staged.diff  # the clear is reflected as a diff
    store.commit(staged)
    assert store.effective_section("proxy")["base_domain"] is None
    # the key is actually gone from the persisted TOML (not left at the old value)
    assert "base_domain" not in store.load_raw().get("proxy", {})


# --- [git] section (P11.5) ----------------------------------------------------


def test_git_section_is_known(tmp_path):
    assert "git" in _store(tmp_path).known_sections()


def test_git_effective_defaults(tmp_path):
    values = _store(tmp_path).effective_section("git")
    assert values["enabled"] is True
    assert values["allowed_hosts"] == ["github.com"]
    assert values["clone_timeout_s"] == 120


def test_git_config_api_round_trip(tmp_path):
    store = _store(tmp_path)
    store.commit(store.stage("git", {"allowed_hosts": ["github.com", "gitlab.com"]}))
    raw = store.load_raw()
    assert raw["git"]["allowed_hosts"] == ["github.com", "gitlab.com"]
    assert store.effective_section("git")["allowed_hosts"] == ["github.com", "gitlab.com"]


def test_git_fields_require_restart(tmp_path):
    store = _store(tmp_path)
    assert store.stage("git", {"enabled": False}).requires_restart is True
    assert store.stage("git", {"allowed_hosts": ["gitlab.com"]}).requires_restart is True
    assert store.stage("git", {"clone_timeout_s": 300}).requires_restart is True


def test_git_watch_interval_default_and_restart(tmp_path):
    """(P24c WP10) The GitWatch poll interval is a restart-required [git] key."""
    store = _store(tmp_path)
    assert store.effective_section("git")["watch_interval_s"] == 60
    assert store.stage("git", {"watch_interval_s": 120}).requires_restart is True


@pytest.mark.parametrize("bad", [14, 0, -1])
def test_git_watch_interval_floor(tmp_path, bad):
    """ge=15 — a misconfiguration must never turn the poller into a hammer."""
    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).stage("git", {"watch_interval_s": bad})
    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"


# --- [notifications] section (P24c WP9) ---------------------------------------


def _target(**overrides) -> dict:
    """One valid https target, overridable field by field."""
    return {"url": "https://hook.example/x", **overrides}


def _refused(store: ConfigStore, targets: list[dict]) -> ConfigError:
    """Stage *targets* and return the 422 the store must raise."""
    with pytest.raises(ConfigError) as exc:
        store.stage("notifications", {"targets": targets})
    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"
    assert exc.value.diagnostics
    return exc.value


def test_notifications_section_is_known(tmp_path):
    assert "notifications" in _store(tmp_path).known_sections()


def test_notifications_effective_defaults(tmp_path):
    """Ships dark: a fresh install POSTs nowhere."""
    values = _store(tmp_path).effective_section("notifications")
    assert values["enabled"] is False
    assert values["targets"] == []
    assert values["timeout_s"] == 10.0
    assert values["retry_max"] == 5
    assert values["batch_max"] == 50
    assert values["poll_interval_s"] == 5.0


def test_notifications_config_api_round_trip(tmp_path):
    store = _store(tmp_path)
    store.commit(
        store.stage(
            "notifications",
            {"enabled": True, "targets": [_target(events=["service.failed"])]},
        )
    )
    raw = store.load_raw()
    assert raw["notifications"]["enabled"] is True
    assert raw["notifications"]["targets"][0]["url"] == "https://hook.example/x"
    effective = store.effective_section("notifications")
    assert effective["targets"][0]["events"] == ["service.failed"]
    # Unset refs are ABSENT from the persisted dump (TOML has no null).
    assert "secret_ref" not in raw["notifications"]["targets"][0]


@pytest.mark.parametrize(
    "key,value",
    [
        ("enabled", True),
        ("targets", [{"url": "https://hook.example/x"}]),
        ("timeout_s", 20.0),
        ("retry_max", 3),
        ("batch_max", 10),
        ("poll_interval_s", 30.0),
    ],
)
def test_notifications_fields_require_restart(tmp_path, key, value):
    """The dispatcher captures the whole section at construction."""
    assert _store(tmp_path).stage("notifications", {key: value}).requires_restart is True


def test_notifications_plain_http_refused_at_model_level(tmp_path):
    """The scheme × allow_http rule is a MODEL validator — loc is the target index.

    A field-level reject on ``url`` would shadow the ``allow_http`` carve-out
    (the flag is declared after ``url``), so the loc must NOT be field-shaped.
    """
    err = _refused(_store(tmp_path), [_target(url="http://example.com/hook")])
    locs = [d.loc for d in err.diagnostics]
    assert locs == [["targets", "0"]]
    assert ["targets", "0", "url"] not in locs


def test_notifications_http_hostname_refused_even_with_allow_http(tmp_path):
    """A hostname re-resolves (DNS rebinding) — IP literals only."""
    err = _refused(_store(tmp_path), [_target(url="http://ntfy.local/x", allow_http=True)])
    assert [d.loc for d in err.diagnostics] == [["targets", "0"]]


def test_notifications_userinfo_url_refused(tmp_path):
    err = _refused(_store(tmp_path), [_target(url="https://user:pw@hook.example/x")])
    assert [d.loc for d in err.diagnostics] == [["targets", "0", "url"]]


def test_notifications_non_http_scheme_refused(tmp_path):
    err = _refused(_store(tmp_path), [_target(url="ftp://hook.example/x")])
    assert [d.loc for d in err.diagnostics] == [["targets", "0", "url"]]


@pytest.mark.parametrize("url", ["https://hook.example:abc/x", "https://hook.example:99999/x"])
def test_notifications_malformed_port_refused(tmp_path, url):
    """A bad port can NEVER work: unchecked, it fails at send time on every
    delivery and retry exhaustion advances the cursor past each batch."""
    err = _refused(_store(tmp_path), [_target(url=url)])
    assert [d.loc for d in err.diagnostics] == [["targets", "0", "url"]]


@pytest.mark.parametrize("field", ["secret_ref", "auth_header_ref"])
def test_notifications_per_service_secret_ref_refused(tmp_path, field):
    """A daemon-level target has no per-service scope — shared refs only."""
    err = _refused(_store(tmp_path), [_target(**{field: "${secrets.HOOK}"})])
    assert [d.loc for d in err.diagnostics] == [["targets", "0", field]]


@pytest.mark.parametrize("field", ["secret_ref", "auth_header_ref"])
def test_notifications_literal_secret_refused(tmp_path, field):
    """A literal value in config would be persisted + backed up — refuse it."""
    err = _refused(_store(tmp_path), [_target(**{field: "zqxjkw-ZQXJKW-9"})])
    assert [d.loc for d in err.diagnostics] == [["targets", "0", field]]


@pytest.mark.parametrize(
    "name", ["X-Nerdit-Signature", "x-nerdit-anything", "Bad Header!", "", "a" * 65]
)
def test_notifications_auth_header_name_refused(tmp_path, name):
    err = _refused(_store(tmp_path), [_target(auth_header_name=name)])
    assert [d.loc for d in err.diagnostics] == [["targets", "0", "auth_header_name"]]


def test_notifications_unknown_event_type_refused(tmp_path):
    err = _refused(_store(tmp_path), [_target(events=["service.exploded"])])
    assert [d.loc for d in err.diagnostics] == [["targets", "0", "events"]]


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.1.5:8080/x",
        "http://127.0.0.1:9000/x",
        "http://[::1]/x",
        "http://10.1.2.3/x",
        "http://172.16.0.9/x",
        # IPv4-mapped loopback normalizes INTO the allow-set.
        "http://[::ffff:127.0.0.1]/x",
    ],
)
def test_notifications_allow_http_accepts_allow_set_literals(tmp_path, url):
    store = _store(tmp_path)
    staged = store.stage("notifications", {"targets": [_target(url=url, allow_http=True)]})
    assert staged.new_raw["notifications"]["targets"][0]["url"] == url
    # …and the SAME url is refused without the flag.
    _refused(store, [_target(url=url)])


@pytest.mark.parametrize(
    "url",
    [
        # The finding-2 pin: ipaddress.is_private would ACCEPT several of these.
        # Each case here is a named member of the refusal set — widening
        # _ALLOWED_HTTP_NETS requires deleting one of these lines.
        "http://169.254.169.254/latest/meta-data",
        "http://169.254.1.1/x",
        "http://0.0.0.0:8080/x",
        "http://192.0.0.1/x",
        "http://100.64.0.1/x",
        "http://[fe80::1]/x",
        "http://[fc00::1]/x",
        "http://[::ffff:169.254.169.254]/x",
        "http://[::]/x",
        "http://224.0.0.1/x",
        "http://198.18.0.1/x",
    ],
)
def test_notifications_allow_http_refuses_outside_the_allow_set(tmp_path, url):
    err = _refused(_store(tmp_path), [_target(url=url, allow_http=True)])
    assert [d.loc for d in err.diagnostics] == [["targets", "0"]]


@pytest.mark.parametrize("key", ["event", "secret", "auth_header"])
def test_notifications_unknown_key_inside_a_target_is_refused(tmp_path, key):
    """``extra='forbid'``: the store only checks unknown keys at SECTION level.

    Without it a near-miss inside a target (``event`` for ``events``, ``secret``
    for ``secret_ref``) validates, is dropped at persist time, and leaves the
    operator with a target that silently ships unfiltered or unsigned."""
    err = _refused(_store(tmp_path), [_target(**{key: "x"})])
    assert [d.loc for d in err.diagnostics] == [["targets", "0", key]]
    assert err.diagnostics[0].type == "extra_forbidden"


def test_notifications_duplicate_targets_refused(tmp_path):
    """Identical targets would share one cursor and starve each other."""
    err = _refused(_store(tmp_path), [_target(), _target()])
    message = " ".join(d.message for d in err.diagnostics)
    assert "targets[0]" in message
    assert "targets[1]" in message
    assert "identical" in message


def test_notifications_same_url_different_filter_accepted(tmp_path):
    """The cursor identity is the whole target, not just its URL."""
    staged = _store(tmp_path).stage(
        "notifications", {"targets": [_target(), _target(events=["service.healthy"])]}
    )
    assert len(staged.new_raw["notifications"]["targets"]) == 2


def test_notifications_same_url_different_auth_header_accepted(tmp_path):
    """``auth_header_name`` is part of the identity — same url, distinct targets."""
    staged = _store(tmp_path).stage(
        "notifications", {"targets": [_target(), _target(auth_header_name="X-Api-Key")]}
    )
    assert len(staged.new_raw["notifications"]["targets"]) == 2


def test_notifications_config_carries_refs_never_values(tmp_path):
    """Custody floor: the config store holds REFERENCES; it never resolves them."""
    sentinel = "zqxjkw-ZQXJKW-9"  # the value an operator put in the shared store
    store = _store(tmp_path)
    store.commit(
        store.stage(
            "notifications",
            {
                "targets": [
                    _target(
                        secret_ref="${secrets.shared.HOOK_HMAC}",
                        auth_header_ref="${secrets.shared.HOOK_TOKEN}",
                    )
                ]
            },
        )
    )
    dumped = json.dumps(store.effective_section("notifications"))
    assert "${secrets.shared.HOOK_HMAC}" in dumped
    assert "${secrets.shared.HOOK_TOKEN}" in dumped
    assert sentinel not in dumped
    assert sentinel not in (tmp_path / "config.toml").read_text()


# --- [retention] section (P14b) -----------------------------------------------


def test_retention_section_is_known(tmp_path):
    assert "retention" in _store(tmp_path).known_sections()


def test_retention_effective_defaults(tmp_path):
    values = _store(tmp_path).effective_section("retention")
    assert values["job_log_days"] == 14
    assert values["audit_days"] == 90
    assert values["audit_archive_dir"] == ""
    assert values["sweep_interval_seconds"] == 3600
    assert values["daemon_log_max_bytes"] == 10_485_760
    assert values["daemon_log_backups"] == 3
    assert values["container_log_max_size"] == "10m"
    assert values["container_log_max_file"] == 3
    assert values["backup_keep_last"] == 0


def test_retention_config_api_round_trip(tmp_path):
    store = _store(tmp_path)
    store.commit(store.stage("retention", {"job_log_days": 30, "backup_keep_last": 5}))
    raw = store.load_raw()
    assert raw["retention"]["job_log_days"] == 30
    assert raw["retention"]["backup_keep_last"] == 5
    values = store.effective_section("retention")
    assert values["job_log_days"] == 30
    assert values["backup_keep_last"] == 5
    # Untouched keys keep their defaults.
    assert values["audit_days"] == 90


def test_retention_fields_require_restart(tmp_path):
    store = _store(tmp_path)
    # Every key is bound at startup ⇒ restart-required.
    assert store.stage("retention", {"job_log_days": 30}).requires_restart is True
    assert store.stage("retention", {"sweep_interval_seconds": 600}).requires_restart is True
    assert store.stage("retention", {"container_log_max_size": "50m"}).requires_restart is True
    assert store.stage("retention", {"backup_keep_last": 3}).requires_restart is True


def test_retention_audit_days_floor(tmp_path):
    store = _store(tmp_path)
    # 0 (never prune) and >= 7 are accepted.
    assert store.stage("retention", {"audit_days": 0}).new_raw["retention"]["audit_days"] == 0
    assert store.stage("retention", {"audit_days": 7}).new_raw["retention"]["audit_days"] == 7
    # A nonzero value below the floor is rejected.
    with pytest.raises(ConfigError) as exc:
        store.stage("retention", {"audit_days": 3})
    assert exc.value.code == "config.invalid"
    with pytest.raises(ConfigError):
        store.stage("retention", {"audit_days": -1})


def test_retention_audit_archive_dir_absolute(tmp_path):
    store = _store(tmp_path)
    # Empty ⇒ default (accepted, stays "").
    empty = store.stage("retention", {"audit_archive_dir": ""})
    assert empty.new_raw.get("retention", {}).get("audit_archive_dir", "") == ""
    staged = store.stage("retention", {"audit_archive_dir": "/srv/nerdit-archive"})
    assert staged.new_raw["retention"]["audit_archive_dir"] == "/srv/nerdit-archive"
    # A relative path is rejected at the model level.
    with pytest.raises(ConfigError) as exc:
        store.stage("retention", {"audit_archive_dir": "relative/archive"})
    assert exc.value.code == "config.invalid"


def test_retention_container_log_max_size_regex(tmp_path):
    store = _store(tmp_path)
    for good in ("10m", "500k", "2g"):
        assert (
            store.stage("retention", {"container_log_max_size": good}).new_raw["retention"][
                "container_log_max_size"
            ]
            == good
        )
    for bad in ("10", "10mb", "abc", "10M"):
        with pytest.raises(ConfigError):
            store.stage("retention", {"container_log_max_size": bad})


def test_retention_numeric_bounds(tmp_path):
    store = _store(tmp_path)
    # sweep_interval_seconds ge=60.
    with pytest.raises(ConfigError):
        store.stage("retention", {"sweep_interval_seconds": 30})
    # container_log_max_file ge=1.
    with pytest.raises(ConfigError):
        store.stage("retention", {"container_log_max_file": 0})
    # daemon_log_max_bytes / backup_keep_last / job_log_days accept 0.
    assert (
        store.stage("retention", {"daemon_log_max_bytes": 0}).new_raw["retention"][
            "daemon_log_max_bytes"
        ]
        == 0
    )
    assert (
        store.stage("retention", {"backup_keep_last": 0})
        .new_raw.get("retention", {})
        .get("backup_keep_last", 0)
        == 0
    )


def test_retention_apply_multi_section(tmp_path):
    store = _store(tmp_path)
    staged = store.stage_many({"retention": {"job_log_days": 7}, "daemon": {"port": 9999}})
    assert staged.new_raw["retention"]["job_log_days"] == 7
    assert staged.new_raw["daemon"]["port"] == 9999
    by_key = {e.key: e for e in staged.diff}
    assert by_key["retention.job_log_days"].requires_restart is True
    assert "retention.job_log_days" in staged.restart_keys


# --- merged-section validation (P14b follow-up, Codex C2) ----------------------


def test_retention_partial_put_validates_against_merged_section(tmp_path):
    # The bug: a partial PUT was validated against MODEL DEFAULTS for the keys it
    # did not supply, but the merge writes the STORED values for them. Stored
    # 0/0 is legal; turning rotation on alone would write max_bytes > 0 with
    # backups = 0 — a config the daemon refuses to boot on.
    store = _store(tmp_path, "[retention]\ndaemon_log_max_bytes = 0\ndaemon_log_backups = 0\n")
    before = (tmp_path / "config.toml").read_bytes()
    with pytest.raises(ConfigError) as exc:
        store.stage("retention", {"daemon_log_max_bytes": 10_485_760})
    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"
    assert exc.value.diagnostics
    assert (tmp_path / "config.toml").read_bytes() == before


def test_retention_partial_put_with_backups_in_same_body_is_accepted(tmp_path):
    # The remedy the hint points at: supply the sibling key in the same request.
    store = _store(tmp_path, "[retention]\ndaemon_log_max_bytes = 0\ndaemon_log_backups = 0\n")
    store.commit(
        store.stage("retention", {"daemon_log_max_bytes": 10_485_760, "daemon_log_backups": 3})
    )
    raw = store.load_raw()
    assert raw["retention"]["daemon_log_max_bytes"] == 10_485_760
    assert raw["retention"]["daemon_log_backups"] == 3
    # And the written file actually boots (the whole point of the merged check).
    settings = load_settings(tmp_path / "config.toml")
    assert settings.retention.daemon_log_backups == 3


def test_merged_validation_allows_unrelated_key(tmp_path):
    # A stored 0/0 retention is legal; an unrelated partial PUT must not 422.
    store = _store(tmp_path, "[retention]\ndaemon_log_max_bytes = 0\ndaemon_log_backups = 0\n")
    store.commit(store.stage("retention", {"job_log_days": 7}))
    assert store.load_raw()["retention"] == {
        "daemon_log_max_bytes": 0,
        "daemon_log_backups": 0,
        "job_log_days": 7,
    }


def test_merged_validation_ignores_unknown_stored_keys(tmp_path):
    # A hand-edited/foreign key in the STORED section must not block an unrelated
    # write (Pydantic's extra='ignore'); the merge still preserves it on disk.
    store = _store(tmp_path, '[proxy]\nlegacy_key = "x"\n')
    store.commit(store.stage("proxy", {"enabled": True}))
    raw = store.load_raw()
    assert raw["proxy"]["enabled"] is True
    assert raw["proxy"]["legacy_key"] == "x"


def test_partial_put_does_not_materialize_sibling_defaults(tmp_path):
    # Merged validation reads; it never writes. Only supplied keys land in TOML.
    store = _store(tmp_path)
    store.commit(store.stage("retention", {"job_log_days": 7}))
    assert store.load_raw()["retention"] == {"job_log_days": 7}


def test_cleared_key_is_validated_as_its_default_not_none(tmp_path):
    # A cleared key is REMOVED from the merged dict (it reverts to the model
    # default) rather than validated as an explicit ``None``.
    store = _store(tmp_path, '[proxy]\nbase_domain = "lan.local"\n')
    staged = store.stage("proxy", {"base_domain": None})
    store.commit(staged)
    assert "base_domain" not in store.load_raw().get("proxy", {})
    assert store.effective_section("proxy")["base_domain"] is None


def test_stage_many_rejects_merged_invalid_retention(tmp_path):
    store = _store(tmp_path, "[retention]\ndaemon_log_max_bytes = 0\ndaemon_log_backups = 0\n")
    before = (tmp_path / "config.toml").read_bytes()
    with pytest.raises(ConfigError) as exc:
        store.stage_many({"retention": {"daemon_log_max_bytes": 10_485_760}})
    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"
    assert all(d.loc[0] == "retention" for d in exc.value.diagnostics)
    assert (tmp_path / "config.toml").read_bytes() == before


# --- multi-section apply (P7) --------------------------------------------------


def test_stage_base_raw_default_matches_file_read(tmp_path):
    # Existing callers pass no base_raw: behavior identical to the file read.
    store = _store(tmp_path, "[monitor]\ninterval_seconds = 5\n")
    a = store.stage("monitor", {"interval_seconds": 9})
    b = store.stage("monitor", {"interval_seconds": 9}, base_raw=store.load_raw())
    assert a.new_raw == b.new_raw
    assert [e.key for e in a.diff] == [e.key for e in b.diff]


def test_stage_many_composes_multiple_sections(tmp_path):
    store = _store(tmp_path, "[monitor]\ninterval_seconds = 5\n")
    staged = store.stage_many({"monitor": {"interval_seconds": 9}, "daemon": {"port": 9999}})
    assert staged.new_raw["monitor"]["interval_seconds"] == 9
    assert staged.new_raw["daemon"]["port"] == 9999
    assert sorted(e.key for e in staged.diff) == ["daemon.port", "monitor.interval_seconds"]
    # Section-tagged diff entries with op classification.
    by_key = {e.key: e for e in staged.diff}
    assert by_key["daemon.port"].section == "daemon"
    assert by_key["daemon.port"].op == "add"  # not previously stored
    assert by_key["monitor.interval_seconds"].op == "change"
    # Nothing written yet.
    assert store.effective_section("monitor")["interval_seconds"] == 5


def test_stage_many_is_all_or_nothing_on_validation_failure(tmp_path):
    store = _store(tmp_path, "[monitor]\ninterval_seconds = 5\n")
    before = (tmp_path / "config.toml").read_bytes()
    with pytest.raises(ConfigError) as exc:
        store.stage_many({"monitor": {"interval_seconds": 9}, "daemon": {"port": "not-an-int"}})
    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"
    # File untouched (stage_many never writes; commit was never reached).
    assert (tmp_path / "config.toml").read_bytes() == before


def test_stage_many_aggregates_diagnostics_across_sections(tmp_path):
    # Validate everything, report everything — diagnostics from ALL invalid
    # sections, namespaced [section, key].
    store = _store(tmp_path)
    with pytest.raises(ConfigError) as exc:
        store.stage_many(
            {
                "daemon": {"port": "not-an-int"},
                "monitor": {"bogus_key": 1},
                "services": {},
            }
        )
    locs = [tuple(d.loc) for d in exc.value.diagnostics]
    assert ("daemon", "port") in locs
    assert ("monitor", "bogus_key") in locs


def test_stage_many_unknown_section_is_422(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ConfigError) as exc:
        store.stage_many({"nope": {"x": 1}})
    assert exc.value.status_code == 422
    assert exc.value.code == "config.unknown_section"
    assert ["nope"] in [d.loc for d in exc.value.diagnostics]


def test_stage_many_blocks_auth_token(tmp_path):
    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).stage_many({"daemon": {"auth_token": "x"}})
    assert exc.value.status_code == 403
    assert exc.value.code == "config.forbidden"


def test_stage_many_restart_keys_aggregation(tmp_path):
    store = _store(tmp_path)
    staged = store.stage_many(
        {
            "daemon": {"port": 9999},
            "proxy": {"enabled": True},
            "client": {"remote_host": "10.0.0.1"},
        }
    )
    assert staged.requires_restart is True
    assert staged.restart_keys == ["daemon.port", "proxy.enabled"]
    by_key = {e.key: e for e in staged.diff}
    assert by_key["daemon.port"].requires_restart is True
    assert by_key["client.remote_host"].requires_restart is False


def test_stage_many_noop_reapply_is_idempotent(tmp_path):
    store = _store(tmp_path)
    doc = {"monitor": {"interval_seconds": 7}, "daemon": {"port": 9999}}
    first = store.stage_many(doc)
    assert first.changed is True
    etag = store.commit(first)
    mtime = (tmp_path / "config.toml").stat().st_mtime_ns

    again = store.stage_many(doc)
    assert again.changed is False
    assert again.diff == []
    # Commit short-circuits: same etag, no rewrite of the file.
    assert store.commit(again) == etag
    assert (tmp_path / "config.toml").stat().st_mtime_ns == mtime
    assert store.current_etag() == etag


def test_stage_many_preserves_unknown_sections_and_auth_token(tmp_path):
    store = _store(
        tmp_path,
        '[future]\nfoo = 1\n[daemon]\nauth_token = "super-secret"\nport = 9321\n',
    )
    store.commit(store.stage_many({"daemon": {"port": 9000}, "monitor": {"interval_seconds": 2}}))
    raw = store.load_raw()
    assert raw["future"] == {"foo": 1}
    assert raw["daemon"]["auth_token"] == "super-secret"
    assert raw["daemon"]["port"] == 9000
    assert raw["monitor"]["interval_seconds"] == 2


def test_stage_many_delete_op_shows_effective_default(tmp_path):
    store = _store(tmp_path, '[proxy]\nbase_domain = "lan.local"\n')
    staged = store.stage_many({"proxy": {"base_domain": None}})
    (entry,) = staged.diff
    assert entry.op == "delete"
    assert entry.old == "lan.local"
    assert entry.new is None  # effective default after the delete
    store.commit(staged)
    assert "base_domain" not in store.load_raw().get("proxy", {})


# --- P25 [security] restart keys + token_default_ttl_s ------------------------


def test_security_section_keys_all_require_restart(tmp_path):
    """(P25 §3.1.5) The whole section is bound once at boot.

    Declaring only ``token_default_ttl_s`` would make ``requires_restart``
    truthful for one key and silently wrong for the other two — and
    ``stage_many``/``apply`` union the flags, so an unflagged key would make a
    multi-key apply under-report.
    """
    store = _store(tmp_path)
    for key, value in (
        ("require_idempotency_key", True),
        ("secrets_key_file", "/tmp/k"),
        ("token_default_ttl_s", 3600),
    ):
        assert store.stage("security", {key: value}).requires_restart is True, key

    staged = store.stage(
        "security",
        {"require_idempotency_key": True, "token_default_ttl_s": 3600},
    )
    assert staged.requires_restart is True
    store.commit(staged)
    assert store.effective_section("security")["token_default_ttl_s"] == 3600


def test_token_default_ttl_s_defaults_to_no_expiry(tmp_path):
    """(D-P25-1 countersign) Shipped default is ``None``: expiry is opt-in."""
    store = _store(tmp_path)
    assert store.effective_section("security")["token_default_ttl_s"] is None


@pytest.mark.parametrize("bad", [59, 31_536_001, -1])
def test_token_default_ttl_s_bounds_are_enforced(tmp_path, bad):
    store = _store(tmp_path)
    with pytest.raises(ConfigError):
        store.stage("security", {"token_default_ttl_s": bad})


# --- P29: the two new keys ---------------------------------------------------


def test_mcp_max_body_bytes_default_and_restart(tmp_path):
    """(P29 §0.1) The transport body cap is configurable but ships UNCHANGED.

    It is bound once into ``_TransportGuard`` at mount time (``build_http_app``),
    so like ``http_enabled`` a config PUT only rewrites TOML until restart.
    """
    store = _store(tmp_path)
    assert store.effective_section("mcp")["max_body_bytes"] == 1_048_576
    assert store.stage("mcp", {"max_body_bytes": 4_194_304}).requires_restart is True
    # Reported together too — stage_many/apply unions the flags (P7).
    staged = store.stage("mcp", {"http_enabled": True, "max_body_bytes": 4_194_304})
    assert staged.requires_restart is True
    store.commit(staged)
    assert store.effective_section("mcp")["max_body_bytes"] == 4_194_304


@pytest.mark.parametrize("bad", [65_535, 16_777_217, 0, -1])
def test_mcp_max_body_bytes_bounds_are_enforced(tmp_path, bad):
    """ge=65_536 / le=16_777_216 — neither a wedge nor an unbounded pipe."""
    store = _store(tmp_path)
    with pytest.raises(ConfigError):
        store.stage("mcp", {"max_body_bytes": bad})


def test_workspace_orphan_days_default_and_restart(tmp_path):
    """(P29 / D-P29-8) The sweep loop captures ``[retention]`` once at startup."""
    store = _store(tmp_path)
    assert store.effective_section("retention")["workspace_orphan_days"] == 30
    assert store.stage("retention", {"workspace_orphan_days": 7}).requires_restart is True
    staged = store.stage("retention", {"workspace_orphan_days": 0})  # 0 = never sweep
    assert staged.requires_restart is True
    store.commit(staged)
    assert store.effective_section("retention")["workspace_orphan_days"] == 0


def test_workspace_orphan_days_rejects_negative(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ConfigError):
        store.stage("retention", {"workspace_orphan_days": -1})


# --- [proxy.acme] — the nested block (P26 WP2 / S-W2-1) -----------------------


def test_proxy_acme_is_one_restart_key(tmp_path):
    """The diff engine compares the nested dict as a UNIT, so any sub-key change
    reads ``requires_restart=True`` — which is the truth: ``ProxyManager``
    captures ``ProxySettings`` at construction and the ``:80`` server is written
    into the bootstrap Caddy is spawned with."""
    from nerdit.config.store import _RESTART_KEYS

    assert "acme" in _RESTART_KEYS["proxy"]

    store = _store(tmp_path, "[proxy]\nenabled = true\n")
    staged = store.stage("proxy", {"acme": {"enabled": True, "email": "ops@example.com"}})

    assert staged.requires_restart is True
    assert "proxy.acme" in staged.restart_keys


def test_proxy_acme_round_trips_through_the_store(tmp_path):
    """The dump must survive ``tomli_w`` (no null leaf) and reload as a table."""
    store = _store(tmp_path, "[proxy]\nenabled = true\n")
    store.commit(
        store.stage(
            "proxy",
            {"acme": {"enabled": True, "email": "ops@example.com", "http_port": 8080}},
        )
    )

    effective = store.effective_section("proxy")["acme"]
    assert effective["enabled"] is True
    assert effective["email"] == "ops@example.com"
    assert effective["http_port"] == 8080
    # Unset optionals are ABSENT, not null — TOML has no null.
    assert "ca_root_file" not in effective
    assert "[proxy.acme]" in (tmp_path / "config.toml").read_text()

    # And the file the store wrote is what load_settings reads back.
    assert load_settings(tmp_path / "config.toml").proxy.acme.http_port == 8080


def test_proxy_acme_enabled_without_email_is_422(tmp_path):
    """A FIRST enable that never names an email is refused: there is no stored
    address to merge over, so the rule has nothing to be satisfied by."""
    store = _store(tmp_path, "[proxy]\nenabled = true\n")

    with pytest.raises(ConfigError) as exc:
        store.stage("proxy", {"acme": {"enabled": True}})

    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"
    assert any("email is required" in diag.message for diag in exc.value.diagnostics)


def test_a_partial_acme_write_keeps_the_stored_sub_keys(tmp_path):
    """(P26 WP2 review round 1) A nested sub-table is merged one level deep, so a
    partial write touches exactly the sub-key it names.

    The bug this pins: ``stage`` splatted the body over the stored SECTION only,
    so a body carrying ``acme`` REPLACED the whole table — a one-key tweak like
    ``{"acme": {"http_redirect": False}}`` came back with ``enabled=False`` and
    no ``email``, i.e. ACME silently off and every public certificate on the node
    reverting to an internal-CA leaf on the next restart, with a 200 and no
    diagnostic. The section merge has always been partial; the nested table now
    obeys the same rule.
    """
    store = _store(tmp_path, "[proxy]\nenabled = true\n")
    store.commit(
        store.stage(
            "proxy",
            {"acme": {"enabled": True, "email": "ops@example.com", "http_port": 8080}},
        )
    )

    staged = store.stage("proxy", {"acme": {"http_redirect": False}})
    store.commit(staged)

    assert staged.requires_restart is True
    effective = store.effective_section("proxy")["acme"]
    assert effective["enabled"] is True
    assert effective["email"] == "ops@example.com"
    assert effective["http_port"] == 8080
    assert effective["http_redirect"] is False
    # And the merged document is what the daemon reads back at boot.
    acme = load_settings(tmp_path / "config.toml").proxy.acme
    assert (acme.enabled, acme.email, acme.http_port, acme.http_redirect) == (
        True,
        "ops@example.com",
        8080,
        False,
    )


def test_a_partial_acme_write_that_breaks_a_cross_field_rule_is_still_422(tmp_path):
    """The deep merge is not a way around validation: the MERGED table is what
    is judged, so a sub-key that makes it invalid is still refused."""
    store = _store(tmp_path, "[proxy]\nenabled = true\nhttps_port = 8443\n")
    store.commit(store.stage("proxy", {"acme": {"enabled": True, "email": "ops@example.com"}}))

    with pytest.raises(ConfigError) as exc:
        store.stage("proxy", {"acme": {"http_port": 8443}})

    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"
    assert store.effective_section("proxy")["acme"]["http_port"] == 80


def test_an_unknown_dotted_key_points_at_the_whole_table_form(tmp_path):
    """``nerdit config set proxy acme.enabled=true`` is the obvious guess and is
    refused — the hint must not leave the operator at a dead end (review round
    1): ``config set`` addresses scalars only, so name the form that works."""
    store = _store(tmp_path, "[proxy]\nenabled = true\n")

    with pytest.raises(ConfigError) as exc:
        store.stage("proxy", {"acme.enabled": True})

    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"
    hint = exc.value.hint or ""
    assert "sub-table" in hint
    assert "config apply" in hint
    assert "[proxy.acme]" in hint


def test_proxy_acme_without_proxy_enabled_is_422(tmp_path):
    store = _store(tmp_path)

    with pytest.raises(ConfigError) as exc:
        store.stage("proxy", {"acme": {"enabled": True, "email": "ops@example.com"}})

    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"


def test_proxy_acme_diagnostics_never_echo_the_directory_credentials(tmp_path):
    """``_diagnostics`` projects ``loc`` + ``msg`` only — never Pydantic's
    ``input_value`` — so a mis-pasted credential cannot ride a 422 body."""
    store = _store(tmp_path, "[proxy]\nenabled = true\n")

    with pytest.raises(ConfigError) as exc:
        store.stage(
            "proxy",
            {
                "acme": {
                    "enabled": True,
                    "email": "ops@example.com",
                    "directory": "https://user:hunter2@ca.example.com/directory",
                }
            },
        )

    body = json.dumps([diag.model_dump() for diag in exc.value.diagnostics]) + exc.value.message
    assert "hunter2" not in body


def test_proxy_acme_unknown_subkey_is_422(tmp_path):
    """``stage`` only checks unknown keys at the SECTION level; ``extra='forbid'``
    on the nested model is what stops a typo from being silently dropped."""
    store = _store(tmp_path, "[proxy]\nenabled = true\n")

    with pytest.raises(ConfigError) as exc:
        store.stage("proxy", {"acme": {"enabled": True, "email": "a@b.co", "mail": "a@b.co"}})

    assert exc.value.code == "config.invalid"
