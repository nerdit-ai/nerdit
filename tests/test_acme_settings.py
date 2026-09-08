"""``[proxy.acme]`` validation (P26 WP2 / D-P26-5, S-W2-1).

Every refusal here is load-time: the daemon must not boot, and
``PUT /config/daemon/proxy`` must not 200, on a configuration that would make Caddy
either bind a second listener it cannot serve or talk to a CA over a channel
that is not authenticated. The positive half is just as load-bearing — the
default dump must round-trip through ``tomli_w`` (no null leaf) or the config
API cannot persist the section at all.
"""

from __future__ import annotations

import pytest
import tomli_w
from pydantic import ValidationError

from nerdit.config.settings import (
    DEFAULT_ACME_DIRECTORY,
    LE_STAGING_DIRECTORY,
    ProxyAcmeSettings,
    ProxySettings,
    load_settings,
)

# --- defaults -----------------------------------------------------------------


def test_acme_defaults_are_off_and_le_production():
    acme = ProxyAcmeSettings()

    assert acme.enabled is False
    assert acme.email is None
    assert (
        acme.directory
        == DEFAULT_ACME_DIRECTORY
        == ("https://acme-v02.api.letsencrypt.org/directory")
    )
    assert acme.http_port == 80
    assert acme.http_redirect is True
    assert acme.ca_root_file is None


def test_le_staging_directory_is_exported_and_valid():
    """Docs, tests and the smoke run must quote ONE staging string."""
    assert LE_STAGING_DIRECTORY == "https://acme-staging-v02.api.letsencrypt.org/directory"
    assert ProxyAcmeSettings(directory=LE_STAGING_DIRECTORY).directory == LE_STAGING_DIRECTORY


def test_proxy_settings_mounts_acme_by_default():
    """The nested block exists on every ProxySettings, off — so every reader can
    say ``settings.proxy.acme.enabled`` without a getattr dance."""
    assert ProxySettings().acme == ProxyAcmeSettings()
    assert ProxySettings().acme.enabled is False


def test_acme_parses_from_toml_as_a_nested_table(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(
        "[proxy]\n"
        "enabled = true\n"
        "\n"
        "[proxy.acme]\n"
        "enabled = true\n"
        'email = "ops@example.com"\n'
        'directory = "https://acme-staging-v02.api.letsencrypt.org/directory"\n'
        "http_port = 8080\n"
        "http_redirect = false\n"
    )

    acme = load_settings(config).proxy.acme

    assert acme.enabled is True
    assert acme.email == "ops@example.com"
    assert acme.directory == LE_STAGING_DIRECTORY
    assert acme.http_port == 8080
    assert acme.http_redirect is False


# --- serialization ------------------------------------------------------------


def test_default_dump_has_no_none_leaf_and_round_trips_through_tomli_w():
    """``tomli_w`` has no null and the store's ``_strip_none`` only reaches
    top-level section keys — a nested ``email = None`` would blow up the write."""
    dumped = ProxySettings().model_dump()

    assert None not in dumped["acme"].values()
    assert set(dumped["acme"]) == {"enabled", "directory", "http_port", "http_redirect"}

    # The whole section, exactly as ConfigStore persists it (top-level None stripped).
    section = {key: value for key, value in dumped.items() if value is not None}
    text = tomli_w.dumps({"proxy": section})

    assert "[proxy.acme]" in text


def test_set_optional_keys_survive_the_dump():
    """Dropping ``None`` must not drop a value the operator actually set."""
    acme = ProxyAcmeSettings(
        enabled=True, email="ops@example.com", ca_root_file="/etc/pki/pebble.pem"
    )

    dumped = acme.model_dump()

    assert dumped["email"] == "ops@example.com"
    assert dumped["ca_root_file"] == "/etc/pki/pebble.pem"
    assert tomli_w.dumps({"acme": dumped})


# --- email --------------------------------------------------------------------


def test_enabled_without_email_is_refused():
    with pytest.raises(ValidationError) as exc:
        ProxyAcmeSettings(enabled=True)

    assert "email is required when acme is enabled" in str(exc.value)


@pytest.mark.parametrize(
    "bad",
    [
        "ops",  # no domain
        "ops@localhost",  # no dot in the domain
        "@example.com",  # no local part
        "ops@@example.com",  # two @
        "Ops <ops@example.com>",  # display name
        "mailto:ops@example.com",  # scheme prefix
        "ops @example.com",  # whitespace
        "",
    ],
)
def test_malformed_email_is_refused(bad):
    with pytest.raises(ValidationError):
        ProxyAcmeSettings(email=bad)


def test_email_is_stripped():
    assert ProxyAcmeSettings(email="  ops@example.com \n").email == "ops@example.com"


def test_email_refusal_message_never_echoes_the_address():
    """The ``msg`` is the only part of a ValidationError that reaches a caller:
    ``ConfigStore._diagnostics`` projects ``loc`` + ``msg`` and drops
    ``input_value``. So the message is what must stay free of the address."""
    with pytest.raises(ValidationError) as exc:
        ProxyAcmeSettings(email="Ops <secret.person@example.com>")

    assert "secret.person" not in exc.value.errors()[0]["msg"]


# --- directory ----------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        DEFAULT_ACME_DIRECTORY,
        LE_STAGING_DIRECTORY,
        "https://ca.internal.example.com/acme/directory",
        "http://127.0.0.1:14000/dir",  # the Pebble carve-out
        "http://localhost:14000/dir",
        "http://[::1]:14000/dir",
    ],
)
def test_accepted_directories(url):
    assert ProxyAcmeSettings(directory=url).directory == url


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/dir",  # plain http off loopback
        "http://192.168.1.10:14000/dir",  # LAN is not loopback
        "acme-v02.api.letsencrypt.org/directory",  # not absolute
        "ftp://example.com/dir",  # wrong scheme
        "https:///directory",  # no host
        "https://example.com:notaport/dir",  # unparsable port
        # (Codex round 2, #3835632986) ``urlsplit`` returns 0 for ``:0`` without
        # raising, so the try/except alone let it through — and Caddy then fails
        # to dial on every order, with a stuck ``pending`` as the only symptom.
        "https://ca.example.com:0/directory",  # port zero
    ],
)
def test_refused_directories(url):
    with pytest.raises(ValidationError):
        ProxyAcmeSettings(directory=url)


def test_a_zero_port_directory_names_the_range_it_violates() -> None:
    """The refusal has to be actionable and must not echo the URL (userinfo)."""
    with pytest.raises(ValidationError) as exc:
        ProxyAcmeSettings(directory="https://ca.example.com:0/directory")

    message = exc.value.errors()[0]["msg"]
    assert "1-65535" in message
    assert "ca.example.com" not in message


def test_directory_with_userinfo_is_refused_without_echoing_it():
    """Same rule as the email message: the refusal names the rule, never the
    credential it just refused (``msg`` is what the API projects)."""
    with pytest.raises(ValidationError) as exc:
        ProxyAcmeSettings(directory="https://user:hunter2@ca.example.com/directory")

    message = exc.value.errors()[0]["msg"]
    assert "userinfo" in message
    assert "hunter2" not in message


def test_http_off_loopback_names_the_rule():
    with pytest.raises(ValidationError) as exc:
        ProxyAcmeSettings(directory="http://ca.example.com/directory")

    assert "loopback" in str(exc.value)


# --- http_port ----------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, 65_536, 100_000])
def test_http_port_bounds(bad):
    with pytest.raises(ValidationError):
        ProxyAcmeSettings(http_port=bad)


def test_http_port_accepts_an_unprivileged_port():
    assert ProxyAcmeSettings(http_port=18080).http_port == 18080


# --- ca_root_file -------------------------------------------------------------


@pytest.mark.parametrize("bad", ["pebble.pem", "./ca/pebble.pem", "  "])
def test_relative_or_empty_ca_root_file_is_refused(bad):
    with pytest.raises(ValidationError):
        ProxyAcmeSettings(ca_root_file=bad)


def test_absolute_ca_root_file_is_accepted_without_being_stat_ed(tmp_path):
    """Existence is Caddy's problem at spawn time — the config store validates
    on a machine that may not hold the file yet."""
    missing = tmp_path / "nope" / "ca.pem"
    assert not missing.exists()

    assert ProxyAcmeSettings(ca_root_file=str(missing)).ca_root_file == str(missing)


# --- cross-field rules on ProxySettings ---------------------------------------


def test_acme_requires_the_proxy_to_be_enabled():
    with pytest.raises(ValidationError) as exc:
        ProxySettings(acme={"enabled": True, "email": "ops@example.com"})

    assert "[proxy.acme].enabled requires [proxy].enabled" in str(exc.value)


def test_acme_http_port_may_not_collide_with_https_port():
    """Caddy refuses to start when ANY listener fails to bind — a collision
    would take the whole proxy down, not just ACME."""
    with pytest.raises(ValidationError) as exc:
        ProxySettings(
            enabled=True,
            https_port=8443,
            acme={"enabled": True, "email": "ops@example.com", "http_port": 8443},
        )

    assert "must differ from" in str(exc.value)


def test_acme_http_port_may_not_collide_with_the_admin_port():
    """(review round 1) Same clash, same consequence, one field away: the Caddy
    admin API is a listener of the same process, so ``http_port = 2019`` takes
    the whole proxy into backoff with the reason buried in ``caddy.log``."""
    with pytest.raises(ValidationError) as exc:
        ProxySettings(
            enabled=True,
            admin_addr="localhost:2019",
            acme={"enabled": True, "email": "ops@example.com", "http_port": 2019},
        )

    assert "[proxy].admin_addr" in str(exc.value)


def test_a_disabled_acme_block_is_inert_config():
    """Both cross-field rules are gated on ``acme.enabled``: an off block
    describes a listener that does not exist, so it cannot 422 a document that
    changes nothing about what the node binds."""
    settings = ProxySettings(enabled=False, https_port=80, acme={"http_port": 80})

    assert settings.acme.enabled is False
    assert settings.acme.http_port == 80


def test_enabled_acme_on_an_enabled_proxy_validates():
    settings = ProxySettings(
        enabled=True,
        acme={
            "enabled": True,
            "email": "ops@example.com",
            "directory": LE_STAGING_DIRECTORY,
            "http_port": 8080,
        },
    )

    assert settings.acme.enabled is True
    assert settings.acme.http_port == 8080


def test_unknown_key_inside_the_block_is_refused():
    """``ConfigStore.stage`` only rejects unknown keys at the SECTION level, so
    without ``extra='forbid'`` here a typo would validate, be dropped at persist
    time, and leave ACME silently mis-configured (the NotificationTarget rule)."""
    with pytest.raises(ValidationError):
        ProxyAcmeSettings(enabled=True, email="ops@example.com", mail="ops@example.com")
