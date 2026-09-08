"""The ``[link]`` config section (P27 WP-C3, item 1).

WP-C3 "identity & config" item 1 mandates ``enabled`` (default false),
``relay_url``, ``key_file`` (default ``<data_dir>/link/node.key``),
``capability_ttl_s`` (default ≤ 900 s, inside the relay's ≤15-min accepted
look-ahead per ADR-W2), ``renew_margin_s``; config-as-API plumbed; all keys
restart-required in v1.

The section ships **dark** (``enabled = false``) and carries **no**
``allow_admin`` and **no** ``role`` key — D-R2 as amended by ADR-W1 makes the
tunnel role permanently ``"submitter"`` and keeps admin operations local-only,
so the ceiling is structural: an unknown key is a 422 at the config API rather
than a knob an operator could raise.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nerdit.config.settings import LinkSettings, NerditSettings, load_settings
from nerdit.config.store import ConfigError, ConfigStore

#: A ``[link]`` block whose every key differs from what the tests stage over it,
#: so each staged key produces a real diff entry.
STORED_LINK = """
[link]
enabled = true
relay_url = "wss://old.example.com/link"
key_file = "/tmp/old-node.key"
capability_ttl_s = 800
renew_margin_s = 200
"""


def _store(tmp_path: Path, body: str = "") -> ConfigStore:
    path = tmp_path / "config.toml"
    if body:
        path.write_text(body)
    return ConfigStore(path)


# --- defaults ----------------------------------------------------------------


def test_defaults():
    link = LinkSettings()
    assert link.enabled is False
    assert link.relay_url == ""
    assert link.key_file is None
    assert link.capability_ttl_s == 600
    assert link.renew_margin_s == 120
    assert link.node_id is None
    assert link.slug is None
    # (P26 D-P26-H5) Unset until a claim or ``nerdit link refresh`` learns it —
    # the hosted URL is computed, never guessed.
    assert link.nodes_base_domain is None
    assert NerditSettings().link == link


def test_default_ttl_within_adr_w2_ceiling():
    # Pins the checklist requirement itself, not the current number: whatever
    # the default becomes, it must stay inside the relay's accepted look-ahead.
    assert LinkSettings().capability_ttl_s <= 900


# --- validators --------------------------------------------------------------


def test_enabled_requires_relay_url():
    # The daemon dials OUT (D-R3); there is no default relay to fall back to.
    with pytest.raises(ValidationError):
        LinkSettings(enabled=True)
    assert LinkSettings(enabled=True, relay_url="wss://relay.example.com/link").enabled is True


@pytest.mark.parametrize(
    "url",
    ["wss://relay.example.com", "https://relay.example.com/path", ""],
)
def test_relay_url_schemes_accepted(url):
    # Empty is accepted while disabled — the section ships dark.
    assert LinkSettings(relay_url=url).relay_url == url


@pytest.mark.parametrize(
    "url",
    [
        "http://x",  # cleartext
        "ws://x",  # unencrypted websocket
        "ftp://x",  # not a relay dial at all
        "relay.example.com",  # no scheme
        "wss://user:pw@x",  # userinfo: auth is the handshake's job, not the URL
        "wss://x/#frag",  # fragment
        # A query string is credentials by another name — and unlike userinfo it
        # would round-trip verbatim through GET /config/daemon, the write diff,
        # the config.apply audit row and the persisted TOML ([link] has no
        # SECRET_LEAF_KEYS leaf to redact it).
        "wss://relay.example.com/connect?token=s3cr3t",
        "https://relay.example.com/connect?a=1",
        " wss://x",  # whitespace is a paste accident, never repaired
    ],
)
def test_relay_url_schemes_rejected(url):
    with pytest.raises(ValidationError):
        LinkSettings(relay_url=url)


def test_key_file_validation():
    assert LinkSettings(key_file="/tmp/k.key").key_file == "/tmp/k.key"
    assert LinkSettings(key_file="~/k.key").key_file == "~/k.key"
    assert LinkSettings(key_file=None).key_file is None
    for bad in ("relative/k.key", "  "):
        with pytest.raises(ValidationError):
            LinkSettings(key_file=bad)


def test_ttl_bounds():
    for bad in (59, 901):
        with pytest.raises(ValidationError):
            LinkSettings(capability_ttl_s=bad, renew_margin_s=10)
    assert LinkSettings(capability_ttl_s=60, renew_margin_s=10).capability_ttl_s == 60
    assert LinkSettings(capability_ttl_s=900, renew_margin_s=10).capability_ttl_s == 900


def test_renew_margin_bounds():
    with pytest.raises(ValidationError):
        LinkSettings(renew_margin_s=9)
    # Renewal at ``expires_at - renew_margin_s`` (D-R8) must fire after minting.
    with pytest.raises(ValidationError):
        LinkSettings(capability_ttl_s=60, renew_margin_s=60)
    with pytest.raises(ValidationError):
        LinkSettings(capability_ttl_s=600, renew_margin_s=600)
    assert LinkSettings(capability_ttl_s=600, renew_margin_s=599).renew_margin_s == 599


# --- the D-R2 ceiling is structural ------------------------------------------


@pytest.mark.parametrize("body", [{"allow_admin": True}, {"role": "admin"}])
def test_no_allow_admin_no_role_keys(tmp_path, body):
    # D-R2 (as amended by ADR-W1): the tunnel role is permanently "submitter"
    # and admin never crosses the tunnel. Neither key exists, so the config API
    # answers 422 rather than storing a knob nothing reads.
    assert "allow_admin" not in LinkSettings.model_fields
    assert "role" not in LinkSettings.model_fields

    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).stage("link", body)
    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"
    assert [d.type for d in exc.value.diagnostics] == ["extra_forbidden"]
    assert [d.loc for d in exc.value.diagnostics] == [[next(iter(body))]]


# --- config-as-API plumbing ---------------------------------------------------


def test_section_registered(tmp_path):
    assert "link" in _store(tmp_path).known_sections()


def test_all_keys_restart_required(tmp_path):
    # The WP-C1 tunnel client captures LinkSettings once at startup and the
    # identity key path is resolved at boot, so a config PUT only rewrites TOML
    # ([git]/[mcp] precedent).
    store = _store(tmp_path, STORED_LINK)
    body = {
        "enabled": False,
        "relay_url": "wss://r.example",
        "key_file": "/tmp/k",
        "capability_ttl_s": 300,
        "renew_margin_s": 60,
        # WP-C1 added the claim result to the same whole-section frozenset:
        # the manager captures node_id/slug at boot like the rest of [link].
        "node_id": "00000000-0000-4000-8000-00000000000a",
        "slug": "dev-node",
        # (P26 D-P26-H5) The hosted base domain joins the same whole-section
        # frozenset: the app-stream resolver captures it when the tunnel
        # manager is built, so writing it is honestly restart-required.
        "nodes_base_domain": "nodes.example",
    }
    staged = store.stage("link", body)

    assert staged.requires_restart is True
    assert all(entry.requires_restart for entry in staged.diff)
    assert {entry.key for entry in staged.diff} == {f"link.{key}" for key in body}
    assert set(staged.restart_keys) == {f"link.{key}" for key in body}

    # Public route to the same claim, key by key: every declared field is
    # restart-required, so the frozenset covers the whole model.
    for key, value in body.items():
        assert store.stage("link", {key: value}).requires_restart is True, key
    assert set(body) == set(LinkSettings.model_fields)


def test_stage_invalid_reports_diagnostics(tmp_path):
    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).stage("link", {"capability_ttl_s": 5000})
    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"
    assert ["capability_ttl_s"] in [d.loc for d in exc.value.diagnostics]


def test_load_settings_maps_link(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[link]\nenabled = false\nrelay_url = "wss://relay.example.com"\ncapability_ttl_s = 300\n'
    )
    link = load_settings(path).link
    assert link.enabled is False
    assert link.relay_url == "wss://relay.example.com"
    assert link.capability_ttl_s == 300
    assert link.key_file is None
    assert link.renew_margin_s == 120


def test_view_section_has_no_secret_material(tmp_path):
    # No [link] leaf is in SECRET_LEAF_KEYS, and none should be: the private key
    # lives only in node.key, never in config — ``key_file`` is a plain path leaf
    # exactly like ``security.secrets_key_file``.
    store = _store(tmp_path, STORED_LINK)
    view = store.view_section("link")
    assert view == store.effective_section("link")
    assert view["key_file"] == "/tmp/old-node.key"


def test_null_clear_key_file(tmp_path):
    store = _store(tmp_path, STORED_LINK)
    staged = store.stage("link", {"key_file": None})

    entry = next(e for e in staged.diff if e.key == "link.key_file")
    assert entry.op == "delete"
    assert entry.new is None
    # Dropped from the file, so the effective value reverts to the data_dir
    # default (<data_dir>/link/node.key) instead of stranding a stale path.
    assert "key_file" not in staged.new_raw["link"]


# --- PR #113 review follow-ups ------------------------------------------------


def test_key_file_unresolvable_home_is_a_422_not_a_500():
    # ``~missinguser/...`` makes pathlib.Path.expanduser raise RuntimeError,
    # which Pydantic would NOT wrap into ValidationError — it escaped the
    # config route as an HTTP 500 instead of the structured 422.
    with pytest.raises(ValidationError, match="unresolvable"):
        LinkSettings(key_file="~nerdit-no-such-user-a1b2/node.key")


def test_partial_put_enabled_validates_against_stored_relay_url(tmp_path):
    # A stored relay_url must satisfy the enabled=>relay_url cross-field rule
    # for a partial PUT of {"enabled": true} — the body used to be judged
    # against the model's DEFAULT (empty) relay_url and falsely 422'd.
    store = _store(tmp_path, '[link]\nrelay_url = "wss://relay.example.com/link"\n')
    staged = store.stage("link", {"enabled": True})
    assert staged.new_raw["link"]["enabled"] is True
    assert staged.new_raw["link"]["relay_url"] == "wss://relay.example.com/link"


def test_partial_put_margin_validates_against_stored_ttl(tmp_path):
    # Stored ttl=900: a margin of 700 is legal (700 < 900) even though it
    # would fail against the DEFAULT ttl of 600.
    store = _store(tmp_path, "[link]\ncapability_ttl_s = 900\n")
    staged = store.stage("link", {"renew_margin_s": 700})
    assert staged.new_raw["link"]["renew_margin_s"] == 700


def test_partial_put_margin_still_rejected_against_stored_ttl(tmp_path):
    # The converse: stored ttl=120 makes a margin of 130 invalid even though
    # it would pass against the default ttl of 600.
    store = _store(tmp_path, "[link]\ncapability_ttl_s = 120\n")
    with pytest.raises(ConfigError) as exc:
        store.stage("link", {"renew_margin_s": 130})
    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"


# --- nodes_base_domain (P26 D-P26-H5) ----------------------------------------


def test_nodes_base_domain_accepts_a_bare_dns_name():
    assert LinkSettings(nodes_base_domain="nodes.nerdit.ai").nodes_base_domain == "nodes.nerdit.ai"
    assert LinkSettings(nodes_base_domain="nodes.localhost").nodes_base_domain == "nodes.localhost"


@pytest.mark.parametrize(
    ("value", "fragment"),
    [
        ("Nodes.Example", "lowercase"),
        ("*.nodes.example", "wildcard"),
        ("nodes.example:443", "port"),
        ("https://nodes.example", "scheme"),
        (" nodes.example", "whitespace"),
        (".nodes.example", "dot"),
        ("nodes.example.", "dot"),
        ("", "empty"),
        # (PR review, P26 WP-H) IP literals: the shared ``[proxy]`` grammar
        # ACCEPTS an IPv4 one — every octet matches the label rule, and
        # ``hostname_override`` is documented as taking a "name/IP" — so this
        # field has to refuse them itself. ``https://demo--gpu-box.10.0.0.1/``
        # is a name nothing can resolve, and D-P26-H5 is "computed, never
        # guessed". IPv6 would also trip the port rule (the colons), but the
        # address check runs first so it reads as the address it is.
        ("10.0.0.1", "IP address"),
        ("192.168.1.50", "IP address"),
        ("::1", "IP address"),
    ],
)
def test_nodes_base_domain_rejects(value, fragment):
    """The value is the SUFFIX of every hosted URL the daemon advertises, so a
    URL-ish, wildcarded, mixed-case or address-literal value must be refused
    loudly rather than silently normalized underneath the operator (the
    ``[proxy]`` hostname rule)."""
    with pytest.raises(ValidationError) as exc:
        LinkSettings(nodes_base_domain=value)
    assert fragment in str(exc.value)


def test_nodes_base_domain_null_is_the_way_to_forget_it():
    """Empty is refused; ``None`` is the single explicit "no hosted domain"."""
    assert LinkSettings(nodes_base_domain=None).nodes_base_domain is None


def test_store_round_trips_nodes_base_domain(tmp_path):
    store = _store(tmp_path, STORED_LINK)
    store.commit(store.stage("link", {"nodes_base_domain": "nodes.example"}))

    assert load_settings(tmp_path / "config.toml").link.nodes_base_domain == "nodes.example"


def test_store_rejects_an_invalid_nodes_base_domain(tmp_path):
    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).stage("link", {"nodes_base_domain": "*.nodes.example"})
    assert exc.value.status_code == 422
    assert ["nodes_base_domain"] in [d.loc for d in exc.value.diagnostics]
