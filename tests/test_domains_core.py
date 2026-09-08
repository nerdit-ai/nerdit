"""P26 WP1 T1 — the pure custom-domain core: validation, ids, URLs, ordering.

Four leaves, no I/O:

1. :func:`normalize_domain` — the total fold every path (including DELETE) runs;
2. :func:`validate_domain` — one assertion per machine reason token (S-W10),
   because the token is what an agent branches on and a refusal that lands on
   the wrong token sends an operator to fix the wrong thing;
3. the composite ``@id`` grammar + :func:`domain_url_for`;
4. :func:`ordering_violations` — the fifth drift dimension, which is the only
   thing standing between a custom domain and a path-mode shadowing bug.
"""

from __future__ import annotations

import pytest

from nerdit.config.settings import ProxySettings
from nerdit.core.proxy import (
    DomainInvalid,
    LiveRoute,
    ReservedNames,
    RouteSpec,
    _domain_route_id,
    _split_route_id,
    domain_url_for,
    host_aliases,
    normalize_domain,
    ordering_violations,
    reserved_names,
    validate_domain,
)

# The node's own names, as a boot-computed set would carry them.
_RESERVED = ReservedNames(names=frozenset({"box", "dev.lan", "192.168.1.50", "nerd.local"}))


def _reason(raw: str, *, reserved: ReservedNames = _RESERVED) -> str:
    """Validate ``raw`` and return the refusal token (fails if it is accepted)."""
    with pytest.raises(DomainInvalid) as exc:
        validate_domain(raw, reserved=reserved)
    return exc.value.reason


# ---------------------------------------------------------------------------
# 1. normalize_domain — total, never raises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("app.example.com", "app.example.com"),
        ("App.Example.COM", "app.example.com"),
        ("  app.example.com  ", "app.example.com"),
        ("app.example.com.", "app.example.com"),
        ("APP.EXAMPLE.COM.", "app.example.com"),
        ("app.example.com..", "app.example.com."),  # exactly ONE trailing dot is dropped
        ("", ""),
        ("   ", ""),
        (".", ""),
        ("https://x.example/", "https://x.example/"),  # folding is not validation
    ],
)
def test_normalize_domain_folds_without_judging(raw, expected):
    assert normalize_domain(raw) == expected


def test_normalize_domain_never_raises_on_garbage():
    """The DELETE path folds whatever was typed; a malformed string must simply
    match no row rather than blow up before the query."""
    for raw in ("*", "::1", "a" * 5000, "\x00", "  \t\n "):
        normalize_domain(raw)


# ---------------------------------------------------------------------------
# 2. validate_domain — one assertion per reason token (S-W10)
# ---------------------------------------------------------------------------


def test_a_valid_domain_comes_back_folded():
    assert validate_domain("App.Example.COM.", reserved=_RESERVED) == "app.example.com"
    assert validate_domain("a.b.c.d.example.org", reserved=_RESERVED) == "a.b.c.d.example.org"


@pytest.mark.parametrize("raw", ["", "   ", ".", " . "])
def test_reason_empty(raw):
    assert _reason(raw) == "empty"


def test_reason_whitespace():
    """Interior whitespace only — the surrounding kind is stripped by the fold."""
    assert _reason("app example.com") == "whitespace"


@pytest.mark.parametrize("raw", ["https://x.example", "x.example/path", "//x.example"])
def test_reason_not_bare(raw):
    assert _reason(raw) == "not_bare"


def test_reason_has_port():
    assert _reason("x.example:8443") == "has_port"


@pytest.mark.parametrize("raw", ["*.example.com", "*", "a.*.example.com"])
def test_reason_wildcard(raw):
    assert _reason(raw) == "wildcard"


@pytest.mark.parametrize("raw", ["192.168.1.50", "10.0.0.1", "[::1]", "::1", "2001:db8::1"])
def test_reason_ip_literal(raw):
    """IPv6 literals answer ``ip_literal``, never ``has_port``.

    The colon in an address is not a port separator; classifying it as one
    would send an operator to delete a port that is not there. This is the one
    deliberate deviation from the spec's literal check order.
    """
    assert _reason(raw) == "ip_literal"


@pytest.mark.parametrize("raw", ["bücher.example", "xn--bcher-kva.example", "münchen.de"])
def test_reason_idn(raw):
    """Both spellings refused, not transcoded: two names for one host would
    defeat "one domain, one service"."""
    assert _reason(raw) == "idn"


@pytest.mark.parametrize(
    "raw",
    [
        "\u017fervice.example.com",  # LATIN SMALL LETTER LONG S -> "s"
        "\u212aing.example.com",  # KELVIN SIGN -> "k"
        "\ufb01le.example.com",  # LATIN SMALL LIGATURE FI -> "fi"
    ],
)
def test_a_casefold_that_lands_on_ascii_is_still_refused(raw):
    """``casefold()`` is not ASCII-preserving, so the check reads the RAW input.

    ``ſ`` (U+017F) folds to ``s``, ``K`` (U+212A, Kelvin sign) to ``k`` and the
    ``ﬁ`` ligature to ``fi``. Tested against the folded form only, each of these
    would validate as a DIFFERENT, perfectly ASCII name — the silent rewriting
    of operator input this module says it refuses.
    """
    assert _reason(raw) == "idn"


@pytest.mark.parametrize("raw", ["intranet", "localhost", "box"])
def test_reason_single_label(raw):
    """A single-label name is a LAN hostname. ``box`` is *also* reserved here,
    but shape is judged before ownership — a name must be a domain before "is
    it ours?" is a meaningful question."""
    assert _reason(raw) == "single_label"


@pytest.mark.parametrize(
    "raw",
    [
        "a..b",
        "-a.example",
        "a-.example",
        ".a.example",
        "a_b.example",
        "a" * 64 + ".example",
        ("a" * 60 + ".") * 4 + "example.com",  # 255 chars — over the DNS name limit
    ],
)
def test_reason_grammar(raw):
    assert _reason(raw) == "grammar"


def test_grammar_message_is_the_shared_proxy_one():
    """The ``[proxy]`` validator already names the offending label; its text is
    passed through rather than replaced with a vaguer one."""
    with pytest.raises(DomainInvalid) as exc:
        validate_domain("-a.example", reserved=_RESERVED)
    assert "invalid label" in exc.value.message
    assert "'-a'" in exc.value.message


@pytest.mark.parametrize(
    "raw",
    [
        "dev.lan",  # equal to base_domain
        "app.dev.lan",  # under base_domain (a subdomain-mode service Host)
        "x.box",  # under the hostname
        "nerd.local",  # equal to an extra_hostnames entry
        "a.b.nerd.local",  # deep under one
        "DEV.LAN",  # case-folded before the comparison
        "app.dev.lan.",  # trailing dot folded before the comparison
    ],
)
def test_reason_reserved(raw):
    assert _reason(raw) == "reserved"


def test_reserved_is_suffix_not_substring():
    """``notdev.lan`` merely ENDS WITH the reserved text; it is a different
    zone and must be accepted."""
    assert validate_domain("notdev.lan", reserved=_RESERVED) == "notdev.lan"
    assert validate_domain("devxlan.example", reserved=_RESERVED) == "devxlan.example"


def test_domain_invalid_is_a_value_error_carrying_both_halves():
    exc = DomainInvalid("wildcard", "no wildcards please")
    assert isinstance(exc, ValueError)
    assert (exc.reason, exc.message) == ("wildcard", "no wildcards please")
    assert str(exc) == "no wildcards please"


# ---------------------------------------------------------------------------
# 2b. reserved_names — the boot-computed set
# ---------------------------------------------------------------------------


def test_reserved_names_unions_every_source_folded():
    proxy = ProxySettings(
        base_domain="dev.lan",
        hostname_override="nerd.local",
        extra_hostnames=["alt.example.org", "other.local"],
    )

    reserved = reserved_names(proxy, "Raw-Host.Local")

    assert reserved.names == {
        "nerd.local",  # hostname_override (the effective hostname)
        "raw-host.local",  # the socket hostname it falls back to, folded
        "dev.lan",
        "alt.example.org",
        "other.local",
    }


def test_reserved_names_drops_the_unset_sources():
    """A default ``[proxy]`` reserves the socket hostname and nothing else — no
    empty string, which would make ``covers`` true for every name."""
    reserved = reserved_names(ProxySettings(), "box.local")

    assert reserved.names == {"box.local"}
    assert not reserved.covers("app.example.com")


def test_reserved_names_is_hashable_and_frozen():
    reserved = reserved_names(ProxySettings(), "box.local")
    with pytest.raises(AttributeError):
        reserved.names = frozenset()  # type: ignore[misc]


def test_covers_on_an_empty_set_is_false():
    assert ReservedNames(names=frozenset()).covers("anything.example") is False


# ---------------------------------------------------------------------------
# 3. Route ids + domain URLs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("service", "domain"),
    [
        ("app", "app.example.com"),
        ("my--app", "a.b.c.d.e.example.org"),  # a service name containing '--'
        ("a", "x.example"),
        ("nerdit-route-lookalike", "deep.sub.domain.example.com"),
    ],
)
def test_route_id_round_trip(service, domain):
    assert _split_route_id(_domain_route_id(service, domain)) == (service, domain)


def test_split_route_id_reads_a_default_id_as_domainless():
    from nerdit.core.proxy import _route_id

    assert _split_route_id(_route_id("app")) == ("app", None)
    assert _split_route_id(_route_id("my--app")) == ("my--app", None)


def test_the_two_id_families_never_collide():
    """``@`` is legal in neither a service name nor a DNS name, so the first
    ``@`` is an exact seam (F7)."""
    default = _route_id_of("app")
    domain = _domain_route_id("app", "app.example.com")
    assert default != domain
    assert domain.startswith(default)  # …and is still recognised as the service's
    assert _split_route_id(domain)[0] == _split_route_id(default)[0]


def _route_id_of(name: str) -> str:
    from nerdit.core.proxy import _route_id

    return _route_id(name)


@pytest.mark.parametrize(
    ("https_port", "public_port", "expected"),
    [
        (443, None, "https://app.example.com/"),
        (8443, None, "https://app.example.com:8443/"),
        (8443, 443, "https://app.example.com/"),  # public_port wins…
        (443, 8443, "https://app.example.com:8443/"),  # …in both directions
    ],
)
def test_domain_url_for_port_rule(https_port, public_port, expected):
    url = domain_url_for(
        "app.example.com", scheme="https", https_port=https_port, public_port=public_port
    )
    assert url == expected


def test_domain_url_is_the_root_in_both_modes():
    """No path segment ever: the Host matcher carries the identity, which is
    the whole point of a custom domain over a path prefix."""
    assert domain_url_for("x.example", scheme="https", https_port=443).endswith("x.example/")


# ---------------------------------------------------------------------------
# 4. ordering_violations — the fifth drift dimension (S-W4)
# ---------------------------------------------------------------------------


def _live(**by_id: tuple[str, int]) -> dict[str, LiveRoute]:
    """Build a live table from ``id -> (shape, index)`` pairs."""
    return {
        rid: LiveRoute(dial="127.0.0.1:1", shape=shape, index=idx)
        for rid, (shape, idx) in by_id.items()
    }


def test_no_routes_no_violations():
    assert ordering_violations({}) == []


def test_domains_first_is_converged():
    live = _live(
        **{
            _domain_route_id("a", "a.example"): ("host", 0),
            _domain_route_id("b", "b.example"): ("host", 1),
            _route_id_of("a"): ("path", 2),
            _route_id_of("b"): ("path", 3),
        }
    )
    assert ordering_violations(live) == []


def test_a_domain_after_the_first_default_is_a_violation():
    live = _live(
        **{
            _route_id_of("a"): ("path", 0),
            _domain_route_id("b", "b.example"): ("host", 1),
        }
    )
    assert ordering_violations(live) == [_domain_route_id("b", "b.example")]


def test_only_the_offenders_are_named_and_they_come_back_in_array_order():
    live = _live(
        **{
            _domain_route_id("a", "a.example"): ("host", 0),  # already ahead — fine
            _route_id_of("a"): ("path", 1),
            _domain_route_id("c", "c.example"): ("host", 3),
            _domain_route_id("b", "b.example"): ("host", 2),
            _route_id_of("b"): ("path", 4),
        }
    )
    assert ordering_violations(live) == [
        _domain_route_id("b", "b.example"),
        _domain_route_id("c", "c.example"),
    ]


def test_domains_only_cannot_shadow_anything():
    """With no default route present there is nothing to sit behind."""
    live = _live(
        **{
            _domain_route_id("a", "a.example"): ("host", 5),
            _domain_route_id("b", "b.example"): ("host", 9),
        }
    )
    assert ordering_violations(live) == []


def test_subdomain_mode_defaults_are_host_shaped_and_the_same_rule_applies():
    """One rule, no mode branch: in subdomain mode the defaults are Host routes
    too, and a domain route behind them still counts as drifted."""
    live = _live(
        **{
            _route_id_of("a"): ("host", 0),
            _route_id_of("b"): ("host", 1),
            _domain_route_id("a", "shop.example.com"): ("host", 2),
        }
    )
    assert ordering_violations(live) == [_domain_route_id("a", "shop.example.com")]


def test_a_foreign_id_is_ignored():
    """The apex route lives outside ``_ID_PREFIX`` by design; a hand-added route
    is not ours to order."""
    live = _live(
        **{
            _domain_route_id("a", "a.example"): ("host", 1),
            _route_id_of("a"): ("path", 2),
        }
    )
    live["nerdit-apex"] = LiveRoute(dial="127.0.0.1:9321", shape="path", index=0)
    live["operator-handmade"] = LiveRoute(dial="127.0.0.1:5", shape="path", index=0)

    assert ordering_violations(live) == []


def test_the_default_index_reads_as_ahead_of_everything():
    """``index=-1`` means "position unknown" — it must never be reported as a
    violation, since every real index is greater than it."""
    live = {
        _route_id_of("a"): LiveRoute(dial="127.0.0.1:1", shape="path"),
        _domain_route_id("a", "a.example"): LiveRoute(dial="127.0.0.1:1", shape="host"),
    }
    assert ordering_violations(live) == []


# ---------------------------------------------------------------------------
# 5. RouteSpec defaults — a pre-WP1 construction is unchanged
# ---------------------------------------------------------------------------


def test_route_spec_defaults_keep_every_pre_wp1_construction_meaning_the_same():
    spec = RouteSpec(service_name="a", host_port=8001, route="/a", caddy_id=_route_id_of("a"))

    assert (spec.kind, spec.shape, spec.host) == ("default", "path", None)
    assert spec.auth is None


def test_a_domain_spec_is_host_shaped_in_either_mode():
    spec = RouteSpec(
        service_name="a",
        host_port=8001,
        route="",
        caddy_id=_domain_route_id("a", "app.example.com"),
        kind="domain",
        shape="host",
        host="app.example.com",
    )

    assert (spec.kind, spec.shape, spec.host) == ("domain", "host", "app.example.com")


def test_live_route_index_defaults_to_unknown():
    assert LiveRoute(dial=None, shape="path").index == -1


def test_host_aliases_cover_the_raw_socket_name_short_label_and_mdns():
    """In path mode the proxy answers on any Host, so the machine's other
    names are names it serves; the boot seam passes them as ``aliases``."""
    assert host_aliases("Box.corp.example.") == ("Box.corp.example", "Box", "Box.local")
    assert host_aliases("box") == ("box", "box.local")
    assert host_aliases("") == ()


def test_reserved_names_folds_caller_supplied_aliases():
    reserved = reserved_names(
        ProxySettings(), "nerd.local", aliases=host_aliases("Box.corp.example")
    )
    assert reserved.covers("box.corp.example")
    assert reserved.covers(normalize_domain("sub.BOX.corp.example."))
    assert reserved.covers("box.local")
    assert not reserved.covers("app.example.com")
