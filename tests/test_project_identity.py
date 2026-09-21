"""Tests for ``nerdit.core.project_identity`` (P40a: D-P40-2 / D-P40-6 / D-P40-14)."""

from __future__ import annotations

import pytest

from nerdit.core.project_identity import (
    DEFAULT_SERVICE,
    ENV_NAME_RE,
    PRODUCTION,
    PROJECT_NAME_RE,
    SERVICE_NAME_RE,
    mint_project_id,
    parse_qualified,
    service_label,
)

# --- grammar (D-P40-14) ---


@pytest.mark.parametrize("name", ["a", "asso", "my-app", "a1-b2", "x" * 40, "a" + "-b" * 19 + "c"])
def test_project_name_accepts_dns_labels_up_to_40(name):
    assert PROJECT_NAME_RE.fullmatch(name)


@pytest.mark.parametrize(
    "name",
    ["", "-a", "a-", "A", "a_b", "a.b", "x" * 41, "a--b", "--", "a/b", "asso\n"],
)
def test_project_name_rejects_bad_labels_and_double_dash(name):
    assert PROJECT_NAME_RE.fullmatch(name) is None


@pytest.mark.parametrize("regex", [SERVICE_NAME_RE, ENV_NAME_RE])
def test_service_and_env_names_are_the_same_grammar_capped_at_20(regex):
    assert regex.fullmatch("web")
    assert regex.fullmatch("x" * 20)
    assert regex.fullmatch("x" * 21) is None
    assert regex.fullmatch("a--b") is None
    assert regex.fullmatch("Web") is None


def test_constants():
    assert PRODUCTION == "production"
    assert DEFAULT_SERVICE == "web"


# --- ids ---


def test_mint_project_id_shape_and_uniqueness():
    ids = {mint_project_id() for _ in range(64)}
    assert len(ids) == 64
    for pid in ids:
        assert pid.startswith("prj_")
        assert len(pid) == 20
        # RFC 4648 base32 alphabet, lowercased, no padding.
        assert set(pid[4:]) <= set("abcdefghijklmnopqrstuvwxyz234567")


# --- service_label (D-P40-2) ---


def test_production_web_is_the_project_name():
    assert service_label("asso", PRODUCTION, DEFAULT_SERVICE) == "asso"
    # A migrated 63-char legacy label stays itself (grammar-exempt, D-P40-14).
    legacy = "x" * 63
    assert service_label(legacy, PRODUCTION, DEFAULT_SERVICE) == legacy


def test_other_production_service_is_service_dashdash_project():
    assert service_label("asso", PRODUCTION, "api") == "api--asso"


def test_non_production_is_the_three_part_form():
    assert service_label("asso", "staging", "api") == "api--staging--asso"
    assert service_label("asso", "staging", DEFAULT_SERVICE) == "web--staging--asso"


def test_label_over_63_octets_is_refused():
    with pytest.raises(ValueError):
        service_label("x" * 60, PRODUCTION, "api")  # 65 octets


@pytest.mark.parametrize(
    ("project", "environment", "service"),
    [
        ("", PRODUCTION, DEFAULT_SERVICE),
        ("Asso", PRODUCTION, DEFAULT_SERVICE),
        ("asso", PRODUCTION, ""),
    ],
)
def test_label_outside_the_dns_grammar_is_refused(project, environment, service):
    with pytest.raises(ValueError):
        service_label(project, environment, service)


# --- parse_qualified (D-P40-6) ---


def test_parse_qualified_segments():
    assert parse_qualified("asso") == ("asso", PRODUCTION, DEFAULT_SERVICE)
    assert parse_qualified("asso/api") == ("asso", PRODUCTION, "api")
    assert parse_qualified("asso/production/api") == ("asso", PRODUCTION, "api")
    # A foreign middle segment is the caller's judgment, not the parser's.
    assert parse_qualified("asso/staging/api") == ("asso", "staging", "api")


@pytest.mark.parametrize("text", ["", "/", "asso/", "/api", "asso//api", "a/b/c/d"])
def test_parse_qualified_refuses_empty_segments_and_more_than_three(text):
    with pytest.raises(ValueError):
        parse_qualified(text)


def test_parse_then_label_round_trips_the_legacy_name():
    assert service_label(*parse_qualified("asso")) == "asso"
    assert service_label(*parse_qualified("asso/api")) == "api--asso"
