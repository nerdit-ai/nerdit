"""P26 WP-H T1 — the pure hosted-name arithmetic (``core/link/hosted.py``, D-P26-H5).

Every function here is I/O-free, so the tests are plain asserts. The cases that
matter are the ones the design turns on: ``--`` is legal *inside* a service
name (so nothing may split on it), the combined label is capped at 63, and the
service-name grammar is the same one the API pins.
"""

from __future__ import annotations

import pytest

from nerdit.core.link.hosted import (
    HOSTED_SEPARATOR,
    MAX_DNS_LABEL,
    hosted_host,
    hosted_label,
    hosted_label_fits,
    hosted_url,
    is_service_name,
)
from nerdit.daemon.schemas.services import ServiceCreateRequest
from nerdit.utils.names import DNS_LABEL_RE

# --- composition --------------------------------------------------------------


def test_label_joins_with_the_double_hyphen():
    assert hosted_label("demo", "gpu-box") == "demo--gpu-box"
    assert HOSTED_SEPARATOR == "--"


def test_host_and_url_are_built_from_the_label():
    assert hosted_host("demo", "gpu-box", "nodes.test") == "demo--gpu-box.nodes.test"
    assert hosted_url("demo", "gpu-box", "nodes.test") == "https://demo--gpu-box.nodes.test/"


def test_url_is_always_https_and_root_relative():
    """The app is served at the ROOT of its own hostname — that is the whole
    point of the hosted path (no base path, unlike path-mode proxying)."""
    url = hosted_url("my-app", "n", "nodes.test")
    assert url.startswith("https://")
    assert url.endswith("/")
    assert url.count("/") == 3  # scheme's two + the trailing one, no path segment


def test_a_service_name_may_itself_contain_the_separator():
    """§0.2-7: ``--`` is legal inside a service name, which is exactly why the
    cloud resolves by SUFFIX match against registered slugs and nothing here
    ever splits."""
    assert is_service_name("my--app")
    assert hosted_label("my--app", "node") == "my--app--node"
    assert hosted_host("my--app", "node", "nodes.test") == "my--app--node.nodes.test"


# --- the 63-octet label ceiling -----------------------------------------------


def test_fits_at_and_past_the_boundary():
    slug = "n"
    exact = "a" * (MAX_DNS_LABEL - len(HOSTED_SEPARATOR) - len(slug))

    assert len(hosted_label(exact, slug)) == MAX_DNS_LABEL
    assert hosted_label_fits(exact, slug) is True
    assert hosted_label_fits(exact + "a", slug) is False


def test_fits_counts_the_separator_not_just_the_two_names():
    """61 + 1 = 62 characters of name and slug still overflow once ``--`` is in."""
    assert hosted_label_fits("a" * 61, "bb") is False
    assert hosted_label_fits("a" * 60, "b") is True


def test_fits_ignores_the_domain():
    """Only the LABEL is capped at 63; the domain rides behind the first dot."""
    assert hosted_label_fits("demo", "n") is True
    assert len(hosted_host("demo", "n", "a" * 200 + ".test")) > MAX_DNS_LABEL


# --- the service-name gate ----------------------------------------------------


@pytest.mark.parametrize("value", ["a", "demo", "my-app", "my--app", "a1", "a" * 63])
def test_valid_service_names(value):
    assert is_service_name(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "",
        "A.b",  # uppercase + dot: two labels, and not lowercase
        "a" * 64,  # past the label ceiling
        "-x",  # leading hyphen
        "x-",  # trailing hyphen
        "--x",  # leading separator
        "a.b",  # a dot is a label separator, never part of one
        "a_b",
        "a b",
        "a/b",
        "demo\n",  # a trailing newline must not slip past a non-fullmatch
    ],
)
def test_invalid_service_names(value):
    assert is_service_name(value) is False


def test_grammar_matches_the_api_service_name_pattern():
    """The header gate and the create-request gate must never drift: a name the
    API accepts must be routable, and one it refuses must not be."""
    field = ServiceCreateRequest.model_fields["name"]
    patterns = [m.pattern for m in field.metadata if hasattr(m, "pattern")]
    assert DNS_LABEL_RE.pattern in patterns
