"""The shared DNS-label grammar refuses a trailing newline at every gate (S7)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nerdit.config.project import DeployConfig
from nerdit.config.settings import _validate_dns_name
from nerdit.daemon.deploy_pipeline import _validate_request_name
from nerdit.daemon.errors import NerditError
from nerdit.daemon.schemas.tokens import TokenCreateRequest
from nerdit.daemon.secret_scope import guard_name
from nerdit.utils.names import DNS_LABEL_RE


def test_fullmatch_refuses_trailing_newline():
    assert DNS_LABEL_RE.match("foo\n") is not None  # why `.match` is never used
    assert DNS_LABEL_RE.fullmatch("foo\n") is None


@pytest.mark.parametrize(
    ("gate", "exc"),
    [
        (lambda: guard_name("foo\n"), NerditError),
        (lambda: _validate_request_name("foo\n"), NerditError),
        (lambda: DeployConfig(name="foo\n"), ValidationError),
        (lambda: TokenCreateRequest(name="t", scope_services=["foo\n"]), ValidationError),
        (lambda: _validate_dns_name("a.b\n", field_name="hostname"), ValueError),
    ],
    ids=["guard_name", "request_name", "deploy_config", "token_scope", "hostname"],
)
def test_trailing_newline_name_is_refused(gate, exc):
    with pytest.raises(exc):
        gate()
