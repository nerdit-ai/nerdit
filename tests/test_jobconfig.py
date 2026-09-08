"""Tests for ``core/jobconfig.py`` (Track B WP12 — job-config unification).

``parse_job_config`` replaces the nine named ``_parse_*_config``/``_job_config``
degrade-to-``{}`` helpers plus the inline ``_row_backend`` parse body. It must
tolerate every input shape the nine predecessors saw: a real
:class:`~nerdit.db.models.Job` row (six of the nine copies), a raw ``config``
string (the two binding-module ``_job_config`` copies), and an object without a
``.config`` attribute at all (the ``_row_backend`` ``getattr`` guard).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from nerdit.core.jobconfig import parse_job_config
from nerdit.db.models import Job, JobKind, JobStatus


def _job(config: str | None) -> Job:
    return Job(
        id="svc-1",
        service_name="my-app",
        name="my-app",
        kind=JobKind.service,
        gpu_count=0,
        status=JobStatus.running,
        desired_state="running",
        restart_policy="always",
        restart_count=0,
        config=config,
    )


def test_valid_dict_config_parses() -> None:
    assert parse_job_config(_job('{"a": 1, "b": "x"}')) == {"a": 1, "b": "x"}


def test_none_config_degrades_to_empty_dict() -> None:
    assert parse_job_config(_job(None)) == {}


def test_empty_string_config_degrades_to_empty_dict() -> None:
    assert parse_job_config(_job("")) == {}


def test_invalid_json_degrades_to_empty_dict() -> None:
    assert parse_job_config(_job("{not json")) == {}


@pytest.mark.parametrize("blob", ["[1, 2, 3]", '"just a string"', "42", "null"])
def test_non_dict_json_degrades_to_empty_dict(blob: str) -> None:
    assert parse_job_config(_job(blob)) == {}


def test_non_str_config_raises_typeerror_internally_and_degrades() -> None:
    # json.loads(int) raises TypeError, not JSONDecodeError — exercises the
    # TypeError arm of the (TypeError, ValueError) tuple.
    row = SimpleNamespace(id="svc-2", config=123)
    assert parse_job_config(row) == {}


def test_object_without_config_attribute_degrades_to_empty_dict() -> None:
    row = SimpleNamespace(id="svc-3")
    assert parse_job_config(row) == {}


def test_bare_string_argument_has_no_config_attribute() -> None:
    # A plain str has no .config attribute, so getattr(job, "config", None)
    # falls through to {} just like any other config-less object — the two
    # former _job_config(raw: str | None) binding-module copies took the raw
    # string directly, but their call sites are repointed to pass the row
    # itself (parse_job_config(row)), never the bare string.
    assert parse_job_config("{}") == {}


def test_warn_true_logs_warning_with_job_id_on_invalid_json(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="nerdit.core.jobconfig"):
        result = parse_job_config(_job("{bad"), warn=True)
    assert result == {}
    assert len(caplog.records) == 1
    assert "svc-1" in caplog.records[0].message
    assert "invalid config JSON" in caplog.records[0].message


def test_warn_false_is_silent_on_invalid_json(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="nerdit.core.jobconfig"):
        result = parse_job_config(_job("{bad"), warn=False)
    assert result == {}
    assert caplog.records == []


def test_warn_true_is_silent_on_valid_config() -> None:
    # warn only fires on the invalid-JSON path, never on a merely-empty config.
    logger = logging.getLogger("nerdit.core.jobconfig")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger.addHandler(handler)
    try:
        assert parse_job_config(_job(None), warn=True) == {}
        assert parse_job_config(_job('{"a": 1}'), warn=True) == {"a": 1}
    finally:
        logger.removeHandler(handler)
    assert records == []
