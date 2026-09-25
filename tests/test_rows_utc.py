"""Naive stored timestamps come back UTC-aware from every row model that carries one."""

from datetime import UTC, datetime

import pytest

from nerdit.db.rows import Event, Project, SecretClaim, ServiceDomain, ServiceShare

NAIVE = datetime(2026, 9, 1, 12, 0, 0)


@pytest.mark.parametrize(
    ("model", "kwargs", "field"),
    [
        (Project, {"id": "prj_1", "name": "asso"}, "created_at"),
        (ServiceShare, {"service_name": "web", "slug": "abc"}, "created_at"),
        (SecretClaim, {"service_name": "web"}, "created_at"),
        (ServiceDomain, {"service_name": "web", "domain": "app.example.com"}, "created_at"),
        (Event, {"id": 1, "type": "service.started"}, "ts"),
    ],
)
def test_naive_timestamp_becomes_utc(model, kwargs, field):
    row = model(**kwargs, **{field: NAIVE})
    assert getattr(row, field).tzinfo is UTC


def test_event_ts_serializes_aware():
    assert Event(id=1, type="x", ts=NAIVE).model_dump(mode="json")["ts"].endswith("+00:00")
