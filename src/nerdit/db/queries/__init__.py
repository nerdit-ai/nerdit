"""Async SQLite queries composed from domain mixins sharing a connection and write lock."""

from __future__ import annotations

from ._base import (
    PortRangeExhausted,
    ProjectExists,
    ProjectOwned,
    ServiceNameClaimed,
    ServiceNameTaken,
    _serialized,
)
from .audit import AuditQueries
from .domains import DomainQueries
from .endpoints import EndpointQueries
from .events import EventQueries
from .gpus import GpuQueries
from .idempotency import IdempotencyQueries
from .logs import LogQueries
from .projects import ProjectQueries
from .secret_claims import SecretClaimQueries
from .service_config import ServiceConfigQueries
from .services import ServiceQueries
from .shares import ShareQueries
from .tokens import TokenQueries
from .workloads import WorkloadQueries

__all__ = [
    "PortRangeExhausted",
    "ProjectExists",
    "ProjectOwned",
    "Queries",
    "ServiceNameClaimed",
    "ServiceNameTaken",
    "_serialized",
]


class Queries(
    GpuQueries,
    TokenQueries,
    AuditQueries,
    EventQueries,
    IdempotencyQueries,
    LogQueries,
    WorkloadQueries,
    ServiceQueries,
    EndpointQueries,
    ServiceConfigQueries,
    ShareQueries,
    DomainQueries,
    SecretClaimQueries,
    ProjectQueries,
):
    """Async query interface shared by routes and scheduling.

    Multi-statement writes use BEGIN IMMEDIATE under the serialized writer lock:
    aiosqlite alone serializes statements, not coroutine transactions. Every mixin
    inherits the same connection and row-conversion helpers from QueriesBase.
    """
