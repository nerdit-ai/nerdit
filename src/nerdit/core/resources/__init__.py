"""Shared backend lookup, image pulls, and readiness support for resource controllers."""

from __future__ import annotations

from nerdit.core.resources.controller import ResourceController
from nerdit.core.resources.registry import BackendRegistry

__all__ = ["BackendRegistry", "ResourceController"]
