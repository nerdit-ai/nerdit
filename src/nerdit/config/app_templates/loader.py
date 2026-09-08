"""Load embedded app templates for the deploy store."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files

from pydantic import ValidationError

from nerdit.db.models import AppTemplate


class AppTemplateLoadError(RuntimeError):
    """Raised when embedded app templates are invalid."""


def load_app_templates() -> list[AppTemplate]:
    """Load and validate built-in app templates."""
    try:
        raw = json.loads(
            (files("nerdit.config.app_templates") / "builtin.json").read_text(encoding="utf-8")
        )
        return [AppTemplate.model_validate(item) for item in raw]
    except (json.JSONDecodeError, ValidationError, OSError) as exc:
        raise AppTemplateLoadError(f"Failed to load app templates: {exc}") from exc


@lru_cache(maxsize=1)
def app_templates_by_id() -> dict[str, AppTemplate]:
    """Return built-in app templates keyed by id, cached in-process.

    The templates are embedded and immutable, so the parsed mapping is built
    once and reused (avoids file I/O + JSON + validation on hot request paths).
    """
    return {template.id: template for template in load_app_templates()}
