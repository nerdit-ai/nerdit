"""Look up resource backends for write paths and reconciliation.

Explicit lookups reject unknown names. Reconcile lookups warn and fall back to
the default so old or malformed rows remain recoverable.
"""

from __future__ import annotations

import logging
from typing import Generic, Protocol, TypeVar

logger = logging.getLogger(__name__)


class _NamedBackend(Protocol):
    """The one attribute every registered backend must expose: its own name."""

    name: str


B = TypeVar("B", bound=_NamedBackend)


class BackendRegistry(Generic[B]):
    """Register a default backend and named alternatives.

    Omitted `default_name` uses the supplied backend's name; an override must be
    registered. `kind_label` identifies fallback warnings.
    """

    def __init__(
        self,
        default: B,
        extra: dict[str, B] | None = None,
        default_name: str | None = None,
        *,
        kind_label: str,
    ) -> None:
        self._default = default
        self._backends: dict[str, B] = {default.name: default}
        if extra:
            self._backends.update(extra)
        self._default_name = default_name or default.name
        self._kind_label = kind_label

    @property
    def default(self) -> B:
        """The default backend."""
        return self._default

    def get(self, name: str | None) -> B | None:
        """Resolve an *explicitly requested* backend by name, no fallback.

        `None` selects the default.
        """
        return self._backends.get(name or self._default_name)

    @property
    def default_name(self) -> str:
        return self._default_name

    @property
    def names(self) -> list[str]:
        """Sorted names of every registered backend."""
        return sorted(self._backends)

    def resolve(self, cfg: dict) -> B:
        """Resolve a row's configured backend, warning and using the default if unknown."""
        name = cfg.get("backend") or self._default_name
        backend = self._backends.get(name)
        if backend is None:
            logger.warning(
                "%s config names unknown backend %r; falling back to %r",
                self._kind_label,
                name,
                self._default_name,
            )
            return self._backends[self._default_name]
        return backend
