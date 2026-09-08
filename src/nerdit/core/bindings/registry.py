"""Map binding kinds to injected key names and shared-secret fields.

Diagnostics use `BINDING_KINDS` to inspect mixed AI/database bindings without
resolving secrets. Kind-specific parsing, resolution, and injection remain in
`models.binding` and `data.binding`, whose result shapes differ.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from nerdit.core.data.binding import inject_db_env_key_names
from nerdit.core.models.binding import inject_env_key_names


@dataclass(frozen=True)
class BindingKind:
    """Per-kind descriptor (see the module docstring for the shape divergence)."""

    key_names_fn: Callable
    shared_secret_fields: tuple[str, ...]


BINDING_KINDS: dict[str, BindingKind] = {
    "ai": BindingKind(
        key_names_fn=inject_env_key_names,
        shared_secret_fields=("api_key",),
    ),
    "db": BindingKind(
        key_names_fn=inject_db_env_key_names,
        shared_secret_fields=("password",),
    ),
}
