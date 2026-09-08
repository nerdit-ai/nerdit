"""Mask secret config leaves consistently in views, diffs, audits and replay storage.

SECRET_LEAF_KEYS identifies secrets by leaf name, covering current and future
nested AI/database bindings without hard-coded section paths.
"""

from __future__ import annotations

from typing import Any

# Leaf key names whose value is a secret and must always be masked. ``password``
# (P15) is a ``${secrets.X}`` ref by grammar — inherently safe — but is masked
# defensively so a hand-forged blob never leaks a literal through a view/diff.
SECRET_LEAF_KEYS: frozenset[str] = frozenset({"auth_token", "api_key", "password"})

# Replacement token written in place of a real secret value.
REDACTED = "***"


def is_secret_key(key: str) -> bool:
    """Return whether a leaf config key holds a secret value."""
    return key.lower() in SECRET_LEAF_KEYS


def redact_value(key: str, value: Any) -> Any:
    """Mask `value` when `key` is secret and the value is actually set.

    `None` (an unset secret) is left as `None` so a redacted view does not
    falsely imply a secret exists.
    """
    if value is not None and is_secret_key(key):
        return REDACTED
    return value


def redact_section(values: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of a config section with secret leaves masked."""
    return {key: redact_value(key, value) for key, value in values.items()}
