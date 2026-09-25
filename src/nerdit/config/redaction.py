"""Mask secret config leaves consistently in views, diffs, audits and replay storage.

SECRET_LEAF_KEYS identifies secrets by leaf name, covering current and future
nested AI/database bindings without hard-coded section paths.
"""

from __future__ import annotations

import re
from typing import Any

# Userinfo through the last `@` before the path, as urlsplit reads it —
# including raw whitespace and an `@` inside the password. Regex, not a parse,
# so it is safe on a URL that failed to parse at all.
_URL_USERINFO_RE = re.compile(r"//[^/]*@")

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
    """Mask secret leaves and credentials embedded in URL values.

    `None` (an unset secret) is left as `None` so a redacted view does not
    falsely imply a secret exists.
    """
    if value is not None and is_secret_key(key):
        return REDACTED
    if key.lower() in {"url", "base_url"}:
        return redact_url_userinfo(value)
    return value


def redact_url_userinfo(value: Any) -> Any:
    """Strip `user:password@` from a URL so it is safe to record.

    Leaf-name masking cannot see a credential embedded IN a value, and a URL
    is the one config value that routinely carries one. Non-strings pass
    through untouched so callers can map it over a spec dict.
    """
    if not isinstance(value, str):
        return value
    return _URL_USERINFO_RE.sub("//", value)


def redact_section(values: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of a config section with secrets masked."""
    return {key: redact_value(key, value) for key, value in values.items()}
