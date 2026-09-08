"""ID generation utilities."""

import secrets
import string

_ALPHABET = string.ascii_lowercase + string.digits
_ID_LENGTH = 12


def generate_id() -> str:
    """Generate a 12-character alphanumeric ID."""
    return "".join(secrets.choice(_ALPHABET) for _ in range(_ID_LENGTH))
