"""Base model for request bodies that reject unknown fields with a 422.

Strictness applies to HTTP body models only. On-disk DeployConfig and Settings
remain tolerant for forward compatibility, as does HealthCheck parsed from DB
JSON. Raw config PUT dictionaries are validated downstream; ConfigApplyRequest
is strict only at the top level, with an open sections mapping.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictRequestModel(BaseModel):
    """A request body that rejects unknown fields instead of dropping them."""

    model_config = ConfigDict(extra="forbid")
