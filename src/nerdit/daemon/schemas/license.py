"""Schemas for installing and removing offline product licenses.

Treat blob as a secret in transit. repr=False protects model representations;
errors._SECRET_INPUT_FIELDS and audit._REDACT_KEYS protect validation echoes
and audit output, including errors raised before a model exists. The length
limit matches MAX_LICENSE_BYTES used by the file reader.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from nerdit.core.license import MAX_LICENSE_BYTES
from nerdit.daemon.schemas._base import StrictRequestModel


class LicenseInstallRequest(StrictRequestModel):
    """`POST /api/license` — one compact JWS, installed as-is."""

    blob: str = Field(min_length=1, max_length=MAX_LICENSE_BYTES, repr=False)


class LicenseInstallView(BaseModel):
    """Verified license claims and computed state; never the submitted blob.

    Customer ID is visible because installation is admin-only. Signature-valid
    expired or grace-period licenses are accepted; the CLI warns about their state.
    """

    lid: str
    plan: str
    features: list[str]
    state: str
    expires_at: str
    expires_in_s: int | None
    customer_id: str
    installed: bool = True


class LicenseRemoveView(BaseModel):
    """`DELETE /api/license` — the unlink report-what-happened idiom.

    `removed` is false on a double delete (a 200, never a 404), so cleanup
    scripts and retries are safe.
    """

    removed: bool
