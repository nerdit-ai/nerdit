"""The ``extra='forbid'`` pass over route request bodies (P22 WP-C, D-P22-3).

An unknown field used to be dropped in silence — the write succeeded and the
caller's intent did not. Every route-bound request model now inherits
:class:`~nerdit.daemon.schemas._base.StrictRequestModel`, so a typo is a 422
naming the offending key.

The out-of-scope models are pinned by their own suites and stay tolerant by
decision: ``DeployConfig`` (``tests/test_deploy_route.py``, forward compat for
``nerdit.toml``), the ``ConfigStore`` Settings models
(``tests/test_config_store.py``), and nested ``HealthCheck`` (also parsed from
stored DB JSON).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from nerdit.daemon.routes.secrets import SecretSetRequest
from nerdit.daemon.routes.system import GcRequest, RestartRequest, VolumeBackupRequest
from nerdit.daemon.schemas._base import StrictRequestModel
from nerdit.daemon.schemas.config import ConfigApplyRequest
from nerdit.daemon.schemas.databases import DatabaseCreateRequest
from nerdit.daemon.schemas.deploy import GitDeployRequest, TemplateDeployRequest
from nerdit.daemon.schemas.models import ModelServeRequest
from nerdit.daemon.schemas.service_run import ServiceRunRequest
from nerdit.daemon.schemas.services import ServiceCreateRequest
from nerdit.daemon.schemas.tokens import TokenCreateRequest
from nerdit.daemon.schemas.workspaces import WorkspaceDeployRequest, WorkspaceWriteRequest

# Every route request-body model in the D-P22-3 inventory, with a MINIMAL valid
# payload. The extra key is added by the test, so a failure here is always the
# extra key and never a missing required field.
_STRICT_MODELS: list[tuple[type[BaseModel], dict]] = [
    (ServiceCreateRequest, {"name": "demo", "image": "nerdit-runtime:0.1"}),
    (TokenCreateRequest, {"name": "ci"}),
    (ConfigApplyRequest, {"sections": {"proxy": {"enabled": True}}}),
    (GitDeployRequest, {"repo_url": "https://github.com/x/y", "name": "demo"}),
    (TemplateDeployRequest, {"name": "demo"}),
    (DatabaseCreateRequest, {}),
    (ModelServeRequest, {"model": "llama3.1:8b"}),
    (ServiceRunRequest, {"command": ["true"]}),
    (SecretSetRequest, {"values": {"K": "v"}}),
    (RestartRequest, {}),
    (GcRequest, {}),
    (VolumeBackupRequest, {"service": "pg"}),
    (WorkspaceWriteRequest, {"files": {"main.py": "x"}}),
    (WorkspaceDeployRequest, {}),
]


@pytest.mark.parametrize("model, payload", _STRICT_MODELS, ids=lambda a: getattr(a, "__name__", ""))
def test_unknown_field_is_rejected(model: type[BaseModel], payload: dict) -> None:
    """A bogus key 422s instead of being silently dropped."""
    assert issubclass(model, StrictRequestModel)
    model(**payload)  # the baseline payload is valid

    with pytest.raises(ValidationError) as exc:
        model(**payload, definitely_not_a_field="oops")
    assert "definitely_not_a_field" in str(exc.value)


def test_config_apply_sections_stay_open() -> None:
    """Strict at the top level only — ``sections`` is an open dict by contract."""
    body = ConfigApplyRequest(sections={"proxy": {"anything": "at", "all": 1}})
    assert body.sections["proxy"]["anything"] == "at"
