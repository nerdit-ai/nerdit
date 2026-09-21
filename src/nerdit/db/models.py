"""Compatibility re-exports for historical model imports; definitions live in domain modules."""

from __future__ import annotations

from nerdit.core.runtime.container import ContainerConfig  # noqa: F401
from nerdit.daemon.schemas.audit import AuditLogPage  # noqa: F401
from nerdit.daemon.schemas.cluster import ClusterInfo, ClusterStats  # noqa: F401
from nerdit.daemon.schemas.config import (  # noqa: F401
    AppConfigView,
    AppConfigWriteResponse,
    ConfigApplyRequest,
    ConfigApplyResponse,
    ConfigDiagnostic,
    ConfigDiffEntry,
    ConfigView,
    ConfigWriteResponse,
)
from nerdit.daemon.schemas.databases import (  # noqa: F401
    DatabaseCreateRequest,
    DatabaseListPage,
    DatabaseResponse,
)
from nerdit.daemon.schemas.deploy import (  # noqa: F401
    AppTemplate,
    AppTemplateDeployDefaults,
    AppTemplateEnvVar,
    GitDeployRequest,
    TemplateDeployRequest,
)
from nerdit.daemon.schemas.gpus import GpuResponse  # noqa: F401
from nerdit.daemon.schemas.health import HealthResponse  # noqa: F401
from nerdit.daemon.schemas.models import (  # noqa: F401
    ModelListPage,
    ModelResponse,
    ModelServeRequest,
)
from nerdit.daemon.schemas.proxy import (  # noqa: F401
    ProxyStatusResponse,
    RouteEndpoint,
    RouteItem,
    RouteListPage,
)
from nerdit.daemon.schemas.service_diagnose import (  # noqa: F401
    DiagnoseBindings,
    DiagnoseBuild,
    DiagnoseError,
    DiagnoseForensics,
    DiagnoseHealth,
    DiagnoseRemediation,
    DiagnoseResponse,
    DiagnoseRestarts,
)
from nerdit.daemon.schemas.service_purge import (  # noqa: F401
    PurgeImages,
    PurgeReport,
    ServiceDeletedResponse,
)
from nerdit.daemon.schemas.service_run import (  # noqa: F401
    ServiceRunRequest,
    ServiceRunResponse,
)
from nerdit.daemon.schemas.service_wait import ServiceWaitResponse  # noqa: F401
from nerdit.daemon.schemas.services import (  # noqa: F401
    ServiceCreateRequest,
    ServiceEndpointView,
    ServiceListPage,
    ServiceResponse,
)
from nerdit.daemon.schemas.system import BackupResponse, VolumeBackupResponse  # noqa: F401
from nerdit.daemon.schemas.tokens import (  # noqa: F401
    TokenCreateRequest,
    TokenCreateResponse,
    TokenRotateRequest,
    TokenSelfView,
    TokenView,
)
from nerdit.db.enums import (  # noqa: F401
    MANAGED_KINDS,
    MANAGED_KINDS_SQL,
    ErrorClass,
    GpuDiscoveryBackend,
    GpuStatus,
    GpuVendor,
    JobKind,
    JobStatus,
    LogStream,
    TokenRole,
)
from nerdit.db.rows import (  # noqa: F401
    ActiveServiceRoute,
    ApiToken,
    AuditLogEntry,
    Gpu,
    GpuMetrics,
    HealthCheck,
    IdempotencyRecord,
    Job,
    LogEntry,
    Project,
    SecretClaim,
    ServiceDomain,
    ServiceEndpoint,
    ServiceShare,
    VariableFlag,
)
