import type { Page, Route } from "@playwright/test";

// JSON snapshots mirroring the daemon's response shapes. Update if Pydantic
// models drift; the matching test will fail on field-name changes.

const SAMPLE_GPUS = [
  {
    id: "GPU-A100-001",
    name: "NVIDIA A100 80GB",
    memory_mb: 81920,
    vendor: "nvidia",
    schedulable: true,
    status: "idle",
    utilization_percent: 12,
    memory_used_mb: 4096,
    temperature_c: 42
  },
  {
    id: "GPU-A100-002",
    name: "NVIDIA A100 80GB",
    memory_mb: 81920,
    vendor: "nvidia",
    schedulable: true,
    status: "idle",
    utilization_percent: 0,
    memory_used_mb: 1024,
    temperature_c: 38
  }
];

// Mixed-vendor variant: an AMD GPU present in inventory but not schedulable
// (enable_amd off). Kept separate so the default specs' viewport is unchanged.
export const SAMPLE_GPUS_WITH_AMD_INVENTORY = [
  ...SAMPLE_GPUS,
  {
    id: "gpu:amd:0",
    name: "AMD Instinct MI300X",
    memory_mb: 196608,
    vendor: "amd",
    schedulable: false,
    status: "idle",
    utilization_percent: null,
    memory_used_mb: null,
    temperature_c: null
  }
];

export const SAMPLE_CLUSTER_STATS = {
  gpus_total: 2,
  gpus_in_use: 0,
  gpus_avg_utilization: 6,
  services_up: 2,
  daemon_uptime_seconds: 120,
  daemon_version: "0.3.0"
};

// Mirrors ServiceListPage / ServiceResponse (db/models.py). Two rows cover the
// public-URL matrix: routed (clickable HTTPS) and proxy-off (public_url null
// while an endpoint exists — an expected state, never an error).
export const SAMPLE_SERVICES = {
  items: [
    {
      id: "svc1aaaa0001",
      name: "my-app",
      // (P40b) The owning project + the service name inside it. A single-service
      // project's one label IS the project name; the UI reads these, never the label.
      project: "my-app",
      project_id: "prj_myapp00000000001",
      service: "web",
      status: "running",
      desired_state: "running",
      kind: "service",
      image: "nerdit-app/my-app:3",
      gpu_count: 0,
      gpu_ids: [],
      restart_policy: "always",
      restart_count: 1,
      health_check: null,
      container_id: "ctr-my-app",
      created_at: "2026-07-04T10:00:00+00:00",
      started_at: "2026-07-04T10:01:00+00:00",
      finished_at: null,
      exit_code: null,
      error_class: null,
      error_message: null,
      submitted_via: "cli",
      // Git provenance (ServiceSource): drives the source-aware surfaces —
      // repo/ref on the app page, and a redeploy that has a source to re-clone.
      source: {
        type: "git",
        repo_url: "https://github.com/nerdit-ai/nerdit-templates",
        ref: "main",
        commit_sha: "9f2c1ab4d5e6f70819a2b3c4d5e6f708192a3b4c"
      },
      endpoint: {
        container_port: 8000,
        host_port: 38000,
        protocol: "tcp",
        route: "/my-app",
        url: "http://127.0.0.1:38000",
        public_url: "https://test-host.nerdit.internal/my-app"
      },
      rollback_available: true,
      build_version: 3,
      last_deploy: {
        version: 3,
        action: "redeploy",
        phase: "healthy",
        image: "nerdit-app/my-app:3",
        started_at: "2026-07-04T10:00:30+00:00",
        updated_at: "2026-07-04T10:01:00+00:00"
      }
    },
    {
      id: "svc2bbbb0002",
      name: "worker-api",
      project: "worker-api",
      project_id: "prj_workerapi0000002",
      service: "web",
      status: "running",
      desired_state: "running",
      kind: "service",
      image: "nerdit-app/worker-api:1",
      gpu_count: 0,
      gpu_ids: [],
      restart_policy: "on-failure",
      restart_count: 0,
      health_check: null,
      container_id: "ctr-worker",
      created_at: "2026-07-04T09:00:00+00:00",
      started_at: "2026-07-04T09:00:30+00:00",
      finished_at: null,
      exit_code: null,
      error_class: null,
      error_message: null,
      submitted_via: "cli",
      // The other half of the provenance matrix: a ZIP-sourced row, which the
      // daemon refuses to redeploy (there is no recorded source to re-clone).
      source: { type: "zip" },
      endpoint: {
        container_port: 3000,
        host_port: 38001,
        protocol: "tcp",
        route: null,
        url: "http://127.0.0.1:38001",
        public_url: null
      },
      rollback_available: false,
      build_version: 1,
      // A redeploy in flight: last_deploy building while the old image still
      // serves (status stays running) — drives the deploy phase line (WP7).
      last_deploy: {
        version: 2,
        action: "redeploy",
        phase: "building",
        image: "nerdit-app/worker-api:2",
        started_at: "2026-07-04T09:05:00+00:00",
        updated_at: "2026-07-04T09:05:10+00:00"
      }
    }
  ],
  next_cursor: null
};

// (P40e) The multi-service variant: project `asso` = `web` (label `asso`) +
// `api` (label `api--asso`), plus a model row whose project fields are null.
// Kept separate (the AMD-inventory precedent) so the default specs' two-row
// list is unchanged. `/api/projects*` is DERIVED from whichever service page a
// spec passes, so the two surfaces cannot disagree.
const ASSO_WEB = {
  ...SAMPLE_SERVICES.items[0],
  id: "svc3cccc0003",
  name: "asso",
  project: "asso",
  project_id: "prj_asso000000000003",
  service: "web",
  image: "nerdit-app/asso:1",
  container_id: "ctr-asso",
  source: { type: "zip" },
  endpoint: {
    container_port: 8000,
    host_port: 38002,
    protocol: "tcp",
    route: "/asso",
    url: "http://127.0.0.1:38002",
    public_url: "https://test-host.nerdit.internal/asso",
    public_urls: [
      {
        url: "https://test-host.nerdit.internal/asso",
        kind: "default",
        state: "ready",
        access: null
      }
    ]
  },
  rollback_available: false,
  build_version: 1,
  last_deploy: { ...SAMPLE_SERVICES.items[0].last_deploy, version: 1, action: "deploy" }
};

export const SAMPLE_SERVICES_WITH_ASSO = {
  items: [
    ...SAMPLE_SERVICES.items,
    ASSO_WEB,
    {
      ...ASSO_WEB,
      id: "svc4dddd0004",
      name: "api--asso",
      service: "api",
      status: "degraded",
      image: "nerdit-app/api--asso:1",
      container_id: "ctr-asso-api",
      endpoint: {
        container_port: 9000,
        host_port: 38003,
        protocol: "tcp",
        route: "/api--asso",
        url: "http://127.0.0.1:38003",
        public_url: "https://test-host.nerdit.internal/api--asso",
        public_urls: []
      }
    },
    {
      ...SAMPLE_SERVICES.items[0],
      id: "mdl1cccc0003",
      name: "ollama-llama3-1-8b",
      kind: "model",
      project: null,
      project_id: null,
      service: null
    }
  ],
  next_cursor: null
};

// Mirrors ModelListPage / ModelResponse (db/models.py). One ready model (weights
// pulled, GPU-backed) and one still pulling (CPU, endpoint pending). Models are
// loopback-only by design — no public URL field exists on this shape at all.
const SAMPLE_MODELS = {
  items: [
    {
      id: "mdl1cccc0003",
      name: "ollama-llama3-1-8b",
      model: "llama3.1:8b",
      backend: "ollama",
      status: "running",
      desired_state: "running",
      model_pulled: true,
      gpu_count: 1,
      gpu_ids: ["GPU-A100-001"],
      gpu_utilization: { "GPU-A100-001": 42 },
      endpoint: "http://127.0.0.1:38100/v1",
      created_at: "2026-07-04T08:00:00+00:00"
    },
    {
      id: "mdl2dddd0004",
      name: "ollama-qwen2-5-3b",
      model: "qwen2.5:3b",
      backend: "ollama",
      status: "running",
      desired_state: "running",
      model_pulled: false,
      gpu_count: 0,
      gpu_ids: [],
      gpu_utilization: {},
      endpoint: null,
      created_at: "2026-07-04T08:30:00+00:00"
    }
  ],
  next_cursor: null
};

// Mirrors DatabaseListPage / DatabaseResponse (db/models.py, P15). One
// ready postgres row (endpoint is a password-free host:port display string —
// never a DSN).
const SAMPLE_DATABASES = {
  items: [
    {
      id: "db1aaaa0001",
      name: "pg",
      backend: "postgres",
      status: "running",
      desired_state: "running",
      db_ready: true,
      endpoint: "127.0.0.1:38200",
      created_at: "2026-07-14T08:00:00+00:00"
    }
  ],
  next_cursor: null
};

function makeAuditEntry(id: number, action: string, overrides: Record<string, unknown> = {}) {
  return {
    id,
    ts: "2026-07-04T11:00:00+00:00",
    principal_id: "tok-admin-01",
    principal_role: "admin",
    action,
    target_type: "service",
    target_id: "my-app",
    params_redacted: { name: "my-app" },
    result: "ok",
    status_code: 201,
    request_id: `req-${id}`,
    idempotency_key: null,
    ...overrides
  };
}

// Mirrors AuditLogPage / AuditLogEntry (db/models.py). Two pages so the
// cursor-pagination path ("Load more") is exercised end-to-end. The deploy-
// shaped rows (all target_type=service, target_id=my-app) back the project
// page's Deployments tab + activity feed (PR3).
const SAMPLE_AUDIT = {
  items: [
    makeAuditEntry(45, "deploy.git_create", { status_code: 201 }),
    makeAuditEntry(44, "template.deploy", {
      status_code: 201,
      params_redacted: { template_id: "fastapi-ai-chat", name: "my-app" }
    }),
    makeAuditEntry(43, "deploy.rollback", { status_code: 200 }),
    makeAuditEntry(42, "deploy.create"),
    makeAuditEntry(41, "secrets.set", {
      action: "secrets.set",
      params_redacted: { values: "***" },
      status_code: 200
    }),
    makeAuditEntry(40, "model.serve", {
      target_type: "model",
      target_id: "ollama-llama3-1-8b"
    })
  ],
  next_cursor: "40"
};

const SAMPLE_AUDIT_PAGE_2 = {
  items: [
    makeAuditEntry(39, "service.stop", { status_code: 200 }),
    makeAuditEntry(38, "auth.denied", {
      principal_id: "tok-readonly-01",
      principal_role: "readonly",
      action: "auth.denied",
      result: "denied",
      status_code: 403
    })
  ],
  next_cursor: null
};

const SAMPLE_CLUSTER_INFO = {
  hostname: "test-host",
  version: "0.3.0",
  uptime_seconds: 120
};

// --- P12/P13 self-knowledge + config surfaces -------------------------------

// Mirrors AppConfigView (db/models.py) — deploy fields + [ai.*] refs, no values.
export const SAMPLE_APP_CONFIG = {
  service_name: "my-app",
  deploy: {
    name: "my-app",
    port: 8000,
    gpus: 0,
    start: "npm start",
    health: "/healthz",
    memory_limit: null,
    cpu_limit: null
  },
  ai: {
    default: {
      provider: "ollama",
      model: "llama3.1:8b",
      base_url: null,
      api_key: null
    }
  },
  db: {},
  env_keys: ["LOG_LEVEL"],
  source: "deploy",
  revision: 0,
  etag: "app-config-etag-1"
};

// Mirrors AppConfigView with an external-API [ai.default] binding. The real
// GET /config/apps/{name} always redacts a set api_key to "***" (never the
// ${secrets.*} ref itself — config/redaction.py), so the fixture carries the
// sentinel: this drives the api → local swap AND the redacted keep-key reality
// (P12.5 WP1).
export const SAMPLE_APP_CONFIG_API = {
  service_name: "my-app",
  deploy: {
    name: "my-app",
    port: 8000,
    gpus: 0,
    start: "npm start",
    health: "/healthz",
    memory_limit: null,
    cpu_limit: null
  },
  ai: {
    default: {
      provider: "api",
      model: "gpt-4o-mini",
      base_url: "https://api.openai.com/v1",
      api_key: "***"
    }
  },
  db: {},
  env_keys: ["LOG_LEVEL"],
  source: "api",
  revision: 1,
  etag: "app-config-etag-api-1"
};

// Mirrors the [proxy] section of GET /config/daemon/proxy (ConfigView).
const SAMPLE_PROXY_CONFIG = {
  section: "proxy",
  values: {
    enabled: true,
    mode: "path",
    scheme: "https",
    https_port: 443,
    base_domain: null,
    hostname_override: "",
    mdns: false
  },
  etag: "proxy-config-etag-1"
};

// Mirrors GET /capabilities — admin projection (paths + admin_addr present).
export const SAMPLE_CAPABILITIES = {
  version: "0.3.0",
  uptime_s: 3600,
  caller: {
    role: "admin",
    token_name: "local",
    quotas: { max_gpus: null, max_concurrent_jobs: null }
  },
  proxy: {
    enabled: true,
    available: true,
    mode: "path",
    base_domain: null,
    hostname: "test-host.local",
    scheme: "https",
    https_port: 443,
    url_shape: "https://test-host.local/<name>",
    dashboard_apex: false,
    mdns: false,
    admin_addr: "127.0.0.1:2019"
  },
  models: {
    backends: ["ollama", "vllm"],
    default_backend: "ollama",
    bridge_host: "172.17.0.1"
  },
  buildpacks: ["dockerfile", "python", "node"],
  deploy: {
    git_enabled: true,
    git_allowed_hosts: ["github.com"],
    max_upload_bytes: 104857600,
    dry_run: true,
    max_concurrent_builds: 2
  },
  gpus: { count: 2, schedulable: 2, vendors: ["nvidia"] },
  mcp: { http_enabled: false },
  limits: {
    wait_timeout_max_s: 300,
    wait_concurrency_max: 16,
    log_tail_max: 1000,
    diagnose_log_tail_max: 200,
    page_limit_max: 200,
    service_port_range: [38000, 38999]
  },
  features: {
    batch: true,
    secrets_shared_scope: true,
    app_templates: true,
    audit_target_filter: true,
    projects: true,
    variables: true,
    project_apply: true
  },
  paths: { data_dir: "/home/test/.nerdit", db_path: "/home/test/.nerdit/nerdit.db" }
};

/** The pre-P40 daemon: no `features.projects|variables|project_apply`, so today's pages render. */
export const SAMPLE_CAPABILITIES_PRE_P40 = {
  ...SAMPLE_CAPABILITIES,
  features: { batch: true, secrets_shared_scope: true, app_templates: true, audit_target_filter: true }
};

// Mirrors GET /doctor — a warn-tinted mix so cards render every status tone.
const SAMPLE_DOCTOR = {
  status: "warn",
  checks: [
    { name: "docker", status: "ok", detail: "Docker runtime responding", latency_ms: 12 },
    { name: "gpu", status: "ok", detail: "2 GPU(s), 2 schedulable", latency_ms: 3 },
    { name: "proxy", status: "ok", detail: "proxy up", latency_ms: 8 },
    { name: "mdns", status: "skipped", detail: "mDNS advertising disabled", latency_ms: 0 },
    { name: "secrets_key", status: "ok", detail: "secrets key present and usable", latency_ms: 5 },
    { name: "disk", status: "ok", detail: "62.4% free", latency_ms: 4 },
    { name: "db", status: "ok", detail: "40960 bytes, quick_check ok", latency_ms: 6 },
    { name: "git", status: "ok", detail: "git binary on PATH", latency_ms: 2 },
    {
      name: "config_restart_pending",
      status: "warn",
      detail: "pending restart: proxy.hostname_override",
      latency_ms: 7
    }
  ]
};

// Mirrors GET /proxy/status (ProxyStatusResponse).
const SAMPLE_PROXY_STATUS = {
  state: "available",
  enabled: true,
  available: true,
  mode: "path",
  base_domain: null,
  hostname: "test-host.local",
  scheme: "https",
  https_port: 443,
  tls: { synced: true, subjects: ["test-host.local"] },
  ca: { fingerprint: "SHA256:ab12cd34ef56" },
  apex: { enabled: false, registered: false },
  respawn: { attempts: 0, last_spawn_ago_s: null, next_retry_in_s: null },
  routes: { count: 1, live_table: "readable" },
  mdns: { enabled: false, registered: false }
};

// Mirrors DiagnoseResponse — a failed service. Includes a trap secret value in
// a place values must never appear so WP5 can assert it stays absent.
const SAMPLE_DIAGNOSE = {
  service_name: "my-app",
  kind: "service",
  status: "failed",
  desired_state: "running",
  last_deploy: { version: 4, action: "redeploy", phase: "failed", reason: "container exited (1)" },
  error: { class: "USER_ERROR", message: "start command not found: npm" },
  forensics: { last_exit_code: 1, oom_killed: false, last_crash_at: "2026-07-04T11:30:00+00:00" },
  restarts: {
    policy: "always",
    count: 3,
    max_restarts: 3,
    window_seconds: 300,
    window_start: "2026-07-04T11:25:00+00:00",
    last_exit_at: "2026-07-04T11:30:00+00:00",
    backoff_s: 30,
    next_retry_in_s: null
  },
  health: { spec: { path: "/healthz" }, probe: null },
  bindings: { waiting: false, messages: [] },
  build: { version: 4, last_result: "failed", reason: "container exited (1)" },
  injected_env_keys: ["LOG_LEVEL", "OPENAI_API_KEY"],
  injected_env_keys_source: "launch",
  pending_env_keys: ["LOG_LEVEL", "OPENAI_API_KEY"],
  logs: [
    { stream: "stderr", line: "sh: npm: not found", ts: "2026-07-04T11:30:00+00:00" }
  ],
  remediation: { code: "fix_start_command", detail: "Fix the start command in [deploy].start." }
};

export function makeWaitResponse(overrides: Record<string, unknown> = {}) {
  return {
    outcome: "converged",
    service_name: "my-app",
    version: 3,
    phase: "healthy",
    status: "running",
    public_url: "https://test-host.nerdit.internal/my-app",
    reason: null,
    error_class: null,
    error_message: null,
    waited_s: 4.2,
    ...overrides
  };
}

export interface MockOverrides {
  gpus?: any[];
  clusterStats?: any;
  /** ServiceListPage returned by GET /api/services (detail resolves from `items`). */
  services?: { items: any[]; next_cursor: string | null };
  /** ModelListPage returned by GET /api/models. */
  models?: { items: any[]; next_cursor: string | null };
  /** DatabaseListPage returned by GET /api/databases (P15). */
  databases?: { items: any[]; next_cursor: string | null };
  /** First AuditLogPage; a `cursor` query param serves `auditPage2`. */
  audit?: { items: any[]; next_cursor: string | null };
  auditPage2?: { items: any[]; next_cursor: string | null };
  /** When true, GET /api/audit returns the admin-required 403 envelope. */
  auditForbidden?: boolean;
  /** When set, POST /api/deploy/:name/rollback returns this status + body. */
  rollbackResponse?: { status: number; body: unknown };
  /** Role returned by GET /api/auth/check (drives admin-only UI). Default "admin". */
  authRole?: string;
  /** AppConfigView returned by GET /api/config/apps/:name (ETag from `.etag`). */
  appConfig?: any;
  /**
   * Per-app AppConfigView overrides, keyed by service name (P15 — the cross-app
   * Bindings pages fan out one GET per app). Falls back to `appConfig` (or
   * SAMPLE_APP_CONFIG) for any name not present here.
   */
  appConfigsByName?: Record<string, any>;
  /** AppConfigWriteResponse returned by PUT /api/config/apps/:name/:section. */
  appConfigWrite?: any;
  /** ConfigView returned by GET /api/config/daemon/proxy (ETag from `.etag`). */
  proxyConfig?: any;
  /** ConfigWriteResponse returned by PUT /api/config/daemon/proxy. */
  proxyConfigWrite?: any;
  /** Capabilities returned by GET /api/capabilities. */
  capabilities?: any;
  /** DoctorReport returned by GET /api/doctor. */
  doctor?: any;
  /** ProxyStatus returned by GET /api/proxy/status. */
  proxyStatus?: any;
  /** DiagnoseResponse returned by GET /api/services/:id/diagnose. */
  diagnose?: any;
  /** ServiceWaitResponse returned by GET /api/services/:id/wait. */
  waitResponse?: any;
  /** When set, POST /api/deploy/git returns this status + body. */
  deployGitResponse?: { status: number; body: unknown };
  /** When set, POST /api/deploy/:name/redeploy returns this status + body. */
  redeployResponse?: { status: number; body: unknown };
  /** When set, POST /api/workspaces/:name/deploy returns this status + body. */
  workspaceDeployResponse?: { status: number; body: unknown };
  /** When set, POST /api/daemon/restart returns this status + body. */
  daemonRestartResponse?: { status: number; body: unknown };
  /** Per-service secret key names for GET /api/secrets/:service ("shared" targets the shared scope). Default keeps ["API_KEY"] for every service. */
  secretNames?: Record<string, string[]>;
  /**
   * (P40e) False = the caller does not own the projects: `GET /projects/{name}`
   * OMITS the `variables` section (D-P40-15) and every variables call plus the
   * project delete answer the one `owner_denial` 403. Default true.
   */
  projectOwner?: boolean;
  /**
   * `GET …/variables` answers the owner 403 although `get_project` listed names —
   * ownership moved between the two reads. The page must stay a page.
   */
  variablesReadForbidden?: boolean;
  /**
   * Seed variables per project name. `scope` is `project` or
   * `production/<service>`; only a PLAIN entry carries a `value` — the mock,
   * like the daemon, has nowhere to return a secret value from.
   */
  variables?: Record<string, MockVariable[]>;
  /** Page size of `GET /api/projects` (default: everything in one page), to exercise the cursor walk. */
  projectsPageSize?: number;
}

export interface MockVariable {
  key: string;
  scope: string;
  plain: boolean;
  value?: string;
}

/** What `mockApi` hands back so a spec can assert on what the page SENT. */
export interface MockHandle {
  /** Every `PUT …/variables`, in order: the `service` query param (null = project scope) + the JSON body. */
  variablePuts: { service: string | null; body: { values?: Record<string, string>; secret?: boolean } }[];
  /** Every `/api/projects*` request as `METHOD path?query` — the older-daemon fallback asserts it stays empty. */
  projectCalls: string[];
}

const TOKEN = "test-token-1234567890abcdef";

function json(route: Route, body: unknown, status = 200) {
  return route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body)
  });
}

/** Like `json` but also sets an ETag header (config GETs round-trip it). */
function jsonWithEtag(route: Route, body: { etag?: string | null } & object) {
  return route.fulfill({
    status: 200,
    contentType: "application/json",
    headers: body.etag ? { ETag: body.etag } : {},
    body: JSON.stringify(body)
  });
}

// The one 403 every owner gate raises (`daemon/auth.py::owner_denial`).
const OWNER_DENIAL = {
  code: "forbidden",
  message: "You do not have permission to act on this job.",
  hint: "Only the submitting token or an admin may manage this job.",
  detail: "You do not have permission to act on this job."
};

export async function mockApi(page: Page, overrides: MockOverrides = {}): Promise<MockHandle> {
  const handle: MockHandle = { variablePuts: [], projectCalls: [] };
  const gpus = overrides.gpus ?? SAMPLE_GPUS;
  const clusterStats = overrides.clusterStats ?? SAMPLE_CLUSTER_STATS;

  // Register first so explicit mocks win; unexpected calls must never reach a daemon.
  await page.route((url) => url.pathname === "/api" || url.pathname.startsWith("/api/"), async (route) => {
    await route.abort("blockedbyclient");
    throw new Error(`Unmocked API request: ${route.request().method()} ${route.request().url()}`);
  });

  await page.route(/\/api\/app-templates(\?.*)?$/, (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    return json(route, []);
  });
  await page.route(/\/api\/routes(\?.*)?$/, (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    return json(route, { items: [], next_cursor: null, live_table: "readable" });
  });
  await page.route(/\/api\/system\/disk(\?.*)?$/, (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    return json(route, {
      docker: { images_bytes: 0, containers_bytes: 0, volumes_bytes: 0, build_cache_bytes: 0 },
      images: { total_bytes: 0, by_repo: {} },
      data_dir: {
        services: [],
        services_total_bytes: 0,
        models: { ollama: 0, huggingface: 0 },
        archive_bytes: 0,
        backups: { bytes: 0, count: 0 },
        volume_backups: { bytes: 0, count: 0 },
        dumps: { bytes: 0, count: 0 },
        dump_staging_bytes: 0
      },
      orphan_images: [],
      orphan_data_dirs: [],
      warnings: []
    });
  });

  await page.route("**/api/auth/check", (route) =>
    json(route, { ok: true, role: overrides.authRole ?? "admin" })
  );
  await page.route("**/api/health", (route) =>
    json(route, { status: "ok", gpu_count: gpus.length, version: "0.3.0" })
  );
  await page.route("**/api/cluster/stats", (route) => json(route, clusterStats));
  await page.route("**/api/cluster/info", (route) => json(route, SAMPLE_CLUSTER_INFO));
  await page.route("**/api/gpus", (route) => json(route, gpus));

  // SSE — return an empty stream so the EventSource opens then idles.
  await page.route("**/api/events/stream", (route) =>
    route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      headers: { "cache-control": "no-cache" },
      body: ":\n\n"
    })
  );

  // Playwright route precedence is LAST-REGISTERED-WINS. Register from least
  // specific to most specific so the specific patterns shadow the generic ones.

  // --- P6 surface: services / models / audit / secrets / deploy -------------

  const services = overrides.services ?? SAMPLE_SERVICES;
  const models = overrides.models ?? SAMPLE_MODELS;
  const audit = overrides.audit ?? SAMPLE_AUDIT;
  const auditPage2 = overrides.auditPage2 ?? SAMPLE_AUDIT_PAGE_2;

  // Generic: GET /api/services (ServiceListPage).
  // (P40e) Projects removed through `DELETE /api/projects/{name}` in this page's
  // lifetime; their rows leave the service list too, like the daemon's cascade.
  const deletedProjects = new Set<string>();
  const liveServices = () => services.items.filter((s) => !deletedProjects.has(s.project));

  await page.route(/\/api\/services(\?.*)?$/, (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    return json(route, { ...services, items: liveServices() });
  });

  // GET single service at /api/services/:ident (by name or id) + DELETE.
  await page.route(/\/api\/services\/[^/?]+(\?.*)?$/, (route) => {
    const url = new URL(route.request().url());
    const ident = decodeURIComponent(url.pathname.split("/").filter(Boolean).pop() ?? "");
    const svc = services.items.find((s) => s.name === ident || s.id === ident);
    if (route.request().method() === "DELETE") {
      return json(route, { id: svc?.id ?? "unknown", name: svc?.name ?? ident, deleted: true });
    }
    if (route.request().method() !== "GET") return route.fallback();
    if (!svc) {
      return json(
        route,
        { code: "service.not_found", message: `No service '${ident}'`, detail: `No service '${ident}'` },
        404
      );
    }
    return json(route, svc);
  });

  // Specific: lifecycle actions echo the (unchanged) service row.
  await page.route(/\/api\/services\/[^/]+\/(stop|restart)$/, (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    const url = new URL(route.request().url());
    const parts = url.pathname.split("/").filter(Boolean);
    const ident = decodeURIComponent(parts[parts.length - 2] ?? "");
    const svc = services.items.find((s) => s.name === ident || s.id === ident);
    return json(route, svc ?? services.items[0]);
  });

  // Specific: polling-only service logs (no SSE variant exists).
  await page.route(/\/api\/services\/[^/]+\/logs(\?.*)?$/, (route) =>
    json(route, [
      { id: 1, stream: "system", message: "container started", timestamp: "2026-07-04T10:01:00+00:00" }
    ])
  );

  // --- P40 surface: projects + variables (`daemon/routes/projects.py`) --------
  // DERIVED from the service page above, grouped by each row's `project` FIELD
  // (never by parsing a label), so the two surfaces cannot disagree. Same
  // ordering rule as everywhere here — last registered wins — so the list comes
  // first and `/variables/resolve` last (it must shadow `/variables/{key}`).

  const owner = overrides.projectOwner ?? true;
  const variableStore = new Map<string, MockVariable[]>(
    Object.entries(overrides.variables ?? {}).map(([name, rows]) => [name, [...rows]])
  );
  const scopeName = (service: string | null) => (service ? `production/${service}` : "project");

  const projectSummaries = () => {
    const byName = new Map<string, { id: string; name: string; services: any[]; addresses: any[] }>();
    for (const svc of liveServices()) {
      if (svc.kind !== "service" || !svc.project) continue;
      const entry = byName.get(svc.project) ?? {
        id: svc.project_id,
        name: svc.project,
        services: [],
        addresses: []
      };
      entry.services.push(svc);
      entry.addresses.push(...(svc.endpoint?.public_urls ?? []));
      byName.set(svc.project, entry);
    }
    return [...byName.values()];
  };
  // Mirrors `_resources`: every `[ai.*]`/`[db.*]` binding of a service, read from
  // the SAME app config its Manage tab gets, judged on the inventory rows only.
  const isUp = (row: any) => row?.status === "running" || row?.status === "degraded";
  const projectResources = (label: string) => {
    const cfg = overrides.appConfigsByName?.[label] ?? overrides.appConfig ?? SAMPLE_APP_CONFIG;
    const modelItems = (overrides.models ?? SAMPLE_MODELS).items;
    const dbItems = (overrides.databases ?? SAMPLE_DATABASES).items;
    const ai = Object.entries<any>(cfg.ai ?? {}).map(([binding, spec]) => ({
      service: label,
      type: "ai",
      binding,
      provider: spec.provider ?? null,
      target: (spec.provider === "api" ? spec.base_url : spec.model) ?? null,
      ready: spec.provider === "ollama" ? isUp(modelItems.find((m) => m.model === spec.model)) : null
    }));
    const db = Object.entries<any>(cfg.db ?? {}).map(([binding, spec]) => ({
      service: label,
      type: "db",
      binding,
      provider: spec.provider ?? null,
      target: spec.database ?? null,
      ready: spec.provider === "managed" ? isUp(dbItems.find((d) => d.name === spec.database)) : null
    }));
    const byBinding = (a: { binding: string }, b: { binding: string }) =>
      a.binding.localeCompare(b.binding);
    return [...ai.sort(byBinding), ...db.sort(byBinding)];
  };
  const projectNotFound = (name: string) => ({
    code: "not_found",
    message: `No project '${name}'.`,
    hint: "List projects with `nerdit projects list` to find a valid name.",
    detail: `No project '${name}'.`
  });
  /** `[project name, …rest]` of the path after `/api/projects/`, plus the `service` query param. */
  const projectRequest = (route: Route) => {
    const url = new URL(route.request().url());
    const parts = url.pathname.split("/").filter(Boolean).map(decodeURIComponent);
    handle.projectCalls.push(`${route.request().method()} ${url.pathname}${url.search}`);
    return {
      parts: parts.slice(parts.indexOf("projects") + 1),
      service: url.searchParams.get("service"),
      url
    };
  };

  // GET /api/projects (ProjectListPage; the cursor is the next index).
  await page.route(/\/api\/projects(\?.*)?$/, (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    const { url } = projectRequest(route);
    const all = projectSummaries();
    const size = overrides.projectsPageSize ?? all.length;
    const start = Number(url.searchParams.get("cursor") ?? 0);
    const end = start + size;
    return json(route, {
      items: all.slice(start, end),
      next_cursor: end < all.length ? String(end) : null
    });
  });

  // GET /api/projects/{name} (ProjectResponse) + DELETE (ProjectDeletedResponse).
  await page.route(/\/api\/projects\/[^/?]+(\?.*)?$/, (route) => {
    const method = route.request().method();
    const { parts } = projectRequest(route);
    const selector = parts[0];
    const summary = projectSummaries().find((p) => p.name === selector || p.id === selector);
    const name = summary?.name ?? selector;
    if (!summary) return json(route, projectNotFound(name), 404);
    if (method === "DELETE") {
      if (!owner) return json(route, OWNER_DENIAL, 403);
      deletedProjects.add(name);
      variableStore.delete(name);
      return json(route, { name, deleted: summary.services.map((s) => s.name) });
    }
    if (method !== "GET") return route.fallback();
    return json(route, {
      ...summary,
      resources: summary.services.flatMap((svc) => projectResources(svc.name)),
      home: { hostname: SAMPLE_CLUSTER_INFO.hostname, node_id: null },
      // D-P40-15: OMITTED, not nulled, for a non-owner. Names only, never a value.
      ...(owner
        ? {
            variables: (variableStore.get(name) ?? []).map(({ key, scope, plain }) => ({
              key,
              scope,
              plain
            }))
          }
        : {})
    });
  });

  // POST /api/projects/{name}/apply (ApplyResponse). No page calls it yet; the
  // mock exists so a future caller meets the real shape, not the catch-all.
  await page.route(/\/api\/projects\/[^/?]+\/apply(\?.*)?$/, (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    const { parts, url } = projectRequest(route);
    const dryRun = url.searchParams.get("dry_run") === "true";
    return json(route, {
      project: parts[0],
      status: dryRun ? "planned" : "applied",
      dry_run: dryRun,
      services: [],
      public_urls: [],
      missing: [],
      hint: null
    });
  });

  // GET/PUT /api/projects/{name}/variables[?service=] — one scope at a time.
  await page.route(/\/api\/projects\/[^/?]+\/variables(\?.*)?$/, (route) => {
    const method = route.request().method();
    const { parts, service } = projectRequest(route);
    const name = parts[0];
    const scope = scopeName(service);
    if (method === "PUT") {
      const body = (route.request().postDataJSON() ?? {}) as MockHandle["variablePuts"][number]["body"];
      handle.variablePuts.push({ service, body });
      if (!owner) return json(route, OWNER_DENIAL, 403);
      // `secret` defaults to TRUE server-side (D-P40-16).
      const plain = body.secret === false;
      const keys = Object.keys(body.values ?? {});
      const rest = (variableStore.get(name) ?? []).filter(
        (row) => row.scope !== scope || !keys.includes(row.key)
      );
      variableStore.set(name, [
        ...rest,
        // A secret's value is dropped on the floor: nothing here can echo it.
        ...keys.map((key) => ({ key, scope, plain, ...(plain ? { value: body.values![key] } : {}) }))
      ]);
      return json(route, { project: name, scope, keys, plain });
    }
    if (method !== "GET") return route.fallback();
    if (!owner || overrides.variablesReadForbidden) return json(route, OWNER_DENIAL, 403);
    return json(route, {
      project: name,
      scope,
      variables: (variableStore.get(name) ?? [])
        .filter((row) => row.scope === scope)
        .sort((a, b) => a.key.localeCompare(b.key))
        .map(({ key, plain, value }) => ({ key, scope, plain, value: plain ? (value ?? null) : null }))
    });
  });

  // DELETE /api/projects/{name}/variables/{key}[?service=].
  await page.route(/\/api\/projects\/[^/?]+\/variables\/[^/?]+(\?.*)?$/, (route) => {
    if (route.request().method() !== "DELETE") return route.fallback();
    const { parts, service } = projectRequest(route);
    if (!owner) return json(route, OWNER_DENIAL, 403);
    const [name, , key] = parts;
    const scope = scopeName(service);
    variableStore.set(
      name,
      (variableStore.get(name) ?? []).filter((row) => row.scope !== scope || row.key !== key)
    );
    return json(route, { project: name, scope, deleted: key });
  });

  // GET /api/projects/{name}/variables/resolve[?service=web] — the winning scope
  // per key (service beats project), never a value. Registered LAST so the
  // literal segment is never taken for a key.
  await page.route(/\/api\/projects\/[^/?]+\/variables\/resolve(\?.*)?$/, (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    const { parts, service } = projectRequest(route);
    if (!owner) return json(route, OWNER_DENIAL, 403);
    const target = service ?? "web";
    const winners = new Map<string, { key: string; scope: string; plain: boolean }>();
    for (const scope of ["project", scopeName(target)]) {
      for (const row of variableStore.get(parts[0]) ?? []) {
        if (row.scope === scope) winners.set(row.key, { key: row.key, scope, plain: row.plain });
      }
    }
    return json(route, { project: parts[0], service: target, variables: [...winners.values()] });
  });

  // GET /api/models (ModelListPage) + POST /api/models (serve).
  await page.route(/\/api\/models(\?.*)?$/, (route) => {
    if (route.request().method() === "POST") {
      return json(route, models.items[0] ?? SAMPLE_MODELS.items[0], 201);
    }
    return json(route, models);
  });

  // GET /api/databases (DatabaseListPage, P15) + POST (provision → 201). The
  // POST echoes a not-yet-ready row: a freshly provisioned database is still
  // starting, and the list is what shows it settling.
  const databases = overrides.databases ?? SAMPLE_DATABASES;
  await page.route(/\/api\/databases(\?.*)?$/, (route) => {
    const method = route.request().method();
    if (method === "POST") {
      const body = (route.request().postDataJSON() ?? {}) as { name?: string; backend?: string };
      return json(
        route,
        {
          id: "db2bbbb0002",
          name: body.name ?? "new-db",
          backend: body.backend ?? "postgres",
          status: "building",
          desired_state: "running",
          db_ready: false,
          endpoint: null,
          created_at: "2026-07-14T09:00:00+00:00"
        },
        201
      );
    }
    if (method !== "GET") return route.fallback();
    return json(route, databases);
  });

  // GET /api/audit — admin-gated, cursor-paginated. Polling-primary by design
  // (D5): the dashboard never opens /audit/stream, so no SSE mock is needed.
  await page.route(/\/api\/audit(\?.*)?$/, (route) => {
    if (overrides.auditForbidden) {
      return json(
        route,
        {
          code: "auth.forbidden",
          message: "Admin role required",
          detail: "Admin role required"
        },
        403
      );
    }
    const url = new URL(route.request().url());
    const page = url.searchParams.has("cursor") ? auditPage2 : audit;
    // PR1 target filter: filter fixture items by target_id / target_type before
    // paging (the real daemon's exact-match query params).
    const target = url.searchParams.get("target");
    const targetType = url.searchParams.get("target_type");
    let items = page.items;
    if (target) items = items.filter((it) => it.target_id === target);
    if (targetType) items = items.filter((it) => it.target_type === targetType);
    return json(route, { items, next_cursor: page.next_cursor });
  });

  // Secrets: names-only reads; writes echo the merged key list. Values never
  // appear in any response (write-only contract).
  await page.route(/\/api\/secrets\/[^/?]+(\/[^/?]+)?(\?.*)?$/, (route) => {
    const url = new URL(route.request().url());
    const parts = url.pathname.split("/").filter(Boolean);
    const afterSecrets = parts.slice(parts.indexOf("secrets") + 1);
    const service = decodeURIComponent(afterSecrets[0] ?? "");
    const key = afterSecrets[1] ? decodeURIComponent(afterSecrets[1]) : null;
    const method = route.request().method();
    if (method === "GET") return json(route, { service, keys: overrides.secretNames?.[service] ?? ["API_KEY"] });
    if (method === "POST") {
      const body = route.request().postDataJSON() as { values?: Record<string, string> };
      return json(route, { service, keys: Object.keys(body?.values ?? {}) });
    }
    if (method === "DELETE") return json(route, { service, deleted: key ?? true });
    return route.fallback();
  });

  // Deploy (multipart) → the first service row; rollback is override-driven so
  // specs can exercise both the happy path and the structured 409.
  await page.route(/\/api\/deploy(\?.*)?$/, (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    return json(route, services.items[0] ?? SAMPLE_SERVICES.items[0], 201);
  });
  await page.route(/\/api\/deploy\/[^/]+\/rollback$/, (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    const rollback = overrides.rollbackResponse;
    if (rollback) return json(route, rollback.body, rollback.status);
    return json(route, services.items[0] ?? SAMPLE_SERVICES.items[0]);
  });

  // POST /api/deploy/:name/redeploy (P24b WP6) — 201 with the row's next
  // generation queued, mirroring the daemon: the old image still serves, and
  // `last_deploy` is what says a rebuild has started.
  await page.route(/\/api\/deploy\/[^/]+\/redeploy(\?.*)?$/, (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    const override = overrides.redeployResponse;
    if (override) return json(route, override.body, override.status);
    const parts = new URL(route.request().url()).pathname.split("/").filter(Boolean);
    const name = decodeURIComponent(parts[parts.length - 2] ?? "");
    const svc =
      services.items.find((s) => s.name === name || s.id === name) ??
      services.items[0] ??
      SAMPLE_SERVICES.items[0];
    const version = (svc.build_version ?? 1) + 1;
    return json(
      route,
      {
        ...svc,
        last_deploy: {
          version,
          action: "redeploy",
          phase: "queued",
          image: null,
          started_at: "2026-07-04T12:00:00+00:00",
          updated_at: "2026-07-04T12:00:00+00:00"
        }
      },
      201
    );
  });

  // POST /api/workspaces/:name/deploy (P29 / Codex PR #147) — the workspace
  // twin of the redeploy mock: 201 with the next generation queued.
  await page.route(/\/api\/workspaces\/[^/]+\/deploy(\?.*)?$/, (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    const override = overrides.workspaceDeployResponse;
    if (override) return json(route, override.body, override.status);
    const parts = new URL(route.request().url()).pathname.split("/").filter(Boolean);
    const name = decodeURIComponent(parts[parts.length - 2] ?? "");
    const svc =
      services.items.find((s) => s.name === name || s.id === name) ??
      services.items[0] ??
      SAMPLE_SERVICES.items[0];
    const version = (svc.build_version ?? 1) + 1;
    return json(
      route,
      {
        ...svc,
        last_deploy: {
          version,
          action: "redeploy",
          phase: "queued",
          image: null,
          started_at: "2026-07-04T12:00:00+00:00",
          updated_at: "2026-07-04T12:00:00+00:00"
        }
      },
      201
    );
  });

  // --- P12/P13 surface: config-as-API + self-knowledge + converge ------------
  // Same ordering rule: generic patterns first, specific ones last.

  const appConfig = overrides.appConfig ?? SAMPLE_APP_CONFIG;
  const capabilities = overrides.capabilities ?? SAMPLE_CAPABILITIES;
  const doctor = overrides.doctor ?? SAMPLE_DOCTOR;
  const proxyStatus = overrides.proxyStatus ?? SAMPLE_PROXY_STATUS;
  const proxyConfig = overrides.proxyConfig ?? SAMPLE_PROXY_CONFIG;
  const diagnose = overrides.diagnose ?? SAMPLE_DIAGNOSE;

  // GET /api/capabilities, GET /api/doctor, GET /api/proxy/status.
  await page.route(/\/api\/capabilities(\?.*)?$/, (route) => json(route, capabilities));
  await page.route(/\/api\/doctor(\?.*)?$/, (route) => json(route, doctor));
  await page.route(/\/api\/proxy\/status(\?.*)?$/, (route) => json(route, proxyStatus));

  // POST /api/daemon/restart (202 by default).
  await page.route(/\/api\/daemon\/restart$/, (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    const r = overrides.daemonRestartResponse;
    if (r) return json(route, r.body, r.status);
    return json(route, { restarting: true, in_flight_builds: 0, drain_timeout_s: 60 }, 202);
  });

  // GET/PUT /api/config/daemon/proxy — ETag round-trips on the GET.
  await page.route(/\/api\/config\/daemon\/proxy(\?.*)?$/, (route) => {
    const method = route.request().method();
    if (method === "GET") return jsonWithEtag(route, proxyConfig);
    if (method === "PUT") {
      const url = new URL(route.request().url());
      const dry = url.searchParams.get("dry_run") === "true";
      return json(
        route,
        overrides.proxyConfigWrite ?? {
          applied: !dry,
          diff: [
            {
              key: "hostname_override",
              old: "",
              new: "my-box.local",
              section: "proxy",
              op: "change",
              requires_restart: true,
              secret: false
            }
          ],
          diagnostics: [],
          requires_restart: true
        }
      );
    }
    return route.fallback();
  });

  // GET /api/config/apps/:name (ETag) + PUT /api/config/apps/:name/:section.
  // Register the section PUT LAST so it shadows the single-segment GET.
  await page.route(/\/api\/config\/apps\/[^/?]+(\?.*)?$/, (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    const url = new URL(route.request().url());
    const name = decodeURIComponent(url.pathname.split("/").filter(Boolean).pop() ?? "");
    const perApp = overrides.appConfigsByName?.[name];
    return jsonWithEtag(route, perApp ?? appConfig);
  });
  await page.route(/\/api\/config\/apps\/[^/?]+\/[^/?]+(\?.*)?$/, (route) => {
    if (route.request().method() !== "PUT") return route.fallback();
    const url = new URL(route.request().url());
    const dry = url.searchParams.get("dry_run") === "true";
    const restart = url.searchParams.get("restart") === "true";
    return json(
      route,
      overrides.appConfigWrite ?? {
        applied: !dry,
        requires_restart: true,
        restarted: !dry && restart,
        view: appConfig
      }
    );
  });

  // POST /api/deploy/git — a fresh service by default, or a plan on ?dry_run.
  await page.route(/\/api\/deploy\/git(\?.*)?$/, (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    const r = overrides.deployGitResponse;
    if (r) return json(route, r.body, r.status);
    const url = new URL(route.request().url());
    if (url.searchParams.get("dry_run") === "true") {
      return json(route, {
        dry_run: true,
        action: "create",
        name: "my-app",
        buildpack: "node",
        effective: {
          port: 8000,
          gpus: 0,
          start: "npm start",
          health: "/",
          memory_limit: null,
          cpu_limit: null
        },
        env_diff: { added: [], changed: [], removed: [] },
        ai_diff: { action: "none", bindings: [] },
        overwrote_api_config: false,
        warnings: []
      });
    }
    return json(route, services.items[0] ?? SAMPLE_SERVICES.items[0], 201);
  });

  // GET /api/services/:id/wait + /diagnose (extra path segment, so they never
  // collide with the single-segment service GET/DELETE route above).
  await page.route(/\/api\/services\/[^/]+\/wait(\?.*)?$/, (route) =>
    json(route, overrides.waitResponse ?? makeWaitResponse())
  );
  await page.route(/\/api\/services\/[^/]+\/diagnose(\?.*)?$/, (route) =>
    json(route, diagnose)
  );

  // Audit live stream — an empty text/event-stream that opens then idles, so
  // the fetch-stream helper connects without emitting rows (specs that want a
  // live row override this route themselves).
  await page.route("**/api/audit/stream", (route) =>
    route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      headers: { "cache-control": "no-cache" },
      body: ":\n\n"
    })
  );

  return handle;
}

export async function loginAsToken(page: Page): Promise<void> {
  await page.goto("/login");
  await page.fill("#token-input", TOKEN);
  await page.getByRole("button", { name: /sign in/i }).click();
  // Sign-in is async (await /auth/check → storeToken → redirect). Wait for the
  // token to actually persist before returning, so a caller's immediate
  // page.goto() to a protected route can't win the race and bounce to /login.
  await page.waitForFunction(() =>
    Boolean(sessionStorage.getItem("nerdit.token") || localStorage.getItem("nerdit.token"))
  );
}
