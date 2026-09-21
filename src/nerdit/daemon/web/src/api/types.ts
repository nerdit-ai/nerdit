/**
 * The statuses a service, model or database row can carry. The daemon keeps the
 * retired batch members in its own enum until v1.0, so a legacy row can still
 * arrive over the wire — that is what the `(string & {})` tail is for, and why
 * every renderer must fall back rather than assume exhaustiveness.
 */
export type JobStatus =
  | "running"
  | "failed"
  | "building"
  | "degraded"
  | "restarting"
  | "stopped"
  | (string & {});

/** Same contract as `JobStatus`: `batch` is retired here, not in the daemon. */
export type WorkloadKind = "service" | "model" | "database" | (string & {});

type ErrorClass =
  | "OOM"
  | "GPU_FAIL"
  | "IMAGE_PULL_FAIL"
  | "USER_ERROR"
  | "TIMEOUT"
  | "CONTAINER_FAIL"
  | "MOUNT_MISSING"
  | "GPU_OOM"
  | "PLATFORM_ERROR"
  | "UNKNOWN"
  | (string & {});

// (P34) `build` is the image build's own output, split off `stdout` so the
// REST filter can express "the app, not the build". Rows written by an older
// daemon still read as `stdout`.
type LogStream = "stdout" | "stderr" | "system" | "crash" | "build" | (string & {});

export type GpuVendor = "nvidia" | "amd";

export interface Gpu {
  id: string;
  name: string;
  memory_mb: number;
  vendor: GpuVendor;
  schedulable: boolean;
  status: string;
  utilization_percent: number | null;
  memory_used_mb: number | null;
  temperature_c: number | null;
}

export interface ClusterStats {
  gpus_total: number;
  gpus_in_use: number;
  gpus_avg_utilization: number;
  services_up?: number;
  daemon_uptime_seconds: number;
  daemon_version: string;
}

export interface ClusterInfo {
  hostname: string;
  version: string;
  uptime_seconds: number;
  /**
   * Publishable PostHog project key (phc_…) + ingest host for the dashboard's
   * product analytics. Both null unless the daemon has analytics enabled with a
   * key configured — the SPA stays inert otherwise (feat/posthog).
   */
  posthog_key?: string | null;
  posthog_host?: string | null;
}

export interface LogEntry {
  id: number;
  stream: LogStream;
  message: string;
  timestamp: string;
}

/**
 * Public projection of a service's durable host-port reservation.
 * Mirrors `ServiceEndpointView` (db/models.py).
 *
 * `route` contract: null = unrouted (proxy off, or a model row — models are
 * never routed); "/name" = path route; "" (empty string) = subdomain route.
 * Test `route !== null`, never truthiness — "" is a valid, routed state.
 */
/**
 * One advertised URL for a service (P26 D-P26-14). `public_url` stays the
 * default-kind scalar and keeps its meaning; this is the additive list beside
 * it, so a hosted share becomes visible without any reader of the scalar
 * changing. `state` is a machine token — branch on it, never on the wording.
 *
 * (P26 WP1) `kind: "domain"` entries carry a direct domain the operator bound
 * to the app, served by this node's own proxy; `domain` names the row and is
 * null on every other kind. Their `state` is `ready` or `withheld` — the
 * daemon computes it per request from the proxy's availability, the service's
 * route and the edge-auth withheld set. Never re-derive it here.
 */
export interface PublicUrlEntry {
  url: string | null;
  kind: "default" | "hosted" | "domain";
  state: "ready" | "link_down" | "not_entitled" | "withheld";
  access: "private" | "public" | null;
  /** Domain entries only: the bound name. Null/absent on `default` and `hosted`. */
  domain?: string | null;
  /**
   * (P26 WP2) Domain entries only: which certificate the name is served with.
   * Orthogonal to `state` — `state` is a fact about the *route*, `cert_state`
   * a fact about the *leaf*, so a `ready` domain can still be on the node's
   * internal CA. `internal` = no public certificate was asked for;
   * `disabled` = the row asks for one but `[proxy.acme]` is off;
   * `pending` = asked for, not issued yet (this also covers "issuance keeps
   * failing" — the daemon has no machine-readable failure record);
   * `issued` / `expired` = a public leaf exists on disk. Null/absent on every
   * other kind and on a pre-WP2 daemon.
   */
  cert_state?: "internal" | "disabled" | "pending" | "issued" | "expired" | null;
}

export interface ServiceEndpoint {
  container_port: number;
  host_port: number;
  /**
   * The port the live container actually answers on (P24b, D-P24-4b).
   * Equal to host_port on every ordinary row; differs only while a
   * cutover-promoted generation is serving on its transient green port.
   */
  effective_host_port: number;
  protocol: string;
  route: string | null;
  /** Always-present loopback URL (http://127.0.0.1:{effective_host_port}). */
  url: string | null;
  /** Proxy-resolved HTTPS URL; null when the proxy is off or the route is unregistered. */
  public_url: string | null;
  /** (P26) Every advertised URL with its kind/state. Optional: a pre-P26 daemon omits it. */
  public_urls?: PublicUrlEntry[];
}

/**
 * Deploy provenance projected from `config['source']` (P11.5 / PR1): records
 * how the app's current image was sourced. Never carries credentials — a
 * `token_ref` (private-repo secret reference) is stripped server-side.
 */
export interface ServiceSource {
  type: "zip" | "git" | (string & {});
  repo_url?: string;
  ref?: string;
  commit_sha?: string;
  subdir?: string;
  template_id?: string;
}

/** Mirrors `ServiceResponse` (db/models.py) — services and model rows alike. */
export interface Service {
  id: string;
  /** The LABEL — the service's wire identity (D-P40-6). Never parsed for parts. */
  name: string;
  /** (P40b) The owning project; null on models, databases and pre-P40 rows, absent on an older daemon. */
  project?: string | null;
  project_id?: string | null;
  /** (P40b) Service name inside the project (`web` for a bare project name). */
  service?: string | null;
  status: JobStatus;
  desired_state: string | null;
  kind: WorkloadKind;
  image: string | null;
  gpu_count: number;
  gpu_ids: string[];
  restart_policy: string | null;
  restart_count: number;
  health_check: Record<string, unknown> | null;
  container_id: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  exit_code: number | null;
  error_class: ErrorClass | null;
  error_message: string | null;
  submitted_via: string;
  endpoint: ServiceEndpoint | null;
  /** True when a redeploy recorded a previous image (rollback returns 409 otherwise). */
  rollback_available: boolean;
  /** Deploy build version currently serving; null for non-deployed services. */
  build_version: number | null;
  /** Current deploy generation's phase object; null for pre-P13 / /services rows. */
  last_deploy: LastDeploy | null;
  /** Deploy provenance (config['source'], PR1); null on older/undeployed rows. */
  source?: ServiceSource | null;
}

/**
 * The `last_deploy` phase object stamped by the deploy pipeline (P13). Drives
 * the deploy progress view and `lib/deployPhase`. `phase` walks
 * queued → building → launching → healthy | failed.
 */
interface LastDeploy {
  phase: string;
  version?: number | null;
  action?: string | null;
  image?: string | null;
  started_at?: string | null;
  updated_at?: string | null;
  reason?: string | null;
  error_class?: ErrorClass | null;
  error_message?: string | null;
  [key: string]: unknown;
}

export interface ServiceListPage {
  items: Service[];
  next_cursor: string | null;
}

/** Wire shape of `DELETE /services/{ident}` (ServiceDeletedResponse). */
export interface ServiceDeleted {
  id: string;
  name: string | null;
  deleted: boolean;
}

/** Mirrors `ModelResponse` (db/models.py) — kind=model rows on /models. */
export interface Model {
  id: string;
  name: string;
  model: string | null;
  backend: string | null;
  status: JobStatus;
  desired_state: string | null;
  /** Flips once the weights finished pulling — "container up, weights absent" is distinct. */
  model_pulled: boolean;
  gpu_count: number;
  gpu_ids: string[];
  /** Live compute utilization (0-100%) per allocated GPU; null when unknown. */
  gpu_utilization: Record<string, number | null>;
  /** Loopback OpenAI-compatible base URL; null until the host port is published. No public URL by design. */
  endpoint: string | null;
  created_at: string;
}

export interface ModelListPage {
  items: Model[];
  next_cursor: string | null;
}

/**
 * Mirrors `DatabaseResponse` (db/models.py) — `kind=database` rows on
 * `/databases`. `endpoint` is a password-free `host:port` display string
 * (never a DSN — the credential-bearing DSN is composed at binding-resolution/
 * launch time only, D7).
 */
export interface Database {
  id: string;
  name: string;
  backend: string | null;
  status: JobStatus;
  desired_state: string | null;
  /** Flips once the wire-protocol readiness probe succeeds ("db_ready"). */
  db_ready: boolean;
  endpoint: string | null;
  created_at: string;
}

export interface DatabaseListPage {
  items: Database[];
  next_cursor: string | null;
}

/** Mirrors `AuditLogEntry` (db/models.py) — one append-only audit record. */
export interface AuditLogEntry {
  id: number;
  ts: string | null;
  principal_id: string | null;
  principal_role: string | null;
  action: string;
  target_type: string | null;
  target_id: string | null;
  params_redacted: unknown;
  result: string | null;
  status_code: number | null;
  request_id: string | null;
  idempotency_key: string | null;
}

export interface AuditLogPage {
  items: AuditLogEntry[];
  next_cursor: string | null;
}

export interface AuditFilters {
  action?: string;
  result?: string;
  /** Exact `target_id` match (PR1). */
  target?: string;
  /** Exact `target_type` match (PR1), e.g. "service". */
  target_type?: string;
  limit?: number;
}

/** Write-only projection: a service's secret key *names* — never values. */
export interface SecretNames {
  service: string;
  keys: string[];
}

/**
 * Wire shape of `DELETE /secrets/{service}[/{key}]`: `deleted` is the key
 * name for single-key deletes, or a boolean (file existed) for delete-all.
 */
export interface SecretDeleted {
  service: string;
  deleted: string | boolean;
}

// --- Projects and variables (P40b/c — mirrors `daemon/routes/projects.py`) ---

/** Mirrors `ProjectSummary`: one row of `GET /projects` (and the create body). */
export interface Project {
  id: string;
  name: string;
  services: Service[];
  addresses: PublicUrlEntry[];
}

export interface ProjectListPage {
  items: Project[];
  next_cursor: string | null;
}

/** Mirrors `ProjectResourceView`. `ready: null` = cannot be judged without the caller's secrets. */
export interface ProjectResourceView {
  /** LABEL of the service that declares the binding. */
  service: string;
  type: string;
  binding: string;
  provider: string | null;
  target: string | null;
  ready: boolean | null;
}

/** Mirrors `VariableName`: a name, its scope and the plain flag. Never a value. */
export interface ProjectVariableName {
  key: string;
  /** `project`, or `production/<service>`. */
  scope: string;
  plain: boolean;
}

/** Mirrors `VariableView`: `value` is set for a PLAIN key only, always null for a secret. */
export interface ProjectVariable extends ProjectVariableName {
  value?: string | null;
}

/** Mirrors `ProjectResponse` (`GET /projects/{project}`). */
export interface ProjectDetail extends Project {
  resources: ProjectResourceView[];
  home: { hostname: string | null; node_id: string | null };
  /** OMITTED (not null) unless the caller owns the project or is an admin (D-P40-15). */
  variables?: ProjectVariableName[] | null;
}

export interface VariableList {
  project: string;
  scope: string;
  variables: ProjectVariable[];
}

/** Mirrors `VariableResolveResponse`: per key, the scope a launch would take it from. */
export interface VariableResolve {
  project: string;
  service: string;
  variables: ProjectVariableName[];
}

/** Body of `PUT …/variables`. `secret` defaults to TRUE server-side (D-P40-16). */
export interface VariableSetRequest {
  values: Record<string, string>;
  secret: boolean;
}

export interface VariableSetResponse {
  project: string;
  scope: string;
  keys: string[];
  plain: boolean;
}

export interface VariableDeleted {
  project: string;
  scope: string;
  deleted: string;
}

export interface ProjectDeleted {
  name: string;
  deleted: string[];
}

/** Multipart payload for `POST /deploy` (field names match the route's Form params). */
export interface DeployRequest {
  archive: File;
  name: string;
  port?: number;
  gpus?: number;
  start?: string;
  health?: string;
  /** JSON object string ({KEY: value|null}) — serialized before upload; null deletes on redeploy. */
  env?: Record<string, string | null>;
  vendor?: GpuVendor;
}

// ---------------------------------------------------------------------------
// App templates (Store — P11.5 git-source deploys)
// ---------------------------------------------------------------------------

/** One declared env var of an app template (`secret: true` collects into secrets). */
interface AppTemplateEnvVar {
  name: string;
  description: string;
  required: boolean;
  secret: boolean;
}

/** Deploy defaults an app template carries; request overrides win over these. */
interface AppTemplateDeployDefaults {
  port?: number | null;
  gpus?: number | null;
  start?: string | null;
  health?: string | null;
}

/** Mirrors `AppTemplate` (db/models.py) — one Store catalog entry (git-sourced). */
export interface AppTemplate {
  id: string;
  name: string;
  description: string;
  /** lucide icon name ("server"/"layout" style). */
  icon: string;
  category: string;
  repo_url: string;
  ref?: string | null;
  subdir?: string | null;
  deploy_defaults: AppTemplateDeployDefaults;
  env_schema: AppTemplateEnvVar[];
  ai_hint?: string | null;
}

/** Body for `POST /app-templates/{id}/deploy` — mirrors `TemplateDeployRequest`. */
export interface TemplateDeployRequest {
  name: string;
  env?: Record<string, string>;
  /** Plaintext secret values (secret-typed env schema entries). */
  secrets?: Record<string, string>;
  port?: number;
  gpus?: number;
  start?: string;
  health?: string;
  vendor?: GpuVendor;
}

// ---------------------------------------------------------------------------
// Per-app config-as-API (P7 — GET/PUT /config/apps/{name})
// ---------------------------------------------------------------------------

/** Mirrors `AppConfigView` — a deployed app's config (secret refs, never values). */
export interface AppConfigView {
  service_name: string;
  /** [deploy] fields: name, port, gpus, start, health, memory_limit, cpu_limit. */
  deploy: Record<string, unknown>;
  /** [ai.*] binding spec; `api_key` is a ${secrets.X} ref. */
  ai: Record<string, Record<string, unknown>>;
  /**
   * [db.*] binding spec (P15). `managed` entries carry `{provider, database}`;
   * `external` entries carry `{provider, url, password}` — `url` is always
   * credential-free and `password` is a `${secrets.X}` ref (or the "***"
   * redaction sentinel once set, mirroring `ai.*.api_key`).
   */
  db: Record<string, Record<string, unknown>>;
  /** Env var names only (values never returned). */
  env_keys: string[];
  source: "deploy" | "api" | (string & {});
  revision: number;
  etag: string | null;
}

/** Mirrors `AppConfigWriteResponse` — result of a section PUT (or dry-run). */
export interface AppConfigWriteResponse {
  applied: boolean;
  requires_restart: boolean;
  restarted: boolean;
  view: AppConfigView;
}

// ---------------------------------------------------------------------------
// Daemon config-as-API (P1/P7 — GET/PUT /config/daemon/{section})
// ---------------------------------------------------------------------------

/** Mirrors `ConfigView` — one daemon config section (secrets redacted). */
export interface ConfigView {
  section: string;
  values: Record<string, unknown>;
  etag: string | null;
}

/** Mirrors `ConfigDiffEntry` — one changed key in a staged daemon config write. */
interface ConfigDiffEntry {
  key: string;
  old: unknown;
  new: unknown;
  section: string;
  op: "add" | "change" | "delete" | (string & {});
  requires_restart: boolean;
  secret: boolean;
}

/** Mirrors `ConfigDiagnostic` — a single validation diagnostic. */
interface ConfigDiagnostic {
  loc: string[];
  message: string;
  type: string | null;
}

/** Mirrors `ConfigWriteResponse` — result of a daemon config write (or dry-run). */
export interface ConfigWriteResponse {
  applied: boolean;
  diff: ConfigDiffEntry[];
  diagnostics: ConfigDiagnostic[];
  requires_restart: boolean;
}

// ---------------------------------------------------------------------------
// Daemon self-knowledge (P13b — /capabilities, /doctor, /proxy/status)
// ---------------------------------------------------------------------------

interface CapabilitiesCaller {
  role: string;
  token_name: string | null;
  quotas: { max_gpus: number | null; max_concurrent_jobs: number | null };
}

interface CapabilitiesProxy {
  enabled: boolean;
  available: boolean;
  mode: string;
  base_domain: string | null;
  hostname: string | null;
  scheme: string;
  https_port: number;
  url_shape: string | null;
  dashboard_apex: boolean;
  mdns: boolean;
  /** Admin-only; absent for non-admin callers. */
  admin_addr?: string;
}

interface CapabilitiesModels {
  backends: string[];
  default_backend: string;
  bridge_host: string;
}

/**
 * The P15 managed-data twin of the models block (`system.py` mirrors the two
 * deliberately). Optional: a pre-P15 daemon omits it, and the Databases form
 * falls back to its module constants.
 */
interface CapabilitiesDatabases {
  backends: string[];
  default_backend: string;
  bridge_host: string;
}

interface CapabilitiesDeploy {
  git_enabled: boolean;
  git_allowed_hosts: string[];
  max_upload_bytes: number;
  dry_run: boolean;
  max_concurrent_builds: number;
}

/**
 * The `link` block of `GET /capabilities` (P27 WP-C1, widened by P26 WP-H and
 * P32; typed here for the first time in P34 D4 — the daemon has emitted it all
 * along, the dashboard simply never read it).
 *
 * The shape is conditional, which is why every field but `enabled` is optional:
 * a daemon with no live link manager — `[link].enabled = false`, or an identity
 * key it could not read — projects only `enabled`, `node_id` and `slug`
 * (`daemon/routes/system.py`, the capabilities `link_block`); the tunnel-state
 * fields belong to a manager that does not exist.
 *
 * `enabled` is the TUNNEL, `node_id` is the ENROLMENT, and reading the first as
 * the second is a real bug this block was widened to fix (P34): a node linked
 * once and then deliberately disabled looks exactly like a node that never
 * linked if you only have the flag. `node_id` is the discriminator the doctor
 * `link` row has always used — written by the claim staging tail, untouched by
 * `enabled = false`, cleared only by `nerdit unlink`. Older daemons omit it in
 * the disabled case, so treat absence as unknown rather than as "never linked"
 * where the difference matters.
 */
export interface CapabilitiesLink {
  enabled: boolean;
  /** Tunnel state — `connected` / `connecting` / `backoff` / `displaced` / `terminal`. */
  state?: string;
  node_id?: string | null;
  /** The node half of every hosted name; absent until a claim persists it. */
  slug?: string | null;
  nodes_base_domain?: string | null;
  hosted_public_entitled?: boolean;
  hosted_public_entitled_at?: string | null;
}

/** Mirrors `GET /capabilities` — role-aware daemon self-knowledge projection. */
export interface Capabilities {
  version: string;
  uptime_s: number;
  caller: CapabilitiesCaller;
  proxy: CapabilitiesProxy;
  models: CapabilitiesModels;
  databases?: CapabilitiesDatabases;
  buildpacks: string[];
  deploy: CapabilitiesDeploy;
  gpus: { count: number; schedulable: number; vendors: string[] };
  mcp: { http_enabled: boolean };
  /**
   * Absent on daemons predating P27's link surface — read it as "not linked"
   * rather than sniffing for the feature, the `features` precedent above.
   */
  link?: CapabilitiesLink;
  limits: Record<string, unknown>;
  features: {
    batch: boolean;
    secrets_shared_scope: boolean;
    app_templates: boolean;
    /**
     * PR1 — absent on older daemons; FastAPI silently ignores unknown query
     * params, so `?target=` against a pre-PR1 daemon returns an UNFILTERED
     * page — gate, never sniff.
     */
    audit_target_filter?: boolean;
    /** P40b — `/api/projects` exists; absent on older daemons (gate, never sniff). */
    projects?: boolean;
    /** P40c — `/api/projects/{project}/variables` exists; absent on older daemons. */
    variables?: boolean;
    /** P40d — `POST /api/projects/{project}/apply` exists; absent on older daemons. */
    project_apply?: boolean;
  };
  /** Admin-only; absent for non-admin callers. */
  paths?: { data_dir: string; db_path: string };
}

export type DoctorStatus = "ok" | "warn" | "fail" | "skipped" | (string & {});

/** One `GET /doctor` check row. */
interface DoctorCheck {
  name: string;
  status: DoctorStatus;
  detail: string;
  latency_ms: number;
}

/** Mirrors `GET /doctor` — worst-of status + per-check rows. */
export interface DoctorReport {
  status: DoctorStatus;
  checks: DoctorCheck[];
}

/** Mirrors `ProxyStatusResponse` (`GET /proxy/status`). Nested blocks stay loose. */
export interface ProxyStatus {
  state: string;
  enabled: boolean;
  available: boolean;
  mode: string;
  base_domain: string | null;
  hostname: string;
  scheme: string;
  https_port: number;
  tls: Record<string, unknown>;
  ca: Record<string, unknown>;
  apex: Record<string, unknown>;
  respawn: Record<string, unknown>;
  routes: Record<string, unknown>;
  mdns: Record<string, unknown>;
}

/**
 * One row of `GET /routes` — mirrors `daemon/schemas/proxy.py::RouteItem`.
 *
 * `route` is a locked **tri-state** (Invariant #2), and the renderer must test
 * it with `=== null` / `=== ""`, never truthiness:
 *   * `null` — unrouted (proxy off, or a `kind=model` row, which is by design);
 *   * `""`   — a subdomain route (the shape carries no path);
 *   * `"/name"` — a path route.
 *
 * `live` is the advisory Caddy annotation and is `null` when the live table
 * could not be read (or the proxy is off) — "unknown", never "not registered".
 */
export interface RouteItem {
  service_name: string;
  kind: WorkloadKind;
  status: JobStatus;
  host_port: number;
  /** Live dial target (P24b, D-P24-4b) — equals host_port outside a cutover window. */
  effective_host_port: number;
  container_port: number;
  protocol: string;
  route: string | null;
  public_url: string | null;
  /** (P26) Every advertised URL with its kind/state. Optional: a pre-P26 daemon omits it. */
  public_urls?: PublicUrlEntry[];
  live: { registered: boolean; dial_matches: boolean } | null;
}

/** Cursor-paginated page of routes (`GET /routes`), DB-authoritative. */
export interface RouteListPage {
  items: RouteItem[];
  next_cursor: string | null;
  /** Page-level live-read state; explains a `null` `live` on every row. */
  live_table: "readable" | "unreadable" | "disabled" | (string & {});
}

// ---------------------------------------------------------------------------
// Service converge + diagnose (P13a — /services/{id}/wait, /diagnose)
// ---------------------------------------------------------------------------

/** Mirrors `ServiceWaitResponse` — always HTTP 200 (timeout is a normal branch). */
export interface ServiceWaitResponse {
  outcome: "converged" | "failed" | "timeout" | "superseded" | (string & {});
  service_name: string;
  version: number | null;
  phase: string | null;
  status: JobStatus;
  public_url: string | null;
  /** (P26) Every advertised URL with its kind/state. Optional: a pre-P26 daemon omits it. */
  public_urls?: PublicUrlEntry[];
  reason: string | null;
  error_class: ErrorClass | null;
  error_message: string | null;
  waited_s: number;
}

/** Diagnose coarse failure classification. The JSON key is `class` (quoted). */
interface DiagnoseError {
  class: string | null;
  message: string | null;
}

interface DiagnoseForensics {
  last_exit_code: number | null;
  oom_killed: boolean;
  /** P21 D2: CUDA/HIP allocator OOM matched on the captured crash tail. */
  gpu_oom: boolean;
  /** P33: the entrypoint was refused a root privilege the sandbox drops. */
  priv_denied: boolean;
  last_crash_at: string | null;
}

interface DiagnoseRestarts {
  policy: string | null;
  count: number;
  max_restarts: number;
  window_seconds: number;
  window_start: string | null;
  last_exit_at: string | null;
  backoff_s: number | null;
  next_retry_in_s: number | null;
}

interface DiagnoseHealth {
  spec: Record<string, unknown> | null;
  probe: Record<string, unknown> | null;
  // Actionable advice derived by the route (e.g. no declared health check and
  // the implicit `/` probe answered 404). Rendered by DiagnosePanel.
  observations: string[];
}

interface DiagnoseBindings {
  waiting: boolean;
  messages: string[];
}

interface DiagnoseBuild {
  version: number | null;
  last_result: "failed" | "ok" | "none" | (string & {});
  reason: string | null;
}

interface DiagnoseRemediation {
  code: string;
  detail: string;
}

/** Mirrors `DiagnoseResponse` — the one-call failure bundle (names only, never values). */
export interface DiagnoseResponse {
  service_name: string;
  kind: WorkloadKind;
  status: JobStatus;
  desired_state: string | null;
  last_deploy: LastDeploy | null;
  error: DiagnoseError;
  forensics: DiagnoseForensics;
  restarts: DiagnoseRestarts;
  health: DiagnoseHealth;
  bindings: DiagnoseBindings;
  build: DiagnoseBuild;
  injected_env_keys: string[];
  injected_env_keys_source: string;
  pending_env_keys: string[];
  logs: Array<Record<string, unknown>>;
  remediation: DiagnoseRemediation;
}

// ---------------------------------------------------------------------------
// Git deploy + deploy plan (P11.5 / P13b)
// ---------------------------------------------------------------------------

/** Body for `POST /deploy/git` — mirrors `GitDeployRequest`. */
export interface GitDeployRequest {
  repo_url: string;
  name: string;
  ref?: string | null;
  subdir?: string | null;
  port?: number | null;
  gpus?: number | null;
  start?: string | null;
  health?: string | null;
  /** Env vars; a `null` value deletes the key on redeploy. */
  env?: Record<string, string | null> | null;
  vendor?: string | null;
  /** Private-repo token as a `${secrets.KEY}` reference. */
  token_ref?: string | null;
}

/** The 1.10 plan diff returned by `?dry_run=true` on /deploy + /deploy/git. */
export interface DeployPlan {
  dry_run: boolean;
  action: "create" | "redeploy" | (string & {});
  name: string;
  buildpack: string | null;
  effective: {
    port: number | null;
    gpus: number | null;
    start: string | null;
    health: string | null;
    memory_limit: string | null;
    cpu_limit: number | null;
  };
  env_diff: Record<string, unknown>;
  ai_diff: { action: string; bindings: unknown };
  overwrote_api_config: boolean;
  warnings: string[];
}

// ---------------------------------------------------------------------------
// Scoped API tokens (P1 surface, P23 UI)
// ---------------------------------------------------------------------------

/** Role granted by a scoped token — mirrors `db.enums.TokenRole`. */
export type TokenRole = "admin" | "submitter" | "readonly";

/**
 * Hash-free projection of an API token — mirrors `daemon/schemas/tokens.py`
 * `TokenView`, and *only* those fields: the plaintext and the stored hash are
 * deliberately absent, and no column that does not exist yet is anticipated.
 */
export interface TokenView {
  id: string;
  name: string;
  role: TokenRole;
  max_gpus: number | null;
  max_concurrent_jobs: number | null;
  created_at: string;
  last_used_at: string | null;
  revoked: boolean;
  /** Absolute expiry instant; `null` = never expires (P25 D-P25-1). */
  expires_at: string | null;
  /** Service names this token may write to; `null` = unscoped (P25 D-P25-3). */
  scope_services: string[] | null;
}

/**
 * `POST /tokens` 201 body — the view plus the plaintext, returned exactly once.
 * The action is in the daemon's `NO_BODY_CACHE_ACTIONS`, so an idempotent
 * *replay* answers with a non-secret envelope carrying no `token` at all; the
 * UI treats that as a real state, not a parse failure (D-P23-5).
 */
export interface TokenCreateResponse extends TokenView {
  token: string;
}

/** Body for `POST /tokens` — mirrors `TokenCreateRequest`. */
export interface TokenCreateRequest {
  name: string;
  role: TokenRole;
  max_gpus?: number | null;
  max_concurrent_jobs?: number | null;
  /**
   * Seconds until expiry. OMITTED means "apply the daemon's
   * `[security].token_default_ttl_s`"; an explicit `null` means "never
   * expires" — the two are distinct on the wire (P25 D-P25-1), so never send
   * `null` to mean "unset".
   */
  expires_in_s?: number | null;
  scope_services?: string[] | null;
}

// ---------------------------------------------------------------------------
// Disk accounting + garbage collection (P14b `/system/disk` + `/system/gc`)
// ---------------------------------------------------------------------------

/**
 * Docker's aggregate usage. The whole block is `null` when `df` failed — the
 * only honest docker-unavailable signal the daemon has, and never rendered as
 * zeros (a zero here would read as "nothing to reclaim", which is a lie).
 */
export interface DockerDiskUsage {
  images_bytes: number;
  containers_bytes: number;
  volumes_bytes: number;
  build_cache_bytes: number;
}

/** A backup tree bucket: total bytes plus file count (absent dir ⇒ 0/0). */
export interface BackupTreeUsage {
  bytes: number;
  count: number;
}

/**
 * `GET /system/disk` — mirrors `daemon/routes/system.py::_build_disk_report`
 * COMPLETELY (the card renders a subset on purpose; the type may not, or the
 * next reader mistakes an omission for a field that does not exist).
 *
 * A `null` byte count means the bounded `du` walk overran its soft budget or
 * failed — paired with a `"scan_timeout"` warning. Render "unknown", never 0.
 * `archive_bytes` is the one exception: `0` there is the honest count when
 * audit archiving is disabled, which is why the daemon never nulls it for that
 * reason.
 */
export interface SystemDiskReport {
  docker: DockerDiskUsage | null;
  images: {
    total_bytes: number | null;
    /** Per-repo attribution, keyed by repo name (`nerdit-app/my-app` → bytes). */
    by_repo: Record<string, number>;
  };
  data_dir: {
    services: { name: string; bytes: number }[];
    /** `null` ⇒ the per-service walk timed out. */
    services_total_bytes: number | null;
    models: { ollama: number | null; huggingface: number | null };
    /** `0` when audit archiving is disabled; `null` when the walk timed out. */
    archive_bytes: number | null;
    backups: BackupTreeUsage;
    /** P15 WP7 per-database volume tars — a disjoint glob from `backups`. */
    volume_backups: BackupTreeUsage;
    /** P37 per-database logical dump tars — disjoint from both other globs. */
    dumps: BackupTreeUsage;
    /** Bytes left in `<data_dir>/dump-staging/`; `null` when the walk timed out. */
    dump_staging_bytes: number | null;
  };
  orphan_images: string[];
  orphan_data_dirs: string[];
  warnings: string[];
}

/**
 * `POST /system/gc` (both `?dry_run=true` and the real run — same shape, with
 * `dry_run` saying which). The CLI renderer
 * `cli/commands/system.py::_render_gc` is the behavioral spec.
 */
export interface GcResult {
  dry_run: boolean;
  images: {
    /** Repos removed — or, on a dry run, the ones that *would* be removed. */
    removed: string[];
    skipped: { repo: string; reason: string }[];
    reclaimed_bytes_estimate: number;
  };
  orphan_data: {
    enabled: boolean;
    removed: string[];
    skipped: { name: string; reason: string }[];
  };
  /** Report-only sizes; `null` values mean a walk overran the soft budget. */
  reports: {
    weights: { ollama: number | null; huggingface: number | null };
    build_cache_bytes: number;
    backups_over_keep: number | null;
  };
  warnings: string[];
}
