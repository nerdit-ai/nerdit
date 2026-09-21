import {
  useInfiniteQuery,
  useMutation,
  useQueries,
  useQuery,
  useQueryClient
} from "@tanstack/react-query";
import { api, apiRaw, ApiError } from "./client";
import { capture } from "../lib/analytics";
import { toastApiError } from "../lib/apiErrors";
import type {
  AppConfigView,
  AppConfigWriteResponse,
  AppTemplate,
  AuditFilters,
  AuditLogPage,
  Capabilities,
  ClusterInfo,
  ClusterStats,
  ConfigView,
  ConfigWriteResponse,
  Database,
  DatabaseListPage,
  DeployPlan,
  DeployRequest,
  DiagnoseResponse,
  DoctorReport,
  GcResult,
  GitDeployRequest,
  Gpu,
  LogEntry,
  Model,
  ModelListPage,
  ProjectDeleted,
  ProjectDetail,
  ProjectListPage,
  ProxyStatus,
  RouteListPage,
  SecretDeleted,
  SecretNames,
  Service,
  ServiceDeleted,
  ServiceListPage,
  ServiceWaitResponse,
  SystemDiskReport,
  TemplateDeployRequest,
  TokenCreateRequest,
  TokenCreateResponse,
  TokenView,
  VariableDeleted,
  VariableList,
  VariableSetRequest,
  VariableSetResponse
} from "./types";

/**
 * RFC 4122 v4 UUID via crypto.getRandomValues — fallback for plain-HTTP
 * dashboards (http://<host>:9321), where crypto.randomUUID is undefined
 * outside secure contexts.
 */
function uuidv4Fallback(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
  bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 10xx
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

/**
 * Fresh Idempotency-Key header for a dashboard mutation — the daemon's
 * IdempotencyMiddleware collapses accidental retries into a single apply,
 * same contract as the CLI/MCP clients.
 */
export function idempotencyHeaders(): Record<string, string> {
  const uuid =
    typeof crypto.randomUUID === "function" ? crypto.randomUUID() : uuidv4Fallback();
  return { "Idempotency-Key": uuid };
}

export type DaemonStatus = "online" | "degraded" | "offline";

export function useDaemonStatus(): { status: DaemonStatus; lastSeen: Date | null } {
  const q = useQuery({
    queryKey: ["health"],
    queryFn: () => api<{ status: string }>("/health"),
    refetchInterval: 5000,
    retry: 1,
    retryDelay: 1000,
    staleTime: 0
  });
  let status: DaemonStatus = "online";
  if (q.isSuccess) status = "online";
  else if (q.failureCount >= 2) status = "offline";
  else if (q.failureCount === 1 || q.isError) status = "degraded";
  const lastSeen = q.dataUpdatedAt ? new Date(q.dataUpdatedAt) : null;
  return { status, lastSeen };
}

/**
 * The authenticated principal's role (from /auth/check) — lets write UI hide
 * up front for non-admins (P8 shared secrets). The server-side role checks
 * stay authoritative; consumers keep their 403 fallback as defense in depth.
 */
export function useAuthRole() {
  return useQuery({
    queryKey: ["auth", "role"],
    queryFn: () => api<{ ok: boolean; role: string }>("/auth/check"),
    staleTime: Infinity
  });
}

export function useClusterStats() {
  return useQuery({
    queryKey: ["cluster", "stats"],
    queryFn: () => api<ClusterStats>("/cluster/stats"),
    refetchInterval: 2000
  });
}

export function useClusterInfo() {
  return useQuery({
    queryKey: ["cluster", "info"],
    queryFn: () => api<ClusterInfo>("/cluster/info")
  });
}

export function useGpus() {
  return useQuery({
    queryKey: ["gpus"],
    queryFn: () => api<Gpu[]>("/gpus"),
    refetchInterval: 2000
  });
}

// ---------------------------------------------------------------------------
// Services (P2-P4 surface)
// ---------------------------------------------------------------------------

export function useService(ident: string) {
  return useQuery({
    queryKey: ["services", ident],
    queryFn: () => api<Service>(`/services/${encodeURIComponent(ident)}`),
    refetchInterval: 5000,
    enabled: Boolean(ident),
    // Surface a 404 (deleted/renamed service) at once so the kind-aware redirect
    // and project-page not-found guard fire promptly; the 5s poll recovers a
    // transient error on its own.
    retry: false
  });
}

/**
 * Block until a service reaches a terminal deploy outcome (`/wait`, P13a).
 * Always HTTP 200 — a `timeout` outcome is a normal branch, not an error. The
 * query keeps polling (short server-side timeout) until the outcome is terminal
 * (converged/failed/superseded). `version` pins the wait to a deploy generation.
 */
export function useServiceWait(ident: string, version?: number | null, enabled = true) {
  return useQuery({
    queryKey: ["services", ident, "wait", version ?? null],
    queryFn: () => {
      const params = new URLSearchParams();
      params.set("timeout", "10");
      if (version != null) params.set("version", String(version));
      return api<ServiceWaitResponse>(
        `/services/${encodeURIComponent(ident)}/wait?${params.toString()}`
      );
    },
    enabled: enabled && Boolean(ident),
    gcTime: 0,
    refetchInterval: (query) => {
      const outcome = query.state.data?.outcome;
      if (outcome === "converged" || outcome === "failed" || outcome === "superseded") {
        return false;
      }
      return 750;
    }
  });
}

/** One-call failure bundle for a service (`/diagnose`, P13a). Names only, never values. */
export function useDiagnose(ident: string, enabled = true) {
  return useQuery({
    queryKey: ["services", ident, "diagnose"],
    queryFn: () => api<DiagnoseResponse>(`/services/${encodeURIComponent(ident)}/diagnose`),
    enabled: enabled && Boolean(ident)
  });
}

export function useServiceLogs(ident: string, sinceId: number, enabled: boolean, tail?: number) {
  const params = new URLSearchParams();
  params.set("since_id", String(sinceId));
  if (tail != null) params.set("tail", String(tail));
  return useQuery({
    queryKey: ["services", ident, "logs", sinceId, tail ?? null],
    queryFn: () =>
      api<LogEntry[]>(`/services/${encodeURIComponent(ident)}/logs?${params.toString()}`),
    refetchInterval: enabled ? 1000 : false,
    enabled
  });
}

export type ServiceAction = "stop" | "restart" | "delete";

/** Argument to a `delete` action: a bare name, or a name plus a `?purge` CSV.
 * A `kind=database` row's delete MUST carry `purge: "data"` (the daemon 409s
 * `db.delete_requires_purge` otherwise); every other kind takes the default. */
export type ServiceActionArg = string | { ident: string; purge?: string };

/** Toast context for a failed lifecycle action ("Stop failed", …). */
function actionContext(action: ServiceAction): string {
  const verb = action === "stop" ? "Stop" : action === "restart" ? "Restart" : "Delete";
  return `${verb} failed`;
}

/** Desired-state lifecycle mutation: stop / restart / delete a service (or model row). */
export function useServiceAction(action: ServiceAction) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (arg: ServiceActionArg): Promise<Service | ServiceDeleted> => {
      const ident = typeof arg === "string" ? arg : arg.ident;
      const purge = typeof arg === "string" ? undefined : arg.purge;
      const path = `/services/${encodeURIComponent(ident)}`;
      if (action === "delete") {
        const qs = purge ? `?purge=${encodeURIComponent(purge)}` : "";
        return api<ServiceDeleted>(`${path}${qs}`, {
          method: "DELETE",
          headers: idempotencyHeaders()
        });
      }
      return api<Service>(`${path}/${action}`, { method: "POST", headers: idempotencyHeaders() });
    },
    onSuccess: () => {
      capture("service_action", { action });
      void qc.invalidateQueries({ queryKey: ["services"] });
      // (P40e) The apps list reads `/projects`; a deleted service must leave it at once.
      void qc.invalidateQueries({ queryKey: ["projects"] });
      void qc.invalidateQueries({ queryKey: ["models"] });
      // A database row rides the same /services quartet, so its own list is
      // stale after stop/restart/delete too — without this the Databases page
      // kept showing a row the operator had just deleted.
      void qc.invalidateQueries({ queryKey: ["databases"] });
      void qc.invalidateQueries({ queryKey: ["cluster"] });
    },
    onError: (err: Error) => toastApiError(err, actionContext(action))
  });
}

// ---------------------------------------------------------------------------
// Models (P5 surface — list-only; lifecycle reuses the /services quartet)
// ---------------------------------------------------------------------------

export function useModels() {
  return useQuery({
    queryKey: ["models"],
    queryFn: () => api<ModelListPage>("/models"),
    refetchInterval: 5000
  });
}

/** Body for `POST /models` — mirrors `ModelServeRequest` (db/models.py). */
export interface ServeModelRequest {
  /** Model reference to serve, e.g. "llama3.1:8b". */
  model: string;
  /** GPUs the model server needs (0 = CPU inference; shared). */
  gpus?: number;
  /** Optional service-name override (default: sanitized model ref). */
  name?: string;
  /** Serving backend ('ollama' | 'vllm'); omitted uses the daemon default (P11, D9). */
  backend?: string;
}

export function useServeModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ServeModelRequest) =>
      api<Model>("/models", {
        method: "POST",
        body: JSON.stringify(body),
        headers: idempotencyHeaders()
      }),
    onSuccess: (_res, body) => {
      capture("serve_model", { backend: body.backend ?? null, gpus: body.gpus ?? 0 });
      void qc.invalidateQueries({ queryKey: ["models"] });
      void qc.invalidateQueries({ queryKey: ["services"] });
      void qc.invalidateQueries({ queryKey: ["cluster"] });
    },
    onError: (err: Error) => toastApiError(err, "Serve failed")
  });
}

// ---------------------------------------------------------------------------
// Databases (P15 — list-only; lifecycle reuses the /services quartet, mirrors
// useModels exactly)
// ---------------------------------------------------------------------------

export function useDatabases() {
  return useQuery({
    queryKey: ["databases"],
    queryFn: () => api<DatabaseListPage>("/databases"),
    refetchInterval: 5000
  });
}

/**
 * `POST /databases` — provision a managed database (P15). The minted
 * credential is never part of the request or the response: the daemon mints
 * it server-side and injects it at binding resolution.
 */
export function useCreateDatabase() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { backend?: string; name?: string }) =>
      api<Database>("/databases", {
        method: "POST",
        body: JSON.stringify(body),
        headers: idempotencyHeaders()
      }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["databases"] });
      void qc.invalidateQueries({ queryKey: ["services"] });
      void qc.invalidateQueries({ queryKey: ["cluster"] });
    },
    onError: (err: Error) => toastApiError(err, "Create failed")
  });
}

// ---------------------------------------------------------------------------
// Cross-app bindings (P15 — client-side aggregation for the Bindings AI/DB
// pages). No dedicated cross-app endpoint exists server-side, so this fans
// out: one bounded /services fetch (kind=service rows only — models/databases
// carry no [ai.*]/[db.*] config of their own) followed by one
// GET /config/apps/{name} per app. Each app-config fetch shares its cache
// (same queryKey/queryFn shape) with `useAppConfig`, so a page already viewed
// via ServiceDetail's GenericBindingsPanel is not re-fetched here.
// ---------------------------------------------------------------------------

/** One app's binding specs (or a fetch error) for the cross-app pages. */
export interface AppBindings {
  name: string;
  ai: Record<string, Record<string, unknown>> | null;
  db: Record<string, Record<string, unknown>> | null;
  error: Error | null;
}

/**
 * Fetch the complete managed-workload inventory by walking the `/services`
 * cursor. The server page cap (200) is shared across kinds, so a fleet with
 * many models/databases can push `kind=service` rows past the first page —
 * a single-shot read would silently drop projects from the grid and the
 * palette. Bounded to 5 pages (1000 rows) so pagination can never spin;
 * shared by `useAllBindings` and the command palette (same queryKey, one
 * cache entry).
 */
export async function fetchAllServices(): Promise<ServiceListPage> {
  const items: ServiceListPage["items"] = [];
  let cursor: string | null = null;
  for (let page = 0; page < 5; page++) {
    const suffix: string = cursor ? `&cursor=${encodeURIComponent(cursor)}` : "";
    const data: ServiceListPage = await api(`/services?limit=200${suffix}`);
    items.push(...data.items);
    cursor = data.next_cursor;
    if (!cursor) break;
  }
  return { items, next_cursor: null };
}

export function useAllBindings(): {
  apps: AppBindings[];
  /** The full `kind === "service"` rows behind `apps` (project grid, PR2). */
  services: Service[];
  isLoading: boolean;
  error: Error | null;
} {
  const services = useQuery({
    queryKey: ["services", "all-for-bindings"],
    queryFn: fetchAllServices,
    staleTime: 10000
  });

  const serviceRows = (services.data?.items ?? []).filter((svc) => svc.kind === "service");
  const serviceNames = serviceRows.map((svc) => svc.name);

  const configQueries = useQueries({
    queries: serviceNames.map((name) => ({
      // Same queryKey + queryFn shape as `useAppConfig` (D — cache-shared).
      queryKey: ["app-config", name],
      queryFn: async (): Promise<ConfigWithEtag<AppConfigView>> => {
        const { data, etag } = await apiRaw<AppConfigView>(
          `/config/apps/${encodeURIComponent(name)}`
        );
        return { view: data, etag: etag ?? data.etag };
      },
      staleTime: 10000
    }))
  });

  const apps: AppBindings[] = serviceNames.map((name, index) => {
    const q = configQueries[index];
    return {
      name,
      ai: q.data?.view.ai ?? null,
      db: q.data?.view.db ?? null,
      error: (q.error as Error | null) ?? null
    };
  });

  return {
    apps,
    services: serviceRows,
    isLoading: services.isLoading || configQueries.some((q) => q.isLoading),
    error: (services.error as Error | null) ?? null
  };
}

// ---------------------------------------------------------------------------
// Audit (polling-primary — the admin-gated SSE stream cannot authenticate
// from EventSource; see clusterEvents.ts)
// ---------------------------------------------------------------------------

export function useAudit(filters?: AuditFilters, enabled = true) {
  const limit = filters?.limit ?? 50;
  return useInfiniteQuery({
    queryKey: ["audit", filters ?? null],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam }) => {
      const params = new URLSearchParams();
      params.set("limit", String(limit));
      if (filters?.action) params.set("action", filters.action);
      if (filters?.result) params.set("result", filters.result);
      if (filters?.target) params.set("target", filters.target);
      if (filters?.target_type) params.set("target_type", filters.target_type);
      if (pageParam) params.set("cursor", pageParam);
      return api<AuditLogPage>(`/audit?${params.toString()}`);
    },
    getNextPageParam: (last) => last.next_cursor,
    enabled,
    // A 403 (non-admin / ownership) is surfaced as the query error and rendered
    // as an admin hint by the project feed — never retried into a storm.
    retry: false
  });
}

// ---------------------------------------------------------------------------
// Deploy + rollback (P4 surface)
// ---------------------------------------------------------------------------

export function useDeploy() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: DeployRequest) => {
      const form = new FormData();
      form.append("archive", body.archive);
      form.append("name", body.name);
      if (body.port != null) form.append("port", String(body.port));
      if (body.gpus != null) form.append("gpus", String(body.gpus));
      if (body.start != null) form.append("start", body.start);
      if (body.health != null) form.append("health", body.health);
      if (body.env != null) form.append("env", JSON.stringify(body.env));
      if (body.vendor != null) form.append("vendor", body.vendor);
      return api<Service>("/deploy", {
        method: "POST",
        body: form,
        headers: idempotencyHeaders()
      });
    },
    onSuccess: () => {
      capture("deploy", { source: "zip" });
      void qc.invalidateQueries({ queryKey: ["services"] });
      void qc.invalidateQueries({ queryKey: ["cluster"] });
    },
    onError: (err: Error) => toastApiError(err, "Deploy failed")
  });
}

export function useRollback() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) =>
      api<Service>(`/deploy/${encodeURIComponent(name)}/rollback`, {
        method: "POST",
        headers: idempotencyHeaders()
      }),
    onSuccess: () => {
      capture("rollback");
      void qc.invalidateQueries({ queryKey: ["services"] });
    },
    onError: (err: Error) => toastApiError(err, "Rollback failed")
  });
}

/**
 * Redeploy a git-sourced app (`POST /deploy/{name}/redeploy`, P24b WP6).
 *
 * Bodyless by design: the daemon reads the repo, ref and subdir back off the
 * recorded source, so the app name is the whole request. Re-cloning the
 * recorded *ref* is what picks up the new HEAD.
 */
export function useRedeployService() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) =>
      api<Service>(`/deploy/${encodeURIComponent(name)}/redeploy`, {
        method: "POST",
        headers: idempotencyHeaders()
      }),
    onSuccess: () => {
      capture("deploy", { source: "redeploy" });
      void qc.invalidateQueries({ queryKey: ["services"] });
      void qc.invalidateQueries({ queryKey: ["cluster"] });
    },
    onError: (err: Error) => toastApiError(err, "Deploy failed")
  });
}

/**
 * Redeploy a workspace-sourced app (`POST /workspaces/{name}/deploy`, P29).
 *
 * The redeploy route above deliberately refuses workspace rows (409
 * `deploy.no_source`, D-P29-7) — the daemon rebuilds these from the app's
 * server-side workspace tree instead, so the two source kinds take two
 * endpoints behind the one Deploy button (Codex review, PR #147).
 */
export function useDeployWorkspace() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) =>
      api<Service>(`/workspaces/${encodeURIComponent(name)}/deploy`, {
        method: "POST",
        headers: idempotencyHeaders()
      }),
    onSuccess: () => {
      capture("deploy", { source: "workspace" });
      void qc.invalidateQueries({ queryKey: ["services"] });
      void qc.invalidateQueries({ queryKey: ["cluster"] });
    },
    onError: (err: Error) => toastApiError(err, "Deploy failed")
  });
}

function deployFormData(body: DeployRequest): FormData {
  const form = new FormData();
  form.append("archive", body.archive);
  form.append("name", body.name);
  if (body.port != null) form.append("port", String(body.port));
  if (body.gpus != null) form.append("gpus", String(body.gpus));
  if (body.start != null) form.append("start", body.start);
  if (body.health != null) form.append("health", body.health);
  if (body.env != null) form.append("env", JSON.stringify(body.env));
  if (body.vendor != null) form.append("vendor", body.vendor);
  return form;
}

/**
 * Dry-run a ZIP deploy (`?dry_run=true`): returns the 1.10 plan diff with zero
 * writes. Sends no Idempotency-Key — the middleware bypasses the key claim on a
 * dry run, so a stray key can never poison the later real deploy.
 */
export function useDeployPlan() {
  return useMutation({
    mutationFn: (body: DeployRequest) =>
      api<DeployPlan>("/deploy?dry_run=true", { method: "POST", body: deployFormData(body) }),
    onError: (err: Error) => toastApiError(err, "Preview failed")
  });
}

/**
 * Deploy from a Git URL (`POST /deploy/git`, P11.5). `dryRun` returns the plan
 * diff (no key, no writes); a real deploy mints an Idempotency-Key.
 */
export function useDeployGit() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ body, dryRun }: { body: GitDeployRequest; dryRun?: boolean }) => {
      const query = dryRun ? "?dry_run=true" : "";
      return api<Service | DeployPlan>(`/deploy/git${query}`, {
        method: "POST",
        body: JSON.stringify(body),
        headers: dryRun ? undefined : idempotencyHeaders()
      });
    },
    onSuccess: (_res, vars) => {
      if (!vars.dryRun) {
        capture("deploy", { source: "git" });
        void qc.invalidateQueries({ queryKey: ["services"] });
        void qc.invalidateQueries({ queryKey: ["cluster"] });
      }
    },
    onError: (err: Error) => toastApiError(err, "Deploy failed")
  });
}

// ---------------------------------------------------------------------------
// App templates (Store — git-sourced one-click deploys)
// ---------------------------------------------------------------------------

export function useAppTemplates() {
  return useQuery({
    queryKey: ["app-templates"],
    queryFn: () => api<AppTemplate[]>("/app-templates"),
    staleTime: 60_000
  });
}

/** Deploy a Store template by id (`POST /app-templates/{id}/deploy`). */
export function useDeployTemplate() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: TemplateDeployRequest }) =>
      api<Service>(`/app-templates/${encodeURIComponent(id)}/deploy`, {
        method: "POST",
        body: JSON.stringify(body),
        headers: idempotencyHeaders()
      }),
    onSuccess: (_res, { id }) => {
      capture("deploy_template", { template_id: id });
      void qc.invalidateQueries({ queryKey: ["services"] });
      void qc.invalidateQueries({ queryKey: ["cluster"] });
    },
    onError: (err: Error) => toastApiError(err, "Deploy failed")
  });
}

// ---------------------------------------------------------------------------
// Secrets (write-only: key names in, never values out)
// ---------------------------------------------------------------------------

export function useSecretNames(service: string) {
  return useQuery({
    queryKey: ["secrets", service],
    queryFn: () => api<SecretNames>(`/secrets/${encodeURIComponent(service)}`),
    enabled: Boolean(service)
  });
}

export function useSetSecrets(service: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (values: Record<string, string>) =>
      api<SecretNames>(`/secrets/${encodeURIComponent(service)}`, {
        method: "POST",
        body: JSON.stringify({ values }),
        headers: idempotencyHeaders()
      }),
    onSuccess: (_res, values) => {
      // Count only — secret keys and values never leave the browser.
      capture("set_secret", { service, key_count: Object.keys(values).length });
      void qc.invalidateQueries({ queryKey: ["secrets", service] });
    },
    onError: (err: Error) => toastApiError(err, "Save failed")
  });
}

/** Delete one secret key, or all of them when `key` is omitted. */
export function useDeleteSecret(service: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (key?: string) =>
      api<SecretDeleted>(
        key
          ? `/secrets/${encodeURIComponent(service)}/${encodeURIComponent(key)}`
          : `/secrets/${encodeURIComponent(service)}`,
        { method: "DELETE", headers: idempotencyHeaders() }
      ),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["secrets", service] });
    },
    onError: (err: Error) => toastApiError(err, "Delete failed")
  });
}

// ---------------------------------------------------------------------------
// Projects and variables (P40e — `/api/projects*`, gated on
// `capabilities.features.projects | variables`). Query keys carry names only,
// never a value.
// ---------------------------------------------------------------------------

/** Every in-scope project, walking the cursor like `fetchAllServices` (bounded: 5 × 200). */
export function useProjects(enabled = true) {
  return useQuery({
    queryKey: ["projects"],
    queryFn: async (): Promise<ProjectListPage> => {
      const items: ProjectListPage["items"] = [];
      let cursor: string | null = null;
      for (let page = 0; page < 5; page++) {
        const suffix: string = cursor ? `&cursor=${encodeURIComponent(cursor)}` : "";
        const data: ProjectListPage = await api(`/projects?limit=200${suffix}`);
        items.push(...data.items);
        cursor = data.next_cursor;
        if (!cursor) break;
      }
      return { items, next_cursor: null };
    },
    refetchInterval: 5000,
    enabled
  });
}

export function useProject(name: string, enabled = true) {
  return useQuery({
    queryKey: ["projects", name],
    queryFn: () => api<ProjectDetail>(`/projects/${encodeURIComponent(name)}`),
    // A 403 is this token's scope, which never changes: stop polling it. A 404
    // keeps polling — another session can create the project any time.
    refetchInterval: (query) => {
      const error = query.state.error;
      return error instanceof ApiError && error.status === 403 ? false : 5000;
    },
    enabled: enabled && Boolean(name),
    // A 404 is an answer (a legacy label deep link resolves through the
    // service row instead), not something to retry.
    retry: false
  });
}

function variablesPath(project: string, service?: string, key?: string): string {
  const base = `/projects/${encodeURIComponent(project)}/variables`;
  const qs = service ? `?service=${encodeURIComponent(service)}` : "";
  return `${base}${key ? `/${encodeURIComponent(key)}` : ""}${qs}`;
}

/**
 * One scope's variables (project scope when `service` is omitted): names, the
 * plain flag, and the value of a PLAIN key only. Owner or admin (D-P40-15) —
 * callers enable it only for an owner view, and a 403 is a quiet state the
 * panel renders as nothing, so it is never retried.
 */
export function useVariables(project: string, service?: string, enabled = true) {
  return useQuery({
    queryKey: ["projects", project, "variables", service ?? null],
    queryFn: () => api<VariableList>(variablesPath(project, service)),
    enabled: enabled && Boolean(project),
    retry: false
  });
}

/**
 * Set/merge variables in one scope. The argument is a THUNK, read once at
 * request time: the mutation cache keeps `variables` for as long as the
 * observer lives, and a closure over the (cleared-on-success) input is the
 * only thing it may hold — never a secret value. `gcTime: 0` drops the entry
 * as soon as it is unobserved.
 */
export function useSetVariables(project: string, service?: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: () => VariableSetRequest) =>
      api<VariableSetResponse>(variablesPath(project, service), {
        method: "PUT",
        body: JSON.stringify(body()),
        headers: idempotencyHeaders()
      }),
    gcTime: 0,
    onSuccess: (res) => {
      // The flag only — variable keys and values never leave the browser.
      capture("set_variable", { plain: res.plain });
      void qc.invalidateQueries({ queryKey: ["projects", project] });
    },
    onError: (err: Error) => toastApiError(err, "Save failed")
  });
}

export function useDeleteVariable(project: string, service?: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (key: string) =>
      api<VariableDeleted>(variablesPath(project, service, key), {
        method: "DELETE",
        headers: idempotencyHeaders()
      }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["projects", project] });
    },
    onError: (err: Error) => toastApiError(err, "Delete failed")
  });
}

/** Delete a project: every service through the `/services` cascade, then the row. */
export function useDeleteProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ name, purge }: { name: string; purge?: string }) =>
      api<ProjectDeleted>(
        `/projects/${encodeURIComponent(name)}${purge ? `?purge=${encodeURIComponent(purge)}` : ""}`,
        { method: "DELETE", headers: idempotencyHeaders() }
      ),
    // Settled, not success: a 409 `project.delete_incomplete` still removed
    // some services.
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ["projects"] });
      void qc.invalidateQueries({ queryKey: ["services"] });
      void qc.invalidateQueries({ queryKey: ["cluster"] });
    },
    onError: (err: Error) => toastApiError(err, "Delete failed")
  });
}

// ---------------------------------------------------------------------------
// Daemon self-knowledge (P13b — /capabilities, /doctor, /proxy/status)
// ---------------------------------------------------------------------------

/** Static daemon capabilities + the caller's role (feeds admin gating + selectors). */
export function useCapabilities() {
  return useQuery({
    queryKey: ["capabilities"],
    queryFn: () => api<Capabilities>("/capabilities"),
    staleTime: 60_000
  });
}

/** Timeout-bounded structured health checks (Settings operator card). */
export function useDoctor(enabled = true) {
  return useQuery({
    queryKey: ["doctor"],
    queryFn: () => api<DoctorReport>("/doctor"),
    enabled,
    staleTime: 5_000
  });
}

/** Typed ProxyManager state projection (Settings proxy card). */
export function useProxyStatus(enabled = true) {
  return useQuery({
    queryKey: ["proxy", "status"],
    queryFn: () => api<ProxyStatus>("/proxy/status"),
    enabled,
    refetchInterval: 10_000
  });
}

/**
 * One page of the DB-authoritative route inventory (`GET /routes`, P13b).
 *
 * The cursor is part of the query key so each page caches on its own; the card
 * advances a local cursor and appends the pages it has seen (an operator table
 * of a handful of rows does not need infinite-query machinery). The cursor is
 * opaque server text, so it is URL-encoded, never interpolated raw.
 */
export function useRoutes(cursor?: string) {
  return useQuery({
    queryKey: ["routes", cursor ?? null],
    queryFn: () =>
      api<RouteListPage>(
        `/routes?limit=50${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ""}`
      )
  });
}

/** Admin-only graceful restart. Mints an Idempotency-Key; 409 = already in progress. */
export function useDaemonRestart() {
  return useMutation({
    mutationFn: (drainTimeoutS?: number) =>
      api<{ restarting: boolean; in_flight_builds: number; drain_timeout_s: number }>(
        "/daemon/restart",
        {
          method: "POST",
          headers: idempotencyHeaders(),
          body:
            drainTimeoutS != null ? JSON.stringify({ drain_timeout_s: drainTimeoutS }) : undefined
        }
      ),
    onSuccess: () => capture("daemon_restart"),
    onError: (err: Error) => toastApiError(err, "Restart failed")
  });
}

// ---------------------------------------------------------------------------
// Config-as-API (P7 — daemon [proxy] section + per-app config)
// ---------------------------------------------------------------------------

/** A config view plus the ETag to round-trip on the matching write (D11). */
export interface ConfigWithEtag<T> {
  view: T;
  etag: string | null;
}

/** Read one daemon config section (proxy) with its ETag (D8/D11 hostname editor). */
export function useDaemonConfigSection(section: string) {
  return useQuery({
    queryKey: ["daemon-config", section],
    queryFn: async (): Promise<ConfigWithEtag<ConfigView>> => {
      const { data, etag } = await apiRaw<ConfigView>(`/config/daemon/${section}`);
      return { view: data, etag: etag ?? data.etag };
    },
    enabled: Boolean(section)
  });
}

export interface UpdateDaemonConfigVars {
  values: Record<string, unknown>;
  /** ETag from the GET; sent as If-Match on a real write (concurrent edits 409). */
  etag?: string | null;
  dryRun?: boolean;
}

/**
 * Write one daemon config section. A dry run previews the diff and sends
 * neither If-Match nor an Idempotency-Key (D11); a real write sends both.
 */
export function useUpdateDaemonConfig(section: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ values, etag, dryRun }: UpdateDaemonConfigVars) => {
      const query = dryRun ? "?dry_run=true" : "";
      const headers: Record<string, string> = {};
      if (!dryRun) {
        if (etag) headers["If-Match"] = etag;
        Object.assign(headers, idempotencyHeaders());
      }
      return api<ConfigWriteResponse>(`/config/daemon/${section}${query}`, {
        method: "PUT",
        body: JSON.stringify(values),
        headers
      });
    },
    onSuccess: (_res, vars) => {
      if (!vars.dryRun) {
        void qc.invalidateQueries({ queryKey: ["daemon-config", section] });
      }
    },
    onError: (err: Error) => toastApiError(err, "Config update failed")
  });
}

/** Read a deployed app's config view with its ETag (Configure panel, D6). */
export function useAppConfig(name: string) {
  return useQuery({
    queryKey: ["app-config", name],
    queryFn: async (): Promise<ConfigWithEtag<AppConfigView>> => {
      const { data, etag } = await apiRaw<AppConfigView>(
        `/config/apps/${encodeURIComponent(name)}`
      );
      return { view: data, etag: etag ?? data.etag };
    },
    enabled: Boolean(name)
  });
}

export interface UpdateAppConfigVars {
  section: "deploy" | "ai" | "db";
  /** Partial section values; a `null` value deletes the key/binding. */
  values: Record<string, unknown>;
  /** ETag from the GET; sent as If-Match on a real write. */
  etag?: string | null;
  dryRun?: boolean;
  /** Opt into a service restart after the write (`?restart=true`). */
  restart?: boolean;
}

/**
 * Assemble the path + headers + body for an app config section write. Pure, so
 * the ETag/Idempotency-Key round-trip is unit-testable without a React render:
 * a real write sends If-Match (from the GET's ETag) + a fresh Idempotency-Key;
 * a dry run sends neither (matches the middleware key-claim bypass).
 */
export function appConfigWriteRequest(
  name: string,
  { section, values, etag, dryRun, restart }: UpdateAppConfigVars
): { path: string; headers: Record<string, string>; body: string } {
  const params = new URLSearchParams();
  if (dryRun) params.set("dry_run", "true");
  if (restart) params.set("restart", "true");
  const query = params.toString() ? `?${params.toString()}` : "";
  const headers: Record<string, string> = {};
  if (!dryRun) {
    if (etag) headers["If-Match"] = etag;
    Object.assign(headers, idempotencyHeaders());
  }
  return {
    path: `/config/apps/${encodeURIComponent(name)}/${section}${query}`,
    headers,
    body: JSON.stringify(values)
  };
}

/**
 * Write one app config section (deploy | ai). A dry run previews the would-be
 * view and sends neither If-Match nor an Idempotency-Key; a real write sends
 * both (the backend 400s a real write without an Idempotency-Key).
 */
export function useUpdateAppConfig(name: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (vars: UpdateAppConfigVars) => {
      const { path, headers, body } = appConfigWriteRequest(name, vars);
      return api<AppConfigWriteResponse>(path, { method: "PUT", body, headers });
    },
    onSuccess: (_res, vars) => {
      if (!vars.dryRun) {
        if (vars.section === "ai") capture("apply_binding", { app: name });
        void qc.invalidateQueries({ queryKey: ["app-config", name] });
        void qc.invalidateQueries({ queryKey: ["services"] });
      }
    },
    onError: (err: Error) => toastApiError(err, "Config update failed")
  });
}

// ---------------------------------------------------------------------------
// Tokens (P1 surface, P23 UI)
// ---------------------------------------------------------------------------

/** Scoped tokens (admin-only server-side). `GET /tokens` is a bare list, not a page. */
export function useTokens(includeRevoked: boolean) {
  return useQuery({
    queryKey: ["tokens", includeRevoked],
    queryFn: () =>
      api<TokenView[]>(`/tokens${includeRevoked ? "?include_revoked=true" : ""}`)
  });
}

/**
 * Mint a scoped token. The plaintext comes back exactly once and is NEVER
 * written to a query cache (D-P23-5) — the caller consumes `data` into local
 * component state and calls `reset()` when the modal closes.
 *
 * `gcTime: 0` is load-bearing, not tuning: `reset()` detaches the observer but
 * does not synchronously evict the `Mutation` from the `MutationCache`, and the
 * default `gcTime` is five minutes — long enough for the plaintext to stay
 * readable via `getMutationCache().getAll()` after the modal is gone. With a
 * zero GC delay the entry is removed as the last observer detaches.
 */
export function useCreateToken() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: TokenCreateRequest) =>
      api<TokenCreateResponse>("/tokens", {
        method: "POST",
        headers: idempotencyHeaders(),
        body: JSON.stringify(body)
      }),
    gcTime: 0,
    // Invalidate only — the response body is never written into the cache.
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["tokens"] });
    },
    onError: (err: Error) => toastApiError(err, "Token creation failed")
  });
}

/** Soft-revoke a token by id. */
export function useRevokeToken() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      api<{ id: string; revoked: boolean }>(`/tokens/${encodeURIComponent(id)}`, {
        method: "DELETE",
        headers: idempotencyHeaders()
      }),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["tokens"] });
    },
    onError: (err: Error) => toastApiError(err, "Revoke failed")
  });
}

// ---------------------------------------------------------------------------
// Disk + garbage collection (P14b /system/disk + /system/gc, P23 UI)
// ---------------------------------------------------------------------------

/**
 * Bounded disk accounting. Deliberately NO `refetchInterval`: the server runs
 * `du` walks over the whole data dir under a lock, so a polling card would keep
 * a walker permanently busy for every open dashboard. Refreshes on mount past
 * the 30 s staleness window, and on demand.
 */
export function useSystemDisk(enabled = true) {
  return useQuery({
    queryKey: ["system", "disk"],
    queryFn: () => api<SystemDiskReport>("/system/disk"),
    enabled,
    staleTime: 30_000
  });
}

/**
 * Garbage-collect orphan app images (and, opt-in, orphan service data dirs).
 *
 * A dry run sends NO Idempotency-Key — the middleware bypasses the key claim on
 * a dry run, so a stray key could otherwise poison the real run that follows
 * (the `useDeployGit` precedent). Only a real run mints a key and invalidates
 * the disk report.
 */
export function useSystemGc() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      includeOrphanData,
      dryRun
    }: {
      includeOrphanData: boolean;
      dryRun?: boolean;
    }) => {
      const query = dryRun ? "?dry_run=true" : "";
      return api<GcResult>(`/system/gc${query}`, {
        method: "POST",
        body: JSON.stringify({ include_orphan_data: includeOrphanData }),
        headers: dryRun ? undefined : idempotencyHeaders()
      });
    },
    onSuccess: (_res, vars) => {
      if (!vars.dryRun) {
        capture("system_gc", { include_orphan_data: vars.includeOrphanData });
        void qc.invalidateQueries({ queryKey: ["system", "disk"] });
      }
    },
    onError: (err: Error) => toastApiError(err, "Cleanup failed")
  });
}
