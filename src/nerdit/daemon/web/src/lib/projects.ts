import type { AppBindings } from "../api/queries";
import type { Database, Model, Service } from "../api/types";
import {
  apiReadiness,
  dbExternalReadiness,
  dbManagedReadiness,
  modelReadiness,
  type BindingReadiness
} from "./bindings";

// The pure-function twin of lib/bindings.ts (D6): a project is one
// `kind === "service"` workload plus the resources its config references. This
// module never invents server state — status and tooltips are derived from the
// service row, its bindings, and the live model/database inventories, so the
// grid, the (PR3) detail page, and the palette can never diverge. No React
// imports: it operates on plain data and is unit-tested in isolation.

export type ProjectStatus =
  | "running" // app running, every binding ready/unknown
  | "deploying" // last_deploy.phase is queued|building|launching
  | "attention" // app itself fine, but a binding is waiting (D6 — distinct from degraded)
  | "degraded" // the app's own status is degraded
  | "stopped" // stopped / desired_state stopped
  | "failed"; // failed, or last_deploy.phase failed with nothing serving

export interface ProjectResource {
  type: "ai" | "db";
  binding: string; // binding name, e.g. "default"
  target: string | null; // model ref / database name / base URL
  readiness: BindingReadiness; // from lib/bindings
}

export interface Project {
  name: string;
  service: Service; // the kind=service row, verbatim
  status: ProjectStatus;
  /** The exact contributing resource/phase for the tooltip (D6 — never invented). */
  statusDetail: string; // e.g. "Waiting on model llama3.1:8b."
  resources: ProjectResource[];
  configError: Error | null; // an app whose config fan-out failed still renders
}

/**
 * The canonical home route for a workload by kind (D8) — shared by
 * `ServiceRedirect` (the kind-aware `/services/:ident` redirect) and
 * `ProjectDetail` (a deep link to a non-service name bounces to its real home).
 * A `service` lands on its project (P40e), read from the row's `project` /
 * `service` FIELDS and never parsed out of the label: the service whose label
 * IS the project name lands on `/projects/:project`, any other on its own
 * `/projects/:project/services/:service`. A row without the fields (an older
 * daemon) keeps the label path. Models and databases land on their inventory
 * pages; anything else falls back to the project grid.
 */
export function kindHomePath(
  svc: Pick<Service, "kind" | "name" | "project" | "service">
): string {
  switch (svc.kind) {
    case "service":
      if (svc.project && svc.service && svc.project !== svc.name) {
        return `/projects/${encodeURIComponent(svc.project)}/services/${encodeURIComponent(svc.service)}`;
      }
      return `/projects/${encodeURIComponent(svc.project ?? svc.name)}`;
    case "model":
      return "/models";
    case "database":
      return "/databases";
    default:
      return "/projects";
  }
}

/** Human phrase for a mid-deploy phase (statusDetail). */
function deployingDetail(phase: string): string {
  switch (phase) {
    case "queued":
      return "Deploy queued.";
    case "building":
      return "Building the new image.";
    case "launching":
      return "Launching the container.";
    default:
      return "Deploying.";
  }
}

/** Tooltip naming the exact resource that is holding the project in attention. */
function waitingDetail(resource: ProjectResource): string {
  const noun = resource.type === "ai" ? "model" : "database";
  if (resource.target) return `Waiting on ${noun} ${resource.target}.`;
  return `Waiting on the ${resource.binding} ${noun} binding.`;
}

/** Project a stored `[ai.*]` binding spec into a resource + its live readiness. */
function aiResource(
  name: string,
  spec: Record<string, unknown>,
  models: Model[]
): ProjectResource {
  const provider = String(spec.provider ?? "ollama");
  if (provider === "ollama") {
    const model = (spec.model as string | null) ?? null;
    return { type: "ai", binding: name, target: model, readiness: modelReadiness(model, models) };
  }
  const baseUrl = (spec.base_url as string | null) ?? null;
  const apiKey = (spec.api_key as string | null) ?? null;
  return { type: "ai", binding: name, target: baseUrl, readiness: apiReadiness(baseUrl, apiKey) };
}

/** Project a stored `[db.*]` binding spec into a resource + its live readiness. */
function dbResource(
  name: string,
  spec: Record<string, unknown>,
  databases: Database[]
): ProjectResource {
  const provider = String(spec.provider ?? "managed");
  if (provider === "managed") {
    const database = (spec.database as string | null) ?? null;
    return {
      type: "db",
      binding: name,
      target: database,
      readiness: dbManagedReadiness(database, databases)
    };
  }
  const url = (spec.url as string | null) ?? null;
  const password = (spec.password as string | null) ?? null;
  return { type: "db", binding: name, target: url, readiness: dbExternalReadiness(url, password) };
}

/**
 * The bound resources of a project. A missing/loading bindings entry
 * contributes nothing (status from the service alone); a fan-out error
 * contributes nothing either (the caller surfaces it via `configError`).
 */
function deriveResources(
  bindings: AppBindings | undefined,
  models: Model[],
  databases: Database[]
): ProjectResource[] {
  if (!bindings || bindings.error) return [];
  const out: ProjectResource[] = [];
  for (const [name, spec] of Object.entries(bindings.ai ?? {})) {
    out.push(aiResource(name, spec, models));
  }
  for (const [name, spec] of Object.entries(bindings.db ?? {})) {
    out.push(dbResource(name, spec, databases));
  }
  return out;
}

/**
 * Derive one project's aggregate. Status precedence (documented + tested
 * exhaustively): failed > deploying > stopped > degraded > attention > running.
 * A `restarting` service reads as attention (transient, worth flagging but not
 * down), ranked below degraded and after the phase checks.
 *
 * The redeploy-failure nuance (D6): an app whose last deploy failed but whose
 * container still serves the old image (status running/degraded) is NOT failed
 * — the card must not claim it is down. A failed deploy only escalates to
 * "failed" when nothing is serving; while serving it reads as "attention" (or
 * stays "degraded" if the app's own health is degraded).
 */
export function deriveProject(
  service: Service,
  bindings: AppBindings | undefined,
  models: Model[],
  databases: Database[]
): Project {
  const resources = deriveResources(bindings, models, databases);
  const configError = bindings?.error ?? null;
  const status = service.status;
  const phase = service.last_deploy?.phase ?? null;
  const serving = status === "running" || status === "degraded";

  const make = (projectStatus: ProjectStatus, statusDetail: string): Project => ({
    name: service.name,
    service,
    status: projectStatus,
    statusDetail,
    resources,
    configError
  });

  // 1. failed — the app's own container failed, or a deploy failed with nothing
  //    left serving.
  if (status === "failed") {
    return make("failed", service.error_message ?? "App failed.");
  }
  if (phase === "failed" && !serving) {
    const reason = service.last_deploy?.reason;
    return make("failed", reason ? `Last deploy failed: ${reason}` : "Last deploy failed.");
  }

  // 2. deploying — a deploy generation is mid-flight (even when the old image
  //    still serves; the card shows the live phase), or the row itself is
  //    still coming up with no deploy phase to speak of (a direct
  //    `POST /services` create sits in building/pending/scheduled).
  if (phase === "queued" || phase === "building" || phase === "launching") {
    return make("deploying", deployingDetail(phase));
  }
  if (status === "building" || status === "pending" || status === "scheduled") {
    return make("deploying", "App starting.");
  }

  // 3. stopped — desired-state stopped / stopped status, and the terminal
  //    exits a service row can end in.
  if (status === "stopped" || service.desired_state === "stopped") {
    return make("stopped", "App stopped.");
  }
  if (status === "completed" || status === "cancelled") {
    return make("stopped", status === "completed" ? "App exited." : "App cancelled.");
  }

  // 4. degraded — the app's own health is degraded (distinct from a waiting
  //    binding).
  if (status === "degraded") {
    return make("degraded", "App degraded, health checks failing.");
  }

  // 4b. restarting — a transient reconcile transition; flag it as attention.
  if (status === "restarting") {
    return make("attention", "App restarting.");
  }

  // 5. attention — the app itself is fine, but the last deploy failed while it
  //    kept serving, or a bound resource is not ready yet (D6).
  if (phase === "failed") {
    return make("attention", "Last deploy failed. Previous version still serving.");
  }
  const waiting = resources.find((resource) => resource.readiness.state === "waiting");
  if (waiting) {
    return make("attention", waitingDetail(waiting));
  }

  // 6. running — serving, every binding ready or unknown. Only an actual
  //    running status may claim it; any state not handled above (paused,
  //    retrying, a future status) reads as attention with the raw status —
  //    never a false "running".
  if (status === "running") {
    return make("running", "App running.");
  }
  return make("attention", `App ${status}.`);
}

/**
 * Reverse index (D9): the names of projects bound to a given resource.
 * Matching mirrors binding resolution: an ai resource targets a model by
 * its model REF (`Model.model`), a db resource targets a managed database
 * by its service NAME. Order follows `projects`; names are unique.
 */
export function projectsBoundTo(
  projects: Project[],
  type: "ai" | "db",
  target: string
): string[] {
  const out: string[] = [];
  for (const project of projects) {
    const bound = project.resources.some(
      (resource) => resource.type === type && resource.target === target
    );
    if (bound && !out.includes(project.name)) out.push(project.name);
  }
  return out;
}

/**
 * Derive every project. `services` is caller-pre-filtered to
 * `kind === "service"`; `apps` are the matching binding fan-out results
 * (order-independent — matched by name). Preserves the `services` order.
 */
export function deriveProjects(
  services: Service[],
  apps: AppBindings[],
  models: Model[],
  databases: Database[]
): Project[] {
  return services.map((service) => {
    const bindings = apps.find((app) => app.name === service.name);
    return deriveProject(service, bindings, models, databases);
  });
}
