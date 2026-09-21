import { useEffect, useMemo, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import { ApiError } from "../api/client";
import { ConfigurePanel } from "../components/ConfigurePanel";
import { DeployDialog } from "../components/DeployDialog";
import { DeployProgress } from "../components/DeployProgress";
import { DiagnosePanel } from "../components/DiagnosePanel";
import { SecretsPanel } from "../components/SecretsPanel";
import { ServiceLogsPanel } from "../components/ServiceLogsPanel";
import { ServicePublicUrl } from "../components/ServicePublicUrl";
import { ServiceShareCard } from "../components/ServiceShareCard";
import { GenericBindingsPanel } from "../components/bindings/GenericBindingsPanel";
import { aiSection, dbSection } from "../components/bindings/sections";
import {
  Badge,
  Button,
  Checkbox,
  Confirm,
  DataRow,
  Dialog,
  Menu,
  Mono,
  Notice,
  PageHeader,
  Panel,
  type MenuItem
} from "../components/ui";
import {
  useAppConfig,
  useAudit,
  useAuthRole,
  useCapabilities,
  useDatabases,
  useModels,
  useDeployWorkspace,
  useRedeployService,
  useRollback,
  useService,
  useServiceAction,
  type AppBindings
} from "../api/queries";
import {
  featuredDomainUrl,
  formatRelative,
  formatUptime,
  hostedShareUrl,
  publicUrlState,
  sourceLine
} from "../lib/format";
import { serviceStatusLabel, serviceStatusTone } from "../lib/status";
import { deriveProject, kindHomePath, type Project, type ProjectResource } from "../lib/projects";
import type { AuditLogEntry, Service } from "../api/types";
import { toast } from "../state/toastStore";

/**
 * The app page (design guidelines §4).
 *
 * Three tabs, one place per fact. The address renders once — in the header.
 * The deploy phase renders once — in Overview's status area. Identity trivia
 * (id, kind, image, ports, created) sits behind the Details disclosure in
 * Manage rather than in a 13-row `<dl>` on the first screen.
 *
 * The header carries exactly one primary action, **Deploy**, and it is
 * source-aware: an app whose recorded source can be re-fetched (git, or a
 * server-side workspace) rebuilds from that source after a confirm; a ZIP row
 * has nothing to re-clone, so Deploy opens the deploy dialog prefilled with
 * its name. Everything else lives in the `…` menu.
 */

const TABS = [
  { key: "overview", label: "Overview" },
  { key: "logs", label: "Logs" },
  { key: "manage", label: "Manage" }
] as const;

type TabKey = (typeof TABS)[number]["key"];

/**
 * The five-tab page collapsed to three, so an old bookmark still resolves:
 * deploy history folded into Overview, the bindings editor and the secrets
 * card into Manage.
 */
const LEGACY_TABS: Record<string, TabKey> = {
  deployments: "overview",
  resources: "manage",
  settings: "manage"
};

/** Deploy phases that mean "a generation is still in flight". */
const IN_FLIGHT = new Set(["queued", "building", "launching"]);

/**
 * A source the daemon can rebuild on its own. The two kinds take two
 * endpoints (Codex review, PR #147): git rides `POST /deploy/{name}/redeploy`,
 * while a workspace row rides `POST /workspaces/{name}/deploy` — the redeploy
 * route deliberately 409s `deploy.no_source` on workspace rows (D-P29-7).
 */
function redeployable(svc: Service): boolean {
  const type = svc.source?.type;
  return type === "git" || type === "workspace";
}

function tabPath(basePath: string, key: TabKey): string {
  return key === "overview" ? basePath : `${basePath}/${key}`;
}

/**
 * (P40e) The page is mounted by `ProjectPage`'s route components, which resolve
 * the URL to a service LABEL through the API (never by parsing one):
 * `/projects/:name` for the single-service project and the pre-P40 daemon,
 * `/projects/:name/services/:service` inside a multi-service project.
 */
export interface ProjectDetailProps {
  /** The service's wire identity (D-P40-6) — every API call here uses it. */
  label: string;
  /** Where this page's tabs live; the overview tab IS this path. */
  basePath: string;
  /** The project page, when the daemon has one; absent on a pre-P40 daemon. */
  projectPath?: string;
  /** True under `/services/:service`: a way back up, and delete returns there. */
  nested?: boolean;
}

export default function ProjectDetail({ label, basePath, projectPath, nested }: ProjectDetailProps) {
  const { tab } = useParams();
  const name = label;
  const navigate = useNavigate();
  const location = useLocation();

  const service = useService(name);
  const config = useAppConfig(name);
  const models = useModels();
  const databases = useDatabases();
  const capabilities = useCapabilities();
  const role = useAuthRole().data?.role;

  const stop = useServiceAction("stop");
  const restart = useServiceAction("restart");
  const remove = useServiceAction("delete");
  const rollback = useRollback();
  const redeploy = useRedeployService();
  const deployWorkspace = useDeployWorkspace();

  const [confirmDelete, setConfirmDelete] = useState(false);
  const [purgeData, setPurgeData] = useState(false);
  const [purgeImages, setPurgeImages] = useState(false);
  const [purgeWorkspace, setPurgeWorkspace] = useState(false);
  const [confirmDeploy, setConfirmDeploy] = useState(false);
  const [progress, setProgress] = useState<Service | null>(null);
  const [deployOpen, setDeployOpen] = useState(false);
  const [configureOpen, setConfigureOpen] = useState(false);

  const svc = service.data;

  // A 404 (or a deep link to a non-service kind) bounces off the app page:
  // 404 → the apps list + a toast; a model/database name → its inventory home.
  const notFound = service.isError && service.error instanceof ApiError && service.error.status === 404;
  useEffect(() => {
    if (notFound) {
      navigate("/", { replace: true });
      toast("error", "App not found");
    }
  }, [notFound, navigate]);
  useEffect(() => {
    if (svc && svc.kind !== "service") {
      navigate(kindHomePath(svc), { replace: true });
    }
  }, [svc, navigate]);

  // A legacy tab slug resolves to its new home and rewrites the URL, so an old
  // bookmark lands on the content it asked for instead of a silent Overview.
  const legacy = tab ? LEGACY_TABS[tab] : undefined;
  useEffect(() => {
    if (legacy) navigate(tabPath(basePath, legacy), { replace: true });
  }, [legacy, basePath, navigate]);

  // The ⌘K palette redeploy quick action hands off via router state. It routes
  // through the SAME source-aware handler as the header Deploy button — a git
  // or workspace app confirms and rebuilds from its recorded source, only a ZIP
  // row opens the deploy dialog — then clears the history state so a
  // refresh/back never re-opens it.
  //
  // The gate matters: `onDeploy` closes over `fromSource`, a const initialised
  // only on the fully rendered path, so the effect must not fire on a loading,
  // erroring or redirecting render.
  // (`project` is derived from `svc` unconditionally, so a service-kind `svc`
  // is exactly the fully rendered path.)
  const deployReady =
    !service.isLoading && !service.error && !notFound && svc?.kind === "service";
  useEffect(() => {
    if (!deployReady) return;
    if ((location.state as { redeploy?: boolean } | null)?.redeploy) {
      onDeploy();
      navigate(location.pathname, { replace: true, state: null });
    }
    // `onDeploy` is a per-render function declaration; `deployReady` is the
    // real trigger alongside the router state, so it is not a dependency.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deployReady, location.state, location.pathname, navigate]);

  const canConfigure = role !== "readonly";
  // `undefined` while the role / capabilities queries are still resolving, so
  // the gate hint stays silent instead of flashing "needs a newer daemon" at an
  // admin on a current daemon. The feed itself stays fail-closed either way.
  const isAdmin = role === undefined ? undefined : role === "admin";
  const auditFilterAvailable = capabilities.isPending
    ? undefined
    : Boolean(capabilities.data?.features?.audit_target_filter);
  const auditEnabled = auditFilterAvailable === true && isAdmin === true;

  const bindings: AppBindings | undefined = useMemo(() => {
    const view = config.data?.view;
    if (view) return { name, ai: view.ai ?? null, db: view.db ?? null, error: null };
    if (config.error) return { name, ai: null, db: null, error: config.error as Error };
    return undefined;
  }, [config.data, config.error, name]);

  const project = useMemo(
    () =>
      svc ? deriveProject(svc, bindings, models.data?.items ?? [], databases.data?.items ?? []) : null,
    [svc, bindings, models.data, databases.data]
  );

  if (service.isLoading) return <p className="text-14 text-muted-foreground">Loading app…</p>;
  if (notFound || (svc && svc.kind !== "service")) {
    return <p className="text-14 text-muted-foreground">Loading app…</p>;
  }
  if (service.error) return <p className="text-14 text-destructive">{(service.error as Error).message}</p>;
  if (!svc || !project) return <p className="text-14 text-destructive">App not found.</p>;

  const activeTab: TabKey =
    legacy ?? (tab && TABS.some((t) => t.key === tab) ? (tab as TabKey) : "overview");
  const busy =
    stop.isPending ||
    restart.isPending ||
    remove.isPending ||
    rollback.isPending ||
    redeploy.isPending ||
    deployWorkspace.isPending;

  // The share card already answers "what address does this app have"; the
  // compact line is the fallback for an app the proxy has not routed, so the
  // address never renders twice and never renders zero times.
  const hasShareSurface = Boolean(
    featuredDomainUrl(svc.endpoint) ||
      hostedShareUrl(svc.endpoint) ||
      (publicUrlState(svc.endpoint) === "routed" && svc.endpoint?.public_url)
  );

  const fromSource = redeployable(svc);

  function doRollback() {
    rollback.mutate(svc!.name, {
      onSuccess: () => toast("success", `Rolling ${svc!.name} back to the previous version`)
    });
  }

  function onDeploy() {
    if (fromSource) setConfirmDeploy(true);
    else setDeployOpen(true);
  }

  function onConfirmDeploy() {
    const fire = svc!.source?.type === "workspace" ? deployWorkspace : redeploy;
    fire.mutate(svc!.name, {
      onSuccess: (updated) => {
        setConfirmDeploy(false);
        setProgress(updated);
      }
    });
  }

  const purge = ["secrets"];
  if (purgeData) purge.push("data");
  if (purgeImages) purge.push("images");
  if (purgeWorkspace) purge.push("workspace");

  function onConfirmDelete() {
    remove.mutate(
      { ident: svc!.name, purge: purge.join(",") },
      {
        onSuccess: () => {
          setConfirmDelete(false);
          toast("success", `App ${svc!.name} deleted`);
          navigate(nested && projectPath ? projectPath : "/");
        }
      }
    );
  }

  const menuItems: MenuItem[] = [
    { label: "Restart", disabled: busy, onSelect: () => restart.mutate(svc.name, {
      onSuccess: () => toast("success", `Restarting ${svc.name}`)
    }) }
  ];
  if (svc.rollback_available) {
    menuItems.push({ label: "Roll back", disabled: busy, onSelect: doRollback });
  }
  menuItems.push("separator");
  menuItems.push({
    label: "Stop",
    disabled: busy || svc.status === "stopped",
    onSelect: () => stop.mutate(svc.name, { onSuccess: () => toast("success", `Stopping ${svc.name}`) })
  });
  menuItems.push({
    label: "Delete…",
    destructive: true,
    disabled: busy,
    onSelect: () => setConfirmDelete(true)
  });

  return (
    <div className="mx-auto max-w-content space-y-6">
      {nested && projectPath && (
        <Link
          to={projectPath}
          className="text-13 text-muted-foreground hover:text-foreground"
          data-testid="app-project-link"
        >
          ← {svc.project ?? "Project"}
        </Link>
      )}
      <PageHeader
        title={nested ? (svc.service ?? svc.name) : svc.name}
        subtitle={
          <Badge
            tone={serviceStatusTone(svc.status)}
            title={project.statusDetail}
            data-testid="app-status"
          >
            {serviceStatusLabel(svc.status)}
          </Badge>
        }
        actions={
          <>
            <Button variant="primary" data-testid="app-deploy" disabled={busy} onClick={onDeploy}>
              Deploy
            </Button>
            <Menu align="end" items={menuItems} />
          </>
        }
      />

      <div className="min-w-0" data-testid="app-address">
        {hasShareSurface ? (
          <ServiceShareCard endpoint={svc.endpoint} />
        ) : (
          <ServicePublicUrl endpoint={svc.endpoint} copyable />
        )}
      </div>

      <nav aria-label="App sections" className="flex gap-6 border-b border-border">
        {TABS.map((t) => {
          const active = t.key === activeTab;
          return (
            <Link
              key={t.key}
              to={tabPath(basePath, t.key)}
              aria-current={active ? "page" : undefined}
              className={`-mb-px border-b-2 px-1 py-2 text-14 ${
                active
                  ? "border-primary text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              {t.label}
            </Link>
          );
        })}
      </nav>

      {activeTab === "overview" && (
        <OverviewTab
          svc={svc}
          project={project}
          name={name}
          managePath={tabPath(basePath, "manage")}
          canConfigure={canConfigure}
          showProgress={progress === null}
          isAdmin={isAdmin}
          auditFilterAvailable={auditFilterAvailable}
          auditEnabled={auditEnabled}
        />
      )}
      {activeTab === "logs" && <ServiceLogsPanel ident={svc.name} status={svc.status} />}
      {activeTab === "manage" && (
        <ManageTab
          svc={svc}
          name={name}
          projectPath={nested ? undefined : projectPath}
          canConfigure={canConfigure}
          onConfigure={() => setConfigureOpen(true)}
        />
      )}

      <Confirm
        open={confirmDeploy}
        title="Deploy the latest source"
        confirmLabel="Deploy"
        busy={redeploy.isPending || deployWorkspace.isPending}
        description={
          <div data-testid="app-deploy-confirm" className="space-y-2">
            <p>
              {svc.source?.type === "workspace" ? (
                <>
                  Rebuilds <Mono>{svc.name}</Mono> from its server-side workspace files. The
                  current version keeps serving while the new one builds.
                </>
              ) : (
                <>
                  Pulls the recorded source for <Mono>{svc.name}</Mono> and rebuilds it. The
                  current version keeps serving while the new one builds.
                </>
              )}
            </p>
            {sourceLine(svc.source) && (
              <p className="text-13 text-subtle-foreground">
                Source: <Mono>{sourceLine(svc.source)}</Mono>
              </p>
            )}
          </div>
        }
        onConfirm={onConfirmDeploy}
        onCancel={() => setConfirmDeploy(false)}
      />

      <Confirm
        open={confirmDelete}
        title="Delete app?"
        confirmLabel="Delete"
        destructive
        busy={remove.isPending}
        confirmPhrase={purgeData ? svc.name : undefined}
        description={
          <div data-testid="app-delete-confirm" className="space-y-3">
            <p>
              Removes the container for <Mono>{svc.name}</Mono> and stops its address answering.
              This app's secrets are deleted with it.
            </p>
            <p>
              Kept: {keptList(purgeData, purgeImages, purgeWorkspace, svc)}
            </p>
            <div className="flex flex-col gap-2">
              <Checkbox
                label="Also delete data volumes"
                checked={purgeData}
                onChange={(event) => setPurgeData(event.target.checked)}
              />
              <Checkbox
                label="Also delete built images"
                checked={purgeImages}
                onChange={(event) => setPurgeImages(event.target.checked)}
              />
              {svc.source?.type === "workspace" && (
                <Checkbox
                  label="Also delete the workspace source"
                  checked={purgeWorkspace}
                  onChange={(event) => setPurgeWorkspace(event.target.checked)}
                />
              )}
            </div>
            {purgeData && (
              <p className="text-destructive">Deleting the data volumes cannot be undone.</p>
            )}
          </div>
        }
        onConfirm={onConfirmDelete}
        onCancel={() => setConfirmDelete(false)}
      />

      {progress && (
        <Dialog
          open
          onClose={() => setProgress(null)}
          title={<span className="sr-only">Deploying {svc.name}</span>}
          data-testid="deploy-progress-dialog"
        >
          <DeployProgress service={progress} onClose={() => setProgress(null)} />
        </Dialog>
      )}

      <DeployDialog open={deployOpen} onClose={() => setDeployOpen(false)} initialName={svc.name} />

      {configureOpen && <ConfigurePanel name={svc.name} onClose={() => setConfigureOpen(false)} />}
    </div>
  );
}

/** The honest other half of the delete dialog: what the purge choices spare. */
function keptList(
  purgeData: boolean,
  purgeImages: boolean,
  purgeWorkspace: boolean,
  svc: Service
): string {
  const kept: string[] = [];
  if (!purgeData) kept.push("data volumes");
  if (!purgeImages) kept.push("built images");
  if (svc.source?.type === "workspace" && !purgeWorkspace) kept.push("the workspace source");
  return kept.length > 0 ? `${kept.join(", ")}.` : "nothing else.";
}

// --- Gating hint shared by the activity feed --------------------------------

function AuditGateHint({
  isAdmin,
  auditFilterAvailable
}: {
  isAdmin: boolean | undefined;
  auditFilterAvailable: boolean | undefined;
}) {
  if (isAdmin === undefined || auditFilterAvailable === undefined) return null;
  if (!auditFilterAvailable) {
    return <p className="text-13 text-subtle-foreground">Activity needs a newer daemon.</p>;
  }
  if (!isAdmin) {
    return <p className="text-13 text-subtle-foreground">Activity needs an admin token.</p>;
  }
  return null;
}

// --- Overview ---------------------------------------------------------------

/** "Running for 2h 05m" when derivable; otherwise the header Badge suffices. */
function statusLine(svc: Service): string | null {
  if (svc.status === "running" && svc.started_at) return `Running for ${formatUptime(svc.started_at)}`;
  if (svc.status === "degraded" && svc.started_at) {
    return `Running for ${formatUptime(svc.started_at)}, health checks failing`;
  }
  return null;
}

/**
 * The last deploy as ONE line. A failed deploy that left the previous version
 * serving says so — that honesty is the product, not decoration.
 */
function deployLine(svc: Service): string | null {
  const last = svc.last_deploy;
  if (!last?.phase) return null;
  if (IN_FLIGHT.has(last.phase)) return null;
  const version = last.version ?? svc.build_version;
  if (last.phase === "failed") {
    const head = last.reason ? `Deploy failed: ${last.reason}` : "Deploy failed";
    const serving = svc.status === "running" || svc.status === "degraded";
    return serving && svc.build_version != null ? `${head} · v${svc.build_version} still serving` : head;
  }
  const when = formatRelative(last.updated_at ?? last.started_at);
  const label = version != null ? `Deployed v${version}` : "Deployed";
  return when === "–" ? label : `${label} · ${when}`;
}

function OverviewTab({
  svc,
  project,
  name,
  managePath,
  canConfigure,
  showProgress,
  isAdmin,
  auditFilterAvailable,
  auditEnabled
}: {
  svc: Service;
  project: Project;
  name: string;
  managePath: string;
  canConfigure: boolean;
  showProgress: boolean;
  isAdmin: boolean | undefined;
  auditFilterAvailable: boolean | undefined;
  auditEnabled: boolean;
}) {
  const phase = svc.last_deploy?.phase;
  const inFlight = Boolean(phase && IN_FLIGHT.has(phase));
  const status = statusLine(svc);
  const deploy = deployLine(svc);

  // The walk is dismissible per generation: closing it hides THIS deploy's
  // progress, and the next generation brings it back on its own.
  const version = svc.last_deploy?.version ?? null;
  const [dismissed, setDismissed] = useState<number | null>(null);
  const walking = inFlight && showProgress && dismissed !== version;

  return (
    <div className="space-y-6">
      <div className="space-y-1" data-testid="app-status-area">
        {status && <p className="text-14 text-foreground">{status}</p>}
        {deploy && (
          <p className="text-14 text-muted-foreground" data-testid="app-deploy-line">
            {deploy}
          </p>
        )}
        {!status && !deploy && (
          <p className="text-14 text-muted-foreground">No deploy recorded for this app yet.</p>
        )}
      </div>

      {walking && (
        <Panel className="p-4" data-testid="app-deploy-progress">
          <DeployProgress service={svc} onClose={() => setDismissed(version)} />
        </Panel>
      )}

      {svc.error_message && <Notice tone="destructive">{svc.error_message}</Notice>}

      <Panel title="Resources">
        {project.resources.length === 0 ? (
          <p className="px-4 py-3 text-13 text-subtle-foreground">
            No models or databases bound.{" "}
            <Link to={managePath} className="text-primary hover:underline">
              Wire one from Manage.
            </Link>
          </p>
        ) : (
          <div className="divide-y divide-border">
            {project.resources.map((resource) => (
              <ResourceSummaryRow key={`${resource.type}:${resource.binding}`} resource={resource} />
            ))}
          </div>
        )}
      </Panel>

      {canConfigure && <DiagnosePanel ident={svc.name} status={svc.status} phase={phase} />}

      <Panel
        title="Recent activity"
        actions={
          <Link to="/activity" className="text-13 text-muted-foreground hover:text-foreground">
            All activity
          </Link>
        }
      >
        <div className="px-4 py-3">
          {auditEnabled ? (
            <ActivityFeed name={name} />
          ) : (
            <AuditGateHint isAdmin={isAdmin} auditFilterAvailable={auditFilterAvailable} />
          )}
        </div>
      </Panel>
    </div>
  );
}

/** Read-only summary of one binding: what it points at, and whether it is ready. */
function ResourceSummaryRow({ resource }: { resource: ProjectResource }) {
  return (
    <DataRow label={`${resource.type === "ai" ? "Model" : "Database"} · ${resource.binding}`}>
      <span className="flex flex-wrap items-baseline justify-end gap-x-3 gap-y-1">
        <Mono>{resource.target ?? `${resource.binding} (unset)`}</Mono>
        <span className="text-13 text-muted-foreground">{resource.readiness.detail}</span>
      </span>
    </DataRow>
  );
}

/** A non-ok audit result is worth a word; an ok one is the expected case. */
function ResultBadge({ result }: { result: string | null }) {
  if (!result || result === "ok") return null;
  const tone = result === "error" ? "destructive" : result === "denied" || result === "replay" ? "warning" : "muted";
  return <Badge tone={tone}>{result}</Badge>;
}

function ActivityFeed({ name }: { name: string }) {
  const query = useAudit({ target: name, limit: 5 }, true);
  const rows = (query.data?.pages ?? []).flatMap((page) => page.items).slice(0, 5);

  if (query.isError) {
    return <p className="text-13 text-subtle-foreground">Activity needs an admin token.</p>;
  }
  if (query.isLoading) return <p className="text-13 text-muted-foreground">Reading activity…</p>;
  if (rows.length === 0) {
    return <p className="text-13 text-subtle-foreground">No activity recorded yet.</p>;
  }

  return (
    <ul className="space-y-1.5">
      {rows.map((entry: AuditLogEntry) => (
        <li key={entry.id} className="flex items-center gap-3 text-14">
          <span className="w-20 shrink-0 text-13 text-subtle-foreground">{formatRelative(entry.ts)}</span>
          <Mono className="min-w-0 flex-1 truncate text-foreground">{entry.action}</Mono>
          <ResultBadge result={entry.result} />
        </li>
      ))}
    </ul>
  );
}

// --- Manage -----------------------------------------------------------------

function ManageTab({
  svc,
  name,
  projectPath,
  canConfigure,
  onConfigure
}: {
  svc: Service;
  name: string;
  projectPath?: string;
  canConfigure: boolean;
  onConfigure: () => void;
}) {
  return (
    <div className="space-y-6">
      <Panel
        title="Configuration"
        actions={
          canConfigure && (
            <Button size="sm" onClick={onConfigure}>
              Configure
            </Button>
          )
        }
      >
        <p className="px-4 py-3 text-13 text-muted-foreground">
          GPUs, start command, health check and resource limits. Name and port are fixed at deploy.
        </p>
      </Panel>

      <Panel title="Secrets">
        <div className="px-4 py-3">
          <SecretsPanel service={svc.name} canWrite={canConfigure} />
        </div>
      </Panel>

      {/* (P40e) A single-service project keeps this page as its home; the
          project-level sections (variables, addresses, delete) are one link away. */}
      {projectPath && (
        <Panel title="Project">
          <p className="px-4 py-3 text-13 text-muted-foreground">
            Variables shared by every service of <Mono>{svc.project ?? svc.name}</Mono>, its
            addresses and the project itself.{" "}
            <Link to={projectPath} className="text-primary hover:underline" data-testid="app-project-link">
              Open the project
            </Link>
          </p>
        </Panel>
      )}

      <Panel title="AI resources">
        <div className="space-y-3 px-4 py-3">
          <GenericBindingsPanel name={name} descriptor={aiSection} />
          <p className="text-13 text-subtle-foreground">
            Serve models from the{" "}
            <Link to="/models" className="text-primary hover:underline">
              Models page
            </Link>
            .
          </p>
        </div>
      </Panel>

      <Panel title="Database resources">
        <div className="space-y-3 px-4 py-3">
          <GenericBindingsPanel name={name} descriptor={dbSection} />
          <p className="text-13 text-subtle-foreground">
            Provision databases from the{" "}
            <Link to="/databases" className="text-primary hover:underline">
              Databases page
            </Link>
            .
          </p>
        </div>
      </Panel>

      <DetailsPanel svc={svc} />
    </div>
  );
}

/** Identity trivia, collapsed. Nothing here is needed to operate the app. */
function DetailsPanel({ svc }: { svc: Service }) {
  const [open, setOpen] = useState(false);
  const source = sourceLine(svc.source);

  return (
    <Panel
      title="Details"
      data-testid="app-details"
      actions={
        <Button size="sm" variant="ghost" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
          {open ? "Hide" : "Show"}
        </Button>
      }
    >
      {open && (
        <div className="divide-y divide-border">
          <DataRow label="ID">
            <Mono>{svc.id}</Mono>
          </DataRow>
          <DataRow label="Kind">{svc.kind}</DataRow>
          <DataRow label="Image">
            <Mono>{svc.image ?? "–"}</Mono>
          </DataRow>
          <DataRow label="Version">
            {svc.build_version != null ? `v${svc.build_version}` : "–"}
            {svc.rollback_available && (
              <span className="ml-2 text-13 text-subtle-foreground">(previous version available)</span>
            )}
          </DataRow>
          <DataRow label="Source">
            <Mono>{source ?? "–"}</Mono>
          </DataRow>
          <DataRow label="Ports">
            <Mono>
              {svc.endpoint
                ? `${svc.endpoint.container_port} → ${svc.endpoint.host_port} (${svc.endpoint.protocol})`
                : "–"}
            </Mono>
          </DataRow>
          <DataRow label="GPUs">
            {svc.gpu_ids.length > 0 ? svc.gpu_ids.join(", ") : `${svc.gpu_count} requested`}
          </DataRow>
          <DataRow label="Restarts">
            {svc.restart_count}
            {svc.restart_policy && (
              <span className="ml-2 text-13 text-subtle-foreground">(policy: {svc.restart_policy})</span>
            )}
          </DataRow>
          <DataRow label="Created">{formatRelative(svc.created_at)}</DataRow>
        </div>
      )}
    </Panel>
  );
}
