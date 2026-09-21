import { useMemo, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { Plus } from "lucide-react";
import { DeployDialog } from "../components/DeployDialog";
import { ServicePublicUrl } from "../components/ServicePublicUrl";
import { TemplateDeployDialog } from "../components/TemplateDeployDialog";
import { Badge, Button, Menu, Mono, Notice, PageHeader, Panel, TableRow } from "../components/ui";
import {
  useAllBindings,
  useCapabilities,
  useDatabases,
  useModels,
  useProjects
} from "../api/queries";
import type { Project as ProjectSummary, Service } from "../api/types";
import { deployPhaseLabel } from "../lib/deployPhase";
import { formatRelative, sourceLine } from "../lib/format";
import { serviceStatusLabel, serviceStatusTone } from "../lib/status";
import {
  deriveProject,
  deriveProjects,
  type Project,
  type ProjectStatus
} from "../lib/projects";

// The apps list — the dashboard's home page (design guidelines §2/§4). A table,
// not a tile grid: four columns an operator scans down (name, run state,
// address, last deploy) instead of nine facts per card repeated sideways.
//
// Two rules shape what is NOT here. Run state comes from `lib/status` alone, so
// this page cannot invent a vocabulary the app page disagrees with; and the
// per-app resource pills moved to the app page, where a resource has room to
// say why it is waiting. Nothing is lost: `deriveProject`'s aggregate sentence
// still reaches the operator as the status tooltip, and an app held up by a
// binding says so in a muted line under its name.

type DeployMode = "upload" | "git";

/** Four columns; the header row is the one place `.label` uppercase is used. */
export function HeadCell({ children, className }: { children: ReactNode; className?: string }) {
  return <th className={`label px-4 py-2 text-left ${className ?? ""}`}>{children}</th>;
}

/**
 * One row of the four-column table. (P40e) The facts are passed apart because a
 * multi-service project has no single service behind it: the badge is its
 * worst service, the address its home service, the deploy its latest one. A
 * single-service project (and the pre-P40 daemon) passes the same row thrice.
 * Shared with the project page's services table.
 */
export function AppRow({
  name,
  to,
  sub,
  badge,
  badgeTitle,
  address,
  deploy
}: {
  name: string;
  to: string;
  sub?: string | null;
  badge: Service | null;
  badgeTitle?: string;
  address: Service | null;
  deploy: Service | null;
}) {
  const navigate = useNavigate();
  // One line, no chip and no dot: the run-state Badge two cells left already
  // carries the tone, and the deploy generation is a fact stated once (§6).
  const phase = deploy?.last_deploy?.phase;
  const phaseLabel = deployPhaseLabel(phase);
  const when = deploy ? formatRelative(deploy.started_at ?? deploy.created_at) : "–";

  return (
    <TableRow onClick={() => navigate(to)}>
      {/* `TableRow` takes no arbitrary props, so the row's test anchor lives on
          its first cell: the one a selector would read anyway. */}
      <td className="px-4 py-3 align-middle" data-testid={`app-row-${name}`}>
        <span className="block truncate text-14 font-medium text-foreground">{name}</span>
        {sub && <span className="mt-0.5 block truncate text-12 text-muted-foreground">{sub}</span>}
      </td>
      <td className="px-4 py-3 align-middle">
        {badge ? (
          <Badge title={badgeTitle} tone={serviceStatusTone(badge.status)}>
            {serviceStatusLabel(badge.status)}
          </Badge>
        ) : (
          <Badge tone="muted">No services</Badge>
        )}
      </td>
      <td className="max-w-[22rem] px-4 py-3 align-middle">
        <ServicePublicUrl endpoint={address?.endpoint ?? null} />
      </td>
      <td className="whitespace-nowrap px-4 py-3 align-middle">
        <span
          data-testid={`app-last-deploy-${name}`}
          className={`text-13 ${phase === "failed" ? "text-destructive" : "text-muted-foreground"}`}
        >
          {phaseLabel ? `${phaseLabel} · ${when}` : when}
        </span>
      </td>
    </TableRow>
  );
}

/** The pre-P40 row: one derived project = one service, bindings and all. */
function derivedRow(project: Project) {
  const { service } = project;
  return (
    <AppRow
      key={project.name}
      name={project.name}
      to={`/projects/${encodeURIComponent(project.name)}`}
      sub={project.status === "attention" ? project.statusDetail : sourceLine(service.source)}
      badge={service}
      badgeTitle={project.statusDetail}
      address={service}
      deploy={service}
    />
  );
}

/** Worst first — the badge a multi-service project shows is its worst service's. */
const STATUS_RANK: ProjectStatus[] = [
  "failed",
  "degraded",
  "attention",
  "deploying",
  "stopped",
  "running"
];

/**
 * (P40e) A `list_projects` row. Status is judged on the service rows alone —
 * the per-app config fan-out is gone from the grid, so a waiting binding no
 * longer reads as "attention" here (the app page still says why).
 */
function summaryRow(summary: ProjectSummary) {
  const derived = summary.services
    .map((service) => deriveProject(service, undefined, [], []))
    .sort((a, b) => STATUS_RANK.indexOf(a.status) - STATUS_RANK.indexOf(b.status));
  const worst = derived[0] ?? null;
  const single = summary.services.length === 1;
  // The home service is the one whose label IS the project name (field compare).
  const home = summary.services.find((svc) => svc.name === summary.name) ?? summary.services[0];
  const latest = [...summary.services].sort((a, b) =>
    (b.started_at ?? b.created_at ?? "").localeCompare(a.started_at ?? a.created_at ?? "")
  )[0];
  const sub = !single
    ? summary.services.length > 1
      ? `${summary.services.length} services`
      : null
    : worst?.status === "attention"
      ? worst.statusDetail
      : sourceLine(worst?.service.source ?? null);
  const detail = worst
    ? single
      ? worst.statusDetail
      : `${worst.service.service ?? worst.service.name}: ${worst.statusDetail}`
    : undefined;
  return (
    <AppRow
      key={summary.name}
      name={summary.name}
      to={`/projects/${encodeURIComponent(summary.name)}`}
      sub={sub}
      badge={worst?.service ?? null}
      badgeTitle={detail}
      address={home ?? null}
      deploy={latest ?? null}
    />
  );
}

/** Quiet placeholder rows in the table's own shape (guidelines §7). */
function PlaceholderRows() {
  return (
    <>
      {[0, 1, 2].map((row) => (
        <tr key={row} className="border-b border-border" data-testid="app-row-placeholder">
          {[0, 1, 2, 3].map((cell) => (
            <td key={cell} className="px-4 py-4">
              <span className="block h-3 w-24 rounded-full bg-surface-hover" />
            </td>
          ))}
        </tr>
      ))}
    </>
  );
}

/** (P40e) The grid off `list_projects`: one bounded read, no per-app fan-out. */
function ProjectsGrid({ onNew }: { onNew: () => void }) {
  const projects = useProjects();
  return (
    <AppsTable
      rows={(projects.data?.items ?? []).map(summaryRow)}
      isLoading={projects.isLoading}
      error={(projects.error as Error | null) ?? null}
      onNew={onNew}
    />
  );
}

/** A daemon without `features.projects`: today's derive-from-services grid. */
function LegacyGrid({ onNew }: { onNew: () => void }) {
  const { services, apps, isLoading, error } = useAllBindings();
  const models = useModels();
  const databases = useDatabases();
  const projects = useMemo(
    () => deriveProjects(services, apps, models.data?.items ?? [], databases.data?.items ?? []),
    [services, apps, models.data, databases.data]
  );
  return (
    <AppsTable rows={projects.map(derivedRow)} isLoading={isLoading} error={error} onNew={onNew} />
  );
}

function AppsTable({
  rows,
  isLoading,
  error,
  onNew
}: {
  rows: ReactNode[];
  isLoading: boolean;
  error: Error | null;
  onNew: () => void;
}) {
  const empty = !isLoading && !error && rows.length === 0;
  return (
    <>
      {error && <Notice tone="destructive">{error.message}</Notice>}

      <Panel>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[40rem] border-collapse" data-testid="apps-table">
            <thead>
              <tr className="border-b border-border">
                <HeadCell>Name</HeadCell>
                <HeadCell>Status</HeadCell>
                <HeadCell>Address</HeadCell>
                <HeadCell>Last deploy</HeadCell>
              </tr>
            </thead>
            <tbody>
              {isLoading && rows.length === 0 && <PlaceholderRows />}
              {rows}
            </tbody>
          </table>
        </div>

        {empty && (
          <div
            className="flex flex-wrap items-center justify-between gap-4 px-4 py-6"
            data-testid="apps-empty"
          >
            <div className="min-w-0">
              <p className="text-14 text-foreground">No apps yet.</p>
              <p className="mt-1 text-13 text-muted-foreground">
                Point Nerdit at a folder and it builds, runs, and gets a URL. From a shell:{" "}
                <Mono>nerdit deploy .</Mono>
              </p>
            </div>
            <Button variant="primary" data-testid="apps-empty-new" onClick={onNew}>
              New app
            </Button>
          </div>
        )}
      </Panel>
    </>
  );
}

export default function Projects() {
  // (P40e) Gate, never sniff: an older daemon has no `/projects`, so it keeps
  // the services-derived grid. While capabilities resolve, the table shows its
  // own placeholder rows instead of guessing a source.
  const capabilities = useCapabilities();
  const [deployOpen, setDeployOpen] = useState(false);
  const [deployMode, setDeployMode] = useState<DeployMode>("upload");
  const [templateOpen, setTemplateOpen] = useState(false);

  function openDeploy(mode: DeployMode) {
    setDeployMode(mode);
    setDeployOpen(true);
  }
  const onNew = () => openDeploy("upload");

  return (
    <div className="mx-auto max-w-content space-y-6">
      <PageHeaderRow onDeploy={openDeploy} onTemplate={() => setTemplateOpen(true)} />

      {capabilities.isPending ? (
        <AppsTable rows={[]} isLoading error={null} onNew={onNew} />
      ) : capabilities.data?.features?.projects ? (
        <ProjectsGrid onNew={onNew} />
      ) : (
        <LegacyGrid onNew={onNew} />
      )}

      <DeployDialog open={deployOpen} onClose={() => setDeployOpen(false)} initialMode={deployMode} />
      <TemplateDeployDialog open={templateOpen} onClose={() => setTemplateOpen(false)} />
    </div>
  );
}

/**
 * The page header and its single action. The three sources are one menu, not
 * three buttons: they all answer "where does the code come from", which is a
 * property of the deploy, not three separate things to do.
 */
function PageHeaderRow({
  onDeploy,
  onTemplate
}: {
  onDeploy: (mode: DeployMode) => void;
  onTemplate: () => void;
}) {
  return (
    <PageHeader
      title="Apps"
      subtitle="Everything deployed on this machine."
      actions={
        <Menu
          triggerVariant="primary"
          triggerSize="md"
          label={
            <>
              <Plus aria-hidden="true" className="h-4 w-4" />
              New app
            </>
          }
          items={[
            { label: "Deploy a folder", onSelect: () => onDeploy("upload") },
            { label: "From a Git repository", onSelect: () => onDeploy("git") },
            { label: "From a template", onSelect: onTemplate }
          ]}
        />
      }
    />
  );
}
