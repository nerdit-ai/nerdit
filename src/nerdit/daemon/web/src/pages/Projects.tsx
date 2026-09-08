import { useMemo, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { Plus } from "lucide-react";
import { DeployDialog } from "../components/DeployDialog";
import { ServicePublicUrl } from "../components/ServicePublicUrl";
import { TemplateDeployDialog } from "../components/TemplateDeployDialog";
import { Badge, Button, Menu, Mono, Notice, PageHeader, Panel, TableRow } from "../components/ui";
import { useAllBindings, useDatabases, useModels } from "../api/queries";
import { deployPhaseLabel } from "../lib/deployPhase";
import { formatRelative, sourceLine } from "../lib/format";
import { serviceStatusLabel, serviceStatusTone } from "../lib/status";
import { deriveProjects, type Project } from "../lib/projects";

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
function HeadCell({ children, className }: { children: ReactNode; className?: string }) {
  return <th className={`label px-4 py-2 text-left ${className ?? ""}`}>{children}</th>;
}

function AppRow({ project }: { project: Project }) {
  const navigate = useNavigate();
  const { service } = project;
  const source = sourceLine(service.source);
  const timestamp = service.started_at ?? service.created_at;
  // One line, no chip and no dot: the run-state Badge two cells left already
  // carries the tone, and the deploy generation is a fact stated once (§6).
  const phase = service.last_deploy?.phase;
  const phaseLabel = deployPhaseLabel(phase);
  const when = formatRelative(timestamp);

  return (
    <TableRow onClick={() => navigate(`/projects/${encodeURIComponent(project.name)}`)}>
      {/* `TableRow` takes no arbitrary props, so the row's test anchor lives on
          its first cell: the one a selector would read anyway. */}
      <td className="px-4 py-3 align-middle" data-testid={`app-row-${project.name}`}>
        <span className="block truncate text-14 font-medium text-foreground">{project.name}</span>
        {project.status === "attention" ? (
          <span className="mt-0.5 block truncate text-12 text-muted-foreground">
            {project.statusDetail}
          </span>
        ) : (
          source && (
            <span className="mt-0.5 block truncate text-12 text-muted-foreground">{source}</span>
          )
        )}
      </td>
      <td className="px-4 py-3 align-middle">
        <Badge title={project.statusDetail} tone={serviceStatusTone(service.status)}>
          {serviceStatusLabel(service.status)}
        </Badge>
      </td>
      <td className="max-w-[22rem] px-4 py-3 align-middle">
        <ServicePublicUrl endpoint={service.endpoint} />
      </td>
      <td className="whitespace-nowrap px-4 py-3 align-middle">
        <span
          data-testid={`app-last-deploy-${project.name}`}
          className={`text-13 ${phase === "failed" ? "text-destructive" : "text-muted-foreground"}`}
        >
          {phaseLabel ? `${phaseLabel} · ${when}` : when}
        </span>
      </td>
    </TableRow>
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

export default function Projects() {
  const { services, apps, isLoading, error } = useAllBindings();
  const models = useModels();
  const databases = useDatabases();
  const [deployOpen, setDeployOpen] = useState(false);
  const [deployMode, setDeployMode] = useState<DeployMode>("upload");
  const [templateOpen, setTemplateOpen] = useState(false);

  const projects = useMemo(
    () => deriveProjects(services, apps, models.data?.items ?? [], databases.data?.items ?? []),
    [services, apps, models.data, databases.data]
  );

  function openDeploy(mode: DeployMode) {
    setDeployMode(mode);
    setDeployOpen(true);
  }

  const empty = !isLoading && !error && projects.length === 0;

  return (
    <div className="mx-auto max-w-content space-y-6">
      <PageHeaderRow onDeploy={openDeploy} onTemplate={() => setTemplateOpen(true)} />

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
              {isLoading && projects.length === 0 && <PlaceholderRows />}
              {projects.map((project) => (
                <AppRow key={project.name} project={project} />
              ))}
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
            <Button variant="primary" data-testid="apps-empty-new" onClick={() => openDeploy("upload")}>
              New app
            </Button>
          </div>
        )}
      </Panel>

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
