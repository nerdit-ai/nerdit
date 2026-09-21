import { useEffect, useId, useRef, useState } from "react";
import { Link, Navigate, useLocation, useNavigate, useParams } from "react-router-dom";
import { KeyRound } from "lucide-react";
import ProjectDetail from "./ProjectDetail";
import { AppRow, HeadCell } from "./Projects";
import {
  Badge,
  Button,
  Checkbox,
  Confirm,
  DataRow,
  Field,
  Input,
  Menu,
  Mono,
  PageHeader,
  Panel,
  Select
} from "../components/ui";
import {
  useAuthRole,
  useCapabilities,
  useDeleteProject,
  useDeleteVariable,
  useProject,
  useService,
  useSetVariables,
  useVariables
} from "../api/queries";
import { deriveProject, kindHomePath } from "../lib/projects";
import type { ProjectDetail as ProjectView, ProjectVariableName } from "../api/types";
import { toast } from "../state/toastStore";
import { ApiError } from "../api/client";

/**
 * The project routes (P40e).
 *
 * `/projects/:name/:tab?` is three things, decided from the API and never by
 * parsing a name: a multi-service (or empty) project renders `ProjectPage`; a
 * project whose only service carries the project's own name renders the
 * service detail exactly as before, with the project page one link away at
 * `/projects/:name/project`; and a name that is no project — a legacy deep
 * link to a service LABEL such as `/projects/api--asso`, or any name on a
 * daemon without `features.projects` — resolves through the service row.
 */

const projectPath = (name: string) => `/projects/${encodeURIComponent(name)}`;

const Loading = () => <p className="text-14 text-muted-foreground">Loading app…</p>;

export default function ProjectRoute() {
  const { name = "", tab } = useParams();
  const location = useLocation();
  const capabilities = useCapabilities();
  const features = capabilities.data?.features;
  const gated = Boolean(features?.projects);
  const project = useProject(name, gated);
  const lastError = useRef<{ name: string; error: Error } | null>(null);
  if (project.data) lastError.current = null;
  else if (project.error) lastError.current = { name, error: project.error };
  // Refetch clears a dataless query's error; retain its last answer until it settles.
  const error = project.error ?? (
    project.isFetching && lastError.current?.name === name ? lastError.current.error : null
  );

  // `isFetched`, not `isPending`: a dataless errored query (the 404 of a label
  // URL) goes back to `pending` on every refetch, and a Loading swap there would
  // unmount the service page below and drop whatever the user was typing.
  if (capabilities.isPending || (gated && !project.isFetched)) return <Loading />;
  if (!gated) return <LabelRoute name={name} tab={tab} />;
  if (!project.data) {
    // Only a definitive compatibility answer can reinterpret the URL as a label.
    if (error instanceof ApiError && (error.status === 403 || error.status === 404)) {
      return <LabelRoute name={name} tab={tab} />;
    }
    return (
      <div className="space-y-3">
        <p role="alert" className="text-14 text-destructive">{error?.message}</p>
        <Button onClick={() => void project.refetch()} loading={project.isFetching}>Retry</Button>
      </div>
    );
  }

  const services = project.data.services;
  const only = services.length === 1 && services[0].name === name ? services[0] : null;
  if (only && tab !== "project") {
    return (
      <ProjectDetail
        key={only.name}
        label={only.name}
        basePath={projectPath(name)}
        projectPath={`${projectPath(name)}/project`}
      />
    );
  }
  // In a multi-service project the label that IS the project name still names a
  // service to a label-keyed caller (the ⌘K palette's logs / redeploy actions):
  // a service tab or the redeploy hand-off goes to that service, never dropped
  // on the project page.
  const home = services.find((svc) => svc.name === name);
  const redeploy = (location.state as { redeploy?: boolean } | null)?.redeploy;
  if (home?.service && ((tab && tab !== "project") || redeploy)) {
    return (
      <Navigate
        to={`${serviceTo(name, home.service)}${tab ? `/${tab}` : ""}`}
        state={location.state}
        replace
      />
    );
  }
  // Keyed: a cached project swaps in without a remount, and a typed secret (or
  // a pending delete confirmation) must never follow the user to another project.
  return (
    <ProjectPage
      key={name}
      detail={project.data}
      variablesEnabled={Boolean(features?.variables)}
    />
  );
}

/** `/projects/<label>`: redirect by the row's project/service FIELDS, else today's page. */
function LabelRoute({ name, tab }: { name: string; tab?: string }) {
  const location = useLocation();
  const service = useService(name);
  const svc = service.data;
  const nested = svc?.kind === "service" && svc.project && svc.service && svc.project !== name;
  // The project-shaped URL renders from `GET /projects/<project>`, which a token
  // scoped to this LABEL alone may not read: redirect only once it answered.
  const home = useProject(svc?.project ?? "", Boolean(nested));
  if (nested && !home.isFetched) return <Loading />; // first answer only, as above
  if (nested && home.data) {
    // `state` rides along: the palette's redeploy hand-off lands on a label URL.
    return (
      <Navigate to={`${kindHomePath(svc)}${tab ? `/${tab}` : ""}`} state={location.state} replace />
    );
  }
  return <ProjectDetail key={name} label={name} basePath={projectPath(name)} />;
}

/** `/projects/:name/services/:service/:tab?` — the label comes from the project's rows. */
export function ServiceRoute() {
  const { name = "", service = "" } = useParams();
  const project = useProject(name);

  // Data first, like ProjectRoute: a failed background poll keeps `data` AND
  // sets `error`, and must not unmount a live service page (typed secret, logs).
  if (!project.data) {
    if (!project.error) return <Loading />;
    return <p className="text-14 text-destructive">{(project.error as Error).message}</p>;
  }
  const row = project.data.services.find((svc) => svc.service === service);
  if (!row) return <Navigate to={projectPath(name)} replace />;
  return (
    <ProjectDetail
      key={row.name}
      label={row.name}
      basePath={`${projectPath(name)}/services/${encodeURIComponent(service)}`}
      projectPath={projectPath(name)}
      nested
    />
  );
}

// --- The project page -------------------------------------------------------

function ProjectPage({
  detail,
  variablesEnabled
}: {
  detail: ProjectView;
  variablesEnabled: boolean;
}) {
  const navigate = useNavigate();
  const role = useAuthRole().data?.role;
  const remove = useDeleteProject();
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [purgeData, setPurgeData] = useState(false);
  const [purgeImages, setPurgeImages] = useState(false);

  const { name, services } = detail;
  // D-P40-15: the daemon OMITS `variables` for a non-owner. That absence is the
  // owner test for everything owner-only here — no variables UI, no delete.
  const owner = detail.variables != null;
  const canWrite = owner && role !== "readonly";
  // The single-service layout links here as `/project`; its way back is the app.
  const home = services.length === 1 && services[0].name === name;
  const serviceName = (label: string) =>
    services.find((svc) => svc.name === label)?.service ?? label;

  const purge = ["secrets"];
  if (purgeData) purge.push("data");
  if (purgeImages) purge.push("images");

  function onConfirmDelete() {
    remove.mutate(
      { name, purge: purge.join(",") },
      {
        onSuccess: () => {
          toast("success", `Project ${name} deleted`);
          navigate("/");
        },
        // A 409 `project.delete_incomplete` is toasted by the hook; the page
        // stays and shows what is left.
        onSettled: () => setConfirmDelete(false)
      }
    );
  }

  return (
    <div className="mx-auto max-w-content space-y-6" data-testid="project-page">
      {home && (
        <Link to={projectPath(name)} className="text-13 text-muted-foreground hover:text-foreground">
          ← {name}
        </Link>
      )}
      <PageHeader
        title={name}
        subtitle={
          <>
            Project · {services.length} service{services.length === 1 ? "" : "s"}
            {detail.home.hostname ? ` · ${detail.home.hostname}` : ""}
          </>
        }
        actions={
          canWrite && (
            <Menu
              align="end"
              items={[
                { label: "Delete project…", destructive: true, onSelect: () => setConfirmDelete(true) }
              ]}
            />
          )
        }
      />

      <Panel title="Services">
        {services.length === 0 ? (
          <p className="px-4 py-3 text-13 text-subtle-foreground">
            No services yet. From the project folder: <Mono>nerdit apply</Mono>
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[40rem] border-collapse" data-testid="project-services">
              <thead>
                <tr className="border-b border-border">
                  <HeadCell>Service</HeadCell>
                  <HeadCell>Status</HeadCell>
                  <HeadCell>Address</HeadCell>
                  <HeadCell>Last deploy</HeadCell>
                </tr>
              </thead>
              <tbody>
                {services.map((svc) => (
                  <AppRow
                    key={svc.name}
                    name={svc.service ?? svc.name}
                    // The single-service home IS `/projects/:name`; every other
                    // row routes by its fields (`kindHomePath`).
                    to={home ? projectPath(name) : serviceTo(name, svc.service) ?? kindHomePath(svc)}
                    badge={svc}
                    badgeTitle={deriveProject(svc, undefined, [], []).statusDetail}
                    address={svc}
                    deploy={svc}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <Panel title="Resources">
        {detail.resources.length === 0 ? (
          <p className="px-4 py-3 text-13 text-subtle-foreground">No models or databases referenced.</p>
        ) : (
          <div className="divide-y divide-border">
            {detail.resources.map((res) => (
              <DataRow
                key={`${res.service}:${res.type}:${res.binding}`}
                label={`${res.type === "ai" ? "Model" : "Database"} · ${res.binding} · ${serviceName(res.service)}`}
              >
                <span className="flex flex-wrap items-baseline justify-end gap-x-3 gap-y-1">
                  <Mono>{res.target ?? `${res.binding} (unset)`}</Mono>
                  <ReadyBadge ready={res.ready} />
                </span>
              </DataRow>
            ))}
          </div>
        )}
      </Panel>

      <Panel title="Addresses">
        {detail.addresses.length === 0 ? (
          <p className="px-4 py-3 text-13 text-subtle-foreground">No address advertised yet.</p>
        ) : (
          <ul className="divide-y divide-border" data-testid="project-addresses">
            {detail.addresses.map((entry, index) => (
              <li key={`${entry.kind}:${entry.url ?? index}`} className="flex items-center gap-3 px-4 py-3">
                {entry.url ? (
                  <a
                    href={entry.url}
                    target="_blank"
                    rel="noreferrer"
                    className="min-w-0 truncate font-mono text-13 text-primary hover:underline"
                  >
                    {entry.url}
                  </a>
                ) : (
                  <span className="text-muted-foreground">–</span>
                )}
                <Badge tone="muted">{entry.kind}</Badge>
                {entry.state !== "ready" && <Badge tone="warning">{entry.state.replace("_", " ")}</Badge>}
                {entry.access === "public" && <Badge tone="primary">public</Badge>}
              </li>
            ))}
          </ul>
        )}
      </Panel>

      {variablesEnabled && owner && (
        <VariablesPanel detail={detail} names={detail.variables ?? []} canWrite={canWrite} />
      )}

      <Confirm
        open={confirmDelete}
        title="Delete project?"
        confirmLabel="Delete"
        destructive
        busy={remove.isPending}
        confirmPhrase={purgeData || services.length > 1 ? name : undefined}
        description={
          <div data-testid="project-delete-confirm" className="space-y-3">
            <p>
              Deletes <Mono>{name}</Mono>
              {services.length > 0 && (
                <>
                  {" "}
                  and its {services.length} service{services.length === 1 ? "" : "s"}
                </>
              )}
              , stops their addresses answering, and frees the name. Project variables and each
              service's secrets are deleted with it.
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
            </div>
            {purgeData && (
              <p className="text-destructive">Deleting the data volumes cannot be undone.</p>
            )}
          </div>
        }
        onConfirm={onConfirmDelete}
        onCancel={() => setConfirmDelete(false)}
      />
    </div>
  );
}

function serviceTo(project: string, service: string | null | undefined): string | null {
  return service ? `${projectPath(project)}/services/${encodeURIComponent(service)}` : null;
}

/** `ready: null` is its own state: the daemon cannot judge it without the caller's secrets. */
function ReadyBadge({ ready }: { ready: boolean | null }) {
  if (ready === null) {
    return (
      <Badge tone="muted" title="Readiness of an external resource cannot be judged from here.">
        Unknown
      </Badge>
    );
  }
  return <Badge tone={ready ? "success" : "warning"}>{ready ? "Ready" : "Not ready"}</Badge>;
}

// --- Variables --------------------------------------------------------------

/** Mirrors the daemon's `_scope_name`: composed for comparison, never parsed back. */
const scopeOf = (service?: string) => (service ? `production/${service}` : "project");

/**
 * Variable names by scope, a plain editor and the secret intake. Rendered for
 * an owner view only. Plain and secret writes are two forms and two PUTs
 * (`secret: false` / `secret: true`) so each audit row carries its flag.
 */
function VariablesPanel({
  detail,
  names,
  canWrite
}: {
  detail: ProjectView;
  names: ProjectVariableName[];
  canWrite: boolean;
}) {
  const fieldId = useId();
  // Only a service that has a row is addressable (the daemon's own rule).
  // ponytail: a name in a scope outside this list would not render; the daemon
  // omits rowless scopes from `get_project.variables`, so there is none today.
  const scopes: (string | undefined)[] = [
    undefined,
    ...detail.services.flatMap((svc) => (svc.service ? [svc.service] : []))
  ];
  const [target, setTarget] = useState("");
  const service = target || undefined;

  return (
    <Panel title="Variables" data-testid="project-variables">
      <div className="space-y-4 px-4 py-3">
        <p className="text-13 text-muted-foreground">
          Injected as env at launch; a service value wins over a project one. A secret is
          write-only: its value is never shown again.
        </p>

        {names.length === 0 && <p className="text-13 text-muted-foreground">No variables set.</p>}
        {scopes.map((scope) => {
          const group = names.filter((entry) => entry.scope === scopeOf(scope));
          if (group.length === 0) return null;
          return (
            <ScopeGroup
              key={scope ?? ""}
              project={detail.name}
              service={scope}
              names={group}
              canWrite={canWrite}
            />
          );
        })}

        {canWrite && (
          <>
            <Field label="Scope" htmlFor={`${fieldId}-scope`} className="max-w-xs">
              <Select
                id={`${fieldId}-scope`}
                value={target}
                onChange={(event) => setTarget(event.target.value)}
              >
                <option value="">Project (every service)</option>
                {scopes.flatMap((scope) =>
                  scope ? (
                    <option key={scope} value={scope}>
                      Service {scope}
                    </option>
                  ) : (
                    []
                  )
                )}
              </Select>
            </Field>
            {/* Keyed by scope: switching it remounts both forms, dropping any typed value. */}
            <PlainForm key={`plain:${target}`} project={detail.name} service={service} />
            <SecretForm key={`secret:${target}`} project={detail.name} service={service} />
          </>
        )}
      </div>
    </Panel>
  );
}

/** One scope's names. Values are fetched only when the scope has a plain key to show. */
function ScopeGroup({
  project,
  service,
  names,
  canWrite
}: {
  project: string;
  service?: string;
  names: ProjectVariableName[];
  canWrite: boolean;
}) {
  // A 403 here is quiet: the names still render, the values simply do not.
  const values = useVariables(project, service, names.some((entry) => entry.plain));
  const remove = useDeleteVariable(project, service);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);

  return (
    <div>
      <h3 className="label mb-1">{service ? `Service ${service}` : "Project"}</h3>
      <ul className="divide-y divide-border border-y border-border">
        {names.map((entry) => {
          const value = entry.plain
            ? values.data?.variables.find((row) => row.key === entry.key)?.value
            : null;
          return (
            <li
              key={entry.key}
              data-testid={`variable-row-${entry.key}`}
              className="flex items-center justify-between gap-3 py-2"
            >
              <Mono className="flex min-w-0 items-center gap-2 text-foreground">
                <KeyRound size={16} className="shrink-0 text-muted-foreground" aria-hidden="true" />
                <span className="truncate">{entry.key}</span>
                {entry.plain ? (
                  value != null && <span className="truncate text-subtle-foreground">= {value}</span>
                ) : (
                  <span className="text-subtle-foreground">= ••••••</span>
                )}
              </Mono>
              <span className="flex shrink-0 items-center gap-2">
                <Badge tone={entry.plain ? "muted" : "primary"}>{entry.plain ? "plain" : "secret"}</Badge>
                {canWrite && (
                  <Button
                    variant="ghost"
                    size="sm"
                    aria-label={`Delete ${entry.key}`}
                    onClick={() => setPendingDelete(entry.key)}
                    disabled={remove.isPending}
                    className="text-destructive"
                  >
                    Delete
                  </Button>
                )}
              </span>
            </li>
          );
        })}
      </ul>

      <Confirm
        open={pendingDelete !== null}
        title="Delete variable?"
        description={
          <>
            Delete <Mono className="text-foreground">{pendingDelete}</Mono> from{" "}
            {service ? `service ${service}` : "the project"}? A running container keeps its env
            until the next restart.
          </>
        }
        destructive
        confirmLabel="Delete"
        busy={remove.isPending}
        onCancel={() => setPendingDelete(null)}
        onConfirm={() => {
          const key = pendingDelete;
          if (key) remove.mutate(key, { onSuccess: () => toast("success", `Variable ${key} deleted`) });
          setPendingDelete(null);
        }}
      />
    </div>
  );
}

/** The plain editor: `secret: false`, the value readable by the owner afterwards. */
function PlainForm({ project, service }: { project: string; service?: string }) {
  const fieldId = useId();
  const setVariables = useSetVariables(project, service);
  const [key, setKey] = useState("");
  const [value, setValue] = useState("");
  const name = key.trim();

  function onSave() {
    setVariables.mutate(() => ({ values: { [name]: value }, secret: false }), {
      onSuccess: () => {
        toast("success", `Variable ${name} saved`);
        setKey("");
        setValue("");
      }
    });
  }

  return (
    <form
      className="grid grid-cols-[1fr_1fr_auto] items-end gap-2"
      data-testid="variable-plain-form"
      onSubmit={(event) => {
        event.preventDefault();
        if (name && !setVariables.isPending) onSave();
      }}
    >
      <Field label="Plain key" htmlFor={`${fieldId}-key`}>
        <Input
          id={`${fieldId}-key`}
          mono
          value={key}
          onChange={(event) => setKey(event.target.value)}
          placeholder="KEY"
          autoComplete="off"
          spellCheck={false}
        />
      </Field>
      <Field label="Plain value" htmlFor={`${fieldId}-value`}>
        <Input
          id={`${fieldId}-value`}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder="value"
          autoComplete="off"
          spellCheck={false}
        />
      </Field>
      <Button type="submit" disabled={!name} loading={setVariables.isPending}>
        Save plain
      </Button>
    </form>
  );
}

/**
 * The secret intake: `secret: true`. The value lives ONLY in the password
 * input's DOM node — never in React state, a query key, storage or a log. The
 * mutation reads it through a thunk at request time, and the node is cleared
 * on success and on unmount.
 */
function SecretForm({ project, service }: { project: string; service?: string }) {
  const fieldId = useId();
  const setVariables = useSetVariables(project, service);
  const input = useRef<HTMLInputElement>(null);
  const [key, setKey] = useState("");
  // Whether the input is non-empty — a boolean, not the value.
  const [filled, setFilled] = useState(false);
  const name = key.trim();

  useEffect(() => {
    const node = input.current;
    return () => {
      if (node) node.value = "";
    };
  }, []);

  function onSave() {
    setVariables.mutate(() => ({ values: { [name]: input.current?.value ?? "" }, secret: true }), {
      onSuccess: () => {
        toast("success", `Secret ${name} saved`);
        if (input.current) input.current.value = "";
        setFilled(false);
        setKey("");
      }
    });
  }

  return (
    <form
      className="grid grid-cols-[1fr_1fr_auto] items-end gap-2"
      data-testid="variable-secret-form"
      onSubmit={(event) => {
        event.preventDefault();
        if (name && filled && !setVariables.isPending) onSave();
      }}
    >
      <Field label="Secret key" htmlFor={`${fieldId}-key`}>
        <Input
          id={`${fieldId}-key`}
          mono
          value={key}
          onChange={(event) => setKey(event.target.value)}
          placeholder="KEY"
          autoComplete="off"
          spellCheck={false}
        />
      </Field>
      <Field label="Secret value" htmlFor={`${fieldId}-value`}>
        <Input
          id={`${fieldId}-value`}
          ref={input}
          type="password"
          onChange={(event) => setFilled(event.target.value !== "")}
          placeholder="value"
          autoComplete="new-password"
        />
      </Field>
      <Button type="submit" variant="primary" disabled={!name || !filled} loading={setVariables.isPending}>
        Save secret
      </Button>
    </form>
  );
}
