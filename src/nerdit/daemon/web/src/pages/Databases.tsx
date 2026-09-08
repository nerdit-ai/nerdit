import { useMemo, useRef, useState, type FormEvent, type RefObject } from "react";
import { Link } from "react-router-dom";
import {
  useAllBindings,
  useCapabilities,
  useCreateDatabase,
  useDatabases,
  useServiceAction
} from "../api/queries";
import {
  Badge,
  Button,
  Confirm,
  CopyField,
  Field,
  Input,
  Menu,
  Mono,
  PageHeader,
  Panel,
  Select,
  TableRow
} from "../components/ui";
import { deriveProjects, projectsBoundTo } from "../lib/projects";
import { serviceStatusLabel, serviceStatusTone } from "../lib/status";
import { toast } from "../state/toastStore";
import type { Database } from "../api/types";

/**
 * Databases: the managed data plane (design guidelines §2/§4), the literal twin
 * of the Models page. One PageHeader with a single primary action, the create
 * form in a Panel built from the shared field primitives, the list as a table,
 * run state from `lib/status`, per-row actions in the `…` menu, and the delete
 * confirm keeping its house copy (it names what is destroyed).
 */

const COLUMNS = ["Name", "Engine", "Status", "Endpoint", ""];

/** The engines `POST /databases` accepts (`core/data/backend.py`). */
/**
 * Fallback engine list for a daemon whose /capabilities omits the databases
 * block (pre-P15). A current daemon supplies both the list and the configured
 * `[databases].default_backend`, and the form defers to it (Codex review,
 * PR #147): hardcoding postgres-first silently overrode an operator's
 * configured default.
 */
const ENGINES_FALLBACK = ["postgres", "redis"];

/**
 * Cross-link chips (D9): the apps bound to this database. Best-effort — an
 * empty list renders nothing, so the fan-out being loading/error stays silent.
 * The `/projects/:name` path is a frozen route (a redirect shim owns the
 * rename); only the word on screen is "app".
 */
function UsedBy({ names }: { names: string[] }) {
  if (names.length === 0) return null;
  return (
    <span className="flex flex-wrap items-center gap-1.5" data-testid="database-used-by">
      <span className="text-12 text-subtle-foreground">Used by</span>
      {names.map((name) => (
        <Link key={name} to={`/projects/${encodeURIComponent(name)}`}>
          <Badge tone="muted" className="font-mono hover:text-foreground">
            {name}
          </Badge>
        </Link>
      ))}
    </span>
  );
}

/**
 * Wire-protocol readiness. "Container up, not yet accepting connections" is a
 * distinct, expected phase, so it reads as in-progress and never as an error:
 * one muted Badge either way, the word carrying the difference.
 */
function ReadyBadge({ ready }: { ready: boolean }) {
  return (
    <Badge tone="muted" data-testid="database-readiness">
      {ready ? "Accepting connections" : "Starting"}
    </Badge>
  );
}

/**
 * Password-free host:port display string. The credential-bearing DSN is only
 * composed at binding-resolution time, never shown here (D7).
 */
function DbEndpoint({ endpoint }: { endpoint: string | null }) {
  if (!endpoint) {
    return (
      <span className="text-13 text-subtle-foreground">
        Endpoint pending: the port is not published yet.
      </span>
    );
  }
  return <CopyField value={endpoint} className="max-w-[18rem]" />;
}

/** Quiet placeholder rows in the table's own shape (guidelines §7). */
function PlaceholderRows({ rows = 2, cells = COLUMNS.length }: { rows?: number; cells?: number }) {
  return (
    <>
      {Array.from({ length: rows }).map((_, row) => (
        <tr key={row} className="border-b border-border">
          {Array.from({ length: cells }).map((__, cell) => (
            <td key={cell} className="px-4 py-4">
              <span className="block h-3 w-full max-w-[8rem] rounded-button bg-surface-hover" />
            </td>
          ))}
        </tr>
      ))}
    </>
  );
}

function CreateDatabasePanel({ inputRef }: { inputRef: RefObject<HTMLInputElement> }) {
  const create = useCreateDatabase();
  const capabilities = useCapabilities();
  // null = untouched: the effective value tracks the daemon's configured
  // default until the user picks, so a late capabilities load never stomps a
  // choice and an untouched submit provisions what the operator configured.
  const [engine, setEngine] = useState<string | null>(null);
  const [name, setName] = useState("");

  const engines = capabilities.data?.databases?.backends?.length
    ? capabilities.data.databases.backends
    : ENGINES_FALLBACK;
  const effectiveEngine = engine ?? capabilities.data?.databases?.default_backend ?? engines[0];

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    create.mutate(
      { backend: effectiveEngine, name: name.trim() || undefined },
      {
        onSuccess: (created) => {
          toast("success", `Creating ${created.name}`);
          setName("");
        }
      }
    );
  }

  return (
    <Panel title="New database" data-testid="create-database-panel">
      <form className="space-y-5 p-4" onSubmit={onSubmit}>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Field label="Engine" htmlFor="db-engine">
            <Select
              id="db-engine"
              value={effectiveEngine}
              onChange={(event) => setEngine(event.target.value)}
            >
              {engines.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Name" htmlFor="db-name" hint="Defaults to the engine's own short name.">
            <Input
              id="db-name"
              ref={inputRef}
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="pg"
            />
          </Field>
        </div>
        <div className="flex items-center justify-between gap-4">
          <p className="text-13 text-muted-foreground">
            Credentials are minted on the daemon and never shown. Apps bind a database with{" "}
            <Mono as="span">[db.*]</Mono> in their nerdit.toml.
          </p>
          <Button type="submit" variant="primary" loading={create.isPending}>
            Create
          </Button>
        </div>
      </form>
    </Panel>
  );
}

export default function Databases() {
  const databases = useDatabases();
  const allBindings = useAllBindings();
  const stop = useServiceAction("stop");
  const restart = useServiceAction("restart");
  const remove = useServiceAction("delete");
  const [pendingDelete, setPendingDelete] = useState<Database | null>(null);
  const [createOpen, setCreateOpen] = useState(false);
  const nameInputRef = useRef<HTMLInputElement>(null);

  // Wrapped so the `?? []` fallback is not a fresh array every render — an
  // unstable reference here defeats the reverse-index memo below.
  const items = useMemo(() => databases.data?.items ?? [], [databases.data]);

  // D9 reverse index: which apps bind each managed database.
  const projects = useMemo(
    () => deriveProjects(allBindings.services, allBindings.apps, [], items),
    [allBindings.services, allBindings.apps, items]
  );

  const empty = !databases.isLoading && !databases.error && items.length === 0;

  function openCreateForm() {
    setCreateOpen(true);
    window.setTimeout(() => nameInputRef.current?.focus(), 0);
  }

  return (
    <div className="mx-auto max-w-content space-y-7">
      <PageHeader
        title="Databases"
        subtitle="Managed Postgres and Redis on this machine, with minted credentials an app reads through its own binding."
        actions={
          <Button variant="primary" data-testid="open-create-database" onClick={openCreateForm}>
            New database
          </Button>
        }
      />

      {createOpen && <CreateDatabasePanel inputRef={nameInputRef} />}

      <Panel title={`Managed databases (${items.length})`}>
        {databases.error && (
          <div className="border-b border-border px-4 py-3 text-14 text-destructive">
            {(databases.error as Error).message}
          </div>
        )}

        {empty ? (
          <div className="flex flex-col items-start gap-2 px-4 py-10" data-testid="databases-empty">
            <p className="text-14 text-foreground">No databases yet.</p>
            <p className="text-14 text-muted-foreground">
              Create one here, or from a shell on this machine:
            </p>
            <Mono className="text-muted-foreground">nerdit db create appdb</Mono>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[46rem] text-14">
              <thead>
                <tr className="border-b border-border">
                  {COLUMNS.map((column, index) => (
                    <th
                      key={column || `actions-${index}`}
                      className="label px-4 py-2 text-left font-medium"
                    >
                      {column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {databases.isLoading && <PlaceholderRows />}
                {items.map((db) => (
                  <TableRow key={db.id}>
                    <td className="max-w-[18rem] px-4 py-3">
                      <Mono className="block truncate text-foreground" title={db.name}>
                        {db.name}
                      </Mono>
                      <span className="mt-1 block">
                        <UsedBy names={projectsBoundTo(projects, "db", db.name)} />
                      </span>
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">{db.backend ?? "–"}</td>
                    <td className="px-4 py-3">
                      <span className="flex flex-wrap items-center gap-2">
                        <Badge tone={serviceStatusTone(db.status)} data-testid="database-status">
                          {serviceStatusLabel(db.status)}
                        </Badge>
                        <ReadyBadge ready={db.db_ready} />
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <DbEndpoint endpoint={db.endpoint} />
                    </td>
                    <td className="px-4 py-3 text-right">
                      <Menu
                        items={[
                          {
                            label: "Restart",
                            disabled: restart.isPending,
                            onSelect: () => restart.mutate(db.name)
                          },
                          {
                            label: "Stop",
                            disabled: stop.isPending || db.status === "stopped",
                            onSelect: () => stop.mutate(db.name)
                          },
                          "separator",
                          {
                            label: "Delete…",
                            destructive: true,
                            disabled: remove.isPending,
                            onSelect: () => setPendingDelete(db)
                          }
                        ]}
                      />
                    </td>
                  </TableRow>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <Confirm
        open={pendingDelete !== null}
        title="Delete database"
        description={
          <>
            Stops{" "}
            <Mono as="span" className="text-foreground">
              {pendingDelete?.name}
            </Mono>{" "}
            and deletes its stored data. This cannot be undone.
          </>
        }
        destructive
        confirmLabel="Delete"
        onCancel={() => setPendingDelete(null)}
        onConfirm={() => {
          if (pendingDelete) {
            // A kind=database row REQUIRES ?purge=data (the daemon 409s
            // db.delete_requires_purge otherwise) — the data dir and the minted
            // password are useless without each other.
            const label = pendingDelete.name;
            remove.mutate(
              { ident: label, purge: "data" },
              { onSuccess: () => toast("success", `Deleted ${label}`) }
            );
          }
          setPendingDelete(null);
        }}
      />
    </div>
  );
}
