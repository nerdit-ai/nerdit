import { useMemo, useRef, useState, type FormEvent, type RefObject } from "react";
import { Link } from "react-router-dom";
import {
  useAllBindings,
  useCapabilities,
  useDatabases,
  useModels,
  useServeModel,
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
  Panel,
  PageHeader,
  Select,
  TableRow
} from "../components/ui";
import { deriveProjects, projectsBoundTo } from "../lib/projects";
import { serviceStatusLabel, serviceStatusTone } from "../lib/status";
import { toast } from "../state/toastStore";
import type { Model } from "../api/types";

/**
 * Models: the served, loopback-only inference plane (design guidelines §2/§4).
 *
 * Anatomy: one PageHeader with a single primary action ("Serve a model", which
 * reveals the form), the form in a Panel built from the shared field
 * primitives, and the served list as a table. Run state comes from
 * `lib/status` — never a page-local pill map — and each row's readiness has
 * exactly ONE rendering (a muted Badge), not a dot *and* a pill.
 */

const COLUMNS = ["Name", "Model", "Backend", "Status", "GPUs", "Endpoint", ""];

/**
 * Cross-link chips (D9): the apps bound to this model. Best-effort — an empty
 * list renders nothing, so the fan-out being loading/error stays silent. The
 * route keeps its frozen `/projects/:name` path (a redirect shim owns the
 * rename); only the word on screen is "app".
 */
function UsedBy({ names }: { names: string[] }) {
  if (names.length === 0) return null;
  return (
    <span className="flex flex-wrap items-center gap-1.5" data-testid="model-used-by">
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
 * Weights-pull readiness. "Container up, weights absent" is a distinct,
 * expected phase, so it reads as in-progress and never as an error: one muted
 * Badge either way, the word carrying the difference.
 */
function PullBadge({ pulled }: { pulled: boolean }) {
  return (
    <Badge tone="muted" data-testid="model-readiness">
      {pulled ? "Pulled" : "Pulling weights"}
    </Badge>
  );
}

/** Allocated GPU ids with live utilization; CPU inference when none. */
function ModelGpus({ model }: { model: Model }) {
  if (model.gpu_ids.length === 0) {
    return <span className="text-muted-foreground">CPU</span>;
  }
  return (
    <span className="flex flex-wrap gap-x-3 gap-y-1">
      {model.gpu_ids.map((id) => {
        const util = model.gpu_utilization[id];
        return (
          <Mono key={id} className="text-muted-foreground">
            {id}
            <span className="ml-1 text-subtle-foreground">{util != null ? `${util}%` : "–"}</span>
          </Mono>
        );
      })}
    </span>
  );
}

/**
 * Loopback OpenAI-compatible endpoint. Models never get a public URL by
 * design: the only reachable address is 127.0.0.1 on the host, plus the Docker
 * bridge gateway from app containers.
 */
function ModelEndpoint({ endpoint }: { endpoint: string | null }) {
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

/** "Serve a model" form → POST /models (name defaults to the sanitized ref). */
function ServeModelPanel({ inputRef }: { inputRef: RefObject<HTMLInputElement> }) {
  const serve = useServeModel();
  const caps = useCapabilities();
  const backends = caps.data?.models.backends ?? ["ollama"];
  const defaultBackend = caps.data?.models.default_backend ?? "ollama";
  const [model, setModel] = useState("");
  const [name, setName] = useState("");
  const [gpus, setGpus] = useState("0");
  const [backend, setBackend] = useState<string | null>(null);

  // Default the selector to the daemon default once capabilities load.
  const selectedBackend = backend ?? defaultBackend;
  const needsGpu = selectedBackend === "vllm" && (Number(gpus) || 0) < 1;

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    const ref = model.trim();
    if (!ref) return;
    serve.mutate(
      {
        model: ref,
        gpus: Math.max(0, Number(gpus) || 0),
        name: name.trim() || undefined,
        // Defer to the server default until capabilities resolve, so a submit
        // racing the load never pins "ollama" over a configured default_backend.
        backend: backend ?? caps.data?.models.default_backend
      },
      {
        onSuccess: (created) => {
          toast("success", `Serving ${created.model ?? ref} as ${created.name}`);
          setModel("");
          setName("");
          setGpus("0");
          setBackend(null);
        }
      }
    );
  }

  return (
    <Panel title="Serve a model" data-testid="serve-panel">
      <form className="space-y-5 p-4" onSubmit={onSubmit}>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Field label="Model reference" htmlFor="serve-model">
            <Input
              id="serve-model"
              ref={inputRef}
              mono
              value={model}
              onChange={(event) => setModel(event.target.value)}
              placeholder="llama3.1:8b"
              required
            />
          </Field>
          <Field label="Name" htmlFor="serve-name" hint="Defaults to the model reference.">
            <Input
              id="serve-name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="ollama-llama3-1-8b"
            />
          </Field>
          <Field label="Backend" htmlFor="serve-backend">
            <Select
              id="serve-backend"
              value={selectedBackend}
              onChange={(event) => setBackend(event.target.value)}
            >
              {backends.map((b) => (
                <option key={b} value={b}>
                  {b}
                </option>
              ))}
            </Select>
          </Field>
          <Field
            label="GPUs"
            htmlFor="serve-gpus"
            hint="0 runs on CPU. GPUs are shared."
            error={needsGpu ? "vLLM needs at least 1 GPU." : undefined}
          >
            <Input
              id="serve-gpus"
              type="number"
              min={0}
              value={gpus}
              aria-invalid={needsGpu}
              onChange={(event) => setGpus(event.target.value)}
            />
          </Field>
        </div>
        <div className="flex items-center justify-between gap-4">
          <p className="text-13 text-muted-foreground">
            The image and the weights pull in the background.
          </p>
          <Button
            type="submit"
            variant="primary"
            loading={serve.isPending}
            disabled={!model.trim()}
          >
            Serve
          </Button>
        </div>
      </form>
    </Panel>
  );
}

export default function Models() {
  const models = useModels();
  const databases = useDatabases();
  const allBindings = useAllBindings();
  const stop = useServiceAction("stop");
  const restart = useServiceAction("restart");
  const remove = useServiceAction("delete");
  const [pendingDelete, setPendingDelete] = useState<Model | null>(null);
  const [serveOpen, setServeOpen] = useState(false);
  const modelInputRef = useRef<HTMLInputElement>(null);

  // Wrapped so the `?? []` fallback is not a fresh array every render — an
  // unstable reference here defeats the reverse-index memo below.
  const items = useMemo(() => models.data?.items ?? [], [models.data]);

  // D9 reverse index: which apps bind each served model.
  const projects = useMemo(
    () => deriveProjects(allBindings.services, allBindings.apps, items, databases.data?.items ?? []),
    [allBindings.services, allBindings.apps, items, databases.data]
  );

  const empty = !models.isLoading && !models.error && items.length === 0;

  function openServeForm() {
    setServeOpen(true);
    // Focus after the panel mounts, so the primary action lands the operator
    // in the one field that matters.
    window.setTimeout(() => modelInputRef.current?.focus(), 0);
  }

  return (
    <div className="mx-auto max-w-content space-y-7">
      <PageHeader
        title="Models"
        subtitle="Served locally over an OpenAI-compatible API. Models stay loopback-only, never public: apps reach them through the injected OPENAI_BASE_URL."
        actions={
          <Button variant="primary" data-testid="open-serve-form" onClick={openServeForm}>
            Serve a model
          </Button>
        }
      />

      {serveOpen && <ServeModelPanel inputRef={modelInputRef} />}

      <Panel title={`Served models (${items.length})`}>
        {models.error && (
          <div className="border-b border-border px-4 py-3 text-14 text-destructive">
            {(models.error as Error).message}
          </div>
        )}

        {empty ? (
          <div className="flex flex-col items-start gap-2 px-4 py-10" data-testid="models-empty">
            <p className="text-14 text-foreground">No models served yet.</p>
            <p className="text-14 text-muted-foreground">
              Serve one here, or from a shell on this machine:
            </p>
            <Mono className="text-muted-foreground">nerdit serve llama3.1:8b</Mono>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[56rem] text-14">
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
                {models.isLoading && <PlaceholderRows />}
                {items.map((m) => (
                  <TableRow key={m.id}>
                    <td className="max-w-[18rem] px-4 py-3">
                      <Mono className="block truncate text-foreground" title={m.name}>
                        {m.name}
                      </Mono>
                      {m.model && (
                        <span className="mt-1 block">
                          <UsedBy names={projectsBoundTo(projects, "ai", m.model)} />
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <Mono className="text-muted-foreground">{m.model ?? "–"}</Mono>
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">{m.backend ?? "–"}</td>
                    <td className="px-4 py-3">
                      <span className="flex flex-wrap items-center gap-2">
                        <Badge tone={serviceStatusTone(m.status)} data-testid="model-status">
                          {serviceStatusLabel(m.status)}
                        </Badge>
                        <PullBadge pulled={m.model_pulled} />
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <ModelGpus model={m} />
                    </td>
                    <td className="px-4 py-3">
                      <ModelEndpoint endpoint={m.endpoint} />
                    </td>
                    <td className="px-4 py-3 text-right">
                      <Menu
                        items={[
                          {
                            label: "Restart",
                            disabled: restart.isPending,
                            onSelect: () => restart.mutate(m.name)
                          },
                          {
                            label: "Stop",
                            disabled: stop.isPending || m.status === "stopped",
                            onSelect: () => stop.mutate(m.name)
                          },
                          "separator",
                          {
                            label: "Delete…",
                            destructive: true,
                            disabled: remove.isPending,
                            onSelect: () => setPendingDelete(m)
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
        title="Delete model"
        description={
          <>
            Stops the model server and unbinds apps using{" "}
            <Mono as="span" className="text-foreground">
              {pendingDelete?.name}
            </Mono>
            .
          </>
        }
        destructive
        confirmLabel="Delete"
        onCancel={() => setPendingDelete(null)}
        onConfirm={() => {
          if (pendingDelete) {
            const label = pendingDelete.name;
            remove.mutate(label, { onSuccess: () => toast("success", `Deleted ${label}`) });
          }
          setPendingDelete(null);
        }}
      />
    </div>
  );
}
