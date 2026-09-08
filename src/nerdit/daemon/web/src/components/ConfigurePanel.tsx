import { useEffect, useId, useRef, useState } from "react";
import { KeyRound, Lock } from "lucide-react";
import { ApiError } from "../api/client";
import { useAppConfig, useUpdateAppConfig } from "../api/queries";
import { apiErrorCopy } from "../lib/apiErrors";
import { toast } from "../state/toastStore";
import { Button, Checkbox, DataRow, Dialog, Field, Input, Mono, Notice } from "./ui";

export interface ConfigurePanelProps {
  /** Service name — app config is keyed by name. */
  name: string;
  onClose: () => void;
}

interface Banner {
  tone: "warning" | "destructive";
  text: string;
}

/**
 * What a Preview learned. `values` is the diff the panel asked about (so the
 * list always describes the pending edit exactly, whatever a daemon chooses to
 * echo back); `requiresRestart` is the daemon's own answer, which is the whole
 * reason for asking.
 */
interface PreviewState {
  values: Record<string, unknown>;
  requiresRestart: boolean;
}

function numOrNull(value: string): number | null {
  const t = value.trim();
  return t === "" ? null : Number(t);
}

function strOrNull(value: string): string | null {
  const t = value.trim();
  return t === "" ? null : t;
}

function asString(value: unknown): string {
  return value == null ? "" : String(value);
}

/** How a would-be value reads in the preview list. */
function describe(value: unknown): string {
  return value == null || value === "" ? "cleared" : String(value);
}

/**
 * Configure a deployed app's [deploy] section over `GET/PUT /config/apps/{name}`
 * (D6). Editable: gpus, start, health, memory_limit, cpu_limit. name/port are
 * read-only (immutable by decision). Preview asks the daemon what the write
 * would do without writing; Apply sends the concurrency header the daemon
 * needs, so a change someone else made in the meantime is refused rather than
 * silently overwritten — that refusal refetches and says so. The [ai.*]/[db.*]
 * editors live in components/bindings/GenericBindingsPanel on the app page.
 *
 * The preview stays (it is the only place an operator learns, before writing,
 * that a change needs a restart to take effect); the machinery behind it is
 * never named on screen.
 */
export function ConfigurePanel({ name, onClose }: ConfigurePanelProps) {
  const config = useAppConfig(name);
  const update = useUpdateAppConfig(name);
  const fieldId = useId();

  const [gpus, setGpus] = useState("");
  const [start, setStart] = useState("");
  const [health, setHealth] = useState("");
  const [memoryLimit, setMemoryLimit] = useState("");
  const [cpuLimit, setCpuLimit] = useState("");
  const [restartNow, setRestartNow] = useState(false);
  const [preview, setPreview] = useState<PreviewState | null>(null);
  const [banner, setBanner] = useState<Banner | null>(null);

  const view = config.data?.view;
  const etag = config.data?.etag ?? null;

  // Deploy-section seeding. A refetch after an apply returns a fresh `view`, so
  // the effect tracks the serialization it last seeded from and re-seeds only
  // when the deploy section's own data changes (first arrival, or after this
  // section's apply refetch) — an apply refetch never clobbers in-flight edits.
  const seededDeploy = useRef<string | null>(null);

  useEffect(() => {
    if (!view) return;
    const d = view.deploy;
    const sig = JSON.stringify([d.gpus, d.start, d.health, d.memory_limit, d.cpu_limit]);
    if (seededDeploy.current === sig) return;
    seededDeploy.current = sig;
    setGpus(asString(d.gpus));
    setStart(asString(d.start));
    setHealth(asString(d.health));
    setMemoryLimit(asString(d.memory_limit));
    setCpuLimit(asString(d.cpu_limit));
    setPreview(null);
  }, [view]);

  function handleWriteError(err: Error) {
    // `config.stale` is the one code that changes what the panel DOES (refetch
    // the fresh values under the operator), so it keeps its own branch; the
    // sentence is still the shared helper's.
    const stale = err instanceof ApiError && err.code === "config.stale";
    setBanner({
      tone: stale ? "warning" : "destructive",
      text: stale
        ? "Settings changed elsewhere. Review and retry."
        : apiErrorCopy(err, "Could not save")
    });
    if (stale) void config.refetch();
  }

  function deployChanges(): Record<string, unknown> {
    if (!view) return {};
    const original = view.deploy;
    const next: Array<[string, unknown]> = [
      ["gpus", numOrNull(gpus)],
      ["start", strOrNull(start)],
      ["health", strOrNull(health)],
      ["memory_limit", strOrNull(memoryLimit)],
      ["cpu_limit", numOrNull(cpuLimit)]
    ];
    const changes: Record<string, unknown> = {};
    for (const [key, value] of next) {
      const orig = original[key] ?? null;
      if (JSON.stringify(orig) !== JSON.stringify(value ?? null)) changes[key] = value;
    }
    return changes;
  }

  function writeSection(values: Record<string, unknown>, previewOnly: boolean) {
    setBanner(null);
    update.mutate(
      {
        section: "deploy",
        values,
        etag,
        dryRun: previewOnly,
        restart: previewOnly ? undefined : restartNow
      },
      {
        onSuccess: (res) => {
          if (previewOnly) {
            setPreview({ values, requiresRestart: res.requires_restart });
            return;
          }
          const restarted = res.restarted ? ", restarted" : "";
          toast("success", `Deploy config saved${restarted}`);
        },
        onError: handleWriteError
      }
    );
  }

  const changes = deployChanges();
  const deployDirty = Object.keys(changes).length > 0;

  return (
    <Dialog
      open
      onClose={onClose}
      width="lg"
      title={`Configure ${name}`}
      className="max-h-[90vh] overflow-y-auto"
      footer={
        <Button variant="secondary" onClick={onClose}>
          Close
        </Button>
      }
    >
      <p className="mt-1 text-14 text-muted-foreground">
        Change deploy settings. Resources are managed on the app page.
      </p>

      <div className="mt-5 space-y-6">
        {/* Quiet placeholder in the form's own shape — never the word "Loading". */}
        {config.isLoading && (
          <div className="space-y-2" data-testid="config-placeholder">
            <span className="block h-3 w-40 rounded-full bg-surface-hover" />
            <span className="block h-3 w-64 rounded-full bg-surface-hover" />
          </div>
        )}
        {config.error && (
          <Notice tone="destructive">{apiErrorCopy(config.error, "Could not load settings")}</Notice>
        )}

        {banner && <Notice tone={banner.tone}>{banner.text}</Notice>}

        {view && (
          <>
            <section className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <ReadOnlyField
                  label="Name"
                  value={asString(view.deploy.name) || name}
                  reason="Fixed at deploy. Redeploy to rename."
                />
                <ReadOnlyField
                  label="Port"
                  value={asString(view.deploy.port) || "–"}
                  reason="Fixed at deploy time."
                />
                <Field label="GPUs" htmlFor={`${fieldId}-gpus`}>
                  <Input
                    id={`${fieldId}-gpus`}
                    value={gpus}
                    onChange={(e) => setGpus(e.target.value.replace(/\D/g, ""))}
                    inputMode="numeric"
                    placeholder="0"
                  />
                </Field>
                <Field label="CPU limit" htmlFor={`${fieldId}-cpu`}>
                  <Input
                    id={`${fieldId}-cpu`}
                    value={cpuLimit}
                    onChange={(e) => setCpuLimit(e.target.value.replace(/[^\d.]/g, ""))}
                    inputMode="decimal"
                    placeholder="e.g. 1.5"
                  />
                </Field>
                <Field label="Memory limit" htmlFor={`${fieldId}-memory`}>
                  <Input
                    id={`${fieldId}-memory`}
                    value={memoryLimit}
                    onChange={(e) => setMemoryLimit(e.target.value)}
                    placeholder="e.g. 512m"
                  />
                </Field>
                <Field label="Health check path" htmlFor={`${fieldId}-health`}>
                  <Input
                    id={`${fieldId}-health`}
                    value={health}
                    onChange={(e) => setHealth(e.target.value)}
                    placeholder="/health"
                    mono
                  />
                </Field>
              </div>

              <Field
                label="Start command"
                htmlFor={`${fieldId}-start`}
                hint="Applied at launch. Restart to take effect."
              >
                <Input
                  id={`${fieldId}-start`}
                  value={start}
                  onChange={(e) => setStart(e.target.value)}
                  placeholder="npm start"
                  mono
                />
              </Field>

              {preview && <PreviewNote preview={preview} />}

              <div className="flex items-center justify-end gap-2">
                <Button
                  disabled={!deployDirty || update.isPending}
                  onClick={() => writeSection(changes, true)}
                >
                  Preview
                </Button>
                <Button
                  variant="primary"
                  disabled={!deployDirty}
                  loading={update.isPending}
                  onClick={() => writeSection(changes, false)}
                >
                  Apply
                </Button>
              </div>
            </section>

            {/* Env keys (managed elsewhere) */}
            <section className="space-y-2 border-t border-border pt-5">
              <h3 className="label">Env</h3>
              {view.env_keys.length === 0 ? (
                <p className="text-13 text-muted-foreground">No env keys set.</p>
              ) : (
                <ul className="flex flex-wrap gap-2">
                  {view.env_keys.map((key) => (
                    <li
                      key={key}
                      className="flex items-center gap-1.5 rounded-button bg-surface-hover px-2 py-1"
                    >
                      <KeyRound size={16} className="text-muted-foreground" aria-hidden="true" />
                      <Mono className="text-foreground">{key}</Mono>
                    </li>
                  ))}
                </ul>
              )}
              <p className="text-13 text-muted-foreground">Managed in Secrets.</p>
            </section>
          </>
        )}
      </div>

      <div className="mt-6 border-t border-border pt-4">
        <Checkbox
          label="Restart now"
          checked={restartNow}
          onChange={(e) => setRestartNow(e.target.checked)}
        />
        <p className="mt-1 text-12 text-muted-foreground">Restart briefly drops the app.</p>
      </div>
    </Dialog>
  );
}

function ReadOnlyField({ label, value, reason }: { label: string; value: string; reason: string }) {
  return (
    <Field
      label={
        <span className="flex items-center gap-1.5">
          <Lock size={16} aria-hidden="true" /> {label}
        </span>
      }
      hint={reason}
    >
      <div className="h-9 rounded-button border border-border bg-surface-hover px-3 py-2 text-14 text-muted-foreground">
        {value}
      </div>
    </Field>
  );
}

/**
 * What Apply would do: the keys it would write, their would-be values, and
 * whether the app has to restart before any of it takes effect. Nothing is
 * saved by asking.
 */
function PreviewNote({ preview }: { preview: PreviewState }) {
  return (
    <Notice tone="info" data-testid="config-preview">
      <p className="font-medium">Preview only, nothing saved.</p>
      <div className="mt-2 divide-y divide-border border-y border-border">
        {Object.entries(preview.values).map(([key, value]) => (
          <DataRow key={key} label={key} className="px-0 py-2">
            <Mono>{describe(value)}</Mono>
          </DataRow>
        ))}
      </div>
      <p className="mt-2 text-13 text-muted-foreground">
        {preview.requiresRestart
          ? "Applying needs a restart to take effect."
          : "Applies without a restart."}
      </p>
    </Notice>
  );
}
