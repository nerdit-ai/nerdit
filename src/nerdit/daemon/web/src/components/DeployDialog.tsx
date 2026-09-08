import { useEffect, useId, useMemo, useState } from "react";
import { ChevronDown, ChevronRight, Plus, X } from "lucide-react";
import { Dropzone } from "./Dropzone";
import { DeployProgress } from "./DeployProgress";
import { Button, Field, Input, Mono, Notice } from "./ui";
import { useCapabilities, useDeploy, useDeployGit, useDeployPlan } from "../api/queries";
import type { DeployPlan, GitDeployRequest, Service } from "../api/types";
import { toast } from "../state/toastStore";

export interface DeployDialogProps {
  open: boolean;
  onClose: () => void;
  /** Prefill the service name (redeploy from a detail page). */
  initialName?: string;
  /** Which tab to open on (default "upload"). */
  initialMode?: Tab;
}

type Tab = "upload" | "git";

interface EnvRow {
  id: number;
  key: string;
  value: string;
}

let envRowSeq = 0;
function newEnvRow(): EnvRow {
  return { id: envRowSeq++, key: "", value: "" };
}

/**
 * Deploy modal for `POST /deploy` (ZIP upload) and `POST /deploy/git` (repo
 * clone). Preview runs a dry-run plan (no writes); Deploy swaps to a progress
 * view that long-polls the service's converge outcome.
 *
 * Only the name and the source chooser are visible (design guidelines §4).
 * Port, GPUs, start command, health path and env are overrides of the app's own
 * `nerdit.toml`, and the product's north-star gesture is that the folder
 * already says all of this — a wall of fields would contradict it. They live
 * behind one collapsed "Advanced" disclosure, collapsed again on every open.
 *
 * The shell is the design system's: `ui.tsx` Confirm's overlay/panel treatment,
 * Field + Input for every control, Button for every action. Nothing here draws
 * its own skin.
 */
export function DeployDialog({
  open,
  onClose,
  initialName = "",
  initialMode = "upload"
}: DeployDialogProps) {
  const caps = useCapabilities();
  const gitEnabled = caps.data?.deploy.git_enabled ?? true;
  const gitHosts = caps.data?.deploy.git_allowed_hosts ?? [];

  const deploy = useDeploy();
  const deployGit = useDeployGit();
  const deployPlan = useDeployPlan();

  const fieldId = useId();
  const titleId = `${fieldId}-title`;

  const [tab, setTab] = useState<Tab>(initialMode);
  const [archive, setArchive] = useState<File | null>(null);
  const [repoUrl, setRepoUrl] = useState("");
  const [ref, setRef] = useState("");
  const [subdir, setSubdir] = useState("");
  const [tokenRef, setTokenRef] = useState("");

  const [name, setName] = useState(initialName);
  const [port, setPort] = useState("");
  const [gpus, setGpus] = useState("");
  const [start, setStart] = useState("");
  const [health, setHealth] = useState("");
  const [envRows, setEnvRows] = useState<EnvRow[]>([]);

  const [advanced, setAdvanced] = useState(false);

  const [plan, setPlan] = useState<DeployPlan | null>(null);
  const [deployed, setDeployed] = useState<Service | null>(null);

  // Reset the form each time the dialog opens.
  useEffect(() => {
    if (!open) return;
    setTab(initialMode);
    setArchive(null);
    setRepoUrl("");
    setRef("");
    setSubdir("");
    setTokenRef("");
    setName(initialName);
    setPort("");
    setGpus("");
    setStart("");
    setHealth("");
    setEnvRows([]);
    setAdvanced(false);
    setPlan(null);
    setDeployed(null);
  }, [open, initialName, initialMode]);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  // Clear a stale preview whenever an input changes.
  const inputsKey = `${tab}|${archive?.name}|${repoUrl}|${ref}|${subdir}|${tokenRef}|${name}|${port}|${gpus}|${start}|${health}|${JSON.stringify(envRows)}`;
  useEffect(() => setPlan(null), [inputsKey]);

  const trimmedName = name.trim();
  const busy = deploy.isPending || deployGit.isPending || deployPlan.isPending;

  /** KEY=VALUE rows → env map; an empty value on a named key deletes it (null). */
  const env = useMemo<Record<string, string | null> | undefined>(() => {
    const rows = envRows.filter((r) => r.key.trim());
    if (!rows.length) return undefined;
    const out: Record<string, string | null> = {};
    for (const r of rows) out[r.key.trim()] = r.value === "" ? null : r.value;
    return out;
  }, [envRows]);

  const shared = {
    name: trimmedName,
    port: port.trim() ? Number(port) : undefined,
    gpus: gpus.trim() ? Number(gpus) : undefined,
    start: start.trim() || undefined,
    health: health.trim() || undefined
  };

  if (!open) return null;

  const canDeploy =
    Boolean(trimmedName) &&
    (tab === "upload" ? Boolean(archive) : gitEnabled && Boolean(repoUrl.trim())) &&
    !busy;

  function gitBody(): GitDeployRequest {
    return {
      repo_url: repoUrl.trim(),
      name: trimmedName,
      ref: ref.trim() || undefined,
      subdir: subdir.trim() || undefined,
      token_ref: tokenRef.trim() || undefined,
      port: shared.port,
      gpus: shared.gpus,
      start: shared.start,
      health: shared.health,
      env: env ?? undefined
    };
  }

  function onPreview() {
    if (!canDeploy) return;
    if (tab === "upload") {
      if (!archive) return;
      deployPlan.mutate({ archive, ...shared, env: env ?? undefined }, { onSuccess: setPlan });
    } else {
      deployGit.mutate(
        { body: gitBody(), dryRun: true },
        { onSuccess: (res) => setPlan(res as DeployPlan) }
      );
    }
  }

  function onDeploy() {
    if (!canDeploy) return;
    if (tab === "upload") {
      if (!archive) return;
      deploy.mutate(
        { archive, ...shared, env: env ?? undefined },
        {
          onSuccess: (svc) => {
            toast("success", `Deploy of ${svc.name} started`);
            setDeployed(svc);
          }
        }
      );
    } else {
      deployGit.mutate(
        { body: gitBody() },
        {
          onSuccess: (res) => {
            const svc = res as Service;
            toast("success", `Deploy of ${svc.name} started`);
            setDeployed(svc);
          }
        }
      );
    }
  }

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-overlay p-4 backdrop-blur"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={deployed ? undefined : titleId}
        data-testid="deploy-dialog"
        className="max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-panel border border-border bg-surface p-6 shadow-lg"
        onClick={(event) => event.stopPropagation()}
      >
        {deployed ? (
          <DeployProgress service={deployed} onClose={onClose} />
        ) : (
          <>
            <h2 id={titleId} className="text-16 font-semibold text-foreground">
              Deploy an app
            </h2>
            <p className="mt-1 text-14 text-muted-foreground">
              Build and run an app from a ZIP or a Git repo. Reusing a name redeploys it.
            </p>

            <div
              className="mt-4 inline-flex items-center gap-1 rounded-button border border-border p-1"
              data-testid="deploy-source-tabs"
            >
              <Button
                size="sm"
                variant={tab === "upload" ? "secondary" : "ghost"}
                aria-pressed={tab === "upload"}
                data-testid="deploy-tab-upload"
                onClick={() => setTab("upload")}
              >
                Upload
              </Button>
              <Button
                size="sm"
                variant={tab === "git" ? "secondary" : "ghost"}
                aria-pressed={tab === "git"}
                data-testid="deploy-tab-git"
                onClick={() => setTab("git")}
              >
                Git
              </Button>
            </div>

            <div className="mt-4 space-y-4">
              {tab === "upload" ? (
                <Dropzone
                  accept=".zip,application/zip"
                  onFiles={(files) => setArchive(files[0] ?? null)}
                  label={archive ? archive.name : "Drop the app ZIP here or click to browse"}
                  hint={archive ? `${(archive.size / 1024).toFixed(0)} KB` : "One .zip archive"}
                  className="py-6"
                />
              ) : !gitEnabled ? (
                <Notice tone="warning">Git deploy is disabled by the daemon.</Notice>
              ) : (
                <div className="space-y-3">
                  <Field label="Repository URL *" htmlFor={`${fieldId}-repo`}>
                    <Input
                      id={`${fieldId}-repo`}
                      data-testid="deploy-repo-url"
                      value={repoUrl}
                      onChange={(event) => setRepoUrl(event.target.value)}
                      placeholder="https://github.com/owner/repo"
                      spellCheck={false}
                      autoComplete="off"
                    />
                  </Field>
                  <div className="grid grid-cols-2 gap-3">
                    <Field label="Ref" htmlFor={`${fieldId}-ref`}>
                      <Input
                        id={`${fieldId}-ref`}
                        data-testid="deploy-ref"
                        value={ref}
                        onChange={(event) => setRef(event.target.value)}
                        placeholder="main"
                        spellCheck={false}
                        autoComplete="off"
                      />
                    </Field>
                    <Field label="Subdir" htmlFor={`${fieldId}-subdir`}>
                      <Input
                        id={`${fieldId}-subdir`}
                        data-testid="deploy-subdir"
                        value={subdir}
                        onChange={(event) => setSubdir(event.target.value)}
                        placeholder="apps/web"
                        spellCheck={false}
                        autoComplete="off"
                      />
                    </Field>
                  </div>
                  <Field
                    label="Token ref"
                    htmlFor={`${fieldId}-token-ref`}
                    hint="For private repos. Use a secret ref, not a raw token."
                  >
                    <Input
                      id={`${fieldId}-token-ref`}
                      data-testid="deploy-token-ref"
                      value={tokenRef}
                      onChange={(event) => setTokenRef(event.target.value)}
                      placeholder="${secrets.GIT_TOKEN}"
                      spellCheck={false}
                      autoComplete="off"
                      mono
                    />
                  </Field>
                  {gitHosts.length > 0 && (
                    <p className="text-12 text-muted-foreground">
                      Allowed hosts: <Mono>{gitHosts.join(", ")}</Mono>
                    </p>
                  )}
                </div>
              )}

              <Field label="Name *" htmlFor={`${fieldId}-name`}>
                <Input
                  id={`${fieldId}-name`}
                  data-testid="deploy-name"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="my-app"
                  spellCheck={false}
                  autoComplete="off"
                />
              </Field>

              <div className="border-t border-border pt-4">
                <Button
                  variant="ghost"
                  size="sm"
                  data-testid="deploy-advanced"
                  aria-expanded={advanced}
                  aria-controls={`${fieldId}-advanced`}
                  className="px-0 text-muted-foreground"
                  onClick={() => setAdvanced((v) => !v)}
                >
                  {advanced ? (
                    <ChevronDown aria-hidden="true" className="h-4 w-4" />
                  ) : (
                    <ChevronRight aria-hidden="true" className="h-4 w-4" />
                  )}
                  Advanced
                </Button>
                <p className="mt-1 text-12 text-muted-foreground">
                  <Mono>nerdit.toml</Mono> in the app folder supplies these.
                </p>
              </div>

              <div
                id={`${fieldId}-advanced`}
                hidden={!advanced}
                className={advanced ? "space-y-4" : undefined}
              >
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Port" htmlFor={`${fieldId}-port`}>
                    <Input
                      id={`${fieldId}-port`}
                      data-testid="deploy-port"
                      value={port}
                      onChange={(event) => setPort(event.target.value.replace(/\D/g, ""))}
                      inputMode="numeric"
                      placeholder="8000"
                    />
                  </Field>
                  <Field label="GPUs" htmlFor={`${fieldId}-gpus`}>
                    <Input
                      id={`${fieldId}-gpus`}
                      data-testid="deploy-gpus"
                      value={gpus}
                      onChange={(event) => setGpus(event.target.value.replace(/\D/g, ""))}
                      inputMode="numeric"
                      placeholder="0"
                    />
                  </Field>
                </div>

                <Field label="Start command" htmlFor={`${fieldId}-start`}>
                  <Input
                    id={`${fieldId}-start`}
                    data-testid="deploy-start"
                    value={start}
                    onChange={(event) => setStart(event.target.value)}
                    placeholder="npm start"
                    spellCheck={false}
                    autoComplete="off"
                    mono
                  />
                </Field>

                <Field label="Health check path" htmlFor={`${fieldId}-health`}>
                  <Input
                    id={`${fieldId}-health`}
                    data-testid="deploy-health"
                    value={health}
                    onChange={(event) => setHealth(event.target.value)}
                    placeholder="/health"
                    spellCheck={false}
                    autoComplete="off"
                    mono
                  />
                </Field>

                <div className="space-y-2">
                  <span className="label block">Environment</span>
                  {envRows.map((row, idx) => (
                    <div key={row.id} className="flex items-center gap-2">
                      <Input
                        value={row.key}
                        aria-label={`Env key ${idx + 1}`}
                        data-testid={`deploy-env-key-${idx}`}
                        onChange={(event) =>
                          setEnvRows((rows) =>
                            rows.map((r) => (r.id === row.id ? { ...r, key: event.target.value } : r))
                          )
                        }
                        placeholder="KEY"
                        spellCheck={false}
                        autoComplete="off"
                        mono
                      />
                      <Input
                        value={row.value}
                        aria-label={`Env value ${idx + 1}`}
                        data-testid={`deploy-env-value-${idx}`}
                        onChange={(event) =>
                          setEnvRows((rows) =>
                            rows.map((r) =>
                              r.id === row.id ? { ...r, value: event.target.value } : r
                            )
                          )
                        }
                        placeholder="value"
                        spellCheck={false}
                        autoComplete="off"
                        mono
                      />
                      <Button
                        variant="ghost"
                        size="sm"
                        aria-label={`Remove env row ${idx + 1}`}
                        data-testid={`deploy-env-remove-${idx}`}
                        className="shrink-0 px-2 text-muted-foreground"
                        onClick={() => setEnvRows((rows) => rows.filter((r) => r.id !== row.id))}
                      >
                        <X aria-hidden="true" className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                  <Button
                    variant="ghost"
                    size="sm"
                    data-testid="deploy-env-add"
                    className="text-muted-foreground"
                    onClick={() => setEnvRows((rows) => [...rows, newEnvRow()])}
                  >
                    <Plus aria-hidden="true" className="h-4 w-4" />
                    Add variable
                  </Button>
                  <p className="text-12 text-muted-foreground">
                    An empty value deletes the key on redeploy.
                  </p>
                </div>
              </div>
            </div>

            {plan && <PlanPreview plan={plan} />}

            <div className="mt-6 flex justify-end gap-2">
              <Button variant="ghost" data-testid="deploy-cancel" onClick={onClose}>
                Cancel
              </Button>
              <Button
                variant="secondary"
                data-testid="deploy-preview"
                disabled={!canDeploy}
                onClick={onPreview}
              >
                {deployPlan.isPending ? "Previewing..." : "Preview"}
              </Button>
              <Button
                variant="primary"
                data-testid="deploy-submit"
                disabled={!canDeploy}
                onClick={onDeploy}
              >
                {deploy.isPending || deployGit.isPending ? "Deploying..." : "Deploy"}
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/** Read-only dry-run plan summary: action, buildpack, effective settings, diffs. */
function PlanPreview({ plan }: { plan: DeployPlan }) {
  const eff = plan.effective;
  const envDiff = plan.env_diff as {
    added?: string[];
    changed?: string[];
    removed?: string[];
  };
  const counts = [
    envDiff.added?.length ? `${envDiff.added.length} added` : null,
    envDiff.changed?.length ? `${envDiff.changed.length} changed` : null,
    envDiff.removed?.length ? `${envDiff.removed.length} removed` : null
  ].filter(Boolean);
  const warnings = plan.warnings ?? [];

  return (
    <div
      data-testid="deploy-plan"
      className="mt-4 rounded-card border border-border bg-background px-4 py-3"
    >
      <p className="text-14 font-medium text-foreground">
        Plan: {plan.action === "redeploy" ? "Redeploy" : "Create"}
        {plan.buildpack ? ` via ${plan.buildpack}` : ""}
      </p>
      <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-12 text-muted-foreground">
        {eff.port != null && (
          <div className="flex justify-between gap-2">
            <dt>Port</dt>
            <dd className="text-foreground">{eff.port}</dd>
          </div>
        )}
        {eff.gpus != null && (
          <div className="flex justify-between gap-2">
            <dt>GPUs</dt>
            <dd className="text-foreground">{eff.gpus}</dd>
          </div>
        )}
        {eff.start && (
          <div className="col-span-2 flex justify-between gap-2">
            <dt>Start</dt>
            <dd className="min-w-0 truncate text-foreground">
              <Mono>{eff.start}</Mono>
            </dd>
          </div>
        )}
        {eff.health && (
          <div className="flex justify-between gap-2">
            <dt>Health</dt>
            <dd className="text-foreground">
              <Mono>{eff.health}</Mono>
            </dd>
          </div>
        )}
      </dl>
      <p className="mt-2 text-12 text-muted-foreground">
        Env: {counts.length ? counts.join(", ") : "no changes"}
      </p>
      <p className="text-12 text-muted-foreground">AI bindings: {plan.ai_diff?.action ?? "none"}</p>
      {(plan.overwrote_api_config || warnings.length > 0) && (
        <Notice tone="warning" className="mt-3 space-y-1">
          {plan.overwrote_api_config && (
            <p className="text-12">This overwrites config set via the API.</p>
          )}
          {warnings.map((w) => (
            <p key={w} className="text-12">
              {w}
            </p>
          ))}
        </Notice>
      )}
    </div>
  );
}
