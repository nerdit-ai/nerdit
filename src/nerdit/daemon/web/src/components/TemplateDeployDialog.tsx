import { useEffect, useId, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button, Dialog, Field, Input, Mono, Notice } from "./ui";
import { useAppTemplates, useDeployTemplate } from "../api/queries";
import { toast } from "../state/toastStore";
import type { AppTemplate } from "../api/types";

/**
 * "New app → From a template", the flow that replaces the Store PAGE.
 *
 * Deploying a template is a way to create an app, not a place to go (design
 * guidelines §2), so the catalog is step 1 of this dialog and the app it
 * produces lands in the same list as every other app. The logic is the Store
 * page's, carried over unchanged: secret-typed env entries collect into
 * `secrets` and everything else into `env`, required entries block submit, and
 * the request goes through the same `POST /app-templates/{id}/deploy` mutation
 * (which already maps a structured error envelope onto a persistent toast).
 *
 * The shell is `ui.tsx` Dialog: the same overlay, the same panel skin, the same
 * Escape/overlay-click close, and it owns the step's title. Nothing here draws
 * its own — the two steps are children, not modals.
 */
export function TemplateDeployDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const templates = useAppTemplates();
  const [selected, setSelected] = useState<AppTemplate | null>(null);

  // A fresh dialog every time it opens: step 1, nothing picked.
  useEffect(() => {
    if (open) setSelected(null);
  }, [open]);

  return (
    <Dialog
      open={open}
      onClose={onClose}
      width="lg"
      data-testid="template-dialog"
      className="max-h-[90vh] overflow-y-auto"
      title={selected ? `Deploy ${selected.name}` : "Start from a template"}
    >
      {selected ? (
        <TemplateForm template={selected} onBack={() => setSelected(null)} onClose={onClose} />
      ) : (
        <TemplatePicker templates={templates} onPick={setSelected} onClose={onClose} />
      )}
    </Dialog>
  );
}

/* ---------------------------------------------------------------- step one */

function TemplatePicker({
  templates,
  onPick,
  onClose
}: {
  templates: ReturnType<typeof useAppTemplates>;
  onPick: (template: AppTemplate) => void;
  onClose: () => void;
}) {
  const items = templates.data ?? [];

  return (
    <>
      <p className="mt-1 text-14 text-muted-foreground">
        A curated app, cloned and built on this machine.
      </p>

      <div className="mt-4 divide-y divide-border border-y border-border">
        {templates.isLoading &&
          [0, 1, 2].map((row) => (
            <div key={row} className="px-1 py-4" data-testid="template-placeholder">
              <span className="block h-3 w-32 rounded-full bg-surface-hover" />
              <span className="mt-2 block h-3 w-56 rounded-full bg-surface-hover" />
            </div>
          ))}

        {!templates.isLoading &&
          items.map((template) => (
            <button
              key={template.id}
              type="button"
              data-testid={`template-row-${template.id}`}
              onClick={() => onPick(template)}
              className="block w-full px-1 py-3 text-left hover:bg-surface-hover"
            >
              <span className="block text-14 font-medium text-foreground">{template.name}</span>
              <span className="mt-0.5 block text-13 text-muted-foreground">
                {template.description}
              </span>
              <span className="mt-1 block text-12 text-subtle-foreground">
                {[template.category, template.ai_hint].filter(Boolean).join(" · ")}
              </span>
            </button>
          ))}
      </div>

      {templates.error && (
        <Notice tone="destructive" className="mt-4">
          {(templates.error as Error).message}
        </Notice>
      )}

      {!templates.isLoading && !templates.error && items.length === 0 && (
        <p className="mt-4 text-13 text-muted-foreground">
          No templates available. The catalog is empty, or Git deploys are off. From a shell:{" "}
          <Mono>nerdit store list</Mono>
        </p>
      )}

      <div className="mt-6 flex justify-end">
        <Button variant="secondary" data-testid="template-cancel" onClick={onClose}>
          Cancel
        </Button>
      </div>
    </>
  );
}

/* ---------------------------------------------------------------- step two */

function TemplateForm({
  template,
  onBack,
  onClose
}: {
  template: AppTemplate;
  onBack: () => void;
  onClose: () => void;
}) {
  const deploy = useDeployTemplate();
  const navigate = useNavigate();
  const fieldId = useId();

  const [name, setName] = useState(template.id);
  const [fields, setFields] = useState<Record<string, string>>({});
  const [advanced, setAdvanced] = useState(false);
  const [port, setPort] = useState("");
  const [gpus, setGpus] = useState("");
  const [start, setStart] = useState("");
  const [health, setHealth] = useState("");

  const trimmedName = name.trim();
  const missingRequired = template.env_schema.some(
    (field) => field.required && !(fields[field.name] ?? "").trim()
  );
  const canSubmit = Boolean(trimmedName) && !missingRequired && !deploy.isPending;

  function onSubmit() {
    if (!canSubmit) return;
    const env: Record<string, string> = {};
    const secrets: Record<string, string> = {};
    for (const field of template.env_schema) {
      const value = (fields[field.name] ?? "").trim();
      if (!value) continue;
      if (field.secret) secrets[field.name] = value;
      else env[field.name] = value;
    }
    deploy.mutate(
      {
        id: template.id,
        body: {
          name: trimmedName,
          port: port.trim() ? Number(port) : undefined,
          gpus: gpus.trim() ? Number(gpus) : undefined,
          start: start.trim() || undefined,
          health: health.trim() || undefined,
          env: Object.keys(env).length ? env : undefined,
          secrets: Object.keys(secrets).length ? secrets : undefined
        }
      },
      {
        onSuccess: (svc) => {
          toast("success", `Deploy of ${svc.name} started`);
          onClose();
          navigate("/");
        }
      }
    );
  }

  return (
    <>
      <p className="mt-1 text-14 text-muted-foreground">{template.description}</p>

      <div className="mt-4 space-y-4">
        <Field label="Name *" htmlFor={`${fieldId}-name`}>
          <Input
            id={`${fieldId}-name`}
            data-testid="template-name"
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="my-app"
            spellCheck={false}
            autoComplete="off"
          />
        </Field>

        {template.env_schema.map((field) => (
          <Field
            key={field.name}
            label={`${field.name}${field.required ? " *" : ""}`}
            htmlFor={`${fieldId}-${field.name}`}
            hint={
              field.secret
                ? [field.description, "Stored as a secret, never shown again."]
                    .filter(Boolean)
                    .join(" ")
                : field.description || undefined
            }
          >
            <Input
              id={`${fieldId}-${field.name}`}
              data-testid={`template-env-${field.name}`}
              type={field.secret ? "password" : "text"}
              value={fields[field.name] ?? ""}
              onChange={(event) =>
                setFields((prev) => ({ ...prev, [field.name]: event.target.value }))
              }
              placeholder={field.secret ? "••••••" : ""}
              autoComplete={field.secret ? "new-password" : "off"}
              spellCheck={false}
              mono={!field.secret}
            />
          </Field>
        ))}

        <Button
          variant="ghost"
          size="sm"
          data-testid="template-advanced"
          aria-expanded={advanced}
          className="text-muted-foreground"
          onClick={() => setAdvanced((v) => !v)}
        >
          Advanced
        </Button>

        {advanced && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Port" htmlFor={`${fieldId}-port`}>
                <Input
                  id={`${fieldId}-port`}
                  data-testid="template-port"
                  value={port}
                  onChange={(event) => setPort(event.target.value.replace(/\D/g, ""))}
                  inputMode="numeric"
                  placeholder={template.deploy_defaults.port?.toString() ?? "8000"}
                />
              </Field>
              <Field label="GPUs" htmlFor={`${fieldId}-gpus`}>
                <Input
                  id={`${fieldId}-gpus`}
                  data-testid="template-gpus"
                  value={gpus}
                  onChange={(event) => setGpus(event.target.value.replace(/\D/g, ""))}
                  inputMode="numeric"
                  placeholder={template.deploy_defaults.gpus?.toString() ?? "0"}
                />
              </Field>
            </div>
            <Field label="Start command" htmlFor={`${fieldId}-start`}>
              <Input
                id={`${fieldId}-start`}
                data-testid="template-start"
                value={start}
                onChange={(event) => setStart(event.target.value)}
                placeholder={template.deploy_defaults.start ?? "auto-detected"}
                spellCheck={false}
                autoComplete="off"
                mono
              />
            </Field>
            <Field label="Health check path" htmlFor={`${fieldId}-health`}>
              <Input
                id={`${fieldId}-health`}
                data-testid="template-health"
                value={health}
                onChange={(event) => setHealth(event.target.value)}
                placeholder={template.deploy_defaults.health ?? "/"}
                spellCheck={false}
                autoComplete="off"
                mono
              />
            </Field>
            <p className="text-12 text-muted-foreground">
              These override the template&apos;s own defaults.
            </p>
          </div>
        )}
      </div>

      <div className="mt-6 flex justify-end gap-2">
        <Button variant="ghost" data-testid="template-back" onClick={onBack}>
          Back
        </Button>
        <Button variant="secondary" data-testid="template-cancel" onClick={onClose}>
          Cancel
        </Button>
        <Button
          variant="primary"
          data-testid="template-submit"
          loading={deploy.isPending}
          disabled={!canSubmit}
          onClick={onSubmit}
        >
          Deploy
        </Button>
      </div>
    </>
  );
}
