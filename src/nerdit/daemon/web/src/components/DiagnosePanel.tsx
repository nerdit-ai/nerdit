import { useState } from "react";
import { ChevronDown, ChevronRight, Stethoscope } from "lucide-react";
import { useDiagnose } from "../api/queries";
import type { JobStatus } from "../api/types";

export interface DiagnosePanelProps {
  ident: string;
  /** Current service status — drives the default-open decision. */
  status: JobStatus;
  /** last_deploy.phase — "failed" also opens the panel by default. */
  phase?: string | null;
}

// Short action label per remediation code. The backend `detail` is shown as a
// second line, so an unknown code still reads sensibly.
const REMEDIATION_LABELS: Record<string, string> = {
  raise_memory_limit: "Raise the memory limit",
  fix_start_command: "Fix the start command",
  image_needs_privileges: "Use a non-root image",
  serve_missing_model: "Serve the missing model",
  set_missing_secret: "Set the missing secret",
  fix_health_check: "Fix the health check",
  rollback: "Roll back",
  retry_later: "Retry later",
  inspect_logs: "Inspect the logs",
  none: "No fix needed"
};

const OPENING_STATUSES = new Set<JobStatus>(["failed", "degraded", "restarting"]);

function factRow(label: string, value: React.ReactNode) {
  return (
    <>
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="min-w-0 break-words">{value}</dd>
    </>
  );
}

/**
 * One-call failure bundle for a service (`GET /services/{id}/diagnose`). Owner/
 * admin only (gated by the caller). Collapsed by default; auto-open when the
 * service is failed/degraded/restarting or the last deploy failed. Renders key
 * names only, never secret values.
 */
export function DiagnosePanel({ ident, status, phase }: DiagnosePanelProps) {
  const shouldOpen = OPENING_STATUSES.has(status) || phase === "failed";
  const [open, setOpen] = useState(shouldOpen);
  const diagnose = useDiagnose(ident, open);
  const data = diagnose.data;

  const remediationLabel = data
    ? (REMEDIATION_LABELS[data.remediation.code] ?? data.remediation.detail)
    : null;

  return (
    <section
      data-testid="diagnose-panel"
      className="rounded-panel border border-border bg-surface p-5"
    >
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="flex w-full items-center justify-between gap-2 text-left"
      >
        <span className="label flex items-center gap-2">
          <Stethoscope size={14} aria-hidden="true" /> Diagnose
        </span>
        {open ? (
          <ChevronDown size={16} className="text-subtle-foreground" aria-hidden="true" />
        ) : (
          <ChevronRight size={16} className="text-subtle-foreground" aria-hidden="true" />
        )}
      </button>

      {open && (
        <div className="mt-4 space-y-4">
          {diagnose.isLoading && <p className="text-14 text-muted-foreground">Running checks...</p>}
          {diagnose.error && (
            <p className="text-14 text-destructive">{(diagnose.error as Error).message}</p>
          )}

          {data && (
            <>
              <div className="rounded-card bg-surface-hover px-4 py-3">
                <p className="label">Suggested fix</p>
                <p className="mt-1 text-14 font-semibold text-foreground">{remediationLabel}</p>
                {data.remediation.detail && remediationLabel !== data.remediation.detail && (
                  <p className="mt-1 text-12 text-muted-foreground">{data.remediation.detail}</p>
                )}
              </div>

              <dl className="grid grid-cols-2 gap-2 text-14">
                {data.error.class &&
                  factRow(
                    "Error",
                    <span className="text-destructive">
                      {data.error.class}
                      {data.error.message ? `: ${data.error.message}` : ""}
                    </span>
                  )}
                {factRow("Exit code", data.forensics.last_exit_code ?? "–")}
                {factRow("OOM killed", data.forensics.oom_killed ? "Yes" : "No")}
                {data.forensics.last_crash_at &&
                  factRow("Last crash", data.forensics.last_crash_at)}
                {factRow(
                  "Restarts",
                  `${data.restarts.count} / ${data.restarts.max_restarts}${
                    data.restarts.backoff_s != null ? ` (backoff ${data.restarts.backoff_s}s)` : ""
                  }`
                )}
                {data.restarts.next_retry_in_s != null &&
                  factRow("Next retry", `${data.restarts.next_retry_in_s}s`)}
                {factRow(
                  "Build",
                  `${data.build.last_result}${data.build.reason ? `: ${data.build.reason}` : ""}`
                )}
                {data.pending_env_keys.length > 0 &&
                  factRow(
                    "Missing env",
                    <span className="font-mono text-12">{data.pending_env_keys.join(", ")}</span>
                  )}
              </dl>

              {data.bindings.waiting && data.bindings.messages.length > 0 && (
                <div className="text-14">
                  <p className="text-muted-foreground">Waiting on AI bindings</p>
                  <ul className="mt-1 list-disc space-y-0.5 pl-5 text-12 text-subtle-foreground">
                    {data.bindings.messages.map((m, i) => (
                      <li key={i}>{m}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* (Codex 3804646869) The route's health advice — mirrors the
                  binding-waits block above; JSX auto-escaping is the same
                  escaping those server-derived messages already rely on. */}
              {(data.health?.observations?.length ?? 0) > 0 && (
                <div className="text-14">
                  <p className="text-muted-foreground">Health observations</p>
                  <ul className="mt-1 list-disc space-y-0.5 pl-5 text-12 text-subtle-foreground">
                    {data.health.observations.map((o, i) => (
                      <li key={i}>{o}</li>
                    ))}
                  </ul>
                </div>
              )}

              {data.logs.length > 0 && (
                <div>
                  <p className="mb-1 text-12 text-muted-foreground">Recent log tail</p>
                  <pre className="max-h-48 overflow-auto rounded-card bg-surface-hover px-3 py-2 font-mono text-12 text-foreground">
                    {data.logs
                      .map((entry) => String(entry.line ?? entry.message ?? ""))
                      .join("\n")}
                  </pre>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </section>
  );
}
