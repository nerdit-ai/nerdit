import { useState } from "react";
import { RefreshCw } from "lucide-react";
import { Badge, Button, Mono, Panel, type BadgeTone } from "../ui";
import { useDoctor } from "../../api/queries";
import type { DoctorStatus } from "../../api/types";

/**
 * Timeout-bounded daemon health checks with a worst-of badge (D8).
 *
 * The status→tone map is the daemon's own four-value vocabulary, rendered with
 * the shared Badge — `skipped` is muted on purpose: it is "not applicable
 * here", and it never worsens the top status server-side either.
 */
const TONES: Record<string, BadgeTone> = {
  ok: "success",
  warn: "warning",
  fail: "destructive",
  skipped: "muted"
};

function tone(status: DoctorStatus): BadgeTone {
  return TONES[status] ?? "muted";
}

/** Summary words and emphasis, in display order; a problem count reads louder than "ok". */
const SUMMARY: [DoctorStatus, (n: number) => string, string][] = [
  ["ok", () => "ok", "text-muted-foreground"],
  ["warn", (n) => (n === 1 ? "warning" : "warnings"), "font-medium text-warning-foreground"],
  ["fail", () => "failed", "font-medium text-destructive"],
  ["skipped", () => "skipped", "text-subtle-foreground"]
];

export function DoctorCard() {
  const doctor = useDoctor();
  const [open, setOpen] = useState(false);
  const checks = doctor.data?.checks ?? [];
  const summary = SUMMARY.map(([status, word, className]) => {
    const n = checks.filter((check) => check.status === status).length;
    return n > 0 ? { status, text: `${n} ${word(n)}`, className } : null;
  }).filter((part) => part !== null);

  return (
    <Panel
      title="Doctor"
      actions={
        <>
          {doctor.data && (
            <Badge data-testid="doctor-status" tone={tone(doctor.data.status)}>
              {doctor.data.status}
            </Badge>
          )}
          <Button size="sm" onClick={() => void doctor.refetch()} disabled={doctor.isFetching}>
            <RefreshCw
              aria-hidden="true"
              className={doctor.isFetching ? "h-4 w-4 animate-spin" : "h-4 w-4"}
            />
            Refresh
          </Button>
        </>
      }
    >
      {doctor.isError && (
        <p className="px-4 py-3 text-14 text-destructive">Could not load checks.</p>
      )}

      {!doctor.data && !doctor.isError && (
        <div className="space-y-2 px-4 py-3" aria-hidden="true">
          <div className="h-4 w-64 rounded-button bg-surface-hover" />
          <div className="h-4 w-48 rounded-button bg-surface-hover" />
        </div>
      )}

      {doctor.data && (
        <div className="flex items-center justify-between gap-3 px-4 py-3 text-14">
          <span data-testid="doctor-summary">
            {summary.map((part, i) => (
              <span key={part.status}>
                {i > 0 && <span className="text-subtle-foreground"> · </span>}
                <span className={part.className}>{part.text}</span>
              </span>
            ))}
          </span>
          <Button size="sm" variant="ghost" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
            {open ? "Hide checks" : "Show checks"}
          </Button>
        </div>
      )}

      {doctor.data && open && (
        <ul className="divide-y divide-border border-t border-border">
          {checks.map((check) => {
            const muted = check.status === "skipped";
            return (
              <li
                key={check.name}
                data-testid="doctor-check"
                className="flex flex-wrap items-center gap-3 px-4 py-3 text-14"
              >
                <Mono
                  className={`w-48 shrink-0 ${muted ? "text-subtle-foreground" : "text-foreground"}`}
                >
                  {check.name}
                </Mono>
                <Badge tone={tone(check.status)}>{check.status}</Badge>
                <span
                  className={`min-w-0 flex-1 ${muted ? "text-subtle-foreground" : "text-muted-foreground"}`}
                >
                  {check.detail}
                </span>
                <Mono className="shrink-0 text-subtle-foreground">{check.latency_ms} ms</Mono>
              </li>
            );
          })}
        </ul>
      )}
    </Panel>
  );
}
