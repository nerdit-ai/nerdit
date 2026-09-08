/**
 * The one run-state vocabulary for the dashboard (design guidelines §3/§6).
 *
 * The audit found ~12 page-local status-pill maps disagreeing about both the
 * word and the colour. This module is the replacement, and a page-local map is
 * a defect: import these two functions and render the result in a `Badge`.
 *
 * Two rulings are load-bearing:
 *
 * - "Degraded" never reaches the screen. A running container whose health check
 *   is failing is **Running · unhealthy** — the honest sentence, and the same
 *   one whether the daemon says `degraded` or says `running` beside a false
 *   `healthy`. The separator is a middle dot; em dashes are banned in rendered
 *   copy repo-wide (pinned by tests/e2e/07-audit.spec.ts).
 * - Unknown values pass through capitalized rather than collapsing to
 *   "Unknown". `JobStatus` carries a `(string & {})` tail precisely because a
 *   legacy row can still arrive over the wire, and showing the daemon's own
 *   word is more useful to an operator mid-incident than erasing it.
 */
import type { JobStatus } from "../api/types";
import type { BadgeTone } from "../components/ui";

export type { JobStatus };

const UNHEALTHY = "Running · unhealthy";

const LABELS: Record<string, string> = {
  running: "Running",
  stopped: "Stopped",
  failed: "Failed",
  building: "Building",
  restarting: "Restarting",
  degraded: UNHEALTHY,
  queued: "Queued"
};

const TONES: Record<string, BadgeTone> = {
  running: "success",
  stopped: "muted",
  failed: "destructive",
  building: "muted",
  restarting: "muted",
  degraded: "warning",
  queued: "muted"
};

function capitalize(value: string): string {
  if (!value) return value;
  return value.charAt(0).toUpperCase() + value.slice(1);
}

/**
 * The word an operator reads. `healthy === false` on a running row renders the
 * unhealthy sentence; `undefined` means "no health check / not known" and is
 * NOT a failure (never invent a state the server did not confirm).
 */
export function serviceStatusLabel(status: string, healthy?: boolean): string {
  if (status === "running" && healthy === false) return UNHEALTHY;
  return LABELS[status] ?? capitalize(status);
}

/** The `Badge` tone for the same pair. Muted is the honest default. */
export function serviceStatusTone(status: string, healthy?: boolean): BadgeTone {
  if (status === "running") return healthy === false ? "warning" : "success";
  return TONES[status] ?? "muted";
}
