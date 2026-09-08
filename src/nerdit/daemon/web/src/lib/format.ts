import type { PublicUrlEntry, ServiceEndpoint, ServiceSource } from "../api/types";

/**
 * Seconds elapsed since `startedAt` (ISO timestamp), or null when the
 * workload has not started. Clamped at 0 for minor clock skew.
 */
export function uptimeSeconds(startedAt: string | null | undefined, now: Date = new Date()): number | null {
  if (!startedAt) return null;
  const started = Date.parse(startedAt);
  if (Number.isNaN(started)) return null;
  return Math.max(0, Math.floor((now.getTime() - started) / 1000));
}

/** Compact human duration: "42s", "3m 12s", "2h 05m", "4d 7h". */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || seconds < 0) return "–";
  const s = Math.floor(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ${String(m % 60).padStart(2, "0")}m`;
  const d = Math.floor(h / 24);
  return `${d}d ${h % 24}h`;
}

/** Derived uptime label from `started_at`; "–" when not started. */
export function formatUptime(startedAt: string | null | undefined, now: Date = new Date()): string {
  return formatDuration(uptimeSeconds(startedAt, now));
}

/**
 * Compact "time ago" label from an ISO timestamp: "just now", "5m ago",
 * "2h ago", "3d ago". Null/unparseable input renders "–". Future timestamps
 * (minor clock skew) clamp to "just now".
 */
export function formatRelative(iso: string | null | undefined, now: Date = new Date()): string {
  if (!iso) return "–";
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return "–";
  const seconds = Math.floor((now.getTime() - then) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export type PublicUrlState =
  /** A public HTTPS URL is resolved and clickable. */
  | "routed"
  /** An endpoint exists but the proxy is off/down — expected state, not an error. */
  | "proxy-off"
  /** No endpoint published yet (service not launched). */
  | "none";

/**
 * Classify a service endpoint's public reachability. `public_url` being null
 * while an endpoint exists is an expected "proxy off" state (trap 8), never
 * an error; the loopback `endpoint.url` remains usable.
 */
export function publicUrlState(endpoint: ServiceEndpoint | null | undefined): PublicUrlState {
  if (!endpoint) return "none";
  return endpoint.public_url != null ? "routed" : "proxy-off";
}

/**
 * The hosted share URL to prefer over the LAN one (P26 D-P26-14), or null.
 *
 * Only a `ready` hosted entry qualifies: a share whose tunnel is down or whose
 * plan does not cover `public` has a URL that does not answer, and offering it
 * as "the link to share" would be worse than offering the LAN one. `state` is
 * the machine token the daemon computes per request — never re-derived here.
 */
export function hostedShareUrl(
  endpoint: ServiceEndpoint | null | undefined,
): PublicUrlEntry | null {
  const entries = endpoint?.public_urls;
  if (!entries) return null;
  return entries.find((e) => e.kind === "hosted" && e.state === "ready" && e.url) ?? null;
}

/**
 * Every direct domain of a service that is actually answering (P26 WP1), in
 * the daemon's own order.
 *
 * Only `ready` entries qualify, for the same reason `hostedShareUrl` filters:
 * a `withheld` domain is stored intent whose URL does not resolve yet (proxy
 * off, app not routed, edge-auth secret unresolved), and offering it as "your
 * URL" would be a lie. `state` is the daemon's machine token — computed per
 * request there, never re-derived here.
 */
export function domainUrls(endpoint: ServiceEndpoint | null | undefined): PublicUrlEntry[] {
  const entries = endpoint?.public_urls;
  if (!entries) return [];
  return entries.filter((e) => e.kind === "domain" && e.state === "ready" && e.url);
}

/**
 * The direct domain to FEATURE as "the URL" — the one the card copies and
 * links — or null when no ready domain can be opened (P26 WP2, Codex round 2
 * #3835632984).
 *
 * `domainUrls` answers "whose route is live", which is not the same question.
 * A `ready` domain whose `cert_state` is `pending` has a live Host route and
 * NO leaf: Caddy's subject-less catch-all policy is not a fallback for an ACME
 * subject, so the handshake is aborted outright (`tlsv1 alert internal error`,
 * SSL alert 80 — measured, see `core/proxy/tls.py`). Featuring it would hand
 * the reader a copy button and an open link for a name that answers nothing —
 * directly above this card's own caption saying it does not answer.
 *
 * Every other `cert_state` completes a handshake and is therefore featurable:
 * `internal`/`disabled` are ordinary internal-CA names (openable after
 * `nerdit trust`), `expired` still serves the stale public leaf (a browser
 * warning, not a dead name), and an absent field is a pre-WP2 daemon. Only
 * `pending` is skipped — and skipped, not hidden: the caller still lists it
 * with its `cert pending` pill, because the card must not lie about what is
 * bound. Nothing is re-derived from the certificate here; this reads the
 * daemon's two tokens exactly as `certBadge` does.
 */
export function featuredDomainUrl(
  endpoint: ServiceEndpoint | null | undefined,
): PublicUrlEntry | null {
  return domainUrls(endpoint).find((e) => e.cert_state !== "pending") ?? null;
}

/** Visual weight for a cert badge; the component owns the actual classes. */
export type CertBadgeTone = "ok" | "warn" | "danger" | "muted";

export interface CertBadge {
  label: string;
  tone: CertBadgeTone;
}

/**
 * The certificate badge for one `public_urls` entry (P26 WP2), or null when
 * there is nothing worth saying.
 *
 * `cert_state` is the daemon's own token — derived there from Caddy's storage,
 * never re-derived here — and it is orthogonal to `state`: a domain can be
 * `ready` (its route answers) while its public leaf is still `pending`. Only
 * the states that change what a visitor's browser will do get a badge:
 * `internal` is the ordinary, documented case for a node that never asked for
 * a public certificate, and an absent field is a pre-WP2 daemon, so both stay
 * silent rather than adding a pill to every row.
 */
export function certBadge(entry: PublicUrlEntry | null | undefined): CertBadge | null {
  switch (entry?.cert_state) {
    case "issued":
      return { label: "public cert", tone: "ok" };
    case "pending":
      return { label: "cert pending", tone: "warn" };
    case "expired":
      return { label: "cert expired", tone: "danger" };
    case "disabled":
      return { label: "ACME off", tone: "muted" };
    default:
      return null;
  }
}

/**
 * A one-line provenance label for a service's `config['source']`:
 * "git · main · a3f2c1" / "zip upload" / "template · fastapi-ai-chat". Null when
 * no provenance was recorded (older/undeployed rows). Shared by the project
 * grid and the project detail Overview so the two never diverge.
 */
export function sourceLine(source: ServiceSource | null | undefined): string | null {
  if (!source) return null;
  if (source.template_id) return `template · ${source.template_id}`;
  if (source.type === "git") {
    const parts = ["git"];
    if (source.ref) parts.push(source.ref);
    if (source.commit_sha) parts.push(source.commit_sha.slice(0, 7));
    return parts.join(" · ");
  }
  if (source.type === "zip") return "zip upload";
  return source.type;
}
