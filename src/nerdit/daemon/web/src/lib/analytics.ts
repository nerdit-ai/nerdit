// Product analytics (PostHog) for the dashboard — a thin wrapper over the
// posthog-js singleton (feat/posthog).
//
// Design:
// - **Inert by default.** Nothing loads or phones home until `initAnalytics`
//   runs with a project key, which arrives from the authenticated
//   `/cluster/info` read (see AppShell). Every `capture*` is a safe no-op
//   before init, so call sites (mutation hooks) never need to guard.
// - **Explicit events only.** DOM autocapture and session recording are off;
//   we emit manual page views + a small set of curated product events.
// - **No secrets ever.** We never pass the bearer token to PostHog (it is a
//   live credential). Identity stays on PostHog's anonymous device id, with
//   `role` / `hostname` / `daemon_version` registered as safe super-properties.
import posthog from "posthog-js";

let initialized = false;

export interface AnalyticsConfig {
  /** Publishable PostHog project key (phc_…) from /cluster/info. */
  key: string;
  /** PostHog ingest host (US/EU cloud, or a self-hosted origin). */
  host: string;
}

/**
 * Initialize PostHog exactly once. Safe to call repeatedly (subsequent calls
 * are ignored) and with a falsy key (stays inert). `superProps` are registered
 * as super-properties attached to every event — keep them low-cardinality and
 * non-sensitive (role, hostname, version).
 */
export function initAnalytics(
  config: AnalyticsConfig,
  superProps?: Record<string, unknown>
): void {
  if (initialized || !config.key) return;
  posthog.init(config.key, {
    api_host: config.host,
    // Explicit events only — this is an ops console, not a marketing site.
    autocapture: false,
    capture_pageview: false, // manual $pageview on router navigation (see below)
    capture_pageleave: true,
    disable_session_recording: true,
    // Plain-HTTP dashboards (http://<host>:9321) are not secure contexts; keep
    // persistence working there too.
    persistence: "localStorage+cookie"
  });
  initialized = true;
  if (superProps) posthog.register(superProps);
}

/** True once `initAnalytics` has run with a key. */
export function isAnalyticsEnabled(): boolean {
  return initialized;
}

/**
 * Register (or update) super-properties attached to every subsequent event.
 * Use for values that resolve after init (e.g. the caller's role). No-op before
 * init. Keep values low-cardinality and non-sensitive.
 */
export function registerProperties(props: Record<string, unknown>): void {
  if (!initialized) return;
  posthog.register(props);
}

/**
 * Normalize a router pathname to a low-cardinality page key: entity-id segments
 * collapse to their route pattern so PostHog groups `/services/foo` and
 * `/services/bar` as one page (the id rides along as a property).
 */
function normalizePath(pathname: string): string {
  if (pathname.startsWith("/services/")) return "/services/:ident";
  if (pathname.startsWith("/projects/")) {
    // Keep the (five-value) tab segment, drop the project name.
    const tab = pathname.split("/").filter(Boolean)[2];
    return tab ? `/projects/:name/${tab}` : "/projects/:name";
  }
  return pathname;
}

/** Emit a manual SPA page view for a router navigation. No-op before init. */
export function capturePageview(pathname: string): void {
  if (!initialized) return;
  const page = normalizePath(pathname);
  posthog.capture("$pageview", {
    // Report the normalized page as the URL so entity ids never inflate
    // cardinality; keep the raw path as a property for drill-down.
    $current_url: `${window.location.origin}${page}`,
    page,
    raw_path: pathname
  });
}

/** Emit a curated product event. No-op before init. */
export function capture(event: string, props?: Record<string, unknown>): void {
  if (!initialized) return;
  posthog.capture(event, props);
}

/** Clear the identity/super-properties on sign-out. No-op before init. */
export function resetAnalytics(): void {
  if (!initialized) return;
  posthog.reset();
}
