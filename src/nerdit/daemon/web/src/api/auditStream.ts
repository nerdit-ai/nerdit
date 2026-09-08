import { useEffect, useState } from "react";
import { streamEvents } from "../lib/sse";

/**
 * Live audit view. Wraps the authenticated fetch-stream (`lib/sse.ts`)
 * on `GET /api/audit/stream`, so the admin-gated SSE stream now authenticates
 * from the dashboard instead of 401-flooding.
 *
 * The stream payload is the raw published event dict (audit.py) — `{type, ts,
 * action, result, status_code, principal_id, principal_role, target_type,
 * target_id, params, request_id}`, NOT an `AuditLogEntry` (there is no `id`).
 * Rows are deduped by `request_id` and merged with the polled pages by the
 * caller. On 401/403 the stream stops permanently (no retry); on transport
 * errors `lib/sse.ts` backs off up to 30 s. Polling stays the fallback.
 */

/** Cap the live buffer so a long-lived tab never grows unbounded. */
const MAX_LIVE_ROWS = 100;

/** A live audit row parsed from the stream (no `id`, unlike `AuditLogEntry`). */
interface AuditStreamRow {
  ts: string | null;
  action: string;
  result: string | null;
  status_code: number | null;
  principal_id: string | null;
  principal_role: string | null;
  target_type: string | null;
  target_id: string | null;
  request_id: string | null;
}

export interface AuditStreamState {
  /** Newest first; deduped by `request_id`. */
  liveRows: AuditStreamRow[];
  /** True while the stream is open. */
  connected: boolean;
}

/** Map a parsed event payload to a row, or null when it lacks an action. */
function toRow(data: unknown): AuditStreamRow | null {
  if (!data || typeof data !== "object") return null;
  const e = data as Record<string, unknown>;
  if (typeof e.action !== "string") return null;
  return {
    ts: typeof e.ts === "string" ? e.ts : null,
    action: e.action,
    result: typeof e.result === "string" ? e.result : null,
    status_code: typeof e.status_code === "number" ? e.status_code : null,
    principal_id: typeof e.principal_id === "string" ? e.principal_id : null,
    principal_role: typeof e.principal_role === "string" ? e.principal_role : null,
    target_type: typeof e.target_type === "string" ? e.target_type : null,
    target_id: typeof e.target_id === "string" ? e.target_id : null,
    request_id: typeof e.request_id === "string" ? e.request_id : null
  };
}

/**
 * Subscribe to the live audit stream while `enabled`. Returns the buffered live
 * rows plus the connection flag. Aborts on unmount or when disabled.
 */
export function useAuditStream(enabled: boolean): AuditStreamState {
  const [state, setState] = useState<AuditStreamState>({ liveRows: [], connected: false });

  useEffect(() => {
    if (!enabled) {
      setState((prev) => (prev.connected ? { ...prev, connected: false } : prev));
      return;
    }
    let cancelled = false;
    const controller = new AbortController();

    void streamEvents("/audit/stream", {
      signal: controller.signal,
      onOpen: () => {
        if (!cancelled) setState((prev) => ({ ...prev, connected: true }));
      },
      onError: () => {
        if (!cancelled) setState((prev) => ({ ...prev, connected: false }));
      },
      onEvent: (event) => {
        if (cancelled || !event.event.startsWith("audit.")) return;
        let data: unknown;
        try {
          data = JSON.parse(event.data);
        } catch {
          return; // ignore malformed payloads
        }
        const row = toRow(data);
        if (!row) return;
        setState((prev) => {
          if (row.request_id && prev.liveRows.some((r) => r.request_id === row.request_id)) {
            return prev; // dedup reconnect replays by request_id
          }
          return { connected: true, liveRows: [row, ...prev.liveRows].slice(0, MAX_LIVE_ROWS) };
        });
      }
    }).then(() => {
      // Terminal (aborted / forbidden): the stream is no longer connected.
      if (!cancelled) setState((prev) => (prev.connected ? { ...prev, connected: false } : prev));
    });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [enabled]);

  return state;
}
