import { useEffect, useId, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useAudit } from "../api/queries";
import { useAuditStream } from "../api/auditStream";
import { apiUrl } from "../api/client";
import {
  Badge,
  Button,
  Field,
  Input,
  Mono,
  PageHeader,
  Panel,
  Select,
  TableRow,
  type BadgeTone
} from "../components/ui";
import { clearStoredToken, getStoredToken, isDeadBearer } from "../lib/auth";
import type { AuditLogEntry } from "../api/types";

// The Activity page (the audit record, in the product's words).
//
// Live-primary with a polling fallback. The admin activity stream authenticates
// via the fetch-stream (lib/sse.ts), so live rows prepend as they happen; GET
// /audit polling stays the fallback and backfills history. Component and file
// keep their `Audit` names: the route and the API are frozen contracts, only
// the words on screen change.

/** Known `result` values on activity rows; anything else renders neutrally. */
const RESULT_TONES: Record<string, BadgeTone> = {
  ok: "success",
  error: "destructive",
  denied: "warning",
  replay: "warning"
};

const RESULT_OPTIONS: { value: string; label: string }[] = [
  { value: "", label: "All results" },
  { value: "ok", label: "OK" },
  { value: "error", label: "Error" },
  { value: "denied", label: "Denied" },
  { value: "replay", label: "Replay" }
];

const COLUMNS = ["Time", "Principal", "Action", "Target", "Result", "Request"];

const FIRST_PAGE_POLL_MS = 10_000;

type AuditAccess = "ok" | "forbidden";

/**
 * One-shot admin-access probe. GET /audit is admin-only; a non-admin token
 * gets a 403 which we surface as *data* ("forbidden"), not a thrown error —
 * so react-query never retries and the page renders a friendly empty state
 * instead of an error-toast loop. The real query only mounts once this probe
 * says "ok".
 *
 * A dead bearer takes the other branch (`isDeadBearer`, shared with
 * `pages/Tokens.tsx`): the daemon answers an unresolvable token with 403
 * `invalid_token`, so reading the status alone left a revoked token wedged on
 * this page — "Admin token required", with /login bounced back by the truthy
 * stale token.
 */
async function probeAuditAccess(): Promise<AuditAccess> {
  const token = getStoredToken();
  const response = await fetch(apiUrl("/audit?limit=1"), {
    headers: token ? { Authorization: `Bearer ${token}` } : {}
  });
  if (response.ok) return "ok";

  // Parse the P1 envelope once: its `code` is the only thing that separates a
  // role refusal from a dead token, and it also carries the human message.
  const body = (await response.json().catch(() => null)) as {
    code?: unknown;
    detail?: unknown;
  } | null;
  const code = typeof body?.code === "string" ? body.code : null;

  if (isDeadBearer(response.status, code)) {
    // Mirror api/client.ts: stale token, back to login.
    clearStoredToken();
    if (typeof window !== "undefined") window.location.assign("/login");
    return "forbidden";
  }
  if (response.status === 403) return "forbidden";

  const detail = body?.detail;
  throw new Error(typeof detail === "string" ? detail : response.statusText);
}

function ResultBadge({ result }: { result: string | null }) {
  const label = result ?? "unknown";
  return <Badge tone={RESULT_TONES[label] ?? "muted"}>{label}</Badge>;
}

/** Fields an activity row renders — shared by polled entries and live rows. */
type AuditDisplayRow = Pick<
  AuditLogEntry,
  | "ts"
  | "principal_id"
  | "principal_role"
  | "action"
  | "target_id"
  | "target_type"
  | "result"
  | "status_code"
  | "request_id"
>;

/** Quiet placeholder rows in the table's own shape (guidelines §7). */
function PlaceholderRows({ rows = 4 }: { rows?: number }) {
  return (
    <>
      {Array.from({ length: rows }).map((_, row) => (
        <tr key={row} className="border-b border-border">
          {COLUMNS.map((column) => (
            <td key={column} className="px-4 py-4">
              <span className="block h-3 w-full max-w-[8rem] rounded-button bg-surface-hover" />
            </td>
          ))}
        </tr>
      ))}
    </>
  );
}

function AuditRow({ entry, live = false }: { entry: AuditDisplayRow; live?: boolean }) {
  return (
    <TableRow className={live ? "bg-primary-subtle" : undefined}>
      <td className="whitespace-nowrap px-4 py-3 text-muted-foreground">
        {entry.ts ? new Date(entry.ts).toLocaleString() : "–"}
      </td>
      <td className="max-w-[12rem] px-4 py-3">
        <Mono className="block truncate text-foreground" title={entry.principal_id ?? undefined}>
          {entry.principal_id ?? "–"}
        </Mono>
        <span className="block truncate text-12 text-subtle-foreground">
          {entry.principal_role ?? "–"}
        </span>
      </td>
      <td className="max-w-[14rem] px-4 py-3">
        {/*
          Raw action strings ("deploy.git_create") are machine-shaped, and they
          ARE the record: this surface reports what the daemon logged, so they
          stay verbatim in Mono rather than being paraphrased into prose.
        */}
        <Mono className="block truncate text-foreground" title={entry.action}>
          {entry.action}
        </Mono>
      </td>
      <td className="max-w-[12rem] px-4 py-3">
        <Mono className="block truncate text-muted-foreground" title={entry.target_id ?? undefined}>
          {entry.target_id ?? "–"}
        </Mono>
        <span className="block truncate text-12 text-subtle-foreground">
          {entry.target_type ?? "–"}
        </span>
      </td>
      <td className="whitespace-nowrap px-4 py-3">
        <span className="flex items-center gap-2">
          <ResultBadge result={entry.result} />
          <span className="text-12 text-subtle-foreground">{entry.status_code ?? ""}</span>
        </span>
      </td>
      <td className="max-w-[10rem] px-4 py-3">
        <Mono
          className="block truncate text-subtle-foreground"
          title={entry.request_id ?? undefined}
        >
          {entry.request_id ?? "–"}
        </Mono>
      </td>
    </TableRow>
  );
}

/** Filters + cursor-paginated table; only mounted once the access probe passed. */
function ActivityTable() {
  // The server's `action` query param is an *exact* match (queries.list_audit_log),
  // so the substring filter is applied client-side over the loaded pages; the
  // `result` filter maps 1:1 to the server param.
  const [actionFilter, setActionFilter] = useState("");
  const [result, setResult] = useState("");
  const [live, setLive] = useState(true);
  const actionId = useId();
  const resultId = useId();

  const filters = useMemo(() => ({ result: result || undefined, limit: 50 }), [result]);
  const query = useAudit(filters);
  const { liveRows, connected } = useAuditStream(live);

  // Freshness fallback: refetch the loaded pages on an interval so history
  // stays current even when the live stream is down. refetch() on an infinite
  // query re-runs the loaded pages, so the newest entries land on page one.
  const { refetch } = query;
  useEffect(() => {
    const id = window.setInterval(() => {
      void refetch();
    }, FIRST_PAGE_POLL_MS);
    return () => window.clearInterval(id);
  }, [refetch]);

  const entries = useMemo(
    () => (query.data?.pages ?? []).flatMap((page) => page.items),
    [query.data]
  );

  // Request ids already present in the polled pages; live rows carrying one of
  // these are dropped so a polled backfill never doubles a live row.
  const polledRequestIds = useMemo(
    () => new Set(entries.map((e) => e.request_id).filter((r): r is string => Boolean(r))),
    [entries]
  );

  // Live rows first (newest), then the polled history.
  const rows = useMemo(() => {
    const merged: { key: string; row: AuditDisplayRow; live: boolean }[] = [];
    liveRows
      .filter(
        (r) =>
          (!result || r.result === result) && (!r.request_id || !polledRequestIds.has(r.request_id))
      )
      .forEach((r, i) =>
        merged.push({ key: `live-${r.request_id ?? `${r.ts}-${i}`}`, row: r, live: true })
      );
    entries.forEach((e) => merged.push({ key: `row-${e.id}`, row: e, live: false }));
    return merged;
  }, [liveRows, entries, polledRequestIds, result]);

  const needle = actionFilter.trim().toLowerCase();
  const visible = needle ? rows.filter((r) => r.row.action.toLowerCase().includes(needle)) : rows;
  const showEmpty = !query.isLoading && visible.length === 0;

  return (
    <div className="space-y-7">
      <Panel title="Filters">
        <div className="grid grid-cols-1 gap-4 p-4 md:grid-cols-2">
          <Field label="Action contains" htmlFor={actionId}>
            <Input
              id={actionId}
              type="text"
              mono
              value={actionFilter}
              onChange={(e) => setActionFilter(e.target.value)}
              placeholder="deploy, secrets.set"
            />
          </Field>
          <Field label="Result" htmlFor={resultId}>
            <Select id={resultId} value={result} onChange={(e) => setResult(e.target.value)}>
              {RESULT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </Select>
          </Field>
        </div>
      </Panel>

      <Panel
        title={`Entries (${visible.length}${query.hasNextPage ? "+" : ""})`}
        actions={
          <Button
            size="sm"
            variant="ghost"
            role="switch"
            aria-checked={live}
            onClick={() => setLive((v) => !v)}
          >
            <span
              className={`text-12 ${
                live && connected ? "text-primary" : "text-muted-foreground"
              }`}
            >
              {live ? (connected ? "Live" : "Connecting") : "Paused"}
            </span>
          </Button>
        }
      >
        {query.error && (
          <div className="border-b border-border px-4 py-3 text-14 text-destructive">
            {(query.error as Error).message}
          </div>
        )}

        {showEmpty ? (
          <div className="flex flex-col items-start gap-2 px-4 py-10">
            <p className="text-14 text-foreground">
              {rows.length === 0
                ? "Nothing recorded yet for these filters."
                : "No loaded entries match this action filter."}
            </p>
            <p className="text-14 text-muted-foreground">
              {rows.length === 0
                ? "Deploys, serves, secrets and stops all land here."
                : "The filter searches loaded pages only. Load more, or clear it."}
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[52rem] text-14">
              <thead>
                <tr className="border-b border-border">
                  {COLUMNS.map((column) => (
                    <th key={column} className="label px-4 py-2 text-left font-medium">
                      {column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {query.isLoading && <PlaceholderRows />}
                {visible.map((item) => (
                  <AuditRow key={item.key} entry={item.row} live={item.live} />
                ))}
              </tbody>
            </table>
          </div>
        )}

        {query.hasNextPage && (
          <div className="flex justify-center px-4 py-4">
            <Button
              onClick={() => query.fetchNextPage()}
              loading={query.isFetchingNextPage}
              disabled={query.isFetchingNextPage}
            >
              Load more
            </Button>
          </div>
        )}
      </Panel>
    </div>
  );
}

export default function Audit() {
  const access = useQuery({
    queryKey: ["audit", "access"],
    queryFn: probeAuditAccess,
    retry: false,
    staleTime: 60_000
  });

  return (
    <div className="mx-auto max-w-content space-y-7">
      <PageHeader
        title="Activity"
        subtitle="Append-only record of every change: who did it, what it touched, how it ended."
      />

      {access.isLoading && (
        <Panel title="Activity">
          <table className="w-full text-14">
            <tbody>
              <PlaceholderRows />
            </tbody>
          </table>
        </Panel>
      )}

      {access.data === "forbidden" && (
        <Panel title="Activity">
          <div className="flex flex-col items-start gap-2 px-4 py-10">
            <p className="text-14 text-foreground">Admin token required</p>
            <p className="text-14 text-muted-foreground">Sign in with an admin token.</p>
          </div>
        </Panel>
      )}

      {access.isError && (
        <Panel title="Activity">
          <div className="flex flex-col items-start gap-3 px-4 py-10">
            <p className="text-14 text-foreground">Could not reach the activity record.</p>
            <p className="text-14 text-muted-foreground">{(access.error as Error).message}</p>
            <Button onClick={() => void access.refetch()}>Retry</Button>
          </div>
        </Panel>
      )}

      {access.data === "ok" && <ActivityTable />}
    </div>
  );
}
