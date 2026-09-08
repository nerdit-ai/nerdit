import { useState } from "react";
import { Button, Mono, Panel, TableRow } from "../ui";
import { useRoutes } from "../../api/queries";
import type { RouteItem } from "../../api/types";

// The DB-authoritative route inventory (`GET /routes`, P13b) as its own Settings
// card (D-P23-7). It replaces — not supplements — the bare route count that used
// to sit in `ProxyStatusCard`: one place per fact.
//
// Two contract details are load-bearing and mirror `cli/commands/routes.py`
// exactly, so the two surfaces can never disagree:
//
//   * `route` is a locked tri-state (Invariant #2). `null` is "unrouted" (proxy
//     off, or a `kind=model` row — models are loopback-only by design), `""` is
//     a subdomain route, anything else is the literal path. Tested with `===`,
//     NEVER truthiness: an empty string is a real route, not a missing one.
//   * `live` is advisory. `null` means the live Caddy table could not be read,
//     which renders "–" (unknown) and never "missing"; the page-level
//     `live_table` tri-state is shown as a caption so the dash is explained.
//
// The first column header is "Name", not "Service": the table lists models and
// databases too, and neither is an app (design guidelines §3).
//
// Paging is a local cursor plus an accumulator — ONE query hook, one page in
// flight. An operator card of a handful of rows does not need infinite-query
// machinery, and a hook-per-cursor would break the rules of hooks.

/** The locked route tri-state. Order matters: `null` before `""`. */
function RouteCell({ route }: { route: string | null }) {
  if (route === null) return <span className="text-subtle-foreground">unrouted</span>;
  if (route === "") return <span className="text-muted-foreground">(subdomain)</span>;
  return <Mono className="text-foreground">{route}</Mono>;
}

/** The advisory live annotation; `null` ⇒ "–" (unknown), never "no". */
function LiveCell({ live }: { live: RouteItem["live"] }) {
  if (live === null) return <span className="text-subtle-foreground">–</span>;
  if (!live.registered) return <span className="text-warning-foreground">missing</span>;
  if (!live.dial_matches) return <span className="text-warning-foreground">stale dial</span>;
  return <span className="text-muted-foreground">ok</span>;
}

const LIVE_TABLE_CAPTION: Record<string, string> = {
  readable: "Live proxy table readable: the live column reflects the running proxy.",
  unreadable: "Live proxy table unreadable: the live column shows – (unknown), not absent.",
  disabled: "Proxy off, so there is no live table to read: the live column shows – (unknown)."
};

const HEADERS = ["Name", "Kind", "Status", "Route", "Address", "Port", "Live"];

export function RoutesCard() {
  const [cursor, setCursor] = useState<string | undefined>(undefined);
  // Pages already walked past, kept so "Load more" appends rather than replaces.
  const [earlier, setEarlier] = useState<RouteItem[]>([]);
  const page = useRoutes(cursor);

  const items = [...earlier, ...(page.data?.items ?? [])];
  const nextCursor = page.data?.next_cursor ?? null;
  const liveTable = page.data?.live_table;

  function loadMore() {
    if (nextCursor === null) return;
    setEarlier(items);
    setCursor(nextCursor);
  }

  if (page.isError) {
    return (
      <Panel title="Routes">
        <p className="px-4 py-3 text-14 text-destructive">
          Could not load routes. {(page.error as Error).message}
        </p>
      </Panel>
    );
  }

  return (
    <Panel title="Routes" data-testid="routes-card">
      {page.isLoading && items.length === 0 && (
        <div className="space-y-2 px-4 py-3" aria-hidden="true">
          <div className="h-4 w-72 rounded-button bg-surface-hover" />
          <div className="h-4 w-56 rounded-button bg-surface-hover" />
        </div>
      )}

      {page.data && items.length === 0 && (
        <p className="px-4 py-3 text-14 text-muted-foreground">No endpoints registered yet.</p>
      )}

      {items.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[46rem] border-collapse text-14">
            <thead>
              <tr className="border-b border-border">
                {HEADERS.map((header) => (
                  <th
                    key={header}
                    scope="col"
                    className={`label px-4 py-2 ${header === "Port" ? "text-right" : "text-left"}`}
                  >
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <TableRow key={`${item.service_name}:${item.host_port}`}>
                  <td className="max-w-[14rem] truncate px-4 py-3">
                    <span className="text-foreground" title={item.service_name}>
                      {item.service_name}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">{item.kind}</td>
                  <td className="px-4 py-3 text-muted-foreground">{item.status}</td>
                  <td className="max-w-[12rem] truncate px-4 py-3">
                    <RouteCell route={item.route} />
                  </td>
                  <td className="max-w-[18rem] truncate px-4 py-3">
                    {item.public_url !== null ? (
                      <a
                        href={item.public_url}
                        target="_blank"
                        rel="noreferrer"
                        className="font-mono text-13 text-primary hover:underline"
                      >
                        {item.public_url}
                      </a>
                    ) : (
                      <span className="text-subtle-foreground">–</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <Mono className="text-muted-foreground">{item.host_port}</Mono>
                  </td>
                  <td className="px-4 py-3">
                    <LiveCell live={item.live} />
                  </td>
                </TableRow>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {liveTable && (
        <p
          data-testid="routes-live-table"
          className="border-t border-border px-4 py-3 text-12 text-subtle-foreground"
        >
          {LIVE_TABLE_CAPTION[liveTable] ?? `Live table: ${liveTable}`}
        </p>
      )}

      {nextCursor !== null && (
        <div className="flex justify-end border-t border-border px-4 py-3">
          <Button size="sm" onClick={loadMore} disabled={page.isFetching}>
            Load more
          </Button>
        </div>
      )}
    </Panel>
  );
}
