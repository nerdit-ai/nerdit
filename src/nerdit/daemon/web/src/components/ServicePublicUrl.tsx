import { ExternalLink } from "lucide-react";
import { Badge, CopyField, Mono } from "./ui";
import { publicUrlState } from "../lib/format";
import type { ServiceEndpoint } from "../api/types";

/**
 * A service endpoint's address, compact enough for a table cell. The app page's
 * own address row is `ServiceShareCard`, which owns the hosted-share and domain
 * machinery; this one only ever renders the proxy URL:
 * - routed  → clickable HTTPS public_url (new tab); with `copyable`, the
 *   `CopyField` form instead (the shareable-link affordance, P9)
 * - proxy-off → "Proxy off" badge + the always-usable loopback url (expected
 *   state when [proxy] is disabled — never an error; trap 8)
 * - none    → dash (no endpoint published yet)
 */
export function ServicePublicUrl({
  endpoint,
  copyable = false
}: {
  endpoint: ServiceEndpoint | null;
  copyable?: boolean;
}) {
  const state = publicUrlState(endpoint);

  if (state === "none" || !endpoint) {
    return <span className="text-muted-foreground">–</span>;
  }

  if (state === "proxy-off") {
    return (
      <span className="inline-flex min-w-0 items-center gap-2" data-testid="public-url-proxy-off">
        <Badge tone="muted">Proxy off</Badge>
        {endpoint.url && <Mono className="truncate text-muted-foreground">{endpoint.url}</Mono>}
      </span>
    );
  }

  if (copyable && endpoint.public_url) {
    return (
      <span
        className="inline-flex min-w-0 items-center gap-2"
        data-testid="public-url"
        onClick={(event) => event.stopPropagation()}
      >
        <CopyField value={endpoint.public_url} className="min-w-0" />
        <a
          href={endpoint.public_url}
          target="_blank"
          rel="noreferrer"
          title="Open in a new tab"
          aria-label="Open in a new tab"
          className="shrink-0 rounded-button p-1 text-muted-foreground hover:text-foreground"
        >
          <ExternalLink size={16} aria-hidden="true" />
        </a>
      </span>
    );
  }

  return (
    <span className="inline-flex min-w-0 items-center gap-2" data-testid="public-url">
      <a
        href={endpoint.public_url ?? undefined}
        target="_blank"
        rel="noreferrer"
        onClick={(event) => event.stopPropagation()}
        className="inline-flex min-w-0 items-center gap-1 font-mono text-13 text-primary hover:underline"
      >
        <span className="truncate">{endpoint.public_url}</span>
        <ExternalLink size={12} className="shrink-0" aria-hidden="true" />
      </a>
    </span>
  );
}
