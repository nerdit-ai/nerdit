import { ExternalLink, Globe } from "lucide-react";
import { Badge, CopyField, DataRow, Mono, Panel } from "./ui";
import type { BadgeTone } from "./ui";
import {
  certBadge,
  domainUrls,
  featuredDomainUrl,
  hostedShareUrl,
  publicUrlState
} from "../lib/format";
import type { CertBadgeTone } from "../lib/format";
import type { PublicUrlEntry, ServiceEndpoint } from "../api/types";

/**
 * `lib/format` owns which certificate states are worth a badge and how heavy
 * each one reads; this map is only the translation into the primitive's tone
 * vocabulary, so the pill is a `ui.Badge` like every other pill on the surface
 * rather than page-local styling.
 */
const CERT_TONE: Record<CertBadgeTone, BadgeTone> = {
  ok: "success",
  warn: "warning",
  danger: "destructive",
  muted: "muted"
};

/**
 * The certificate pill beside a domain URL. Renders nothing for an
 * internal-CA name or a daemon that predates certificate reporting, so a node
 * that never asked for public certificates looks exactly as it did before.
 */
function CertPill({ entry }: { entry: PublicUrlEntry }) {
  const badge = certBadge(entry);
  if (!badge) return null;
  return (
    <Badge tone={CERT_TONE[badge.tone]} className="shrink-0">
      {badge.label}
    </Badge>
  );
}

/** The open-in-a-new-tab affordance beside a copyable address. */
function OpenLink({ url, label }: { url: string; label: string }) {
  return (
    <a
      href={url}
      target="_blank"
      rel="noreferrer"
      title={label}
      aria-label={label}
      className="shrink-0 rounded-button p-1 text-muted-foreground hover:text-foreground"
    >
      <ExternalLink size={16} aria-hidden="true" />
    </a>
  );
}

/**
 * An app's address (design guidelines §4: the address renders once, here).
 *
 * The common self-hosted case is ONE calm row: the URL as a `CopyField` and a
 * link that opens it. Everything else on this panel — certificate pills, the
 * list of bound domains, the `Public` state — mounts only when a hosted share
 * or a direct domain actually exists on the row, so a node that never published
 * anything never reads about publishing.
 *
 * Which URL is featured is `lib/format`'s ruling, not this component's:
 *
 * A `ready` hosted share OUTRANKS the LAN URL and replaces it here: it works
 * from anywhere, needs no certificate install, and it is the one a remote agent
 * can open. The LAN copy is dropped in that case rather than shown beside it —
 * two "the URL" strings in one place is how a reader ends up pasting the wrong
 * one.
 *
 * A direct domain outranks both: it is the name the operator chose and the one
 * a human will recognise. The remaining domains are listed underneath — they
 * are all really bound, and hiding them would make the panel lie about what
 * this app answers to.
 *
 * "Featured" demands one fact more than `state === "ready"`: a domain whose
 * `cert_state` is `pending` has a live route and no leaf, and aborts the
 * handshake rather than falling back to the internal CA — so
 * `featuredDomainUrl` skips it and we fall through to the hosted or LAN URL,
 * which do answer. The pending name still appears in the list below with its
 * pill; it is just never the string the reader copies.
 *
 * Each domain row carries a certificate pill when — and only when — the answer
 * differs from the documented default: an issued public leaf, one still
 * pending, an expired one, or a row that asked for ACME on a node where it is
 * off. The internal-CA case stays badge-free, and the caption says which trust
 * step (if any) a visitor still owes. The pill reads the daemon's `cert_state`;
 * nothing about the certificate is derived here.
 *
 * `Public` is a fact about the app's exposure, not about the featured string,
 * so it renders whenever a ready hosted share is public — even when a domain is
 * the address on show. Private shares render nothing: private is the default
 * and the default is silence.
 */
export function ServiceShareCard({ endpoint }: { endpoint: ServiceEndpoint | null }) {
  const featured = featuredDomainUrl(endpoint);
  const extraDomains = domainUrls(endpoint).filter((e) => e !== featured);
  const pendingDomain = extraDomains.find((e) => e.cert_state === "pending") ?? null;
  const hosted = hostedShareUrl(endpoint);
  const url = featured?.url ?? hosted?.url ?? endpoint?.public_url;
  if (!featured && !hosted && (publicUrlState(endpoint) !== "routed" || !endpoint?.public_url)) {
    return null;
  }
  if (!url) return null;

  const isPublic = hosted?.access === "public";

  return (
    <Panel data-testid="service-address" className="divide-y divide-border">
      <DataRow label="Address">
        <div className="flex min-w-0 flex-wrap items-center gap-2 sm:justify-end">
          <CopyField value={url} data-testid="service-address-url" className="min-w-0" />
          <OpenLink url={url} label="Open in a new tab" />
          {featured ? <CertPill entry={featured} /> : null}
          {isPublic ? (
            <Badge tone="primary" data-testid="service-public" className="shrink-0">
              <Globe size={12} aria-hidden="true" />
              Public
            </Badge>
          ) : null}
        </div>
      </DataRow>

      {extraDomains.length > 0 ? (
        <DataRow label="Domains">
          <ul className="flex min-w-0 flex-col gap-2 sm:items-end" data-testid="service-domains">
            {extraDomains.map((entry) => (
              <li key={entry.url} className="flex min-w-0 flex-wrap items-center gap-2 sm:justify-end">
                <CopyField value={entry.url ?? ""} className="min-w-0" />
                <OpenLink
                  url={entry.url ?? ""}
                  label={`Open ${entry.domain ?? entry.url} in a new tab`}
                />
                <CertPill entry={entry} />
              </li>
            ))}
          </ul>
        </DataRow>
      ) : null}

      {pendingDomain ? (
        <p className="px-4 py-3 text-12 text-muted-foreground" data-testid="service-cert-pending">
          <Mono className="text-foreground">{pendingDomain.domain}</Mono> is bound, but its public
          certificate is pending, so the name does not answer HTTPS at all. Running{" "}
          <Mono className="text-foreground">nerdit trust</Mono> will not help: check DNS and the{" "}
          <Mono className="text-foreground">acme_http_port</Mono> doctor row.
        </p>
      ) : null}

      <p className="px-4 py-3 text-12 text-muted-foreground" data-testid="service-address-note">
        {!featured && !hosted ? (
          // The plain LAN address is served by this node's internal CA, so a
          // visitor's browser warns until the root is trusted. That is
          // operational truth, not decoration: it is the single most common
          // "why is this insecure?" on a fresh install, and the domain
          // branches below already explain the same step. Leaving it out here
          // would explain the trust step everywhere except the one address
          // most nodes actually serve.
          <>
            Served by this node over its own certificate authority. Visitors trust it once by
            running <Mono className="text-foreground">nerdit trust</Mono>.
          </>
        ) : featured?.cert_state === "issued" ? (
          "Your own domain with a public certificate. Visitors need no trust step."
        ) : featured?.cert_state === "expired" ? (
          <>
            Your own domain, served by this node. Its public certificate has expired and renewal has
            not landed yet, so visitors see a browser warning;{" "}
            <Mono className="text-foreground">nerdit trust</Mono> does not cover a public
            certificate.
          </>
        ) : featured ? (
          <>
            Your own domain, served by this node. Visitors trust the certificate once by running{" "}
            <Mono className="text-foreground">nerdit trust</Mono>.
          </>
        ) : (
          <>
            Hosted through your Nerdit account, so it works from anywhere with no certificate to
            trust.{" "}
            {isPublic
              ? "Anyone with the link can open it."
              : "It opens for signed in owners of this node."}
          </>
        )}
      </p>
    </Panel>
  );
}
