import { describe, expect, it } from "vitest";
import {
  certBadge,
  domainUrls,
  featuredDomainUrl,
  formatDuration,
  formatUptime,
  hostedShareUrl,
  publicUrlState,
  uptimeSeconds
} from "./format";
import type { PublicUrlEntry, ServiceEndpoint } from "../api/types";

const NOW = new Date("2026-07-04T12:00:00Z");

describe("uptimeSeconds", () => {
  it("returns null when the workload has not started", () => {
    expect(uptimeSeconds(null, NOW)).toBeNull();
    expect(uptimeSeconds(undefined, NOW)).toBeNull();
  });

  it("returns null on an unparsable timestamp", () => {
    expect(uptimeSeconds("not-a-date", NOW)).toBeNull();
  });

  it("derives whole seconds since started_at", () => {
    expect(uptimeSeconds("2026-07-04T11:59:18Z", NOW)).toBe(42);
    expect(uptimeSeconds("2026-07-04T12:00:00Z", NOW)).toBe(0);
  });

  it("clamps minor clock skew (started_at in the future) to 0", () => {
    expect(uptimeSeconds("2026-07-04T12:00:05Z", NOW)).toBe(0);
  });
});

describe("formatDuration / formatUptime", () => {
  it("renders a dash for missing or negative values", () => {
    expect(formatDuration(null)).toBe("–");
    expect(formatDuration(undefined)).toBe("–");
    expect(formatDuration(-1)).toBe("–");
    expect(formatUptime(null, NOW)).toBe("–");
  });

  it("scales through seconds, minutes, hours and days", () => {
    expect(formatDuration(42)).toBe("42s");
    expect(formatDuration(3 * 60 + 12)).toBe("3m 12s");
    expect(formatDuration(2 * 3600 + 5 * 60)).toBe("2h 05m");
    expect(formatDuration(4 * 86400 + 7 * 3600)).toBe("4d 7h");
  });

  it("derives the label from started_at", () => {
    expect(formatUptime("2026-07-04T11:56:48Z", NOW)).toBe("3m 12s");
  });
});

function endpoint(overrides: Partial<ServiceEndpoint> = {}): ServiceEndpoint {
  return {
    container_port: 8000,
    host_port: 38000,
    effective_host_port: 38000,
    protocol: "tcp",
    route: null,
    url: "http://127.0.0.1:38000",
    public_url: null,
    ...overrides
  };
}

describe("publicUrlState", () => {
  it("is 'none' when no endpoint is published yet", () => {
    expect(publicUrlState(null)).toBe("none");
    expect(publicUrlState(undefined)).toBe("none");
  });

  it("is 'proxy-off' when the endpoint exists but public_url is null (expected, not an error)", () => {
    expect(publicUrlState(endpoint())).toBe("proxy-off");
  });

  it("is 'routed' when public_url is resolved", () => {
    expect(
      publicUrlState(endpoint({ route: "/my-app", public_url: "https://host.example/my-app" }))
    ).toBe("routed");
  });

  it("treats an empty-string subdomain route with a resolved URL as routed", () => {
    // route === "" is a valid, routed state (subdomain mode) — never test truthiness.
    expect(
      publicUrlState(endpoint({ route: "", public_url: "https://my-app.host.example/" }))
    ).toBe("routed");
  });
});

const HOSTED_URL = "https://my-app--gpu-box.nodes.test/";

function hosted(overrides: Partial<PublicUrlEntry> = {}): PublicUrlEntry {
  return { url: HOSTED_URL, kind: "hosted", state: "ready", access: "private", ...overrides };
}

describe("hostedShareUrl", () => {
  it("is null with no endpoint at all", () => {
    expect(hostedShareUrl(null)).toBeNull();
    expect(hostedShareUrl(undefined)).toBeNull();
  });

  it("is null when the daemon predates public_urls (field absent)", () => {
    // Optional on purpose: a dashboard talking to a pre-P26 daemon must render.
    expect(hostedShareUrl(endpoint())).toBeNull();
  });

  it("is null when only the LAN (default-kind) URL is advertised", () => {
    expect(
      hostedShareUrl(
        endpoint({
          public_url: "https://host.example/my-app",
          public_urls: [
            { url: "https://host.example/my-app", kind: "default", state: "ready", access: null }
          ]
        })
      )
    ).toBeNull();
  });

  it("is null for a hosted share whose tunnel is down — a URL that does not answer", () => {
    expect(hostedShareUrl(endpoint({ public_urls: [hosted({ state: "link_down" })] }))).toBeNull();
  });

  it("is null for an unentitled public share (the URL exists, the plan does not)", () => {
    expect(
      hostedShareUrl(
        endpoint({ public_urls: [hosted({ state: "not_entitled", access: "public" })] })
      )
    ).toBeNull();
  });

  it("returns the ready hosted entry, preferring it over the LAN URL", () => {
    const entry = hostedShareUrl(
      endpoint({
        public_url: "https://host.example/my-app",
        public_urls: [
          { url: "https://host.example/my-app", kind: "default", state: "ready", access: null },
          hosted()
        ]
      })
    );
    expect(entry?.url).toBe(HOSTED_URL);
    expect(entry?.access).toBe("private");
  });

  it("ignores a ready hosted entry with no url (shared, then unlinked)", () => {
    expect(hostedShareUrl(endpoint({ public_urls: [hosted({ url: null })] }))).toBeNull();
  });
});

const DOMAIN_URL = "https://app.example.com/";

function domain(overrides: Partial<PublicUrlEntry> = {}): PublicUrlEntry {
  return {
    url: DOMAIN_URL,
    kind: "domain",
    state: "ready",
    access: null,
    domain: "app.example.com",
    ...overrides
  };
}

describe("domainUrls", () => {
  it("is empty with no endpoint at all", () => {
    expect(domainUrls(null)).toEqual([]);
    expect(domainUrls(undefined)).toEqual([]);
  });

  it("is empty when the daemon predates public_urls (field absent)", () => {
    // Optional on purpose: a dashboard talking to a pre-P26 daemon must render.
    expect(domainUrls(endpoint())).toEqual([]);
  });

  it("ignores the LAN and hosted entries", () => {
    expect(
      domainUrls(
        endpoint({
          public_url: "https://host.example/my-app",
          public_urls: [
            { url: "https://host.example/my-app", kind: "default", state: "ready", access: null },
            hosted()
          ]
        })
      )
    ).toEqual([]);
  });

  it("is empty for a withheld domain — stored intent whose URL does not answer", () => {
    expect(domainUrls(endpoint({ public_urls: [domain({ state: "withheld" })] }))).toEqual([]);
  });

  it("ignores a ready domain entry with no url", () => {
    expect(domainUrls(endpoint({ public_urls: [domain({ url: null })] }))).toEqual([]);
  });

  it("returns the ready domain entry with its name", () => {
    const entries = domainUrls(endpoint({ public_urls: [hosted(), domain()] }));
    expect(entries).toHaveLength(1);
    expect(entries[0]?.url).toBe(DOMAIN_URL);
    expect(entries[0]?.domain).toBe("app.example.com");
    expect(entries[0]?.access).toBeNull();
  });

  it("keeps every ready domain, in the daemon's order, dropping the withheld ones", () => {
    const second = domain({ url: "https://beta.example.com/", domain: "beta.example.com" });
    const third = domain({
      url: "https://gamma.example.com/",
      domain: "gamma.example.com",
      state: "withheld"
    });
    expect(
      domainUrls(endpoint({ public_urls: [domain(), second, third] })).map((e) => e.url)
    ).toEqual([DOMAIN_URL, "https://beta.example.com/"]);
  });

  it("still lists a ready domain whose certificate is pending", () => {
    // The list is the card's diagnostics: a bound name must stay visible with
    // its pill. Only `featuredDomainUrl` refuses to make it "the URL".
    expect(domainUrls(endpoint({ public_urls: [domain({ cert_state: "pending" })] }))).toHaveLength(
      1
    );
  });
});

describe("featuredDomainUrl", () => {
  it("is null with no endpoint and with no domains", () => {
    expect(featuredDomainUrl(null)).toBeNull();
    expect(featuredDomainUrl(undefined)).toBeNull();
    expect(featuredDomainUrl(endpoint({ public_urls: [hosted()] }))).toBeNull();
  });

  it("skips a ready domain whose public certificate is pending", () => {
    // (Codex round 2 #3835632984) `ready` is the ROUTE's verdict. With
    // `cert_state: pending` there is no leaf and the catch-all policy is not a
    // fallback for an ACME subject, so the handshake aborts (SSL alert 80) —
    // featuring the name would offer a copy button for a dead link.
    expect(featuredDomainUrl(endpoint({ public_urls: [domain({ cert_state: "pending" })] })))
      .toBeNull();
  });

  it("features the next ready domain that can actually complete a handshake", () => {
    const pending = domain({
      url: "https://pending.example.com/",
      domain: "pending.example.com",
      cert_state: "pending"
    });
    expect(featuredDomainUrl(endpoint({ public_urls: [pending, domain()] }))?.url).toBe(DOMAIN_URL);
  });

  it("features every other cert state — all of them answer HTTPS", () => {
    // `internal`/`disabled` are internal-CA names (openable after `nerdit
    // trust`), `expired` still serves the stale public leaf (a warning, not a
    // dead name), and an absent field is a pre-WP2 daemon.
    for (const state of ["issued", "internal", "disabled", "expired", null] as const) {
      expect(featuredDomainUrl(endpoint({ public_urls: [domain({ cert_state: state })] }))?.url)
        .toBe(DOMAIN_URL);
    }
    expect(featuredDomainUrl(endpoint({ public_urls: [domain()] }))?.url).toBe(DOMAIN_URL);
  });

  it("ignores a withheld domain exactly as domainUrls does", () => {
    expect(
      featuredDomainUrl(endpoint({ public_urls: [domain({ state: "withheld" })] }))
    ).toBeNull();
  });
});

describe("certBadge", () => {
  it("is null with no entry at all", () => {
    expect(certBadge(null)).toBeNull();
    expect(certBadge(undefined)).toBeNull();
  });

  it("is null when the daemon predates cert_state (field absent)", () => {
    // A pre-WP2 daemon sends no cert_state; the row still renders, badge-free.
    expect(certBadge(domain())).toBeNull();
  });

  it("is null for an internal-CA domain — the ordinary, documented case", () => {
    expect(certBadge(domain({ cert_state: "internal" }))).toBeNull();
    expect(certBadge(domain({ cert_state: null }))).toBeNull();
  });

  it("badges an issued public certificate", () => {
    expect(certBadge(domain({ cert_state: "issued" }))).toEqual({
      label: "public cert",
      tone: "ok"
    });
  });

  it("badges a pending certificate as a warning, not a failure", () => {
    // `pending` covers "in progress" AND "failing" — the daemon cannot tell
    // them apart, so the badge must not claim either.
    expect(certBadge(domain({ cert_state: "pending" }))).toEqual({
      label: "cert pending",
      tone: "warn"
    });
  });

  it("badges an expired certificate as a failure", () => {
    expect(certBadge(domain({ cert_state: "expired" }))).toEqual({
      label: "cert expired",
      tone: "danger"
    });
  });

  it("badges a row that wants ACME on a node where it is off", () => {
    expect(certBadge(domain({ cert_state: "disabled" }))).toEqual({
      label: "ACME off",
      tone: "muted"
    });
  });

  it("reads cert_state on any kind of entry, ignoring state", () => {
    // state is a route fact, cert_state a leaf fact: they never gate each other.
    expect(certBadge(domain({ state: "withheld", cert_state: "issued" }))?.tone).toBe("ok");
  });
});
