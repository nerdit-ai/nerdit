// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { ServiceShareCard } from "./ServiceShareCard";
import type { PublicUrlEntry, ServiceEndpoint } from "../api/types";

/**
 * (Codex round 2, #3835632984) The panel's one job is to hand the reader a URL
 * they can open. A `ready` domain whose `cert_state` is `pending` has a live
 * Host route and no leaf — it aborts the handshake instead of falling back to
 * the internal CA — so it must never be the featured string, even though the
 * daemon calls its route `ready`. The panel used to feature exactly that, with
 * a copy button and an open link, directly above its own caption saying the
 * name does not answer HTTPS at all.
 *
 * (W4) It is now the app's ADDRESS row: the plain self-hosted case is the URL
 * and nothing else, and the hosted/domain vocabulary mounts only when a share
 * or a domain really exists on the row.
 */
const LAN_URL = "https://box.local/my-app";
const HOSTED_URL = "https://my-app--gpu-box.nodes.nerdit.ai/";

function endpoint(public_urls: PublicUrlEntry[]): ServiceEndpoint {
  return {
    container_port: 8000,
    host_port: 9400,
    effective_host_port: 9400,
    protocol: "http",
    route: "/my-app",
    url: "http://127.0.0.1:9400",
    public_url: LAN_URL,
    public_urls
  } as ServiceEndpoint;
}

function domain(overrides: Partial<PublicUrlEntry> = {}): PublicUrlEntry {
  return {
    url: "https://app.example.com/",
    kind: "domain",
    state: "ready",
    access: null,
    domain: "app.example.com",
    ...overrides
  };
}

const hosted: PublicUrlEntry = {
  url: HOSTED_URL,
  kind: "hosted",
  state: "ready",
  access: "private"
};

/** The featured address — the string that is copied and opened. */
function featuredUrl(): string | null {
  return screen.getByTestId("service-address-url").textContent;
}

describe("ServiceShareCard", () => {
  afterEach(cleanup);

  it("features a ready domain whose certificate can complete a handshake", () => {
    render(<ServiceShareCard endpoint={endpoint([hosted, domain({ cert_state: "issued" })])} />);
    expect(featuredUrl()).toContain("https://app.example.com/");
    expect(screen.getByRole("link", { name: "Open in a new tab" }).getAttribute("href")).toBe(
      "https://app.example.com/"
    );
  });

  it("falls through to the hosted URL when the only domain's cert is pending", () => {
    render(<ServiceShareCard endpoint={endpoint([hosted, domain({ cert_state: "pending" })])} />);

    // The featured string — copied and opened — is the one that answers.
    expect(featuredUrl()).toContain(HOSTED_URL);
    expect(screen.getByRole("link", { name: "Open in a new tab" }).getAttribute("href")).toBe(
      HOSTED_URL
    );
    // ...and the pending name is still shown, so the panel does not hide what is
    // bound; it just is not offered as "your URL".
    expect(screen.getByText("https://app.example.com/")).toBeTruthy();
    expect(screen.getByText("cert pending")).toBeTruthy();
    expect(screen.getByTestId("service-cert-pending").textContent).toMatch(
      /does not answer HTTPS at all/
    );
  });

  it("falls all the way through to the LAN URL when there is no hosted share", () => {
    render(<ServiceShareCard endpoint={endpoint([domain({ cert_state: "pending" })])} />);
    expect(featuredUrl()).toContain(LAN_URL);
    expect(screen.getByRole("link", { name: "Open in a new tab" }).getAttribute("href")).toBe(
      LAN_URL
    );
  });

  it("lists every bound domain that is not the featured one, with its cert word", () => {
    render(
      <ServiceShareCard
        endpoint={endpoint([
          domain({ cert_state: "issued" }),
          domain({ url: "https://alt.example.com/", domain: "alt.example.com", cert_state: "expired" })
        ])}
      />
    );
    const domains = screen.getByTestId("service-domains");
    expect(domains.textContent).toContain("https://alt.example.com/");
    expect(domains.textContent).toContain("cert expired");
    // The featured one is the address row, not a list entry: one place per fact.
    expect(domains.textContent).not.toContain("https://app.example.com/");
  });

  it("renders the address plus the internal-CA trust step when no share and no domain exist", () => {
    render(<ServiceShareCard endpoint={endpoint([])} />);

    expect(featuredUrl()).toContain(LAN_URL);
    // None of the hosted/domain vocabulary: private is the default, and the
    // default is silence.
    expect(screen.queryByTestId("service-public")).toBeNull();
    expect(screen.queryByTestId("service-domains")).toBeNull();
    expect(screen.queryByTestId("service-cert-pending")).toBeNull();
    // The trust step is NOT decoration: this URL is served by the node's own
    // CA, so a visitor's browser warns until `nerdit trust` has run. W4 keeps
    // it as the one caption on the plain LAN address.
    expect(screen.getByTestId("service-address-note").textContent).toContain("nerdit trust");
  });

  it("says Public only for a public hosted share", () => {
    render(<ServiceShareCard endpoint={endpoint([hosted])} />);
    expect(screen.queryByTestId("service-public")).toBeNull();
    expect(screen.getByTestId("service-address-note").textContent).toMatch(
      /opens for signed in owners/i
    );

    cleanup();
    render(<ServiceShareCard endpoint={endpoint([{ ...hosted, access: "public" }])} />);
    expect(screen.getByTestId("service-public").textContent).toContain("Public");
    expect(screen.getByTestId("service-address-note").textContent).toMatch(
      /anyone with the link can open it/i
    );
  });

  it("renders nothing when the service is not routed and nothing is published", () => {
    const { container } = render(
      <ServiceShareCard endpoint={{ ...endpoint([]), public_url: null }} />
    );
    expect(container.firstChild).toBeNull();
  });
});
