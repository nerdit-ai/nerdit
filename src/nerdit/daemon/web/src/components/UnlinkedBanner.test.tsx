// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { UnlinkedBanner } from "./UnlinkedBanner";
import type { Capabilities, CapabilitiesLink } from "../api/types";

/**
 * (P34 D4) The banner is a rendering decision over one field, and every case
 * below is a way that decision can go wrong in a user's face: a flash on a cold
 * load, a "you are not linked" shown to a linked node whose tunnel is simply
 * reconnecting, or a second fixed strip stacked on the daemon-offline one.
 *
 * The two hooks are mocked rather than driven through a `QueryClientProvider`:
 * the component makes no request of its own, it reads two shared queries, and
 * the states worth pinning here (in flight, success-but-offline) are states of
 * those queries, not of the network.
 */
const daemonStatus = vi.fn(() => ({ status: "online", lastSeen: new Date() }));
const capabilities = vi.fn();

vi.mock("../api/queries", () => ({
  useDaemonStatus: () => daemonStatus(),
  useCapabilities: () => capabilities()
}));

/** A capabilities response carrying only the block under test. */
function caps(link?: CapabilitiesLink) {
  return { isSuccess: true, data: { link } as unknown as Capabilities };
}

const BANNER = /not linked to a Nerdit account/;

describe("UnlinkedBanner", () => {
  afterEach(() => {
    cleanup();
    daemonStatus.mockReturnValue({ status: "online", lastSeen: new Date() });
  });

  it("stays silent on a node that linked once and then disabled the feature", () => {
    // (P34) The bug this pins: `enabled` is the TUNNEL, `node_id` is the
    // ENROLMENT. Reading the first as the second put a permanent,
    // non-dismissable warning on top of a deliberate opt-out — and on top of a
    // linked node whose identity key had merely become unreadable, which the
    // daemon reports the same way.
    capabilities.mockReturnValue(
      caps({ enabled: false, node_id: "5f0b7c1e", slug: "gpu-box" })
    );
    render(<UnlinkedBanner />);

    expect(screen.queryByRole("status")).toBeNull();
  });

  it("shows on a node that has genuinely never linked", () => {
    // Same `enabled: false`, no enrolment — the case the banner is FOR.
    capabilities.mockReturnValue(caps({ enabled: false, node_id: null, slug: null }));
    render(<UnlinkedBanner />);

    expect(screen.getByText(BANNER)).toBeTruthy();
  });

  it("shows on a node whose link block reports the feature off", () => {
    // An older daemon omits the enrolment keys entirely; absent reads as
    // unlinked, which is the pre-P34 behaviour and the safe default here.
    capabilities.mockReturnValue(caps({ enabled: false }));
    render(<UnlinkedBanner />);

    expect(screen.getByRole("status")).toBeTruthy();
    expect(screen.getByTestId("unlinked-banner")).toBeTruthy();
    expect(screen.getByText(BANNER)).toBeTruthy();
  });

  it("names the resume command and never implies the local plane is degraded", () => {
    // D-ENT-2 rendered as UI: the banner may say the cloud half is off, and
    // must not let a reader infer that deploys, the proxy or models are. The
    // "linking is optional" clause is the design guidelines' §4 wording of the
    // same guarantee, stated about the operator's choice rather than the box.
    capabilities.mockReturnValue(caps({ enabled: false }));
    render(<UnlinkedBanner />);

    const text = screen.getByRole("status").textContent ?? "";
    expect(text).toContain("nerdit link --device");
    expect(text).toContain("Linking is optional");
    expect(text).toContain("everything local works without it");
    expect(text).not.toMatch(/degraded|unavailable|broken/i);
    // Repo convention, pinned for every route by tests/e2e/07-audit.spec.ts.
    expect(text).not.toContain("—");
  });

  it("shows when the daemon predates the link surface and omits the block", () => {
    capabilities.mockReturnValue(caps(undefined));
    render(<UnlinkedBanner />);

    expect(screen.getByText(BANNER)).toBeTruthy();
  });

  it("hides on a linked node", () => {
    capabilities.mockReturnValue(caps({ enabled: true, state: "connected", slug: "gpu-box" }));
    render(<UnlinkedBanner />);

    expect(screen.queryByText(BANNER)).toBeNull();
  });

  it("hides on a linked node whose tunnel is merely reconnecting", () => {
    // The banner keys off enrolment, never connection health: a `backoff`
    // episode is the doctor's and DaemonStatusBanner's story, not "unlinked".
    capabilities.mockReturnValue(caps({ enabled: true, state: "backoff", slug: "gpu-box" }));
    render(<UnlinkedBanner />);

    expect(screen.queryByText(BANNER)).toBeNull();
  });

  it("hides while the capabilities query is still in flight", () => {
    // `data` is undefined both while loading and on error; rendering off that
    // would flash "not linked" on every cold load of a linked node.
    capabilities.mockReturnValue({ isSuccess: false, data: undefined });
    render(<UnlinkedBanner />);

    expect(screen.queryByText(BANNER)).toBeNull();
  });

  it("no longer decides precedence against the daemon banner itself", () => {
    // The shell owns one banner slot with an ordered precedence list (daemon
    // offline > unlinked), so this component renders purely off enrolment and
    // must NOT read the daemon status — two components cross-referencing each
    // other's conditions is the arrangement the redesign removed. Pinned by
    // driving the offline case: the decision is made one level up now, and
    // `AppShell` never mounts this while the daemon is unreachable.
    capabilities.mockReturnValue(caps({ enabled: false }));
    daemonStatus.mockReturnValue({ status: "offline", lastSeen: new Date() });
    render(<UnlinkedBanner />);

    expect(screen.getByText(BANNER)).toBeTruthy();
    expect(daemonStatus).not.toHaveBeenCalled();
  });
});
