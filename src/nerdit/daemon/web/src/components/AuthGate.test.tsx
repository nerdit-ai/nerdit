// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AuthGate } from "./AuthGate";
import { clearStoredToken, getStoredToken, handleAuthFailure, setSessionMode, storeToken } from "../lib/auth";

function mount(path = "/") {
  return render(<MemoryRouter initialEntries={[path]}><Routes>
    <Route path="/" element={<AuthGate><div>Dashboard</div></AuthGate>} />
    <Route path="/login" element={<AuthGate login><div>Local login</div></AuthGate>} />
  </Routes></MemoryRouter>);
}

afterEach(() => {
  cleanup();
  clearStoredToken();
  setSessionMode(undefined);
  vi.unstubAllGlobals();
});

describe("authenticated dashboard entry", () => {
  it("opens a tunnel session without writing a browser token or bearer", async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ok: true, role: "submitter", mode: "tunnel" })));
    vi.stubGlobal("fetch", fetch);
    mount();
    await screen.findByText("Dashboard");
    expect(getStoredToken()).toBeNull();
    expect(fetch.mock.calls[0][1].headers).toEqual({});
    handleAuthFailure();
    await screen.findByText("Session ended");
    expect(screen.queryByText("Local login")).toBeNull();
  });

  it("recognizes an existing proxy session even from the login URL", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation(() => Promise.resolve(
      new Response(JSON.stringify({ ok: true, role: "submitter", mode: "tunnel" }))
    )));
    mount("/login");
    await screen.findByText("Dashboard");
    expect(screen.queryByText("Local login")).toBeNull();
  });

  it("preserves local bearer login when the server requires it", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation(() => Promise.resolve(
      new Response(JSON.stringify({ code: "unauthenticated" }), { status: 401 })
    )));
    mount();
    await screen.findByText("Local login");
    expect(screen.queryByText("Dashboard")).toBeNull();
  });

  it("validates stored tokens instead of trusting their presence", async () => {
    storeToken("revoked-test-token", false);
    vi.stubGlobal("fetch", vi.fn().mockImplementation(() => Promise.resolve(
      new Response(JSON.stringify({ code: "invalid_token" }), { status: 403 })
    )));
    mount();
    await screen.findByText("Local login");
    expect(getStoredToken()).toBeNull();
  });

  it.each([401, 403])("offers recovery for expired Cloud session %s, never local token entry", async (status) => {
    const fetch = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify({ error: { code: "proxy_session_expired", message: "The session expired", status } }), { status }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ok: true, role: "submitter", mode: "tunnel" })));
    vi.stubGlobal("fetch", fetch);
    mount();
    await screen.findByText("Reopen this machine from Nerdit Cloud.");
    expect(screen.queryByText("Local login")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    await screen.findByText("Dashboard");
  });

  it("does not mount the dashboard when the auth probe is unavailable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    mount();
    await screen.findByRole("alert");
    expect(screen.queryByText("Dashboard")).toBeNull();
    expect(screen.queryByText("Local login")).toBeNull();
  });
});
