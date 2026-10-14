// @vitest-environment jsdom
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DashboardNavigation } from "./AppShell";

let session: unknown = null;
vi.mock("../api/client", async (original) => ({
  ...(await original<typeof import("../api/client")>()),
  api: vi.fn(async () => session)
}));

function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <DashboardNavigation pathname="/" />
      </MemoryRouter>
    </QueryClientProvider>
  );
}

afterEach(cleanup);

describe("dashboard navigation", () => {
  it("hides admin-only pages from a Cloud session", async () => {
    session = { ok: true, role: "submitter", mode: "tunnel" };
    mount();
    await waitFor(() => expect(screen.queryByText("Tokens")).toBeNull());
    expect(screen.queryByText("Activity")).toBeNull();
    expect(screen.getByText("Apps")).toBeTruthy();
  });

  it("shows them to an admin", async () => {
    session = { ok: true, role: "admin", mode: "token" };
    mount();
    expect(await screen.findByText("Tokens")).toBeTruthy();
    expect(screen.getByText("Activity")).toBeTruthy();
  });
});
