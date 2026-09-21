// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, expect, it, vi } from "vitest";
import ProjectRoute from "./ProjectPage";
import { ApiError } from "../api/client";

// A dataless errored query goes back to `pending` on every refetch; the service
// page a label-scoped token reached must stay mounted through it, draft included.
let hold = false;
let release: (() => void) | undefined;

vi.mock("./ProjectDetail", () => ({
  default: () => <input aria-label="Draft secret" type="password" />
}));

vi.mock("../api/client", async (original) => ({
  ...(await original<typeof import("../api/client")>()),
  api: vi.fn(async (path: string) => {
    if (path === "/capabilities") return { features: { projects: true } };
    if (path === "/services/api--asso") {
      return { name: "api--asso", kind: "service", project: "asso", service: "api" };
    }
    if (hold) {
      await new Promise<void>((resolve) => {
        release = resolve;
      });
    }
    throw new ApiError("scope denied", 403, "scope_denied");
  })
}));

afterEach(() => {
  release?.();
  cleanup();
});

it.each([["projects", "asso"], ["projects", "api--asso"]])(
  "keeps the label page and its draft mounted while %s/%s is re-checked",
  async (...queryKey) => {
    hold = false;
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={["/projects/api--asso/logs"]}>
          <Routes>
            <Route path="/projects/:name/:tab?" element={<ProjectRoute />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>
    );
    await waitFor(() => expect(client.getQueryState(["projects", "asso"])?.status).toBe("error"));
    const input = (await screen.findByLabelText("Draft secret")) as HTMLInputElement;
    fireEvent.change(input, { target: { value: "unsaved synthetic draft" } });

    hold = true;
    void client.refetchQueries({ queryKey, exact: true });
    await waitFor(() => expect(client.getQueryState(queryKey)?.status).toBe("pending"));
    expect(screen.getByLabelText("Draft secret")).toBe(input);
    expect(input.value).toBe("unsaved synthetic draft");
    release?.();
    client.clear();
  }
);
