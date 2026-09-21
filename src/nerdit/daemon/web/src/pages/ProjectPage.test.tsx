// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import ProjectRoute, { ServiceRoute } from "./ProjectPage";
import { ApiError } from "../api/client";

// The service detail is covered by the e2e suite; here it is a marker that
// says which label and base path the route resolved to.
vi.mock("./ProjectDetail", () => ({
  default: ({ label, basePath }: { label: string; basePath: string }) => (
    <p data-testid="detail">
      {label} @ {basePath}
    </p>
  )
}));

const calls: { path: string; init?: RequestInit }[] = [];
let routes: Record<string, unknown> = {};

vi.mock("../api/client", async (original) => ({
  ...(await original<typeof import("../api/client")>()),
  api: vi.fn(async (path: string, init?: RequestInit) => {
    calls.push({ path, init });
    if (init?.method === "PUT") return { project: "asso", scope: "project", keys: [], plain: false };
    if (routes[path] instanceof Error) throw routes[path];
    if (path in routes) return routes[path];
    throw new ApiError("not found", 404, "not_found");
  })
}));

const svc = (name: string, project: string | null, service: string | null) => ({
  id: name,
  name,
  project,
  service,
  kind: "service",
  status: "running",
  desired_state: "running",
  endpoint: null,
  source: null,
  last_deploy: null,
  started_at: null,
  created_at: null
});

const CAPS = { features: { projects: true, variables: true } };
const ASSO = {
  id: "prj_a",
  name: "asso",
  services: [svc("asso", "asso", "web"), svc("api--asso", "asso", "api")],
  addresses: [],
  resources: [
    { service: "api--asso", type: "db", binding: "default", provider: "external", target: null, ready: null }
  ],
  home: { hostname: "box", node_id: null },
  variables: [
    { key: "PUBLIC_URL", scope: "project", plain: true },
    { key: "API_KEY", scope: "production/api", plain: false }
  ]
};

function Where() {
  const location = useLocation();
  return <p data-testid="where">{location.pathname}</p>;
}

function Jump() {
  const navigate = useNavigate();
  return <button onClick={() => navigate("/projects/other")}>jump</button>;
}

function mount(path: string, cached?: { name: string }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  // A cached project swaps in with no loading state, so nothing unmounts by accident.
  if (cached) client.setQueryData(["projects", cached.name], cached);
  const view = render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <Jump />
        <Routes>
          <Route path="/projects/:name/services/:service/:tab?" element={<Where />} />
          <Route path="/projects/:name/:tab?" element={<ProjectRoute />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  );
  return { ...view, client };
}

afterEach(() => {
  cleanup();
  calls.length = 0;
  routes = {};
});

describe("ProjectRoute", () => {
  it.each([new ApiError("Temporarily unavailable", 503), new TypeError("Failed to fetch")])(
    "keeps an initial %s recoverable without falling back to the service route",
    async (error) => {
      routes = { "/capabilities": CAPS, "/projects/asso": error };
      mount("/projects/asso");
      expect(await screen.findByRole("alert")).toHaveProperty("textContent", error.message);
      expect(screen.queryByTestId("detail")).toBeNull();
      expect(calls.some((call) => call.path.startsWith("/services/"))).toBe(false);

      routes["/projects/asso"] = ASSO;
      fireEvent.click(screen.getByRole("button", { name: "Retry" }));
      expect(await screen.findByTestId("project-page")).toBeTruthy();
    }
  );

  it.each([403, 404])("falls back to the legacy service for a definitive %s", async (status) => {
    routes = { "/capabilities": CAPS, "/projects/asso": new ApiError("Unavailable", status) };
    mount("/projects/asso");
    expect((await screen.findByTestId("detail")).textContent).toBe("asso @ /projects/asso");
  });

  it("recovers on the next poll and preserves the project if a later poll fails", async () => {
    routes = { "/capabilities": CAPS, "/projects/asso": new ApiError("Try later", 503) };
    const { client } = mount("/projects/asso");
    await screen.findByRole("alert");
    routes["/projects/asso"] = ASSO;
    const page = await screen.findByTestId("project-page", {}, { timeout: 7000 });
    fireEvent.change(screen.getByLabelText("Secret value"), { target: { value: "pending-value" } });
    routes["/projects/asso"] = new ApiError("Try later", 503);
    await client.refetchQueries({ queryKey: ["projects", "asso"] });
    expect(client.getQueryState(["projects", "asso"])?.status).toBe("error");
    expect(screen.getByTestId("project-page")).toBe(page);
    expect((screen.getByLabelText("Secret value") as HTMLInputElement).value).toBe("pending-value");
  }, 10000);

  it("renders the project page: names by scope, the null-ready state, never a secret value", async () => {
    routes = {
      "/capabilities": CAPS,
      "/auth/check": { role: "admin" },
      "/projects/asso": ASSO,
      "/projects/asso/variables": {
        project: "asso",
        scope: "project",
        variables: [{ key: "PUBLIC_URL", scope: "project", plain: true, value: "https://asso.test" }]
      }
    };
    mount("/projects/asso");
    expect(await screen.findByTestId("project-page")).toBeTruthy();
    expect(await screen.findByText("= https://asso.test")).toBeTruthy();
    expect(screen.getByTestId("variable-row-API_KEY").textContent).toContain("secret");
    expect(screen.getByText("Unknown")).toBeTruthy();
    // The secret scope has no plain key, so its values are never even asked for.
    expect(calls.some((call) => call.path.includes("service=api"))).toBe(false);
  });

  it("shows no variables UI (and no delete) to a non-owner", async () => {
    const foreign: Partial<typeof ASSO> = { ...ASSO };
    delete foreign.variables; // OMITTED, not nulled (D-P40-15)
    routes = { "/capabilities": CAPS, "/auth/check": { role: "submitter" }, "/projects/asso": foreign };
    mount("/projects/asso");
    expect(await screen.findByTestId("project-page")).toBeTruthy();
    expect(screen.queryByTestId("project-variables")).toBeNull();
    expect(screen.queryByRole("button", { name: /save secret/i })).toBeNull();
  });

  it("writes plain and secret through separate PUTs and clears the hidden input", async () => {
    routes = { "/capabilities": CAPS, "/auth/check": { role: "admin" }, "/projects/asso": ASSO };
    mount("/projects/asso");
    const value = (await screen.findByLabelText("Secret value")) as HTMLInputElement;
    expect(value.type).toBe("password");
    fireEvent.change(screen.getByLabelText("Secret key"), { target: { value: "TOKEN" } });
    fireEvent.change(value, { target: { value: "s3cr3t" } });
    fireEvent.click(screen.getByRole("button", { name: /save secret/i }));
    await waitFor(() => expect(value.value).toBe(""));

    fireEvent.change(screen.getByLabelText("Plain key"), { target: { value: "MODE" } });
    fireEvent.change(screen.getByLabelText("Plain value"), { target: { value: "fast" } });
    fireEvent.click(screen.getByRole("button", { name: /save plain/i }));
    await waitFor(() => expect(calls.filter((call) => call.init?.method === "PUT")).toHaveLength(2));

    const bodies = calls
      .filter((call) => call.init?.method === "PUT")
      .map((call) => JSON.parse(String(call.init?.body)));
    expect(bodies).toEqual([
      { values: { TOKEN: "s3cr3t" }, secret: true },
      { values: { MODE: "fast" }, secret: false }
    ]);
    // Never in a URL, and never rendered back.
    expect(calls.every((call) => !call.path.includes("s3cr3t"))).toBe(true);
    expect(document.body.textContent).not.toContain("s3cr3t");
  });

  it("renders a single-service project as the service detail, as before", async () => {
    routes = {
      "/capabilities": CAPS,
      "/projects/my-app": { ...ASSO, name: "my-app", services: [svc("my-app", "my-app", "web")] }
    };
    mount("/projects/my-app/logs");
    expect((await screen.findByTestId("detail")).textContent).toBe("my-app @ /projects/my-app");
  });

  it("sends a service tab on a multi-service project to the home service, not the project page", async () => {
    routes = { "/capabilities": CAPS, "/auth/check": { role: "admin" }, "/projects/asso": ASSO };
    mount("/projects/asso/logs");
    expect((await screen.findByTestId("where")).textContent).toBe("/projects/asso/services/web/logs");
  });

  it("keeps a multi-service project with no home service on the project page", async () => {
    routes = {
      "/capabilities": CAPS,
      "/auth/check": { role: "admin" },
      "/projects/asso": { ...ASSO, services: [svc("api--asso", "asso", "api")] }
    };
    mount("/projects/asso/logs");
    expect(await screen.findByTestId("project-page")).toBeTruthy();
  });

  it("redirects a legacy label deep link by the row's FIELDS, keeping the tab", async () => {
    // The fields deliberately disagree with the label: a label parser would land elsewhere.
    routes = {
      "/capabilities": CAPS,
      "/services/api--asso": svc("api--asso", "other", "worker"),
      "/projects/other": { ...ASSO, name: "other" }
    };
    mount("/projects/api--asso/logs");
    expect((await screen.findByTestId("where")).textContent).toBe(
      "/projects/other/services/worker/logs"
    );
  });

  it("keeps a label-scoped token on its service page when the project is out of scope", async () => {
    // `/projects/asso` is not in `routes`: the project read fails, the row's does not.
    routes = { "/capabilities": CAPS, "/services/api--asso": svc("api--asso", "asso", "api") };
    mount("/projects/api--asso/logs");
    // The detail also renders while the row loads: wait for the project's refusal.
    await waitFor(() => expect(calls.some((call) => call.path === "/projects/asso")).toBe(true));
    expect((await screen.findByTestId("detail")).textContent).toBe(
      "api--asso @ /projects/api--asso"
    );
    expect(screen.queryByTestId("where")).toBeNull();
  });

  it("drops a typed secret when the route swaps to another cached project", async () => {
    routes = {
      "/capabilities": CAPS,
      "/auth/check": { role: "admin" },
      "/projects/asso": ASSO,
      "/projects/other": { ...ASSO, name: "other" }
    };
    mount("/projects/asso", { ...ASSO, name: "other" });
    fireEvent.change(await screen.findByLabelText("Secret value"), {
      target: { value: "old-project-secret" }
    });
    fireEvent.click(screen.getByText("jump"));
    await screen.findByRole("heading", { name: "other" });
    expect((screen.getByLabelText("Secret value") as HTMLInputElement).value).toBe("");
  });

  it("keeps today's page on a daemon without features.projects", async () => {
    routes = { "/capabilities": { features: {} } };
    mount("/projects/my-app");
    expect((await screen.findByTestId("detail")).textContent).toBe("my-app @ /projects/my-app");
    expect(calls.some((call) => call.path.startsWith("/projects"))).toBe(false);
  });
});

describe("ServiceRoute", () => {
  it("keeps the service page mounted when a background project poll fails", async () => {
    routes = { "/projects/asso": ASSO };
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={["/projects/asso/services/api"]}>
          <Routes>
            <Route path="/projects/:name/services/:service/:tab?" element={<ServiceRoute />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>
    );
    const detail = await screen.findByTestId("detail");
    expect(detail.textContent).toBe("api--asso @ /projects/asso/services/api");

    routes = {}; // the next poll 404s: error set, data kept
    await client.refetchQueries({ queryKey: ["projects", "asso"] });
    expect(client.getQueryState(["projects", "asso"])?.status).toBe("error");
    expect(screen.getByTestId("detail")).toBe(detail);
  });
});
