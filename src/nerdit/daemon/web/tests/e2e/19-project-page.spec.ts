import { expect, test } from "@playwright/test";
import {
  SAMPLE_APP_CONFIG,
  SAMPLE_CAPABILITIES,
  SAMPLE_SERVICES,
  SAMPLE_SERVICES_WITH_ASSO,
  loginAsToken,
  mockApi
} from "./fixtures";

// (P40e) /projects/:name is decided from the API, never by parsing a name: a
// single-service project renders the app page exactly as before (the first half
// of this file, unchanged), a multi-service project renders the project page
// with each service at /projects/:name/services/:service/:tab?, and a legacy
// deep link naming a service LABEL redirects there. The second half covers the
// project page: services, variables (a secret value never reaches the DOM),
// the non-owner view and the project delete.
//
// The app page (dashboard redesign W4). /projects/:name/:tab? aggregates a
// kind=service app across THREE tabs (Overview / Logs / Manage); the five-tab
// page's slugs (deployments / resources / settings) redirect to their new home
// so old bookmarks still resolve. The header carries one primary action —
// Deploy, source-aware — and everything else moved into the `…` menu. The
// activity view is gated on the PR1 audit_target_filter capability + an admin
// role, and the retired /services/:ident URL is a kind-aware redirect (D8).

test("apps table navigates to the app page with aggregated status and address", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.getByText("my-app", { exact: true }).click();
  await expect(page).toHaveURL("/projects/my-app");
  await expect(page.getByRole("heading", { name: "my-app" })).toBeVisible();

  // Aggregated status (D6): my-app healthy + bound to a ready model → running.
  await expect(page.getByTitle("App running.")).toBeVisible();
  // The address renders ONCE, in the header block.
  const address = page.getByTestId("app-address");
  await expect(address.getByText("https://test-host.nerdit.internal/my-app")).toBeVisible();
});

test("three tabs render and the legacy slugs redirect to their new home", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app");
  const tabs = page.getByRole("navigation", { name: "App sections" });
  for (const label of ["Overview", "Logs", "Manage"]) {
    await expect(tabs.getByRole("link", { name: label })).toBeVisible();
  }
  await expect(tabs.getByRole("link", { name: "Overview" })).toHaveAttribute("aria-current", "page");

  // Old bookmarks resolve: deployments → overview, resources/settings → manage.
  await page.goto("/projects/my-app/deployments");
  await expect(page).toHaveURL("/projects/my-app");
  await page.goto("/projects/my-app/settings");
  await expect(page).toHaveURL("/projects/my-app/manage");
  await page.goto("/projects/my-app/resources");
  await expect(page).toHaveURL("/projects/my-app/manage");
  await expect(page.getByRole("heading", { name: "AI resources" })).toBeVisible();
});

test("overview shows the deploy outcome, the resources summary and recent activity", async ({
  page
}) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app");

  // The last deploy is ONE line, not a phase chip repeated across the page.
  await expect(page.getByTestId("app-deploy-line")).toContainText("Deployed v3");

  const resources = page
    .locator("section")
    .filter({ has: page.getByRole("heading", { name: "Resources" }) });
  await expect(resources.getByText("llama3.1:8b")).toBeVisible();
  await expect(resources.getByText("Model running, weights pulled.")).toBeVisible();

  // Admin + the audit_target_filter flag (fixtures) → the activity feed renders.
  const activity = page
    .locator("section")
    .filter({ has: page.getByRole("heading", { name: "Recent activity" }) });
  await expect(activity.getByText("deploy.git_create")).toBeVisible();
  await expect(activity.getByRole("link", { name: "All activity" })).toBeVisible();
});

test("without the audit flag the overview hints upgrade and fires no audit query", async ({
  page
}) => {
  await mockApi(page, {
    capabilities: {
      ...SAMPLE_CAPABILITIES,
      features: { ...SAMPLE_CAPABILITIES.features, audit_target_filter: false }
    }
  });
  await loginAsToken(page);

  let targetAuditRequested = false;
  page.on("request", (r) => {
    if (r.url().includes("/api/audit") && r.url().includes("target=")) targetAuditRequested = true;
  });

  await page.goto("/projects/my-app");
  // The rest of the overview still renders (not audit-gated).
  await expect(page.getByTestId("app-deploy-line")).toBeVisible();
  await expect(page.getByText(/activity needs a newer daemon/i)).toBeVisible();

  await page.waitForTimeout(600);
  expect(targetAuditRequested).toBe(false);
});

test("an audit 403 (defense in depth) shows the admin hint, the overview intact", async ({
  page
}) => {
  await mockApi(page, { auditForbidden: true });
  await loginAsToken(page);

  await page.goto("/projects/my-app");
  await expect(page.getByTestId("app-deploy-line")).toBeVisible();
  await expect(page.getByText(/activity needs an admin token/i)).toBeVisible();
});

test("manage tab: a managed db binding renders and edits to external write secret then ref", async ({
  page
}) => {
  const cfg = {
    ...SAMPLE_APP_CONFIG,
    db: { default: { provider: "managed", database: "pg", url: null, password: null } }
  };
  await mockApi(page, { appConfig: cfg });
  await loginAsToken(page);

  await page.goto("/projects/my-app/manage");

  const dbCard = page
    .locator("section")
    .filter({ has: page.getByRole("heading", { name: "Database resources" }) });
  const card = dbCard.getByTestId("binding-card-default");

  // Managed binding is ready against the live "pg" row (SAMPLE_DATABASES).
  await expect(card.getByText("pg")).toBeVisible();
  await expect(card.getByText("Database running, ready.")).toBeVisible();
  await expect(card.getByText("NERDIT_DB_DEFAULT_URL")).toBeVisible();

  await card.getByRole("button", { name: /edit/i }).click();
  await card.getByRole("button", { name: "External", exact: true }).click();
  await card
    .getByPlaceholder("postgresql://user@host:5432/db")
    .fill("postgresql://db.example.com:5432/app");

  await card.getByLabel("Paste a new key").check();
  // The derived password secret name pre-fills (second input in the paste block).
  const nameInput = card.locator("div.pl-6 input").nth(1);
  await expect(nameInput).toHaveValue("DB_DEFAULT_PASSWORD");
  await card.getByPlaceholder("sk-...").fill("pg-secret-123");

  const secretsReq = page.waitForRequest(
    (r) => r.method() === "POST" && r.url().includes("/api/secrets/my-app")
  );
  const putReq = page.waitForRequest(
    (r) =>
      r.method() === "PUT" &&
      r.url().includes("/config/apps/my-app/db") &&
      r.url().includes("restart=true") &&
      !r.url().includes("dry_run")
  );

  await card.getByRole("button", { name: /^save$/i }).click();

  const secrets = await secretsReq;
  expect(secrets.postDataJSON()).toEqual({ values: { DB_DEFAULT_PASSWORD: "pg-secret-123" } });

  const put = await putReq;
  const body = put.postDataJSON();
  expect(body.default.provider).toBe("external");
  expect(body.default.url).toBe("postgresql://db.example.com:5432/app");
  expect(body.default.password).toBe("${secrets.DB_DEFAULT_PASSWORD}");
  // The raw password is never a literal in the db section.
  expect(put.postData() ?? "").not.toContain("pg-secret-123");

  await expect(page.getByText(/resource default saved/i)).toBeVisible();
});

test("manage tab hides identity trivia behind the Details disclosure", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app/manage");

  const details = page.getByTestId("app-details");
  await expect(details.getByText("svc1aaaa0001")).toHaveCount(0);
  await details.getByRole("button", { name: "Show" }).click();
  await expect(details.getByText("svc1aaaa0001")).toBeVisible();
  await expect(details.getByText("nerdit-app/my-app:3")).toBeVisible();
});

test("Deploy on a git-sourced app confirms, then redeploys from the recorded source", async ({
  page
}) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app");

  const redeployReq = page.waitForRequest(
    (r) => r.method() === "POST" && r.url().includes("/api/deploy/my-app/redeploy")
  );

  await page.getByTestId("app-deploy").click();
  const confirm = page.getByTestId("app-deploy-confirm");
  await expect(confirm).toContainText("keeps serving");
  await page.getByRole("dialog").getByRole("button", { name: "Deploy", exact: true }).click();

  await redeployReq;
  // The same phase walk a dialog-triggered deploy shows.
  await expect(page.getByTestId("deploy-progress-dialog")).toBeVisible();
  await expect(page.getByTestId("deploy-phases")).toBeVisible();
});

test("Deploy on a workspace-sourced app hits the workspace endpoint, never /redeploy", async ({
  page
}) => {
  // The redeploy route 409s deploy.no_source on workspace rows (D-P29-7), so
  // the header Deploy must take POST /workspaces/{name}/deploy instead
  // (Codex review, PR #147).
  const services = JSON.parse(JSON.stringify(SAMPLE_SERVICES));
  services.items[0].source = { type: "workspace" };
  await mockApi(page, { services });
  await loginAsToken(page);

  let redeployRequested = false;
  page.on("request", (r) => {
    if (r.url().includes("/redeploy")) redeployRequested = true;
  });
  const workspaceReq = page.waitForRequest(
    (r) => r.method() === "POST" && r.url().includes("/api/workspaces/my-app/deploy")
  );

  await page.goto("/projects/my-app");
  await page.getByTestId("app-deploy").click();
  const confirm = page.getByTestId("app-deploy-confirm");
  await expect(confirm).toContainText("server-side workspace files");
  await page.getByRole("dialog").getByRole("button", { name: "Deploy", exact: true }).click();

  await workspaceReq;
  await expect(page.getByTestId("deploy-progress-dialog")).toBeVisible();
  expect(redeployRequested).toBe(false);
});

test("Deploy on a zip-sourced app opens the deploy dialog instead", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  let redeployRequested = false;
  page.on("request", (r) => {
    if (r.url().includes("/redeploy")) redeployRequested = true;
  });

  await page.goto("/projects/worker-api");
  await page.getByTestId("app-deploy").click();

  await expect(page.getByTestId("deploy-dialog")).toBeVisible();
  expect(redeployRequested).toBe(false);
});

test("the delete dialog states what survives and purges only secrets by default", async ({
  page
}) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app");

  const deleteReq = page.waitForRequest(
    (r) => r.method() === "DELETE" && r.url().includes("/api/services/my-app")
  );

  await page.getByRole("button", { name: "More actions" }).click();
  await page.getByRole("menuitem", { name: /^Delete/ }).click();

  const dialog = page.getByTestId("app-delete-confirm");
  await expect(dialog).toContainText("secrets are deleted");
  await expect(dialog).toContainText("Kept: data volumes, built images.");
  // No typed confirm without a data purge.
  await expect(page.getByTestId("confirm-phrase-input")).toHaveCount(0);

  await page.getByRole("dialog").getByRole("button", { name: "Delete", exact: true }).click();

  const del = await deleteReq;
  expect(new URL(del.url()).searchParams.get("purge")).toBe("secrets");
  await expect(page).toHaveURL("/");
});

test("deleting the data volumes requires typing the app name", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app");

  await page.getByRole("button", { name: "More actions" }).click();
  await page.getByRole("menuitem", { name: /^Delete/ }).click();

  const dialog = page.getByRole("dialog");
  await dialog.getByLabel("Also delete data volumes").check();
  await expect(page.getByTestId("app-delete-confirm")).toContainText("Kept: built images.");
  await expect(page.getByTestId("app-delete-confirm")).toContainText("cannot be undone");

  const confirmButton = dialog.getByRole("button", { name: "Delete", exact: true });
  await expect(confirmButton).toBeDisabled();

  const deleteReq = page.waitForRequest(
    (r) => r.method() === "DELETE" && r.url().includes("/api/services/my-app")
  );
  await page.getByTestId("confirm-phrase-input").fill("my-app");
  await confirmButton.click();

  const del = await deleteReq;
  expect(new URL(del.url()).searchParams.get("purge")).toBe("secrets,data");
});

test("the /services/:ident redirect is kind-aware", async ({ page }) => {
  const modelRow = {
    ...SAMPLE_SERVICES.items[0],
    id: "mdl1cccc0003",
    name: "ollama-llama3-1-8b",
    kind: "model"
  };
  const dbRow = { ...SAMPLE_SERVICES.items[0], id: "db1aaaa0001", name: "pg", kind: "database" };
  await mockApi(page, {
    services: { items: [SAMPLE_SERVICES.items[0], modelRow, dbRow], next_cursor: null }
  });
  await loginAsToken(page);

  await page.goto("/services/my-app");
  await expect(page).toHaveURL("/projects/my-app");

  await page.goto("/services/ollama-llama3-1-8b");
  await expect(page).toHaveURL("/models");

  await page.goto("/services/pg");
  await expect(page).toHaveURL("/databases");
});

test("the /services/:ident redirect lands a project's service on its nested page", async ({
  page
}) => {
  await mockApi(page, { services: SAMPLE_SERVICES_WITH_ASSO });
  await loginAsToken(page);

  await page.goto("/services/api--asso");
  await expect(page).toHaveURL("/projects/asso/services/api");
  // The home service's label IS the project name: it lands on the project.
  await page.goto("/services/asso");
  await expect(page).toHaveURL("/projects/asso");
});

test("an unknown /services/:ident redirects to the apps list with a toast", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/services/unknown-name");
  await expect(page).toHaveURL("/");
  await expect(page.getByText(/app not found/i)).toBeVisible();
});

// --- P40e: the project page --------------------------------------------------

const ASSO_VARIABLES = {
  asso: [
    { key: "PUBLIC_URL", scope: "project", plain: true, value: "https://asso.example" },
    { key: "SESSION_KEY", scope: "project", plain: false },
    { key: "API_TOKEN", scope: "production/api", plain: false }
  ]
};

test("a multi-service project lists both services and links each to its own page", async ({
  page
}) => {
  await mockApi(page, { services: SAMPLE_SERVICES_WITH_ASSO });
  await loginAsToken(page);

  await page.goto("/projects/asso");
  const project = page.getByTestId("project-page");
  await expect(project.getByRole("heading", { name: "asso", level: 1 })).toBeVisible();
  await expect(project).toContainText("Project · 2 services");

  // Rows are named by the `service` FIELD (`web`, `api`), never the label.
  const services = page.getByTestId("project-services");
  await expect(services.getByTestId("app-row-web")).toBeVisible();
  await expect(services.getByTestId("app-row-api")).toBeVisible();
  await expect(services.getByText("api--asso", { exact: true })).toHaveCount(0);
  await expect(services.getByText("Running · unhealthy")).toBeVisible();

  // Referenced resources and every advertised address, one call's worth.
  const resources = project
    .locator("section")
    .filter({ has: page.getByRole("heading", { name: "Resources" }) });
  await expect(resources.getByText("llama3.1:8b").first()).toBeVisible();
  await expect(
    page.getByTestId("project-addresses").getByRole("link", {
      name: "https://test-host.nerdit.internal/asso"
    })
  ).toBeVisible();

  // The service page talks to the daemon by LABEL (D-P40-6) under a field-built URL.
  const labelRead = page.waitForRequest(
    (r) => r.method() === "GET" && new URL(r.url()).pathname === "/api/services/api--asso"
  );
  await services.getByTestId("app-row-api").click();
  await expect(page).toHaveURL("/projects/asso/services/api");
  await labelRead;
  await expect(page.getByRole("heading", { name: "api", level: 1 })).toBeVisible();

  // Its tabs live under the nested path, and the way back up is the project.
  const tabs = page.getByRole("navigation", { name: "App sections" });
  await tabs.getByRole("link", { name: "Manage" }).click();
  await expect(page).toHaveURL("/projects/asso/services/api/manage");
  await expect(page.getByRole("heading", { name: "AI resources" })).toBeVisible();
  await page.getByTestId("app-project-link").click();
  await expect(page).toHaveURL("/projects/asso");
  await expect(page.getByTestId("project-page")).toBeVisible();
});

test("a legacy deep link naming a service label redirects into its project", async ({ page }) => {
  await mockApi(page, { services: SAMPLE_SERVICES_WITH_ASSO });
  await loginAsToken(page);

  await page.goto("/projects/api--asso");
  await expect(page).toHaveURL("/projects/asso/services/api");
  await expect(page.getByRole("heading", { name: "api", level: 1 })).toBeVisible();

  // The tab segment rides along.
  await page.goto("/projects/api--asso/logs");
  await expect(page).toHaveURL("/projects/asso/services/api/logs");
  await expect(
    page.getByRole("navigation", { name: "App sections" }).getByRole("link", { name: "Logs" })
  ).toHaveAttribute("aria-current", "page");

  // A service the project does not have falls back to the project page.
  await page.goto("/projects/asso/services/ghost");
  await expect(page).toHaveURL("/projects/asso");
});

test("a single-service project keeps the app page, with the project one link away", async ({
  page
}) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/projects/my-app");
  // Today's page: the app's name, its three tabs, no project chrome on top.
  await expect(page.getByRole("heading", { name: "my-app", level: 1 })).toBeVisible();
  await expect(page.getByTestId("project-page")).toHaveCount(0);
  await expect(page.getByTestId("app-deploy-line")).toContainText("Deployed v3");

  await page.goto("/projects/my-app/manage");
  await page.getByTestId("app-project-link").click();
  await expect(page).toHaveURL("/projects/my-app/project");
  const project = page.getByTestId("project-page");
  await expect(project).toContainText("Project · 1 service");
  await expect(page.getByTestId("project-variables")).toBeVisible();

  // Its one service row leads back to the app page, not to a nested URL.
  await page.getByTestId("project-services").getByTestId("app-row-web").click();
  await expect(page).toHaveURL("/projects/my-app");
});

test("variables render by scope: a plain value is shown, a secret is only a name", async ({
  page
}) => {
  await mockApi(page, { services: SAMPLE_SERVICES_WITH_ASSO, variables: ASSO_VARIABLES });
  await loginAsToken(page);

  await page.goto("/projects/asso");
  const panel = page.getByTestId("project-variables");
  await expect(panel.getByRole("heading", { name: "Project", exact: true })).toBeVisible();
  await expect(panel.getByRole("heading", { name: "Service api" })).toBeVisible();
  // `web` has no variable of its own: no empty group.
  await expect(panel.getByRole("heading", { name: "Service web" })).toHaveCount(0);

  const plain = panel.getByTestId("variable-row-PUBLIC_URL");
  await expect(plain).toContainText("https://asso.example");
  await expect(plain.getByText("plain", { exact: true })).toBeVisible();

  for (const key of ["SESSION_KEY", "API_TOKEN"]) {
    const row = panel.getByTestId(`variable-row-${key}`);
    await expect(row.getByText("secret", { exact: true })).toBeVisible();
    await expect(row).toContainText("••••••");
  }
});

test("a secret value never reaches the DOM; plain and secret are two PUTs with their flag", async ({
  page
}) => {
  const PLAIN = "plain-value-visible-7c1d";
  const SECRET = "s3cr3t-sentinel-never-rendered-9f4e2b";

  const api = await mockApi(page, { services: SAMPLE_SERVICES_WITH_ASSO });
  // Everything that could carry the value out of the input: URLs and the console.
  const urls: string[] = [];
  const logs: string[] = [];
  page.on("request", (r) => urls.push(r.url()));
  page.on("console", (m) => logs.push(m.text()));
  await loginAsToken(page);

  await page.goto("/projects/asso");
  const panel = page.getByTestId("project-variables");
  await expect(panel.getByText("No variables set.")).toBeVisible();

  // The plain editor: project scope.
  const plainForm = panel.getByTestId("variable-plain-form");
  await plainForm.getByLabel("Plain key").fill("PUBLIC_URL");
  await plainForm.getByLabel("Plain value").fill(PLAIN);
  await plainForm.getByRole("button", { name: "Save plain" }).click();
  await expect(panel.getByTestId("variable-row-PUBLIC_URL")).toContainText(PLAIN);

  // The secret intake: a hidden input, scoped to the `api` service.
  await panel.getByLabel("Scope").selectOption("api");
  const secretForm = panel.getByTestId("variable-secret-form");
  const secretInput = secretForm.getByLabel("Secret value");
  await expect(secretInput).toHaveAttribute("type", "password");
  await secretForm.getByLabel("Secret key").fill("API_TOKEN");
  await secretInput.fill(SECRET);
  await secretForm.getByRole("button", { name: "Save secret" }).click();

  // Saved: the NAME is listed under its scope, the input is cleared.
  const row = panel.getByTestId("variable-row-API_TOKEN");
  await expect(row.getByText("secret", { exact: true })).toBeVisible();
  await expect(secretInput).toHaveValue("");
  await expect(secretForm.getByLabel("Secret key")).toHaveValue("");

  // Two SEPARATE calls, each carrying its own flag (the audit rows need it).
  expect(api.variablePuts).toEqual([
    { service: null, body: { values: { PUBLIC_URL: PLAIN }, secret: false } },
    { service: "api", body: { values: { API_TOKEN: SECRET }, secret: true } }
  ]);

  // The sentinel is nowhere: not in the markup, not in any input's live value,
  // not in a URL, the console, or either browser store.
  expect(await page.content()).not.toContain(SECRET);
  const leaked = await page.evaluate((needle) => {
    const inputs = [...document.querySelectorAll("input, textarea")].some((el) =>
      (el as HTMLInputElement).value.includes(needle)
    );
    const stores = [localStorage, sessionStorage].some((store) =>
      JSON.stringify({ ...store }).includes(needle)
    );
    return inputs || stores || (document.body.textContent ?? "").includes(needle);
  }, SECRET);
  expect(leaked).toBe(false);
  expect(urls.filter((url) => url.includes(SECRET))).toEqual([]);
  // D-P40-11: the scope is a `service` param; nothing is ever named `environment`
  // (the exact-key `toEqual` above already pins the bodies).
  expect(urls.filter((url) => /environment/i.test(url))).toEqual([]);
  expect(logs.filter((line) => line.includes(SECRET))).toEqual([]);

  // A poll later (the project refetches every 5 s) it is still only a name.
  await page.reload();
  await expect(page.getByTestId("variable-row-API_TOKEN")).toContainText("••••••");
  expect(await page.content()).not.toContain(SECRET);
});

test("switching scope or leaving the page drops a typed, unsaved secret", async ({ page }) => {
  const SECRET = "typed-but-never-saved-51aa";
  const api = await mockApi(page, { services: SAMPLE_SERVICES_WITH_ASSO });
  await loginAsToken(page);

  await page.goto("/projects/asso");
  const panel = page.getByTestId("project-variables");
  const secretInput = panel.getByTestId("variable-secret-form").getByLabel("Secret value");
  await secretInput.fill(SECRET);
  await panel.getByLabel("Scope").selectOption("api");
  await expect(secretInput).toHaveValue("");

  await secretInput.fill(SECRET);
  await page.getByTestId("project-services").getByTestId("app-row-api").click();
  await expect(page).toHaveURL("/projects/asso/services/api");
  await page.goBack();
  await expect(panel.getByTestId("variable-secret-form").getByLabel("Secret value")).toHaveValue("");
  expect(api.variablePuts).toEqual([]);
});

test("deleting a variable confirms, names its scope, and sends the service param", async ({
  page
}) => {
  await mockApi(page, { services: SAMPLE_SERVICES_WITH_ASSO, variables: ASSO_VARIABLES });
  await loginAsToken(page);

  await page.goto("/projects/asso");
  const panel = page.getByTestId("project-variables");
  const deleteReq = page.waitForRequest((r) => r.method() === "DELETE");
  await panel.getByRole("button", { name: "Delete API_TOKEN" }).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toContainText("service api");
  await dialog.getByRole("button", { name: "Delete", exact: true }).click();

  const url = new URL((await deleteReq).url());
  expect(url.pathname).toBe("/api/projects/asso/variables/API_TOKEN");
  expect(url.searchParams.get("service")).toBe("api");
  await expect(panel.getByTestId("variable-row-API_TOKEN")).toHaveCount(0);
  await expect(panel.getByTestId("variable-row-SESSION_KEY")).toBeVisible();
});

test("a non-owner sees the project without any variables UI, and no error page", async ({
  page
}) => {
  const api = await mockApi(page, {
    services: SAMPLE_SERVICES_WITH_ASSO,
    variables: ASSO_VARIABLES,
    projectOwner: false,
    authRole: "submitter"
  });
  await loginAsToken(page);

  await page.goto("/projects/asso");
  // The page is a page: services, resources and addresses all render.
  await expect(page.getByTestId("project-services").getByTestId("app-row-api")).toBeVisible();
  await expect(page.getByTestId("project-addresses")).toBeVisible();

  // No variables section, no intake, no owner-only delete — and nothing that
  // reads as a failure.
  await expect(page.getByTestId("project-variables")).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Variables" })).toHaveCount(0);
  await expect(page.getByLabel("Secret value")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "More actions" })).toHaveCount(0);
  await expect(page.getByText(/permission|forbidden|error/i)).toHaveCount(0);

  // It did not even ask: the omitted section is the answer (D-P40-15).
  expect(api.projectCalls.filter((call) => call.includes("/variables"))).toEqual([]);
});

test("a 403 on the variable values stays quiet: names render, no value, no error", async ({
  page
}) => {
  await mockApi(page, {
    services: SAMPLE_SERVICES_WITH_ASSO,
    variables: ASSO_VARIABLES,
    variablesReadForbidden: true
  });
  await loginAsToken(page);

  const denied = page.waitForResponse(
    (r) => r.url().includes("/api/projects/asso/variables") && r.status() === 403
  );
  await page.goto("/projects/asso");
  await denied;

  const row = page.getByTestId("variable-row-PUBLIC_URL");
  await expect(row).toBeVisible();
  await expect(row).not.toContainText("https://asso.example");
  await expect(page.getByTestId("project-services")).toBeVisible();
  await expect(page.getByText(/permission|forbidden/i)).toHaveCount(0);
});

test("a daemon with projects but without variables shows the project page, no variables UI", async ({
  page
}) => {
  const api = await mockApi(page, {
    services: SAMPLE_SERVICES_WITH_ASSO,
    variables: ASSO_VARIABLES,
    capabilities: {
      ...SAMPLE_CAPABILITIES,
      features: { ...SAMPLE_CAPABILITIES.features, variables: false }
    }
  });
  await loginAsToken(page);

  await page.goto("/projects/asso");
  await expect(page.getByTestId("project-services")).toBeVisible();
  await expect(page.getByTestId("project-variables")).toHaveCount(0);
  expect(api.projectCalls.filter((call) => call.includes("/variables"))).toEqual([]);
});

test("deleting a multi-service project needs its name typed, then returns to the list", async ({
  page
}) => {
  await mockApi(page, { services: SAMPLE_SERVICES_WITH_ASSO });
  await loginAsToken(page);

  await page.goto("/projects/asso");
  await page.getByRole("button", { name: "More actions" }).click();
  await page.getByRole("menuitem", { name: /^Delete project/ }).click();

  const dialog = page.getByRole("dialog");
  await expect(page.getByTestId("project-delete-confirm")).toContainText("its 2 services");
  const confirmButton = dialog.getByRole("button", { name: "Delete", exact: true });
  await expect(confirmButton).toBeDisabled();
  await page.getByTestId("confirm-phrase-input").fill("asso");

  const deleteReq = page.waitForRequest(
    (r) => r.method() === "DELETE" && new URL(r.url()).pathname === "/api/projects/prj_asso000000000003"
  );
  await confirmButton.click();

  const del = await deleteReq;
  expect(new URL(del.url()).searchParams.get("purge")).toBe("secrets");
  expect(del.headers()["idempotency-key"]).toBeTruthy();

  await expect(page).toHaveURL("/");
  await expect(page.getByText("Project asso deleted")).toBeVisible();
  await expect(page.getByTestId("app-row-asso")).toHaveCount(0);
  await expect(page.getByTestId("app-row-my-app")).toBeVisible();
});

test("the project delete carries the data purge only when asked", async ({ page }) => {
  await mockApi(page, { services: SAMPLE_SERVICES_WITH_ASSO });
  await loginAsToken(page);

  await page.goto("/projects/asso");
  await page.getByRole("button", { name: "More actions" }).click();
  await page.getByRole("menuitem", { name: /^Delete project/ }).click();

  const dialog = page.getByRole("dialog");
  await dialog.getByLabel("Also delete data volumes").check();
  await expect(page.getByTestId("project-delete-confirm")).toContainText("cannot be undone");
  await page.getByTestId("confirm-phrase-input").fill("asso");

  const deleteReq = page.waitForRequest(
    (r) => r.method() === "DELETE" && new URL(r.url()).pathname === "/api/projects/prj_asso000000000003"
  );
  await dialog.getByRole("button", { name: "Delete", exact: true }).click();
  expect(new URL((await deleteReq).url()).searchParams.get("purge")).toBe("secrets,data");
});

test("a read-only token sees variable names but no intake and no delete", async ({ page }) => {
  await mockApi(page, {
    services: SAMPLE_SERVICES_WITH_ASSO,
    variables: ASSO_VARIABLES,
    authRole: "readonly"
  });
  await loginAsToken(page);

  await page.goto("/projects/asso");
  const panel = page.getByTestId("project-variables");
  await expect(panel.getByTestId("variable-row-SESSION_KEY")).toBeVisible();
  await expect(panel.getByTestId("variable-secret-form")).toHaveCount(0);
  await expect(panel.getByTestId("variable-plain-form")).toHaveCount(0);
  await expect(panel.getByRole("button", { name: /^Delete / })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "More actions" })).toHaveCount(0);
});
