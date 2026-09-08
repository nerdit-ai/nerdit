import { expect, test } from "@playwright/test";
import { SAMPLE_APP_CONFIG, SAMPLE_CAPABILITIES, SAMPLE_SERVICES, loginAsToken, mockApi } from "./fixtures";

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

test("an unknown /services/:ident redirects to the apps list with a toast", async ({ page }) => {
  await mockApi(page);
  await loginAsToken(page);

  await page.goto("/services/unknown-name");
  await expect(page).toHaveURL("/");
  await expect(page.getByText(/app not found/i)).toBeVisible();
});
